import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_aoi.gnn.features import N_FEATURES

NEG = -1e9


def _row_norm(A):
    return A / (A.sum(-1, keepdim=True) + 1e-6)


class GraphLayer(nn.Module):
    """Dense message passing over two weighted graphs plus a global context
    term. Dense adjacency is the right call at these sizes (M <= ~100):
    one batched matmul instead of scatter ops."""

    def __init__(self, d):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h, A1, A2):
        ctx = h.mean(1, keepdim=True).expand_as(h)
        m = self.msg(torch.cat([h, A1 @ h, A2 @ h, ctx], -1))
        return self.norm(h + m)


class GraphEncoder(nn.Module):
    def __init__(self, d=64, layers=3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(N_FEATURES, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList(GraphLayer(d) for _ in range(layers))

    def forward(self, X, A_corr, A_post):
        A1, A2 = _row_norm(A_corr), _row_norm(A_post)
        h = self.inp(X)
        for layer in self.layers:
            h = layer(h, A1, A2)
        return h


class Actor(nn.Module):
    """GNN encoder + residual autoregressive pointer decoder.

    The decoder picks sensors one at a time. Sensor i's logit at each step is

        alpha * log(gain_i)  +  f_theta(h)_i  +  sum_{j picked} G_ij

    where gain_i is the exact drop in total posterior variance if i were
    polled next (recomputed by a Kalman conditioning step after every pick),
    f_theta is a learned per-sensor correction from the GNN embedding, and G
    is a learned query/key interaction between sensors. With f_theta = G = 0
    and large alpha this is exactly Kalman-greedy, and that is how the actor is
    initialised. RL then only has to learn *when deviating from the myopic
    greedy choice pays off later* (e.g. polling a sensor before the freshness
    guard forces it at a bad moment).

    The probability of an ordered pick list is a product of softmaxes
    (Plackett-Luce), which gives exact log-probs for PPO.
    """

    def __init__(self, d=64, layers=3, alpha=4.0):
        super().__init__()
        self.enc = GraphEncoder(d, layers)
        self.base = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.q = nn.Linear(d, d // 2)
        self.k = nn.Linear(d, d // 2)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))
        self.dk = d // 2
        for lin in (self.base[-1], self.q):  # start as pure Kalman-greedy
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def forward(self, batch):
        h = self.enc(batch.X, batch.A_corr, batch.A_post)
        corr = self.base(h).squeeze(-1)
        G = self.q(h) @ self.k(h).transpose(1, 2) / math.sqrt(self.dk)
        return corr, G

    def decode(self, batch, actions=None, greedy=False, generator=None):
        """Pick up to batch.n_pick[b] sensors per row (forced ones are
        already committed). If `actions` (B,K) is given, score those picks
        instead of choosing. Returns picks (B,K) padded with -1,
        log-prob (B,), entropy (B,)."""
        corr, G = self(batch)
        B, M = corr.shape
        conflict, forced, n_pick, r = batch.conflict, batch.forced, batch.n_pick, batch.r
        P = batch.P.clone()
        K = int(n_pick.max().item()) if actions is None else actions.shape[1]
        corr = corr + (G * forced[:, None, :].float()).sum(-1)
        avail = ~forced & ~(conflict & forced[:, :, None]).any(1)
        picks = torch.full((B, K), -1, dtype=torch.long)
        logp = torch.zeros(B)
        ent = torch.zeros(B)
        rows = torch.arange(B)
        for k in range(K):
            active = (k < n_pick) & avail.any(1)
            if actions is not None:
                active = active & (actions[:, k] >= 0)
            if not active.any():
                break
            diag = torch.diagonal(P, dim1=1, dim2=2)
            gain = (P**2).sum(1) / (diag + r[:, None])
            logits = self.alpha * torch.log(gain.clamp(min=1e-8)) + corr
            masked = logits.masked_fill(~avail, NEG)
            lp = F.log_softmax(masked, -1)
            if k == 0:  # first-pick distribution, kept for explanations / the live demo
                self.last_priority = lp.detach().exp()
            if actions is not None:
                a = actions[:, k].clamp(min=0)
            elif greedy:
                a = masked.argmax(-1)
            else:
                a = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(-1)
            step_ent = -(lp.exp() * lp.masked_fill(~avail, 0.0)).sum(-1)
            zero = torch.zeros_like(logp)
            logp = logp + torch.where(active, lp[rows, a], zero)
            ent = ent + torch.where(active, step_ent, zero)
            picks[:, k] = torch.where(active, a, torch.full_like(a, -1))
            # commit the pick for active rows
            act = active[:, None]
            avail = avail & ~(F.one_hot(a, M).bool() & act) & ~(conflict[rows, a] & act)
            corr = corr + G[rows, :, a] * act.float()
            pa = P[rows, :, a]
            upd = pa[:, :, None] * pa[:, None, :] / (pa[rows, a] + r)[:, None, None]
            P = P - upd * act[:, :, None].float()
        return picks, logp, ent


class Critic(nn.Module):
    def __init__(self, d=64, layers=2):
        super().__init__()
        self.enc = GraphEncoder(d, layers)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, batch):
        h = self.enc(batch.X, batch.A_corr, batch.A_post)
        return self.head(torch.cat([h.mean(1), h.max(1).values], -1)).squeeze(-1)
