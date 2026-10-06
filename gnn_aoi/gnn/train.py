"""Train the GNN scheduler with PPO on the true objective.

The actor is initialised to reproduce Kalman-greedy exactly (see
model.Actor), so training starts from the strongest model-based baseline
and only has to learn where looking ahead beats it.

Training mixes two sources:

* synthetic fields with randomised size (10-100 sensors), capacity,
  correlation length, per-sensor dynamics, layout, freshness bound and
  interference; reward = -log(true posterior variance);
* replays of the real datasets' *training period*, on sensor sub-networks;
  the sink's model is fitted on the first 70% of that period and episodes
  run on the next 20%, scored on the actual estimation error, so the policy
  learns to cope with a model that does not match reality. The last 10% is
  validation. Test periods / test layouts (seeds >= 10_000) are never seen.
"""
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from gnn_aoi.config import Scenario
from gnn_aoi.env import SchedulingEnv
from gnn_aoi.gnn.features import Batch, featurize
from gnn_aoi.gnn.model import Actor, Critic
from gnn_aoi.selection import greedy_variance_reduction

VAL_SEED_BASE = 9_000  # validation layouts: 9_000..; training draws from [0, 9_000)


@dataclass
class TrainConfig:
    seed: int = 0
    n_envs: int = 32               # at M = 40; scaled down for larger networks
    rollout_len: int = 96
    ppo_iters: int = 500
    critic_warmup: int = 15
    lr_actor: float = 1e-4
    lr_critic: float = 5e-4
    gamma: float = 0.95
    lam: float = 0.9
    clip: float = 0.15
    epochs: int = 4
    minibatch: int = 512
    ent_coef: float = 0.002
    max_grad: float = 0.5
    M_range: tuple = (10, 100)
    real_frac: float = 0.35        # share of iterations that replay real training data
    real_datasets: tuple = ("intel_lab", "pems_bay")
    d: int = 64
    layers: int = 3
    eval_every: int = 10
    threads: int = 8
    out: str = "checkpoints/gnn_scheduler.pt"
    init_from: str = ""            # warm-start the actor from a saved checkpoint
    log: str = "results/data/training_log.json"
    extra: dict = field(default_factory=dict)


FIT, TUNE = 0.7, 0.9  # fractions of a real dataset's training period


def sample_capacity(rng, M):
    return max(1, int(round(M * rng.uniform(0.08, 0.35))))


def sample_scenario(rng, M):
    C = sample_capacity(rng, M)
    return Scenario(
        M=M, C=C,
        length_scale=float(np.exp(rng.uniform(np.log(0.05), np.log(0.45)))),
        ar_rho=float(rng.uniform(0.6, 0.995)),
        rho_spread=float(rng.uniform(0, 0.2)) if rng.random() < 0.6 else 0.0,
        layout="clustered" if rng.random() < 0.4 else "uniform",
        n_clusters=int(rng.integers(2, 8)),
        freshness=int(3 * M / C) if rng.random() < 0.7 else 0,
        interference_radius=float(rng.uniform(0.05, 0.25)) * np.sqrt(20 / M) if rng.random() < 0.2 else 0.0,
    )


def nearby_sensors(ds, M, rng):
    """A geographically compact sub-network: a random sensor and its M-1
    nearest neighbours."""
    if M >= ds.M:
        return np.arange(ds.M)
    c = rng.integers(ds.M)
    d = np.linalg.norm(ds.positions - ds.positions[c], axis=1)
    return np.sort(np.argsort(d)[:M])


def real_env(ds, M, C, freshness, start, rng=None, sensors=None):
    if sensors is None:
        sensors = nearby_sensors(ds, M, rng)
    sc = Scenario(M=len(sensors), C=C, freshness=freshness)
    field = ds.field(start, sensors, train_slice=slice(0, int(FIT * ds.train_end)))
    return SchedulingEnv(sc, field=field)


def validation_envs():
    """Fixed validation set, disjoint from training and test: synthetic
    layouts in the validation seed range, and the last 10% of each real
    dataset's training period."""
    from gnn_aoi.datasets import load
    base = Scenario(freshness=12)
    scs = [base, base.with_(layout="clustered"), base.with_(rho_spread=0.15, ar_rho=0.85),
           base.with_(M=60, C=8, freshness=22, length_scale=0.15)]
    envs = [SchedulingEnv(sc, layout_seed=VAL_SEED_BASE + 10 * i + k)
            for i, sc in enumerate(scs) for k in range(2)]
    intel, pems = load("intel_lab"), load("pems_bay")
    rng = np.random.default_rng(123)
    for k in range(2):
        envs.append(real_env(intel, intel.M, 6, 22, int(TUNE * intel.train_end) + 100 * k,
                             sensors=np.arange(intel.M)))
        envs.append(real_env(pems, 120, 15, 24, int(TUNE * pems.train_end) + 1500 * k, rng))
    return envs


def score(info, env):
    """What validation and the real-data reward measure: true posterior
    variance when the truth is known, otherwise the actual squared error."""
    return info["pvar"] if env.oracle is not None else info["mse"]


def full_picks(obs, row):
    return list(obs.forced) + [int(a) for a in row if a >= 0]


def teacher_actions(obs):
    picks = greedy_variance_reduction(obs.P_hat, obs.C, obs.obs_noise_var, obs.conflict, obs.forced)
    return picks[len(obs.forced):]


@torch.no_grad()
def run_batched(actor, envs, T, burn=30, greedy=True):
    """Roll the actor through a list of envs (grouped by M); return the
    steady-state score per env (see `score`)."""
    actor.eval()
    out = [None] * len(envs)
    groups = {}
    for i, e in enumerate(envs):
        groups.setdefault(e.M, []).append(i)
    for idx in groups.values():
        es = [envs[i] for i in idx]
        obs = [e.reset() for e in es]
        acc = np.zeros(len(es))
        for t in range(T):
            picks, _, _ = actor.decode(featurize(es, obs), greedy=greedy)
            picks = picks.numpy()
            for b, e in enumerate(es):
                obs[b], _, info = e.step(full_picks(obs[b], picks[b]))
                if t >= burn:
                    acc[b] += score(info, e)
        for b, i in enumerate(idx):
            out[i] = acc[b] / (T - burn)
    return np.array(out)


def run_teacher(envs, T, burn=30):
    out = []
    for e in envs:
        obs = e.reset()
        acc = 0.0
        for t in range(T):
            obs, _, info = e.step(obs.forced + teacher_actions(obs))
            if t >= burn:
                acc += score(info, e)
        out.append(acc / (T - burn))
    return np.array(out)


def train(cfg: TrainConfig, verbose=True, resume=True):
    """Run PPO. Full training state is saved next to `cfg.out` every
    `eval_every` iterations, and with resume=True an interrupted run picks up
    where it stopped."""
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    actor, critic = Actor(cfg.d, cfg.layers), Critic(cfg.d, 2)
    log = {"config": asdict(cfg), "ppo": [], "val": []}
    from gnn_aoi.datasets import load
    real = [load(n) for n in cfg.real_datasets]
    val_envs = validation_envs()
    is_real = np.array([e.oracle is None for e in val_envs])
    teacher_val = run_teacher(val_envs, T=150)
    log["teacher_val"] = teacher_val.tolist()
    best = {"score": np.inf}
    t_start = time.time()

    def say(*a):
        if verbose:
            print(f"[{time.time() - t_start:6.0f}s]", *a, flush=True)

    def validate(stage, it):
        actor.eval()
        pv = run_batched(actor, val_envs, T=150)
        r = pv / teacher_val  # < 1 => better than Kalman-greedy
        syn, rl = float(r[~is_real].mean()), float(r[is_real].mean())
        ratio = (syn + rl) / 2
        log["val"].append({"stage": stage, "iter": it, "ratio_vs_teacher": ratio,
                           "synthetic": syn, "real": rl, "score": pv.tolist()})
        say(f"  validate {stage}@{it}: vs Kalman-greedy  synthetic={syn:.4f}  real={rl:.4f}  mean={ratio:.4f}")
        if ratio < best["score"]:
            best.update(score=ratio, stage=stage, iter=it)
            save(actor, cfg, Path(cfg.out), extra={"val_ratio": ratio, "stage": stage, "iter": it})
        actor.train()

    def new_envs():
        M = int(np.exp(rng.uniform(*np.log(cfg.M_range))).round())
        B = int(np.clip(cfg.n_envs * (40 / M) ** 2, 8, cfg.n_envs))
        if real and rng.random() < cfg.real_frac:
            ds = real[rng.integers(len(real))]
            M = min(M, ds.M)
            lo, hi = int(FIT * ds.train_end), int(TUNE * ds.train_end) - cfg.rollout_len - 2
            envs = []
            for _ in range(B):
                C = sample_capacity(rng, M)
                F = int(3 * M / C) if rng.random() < 0.7 else 0
                envs.append(real_env(ds, M, C, F, int(rng.integers(lo, hi)), rng))
            return envs
        return [SchedulingEnv(sample_scenario(rng, M), layout_seed=int(rng.integers(0, VAL_SEED_BASE)))
                for _ in range(B)]

    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)
    state_path = Path(cfg.out).with_suffix(".state.pt")

    def save_state(it):
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                    "opt_a": opt_a.state_dict(), "opt_c": opt_c.state_dict(),
                    "rng": rng.bit_generator.state, "gen": gen.get_state(), "it": it,
                    "log": log, "best": dict(best), "elapsed": time.time() - t_start}, state_path)

    start = 0
    if resume and state_path.exists():
        st = torch.load(state_path, weights_only=False)
        actor.load_state_dict(st["actor"]); critic.load_state_dict(st["critic"])
        opt_a.load_state_dict(st["opt_a"]); opt_c.load_state_dict(st["opt_c"])
        rng.bit_generator.state = st["rng"]; gen.set_state(st["gen"])
        log.clear(); log.update(st["log"]); best.update(st["best"])
        start = st["it"] + 1
        t_start -= st["elapsed"]
        say(f"resumed at iteration {start} (best so far {best})")
    elif cfg.init_from:
        actor.load_state_dict(torch.load(cfg.init_from, weights_only=False)["state_dict"])
        log["init_from"] = cfg.init_from
        validate("init (warm start)", -1)
    else:
        validate("init (= Kalman-greedy)", -1)

    # ---------------- PPO ----------------
    for it in range(start, cfg.ppo_iters):
        envs = new_envs()
        obs = [e.reset() for e in envs]
        T, B = cfg.rollout_len, len(envs)
        buf = {k: [] for k in ("batch", "act", "logp", "val", "rew")}
        actor.eval(); critic.eval()
        for t in range(T):
            batch = featurize(envs, obs)
            with torch.no_grad():
                picks, logp, _ = actor.decode(batch, generator=gen)
                v = critic(batch)
            rew = np.zeros(B)
            for b, e in enumerate(envs):
                obs[b], _, info = e.step(full_picks(obs[b], picks[b].numpy()))
                # returns ~ mean log error; the floor keeps near-perfect slots from dominating
                rew[b] = -np.log(score(info, e) + 0.005) * (1 - cfg.gamma)
            for k, val in zip(buf, (batch, picks, logp, v, torch.as_tensor(rew, dtype=torch.float32))):
                buf[k].append(val)
        with torch.no_grad():
            last_v = critic(featurize(envs, obs))

        rew = torch.stack(buf["rew"]); vals = torch.stack(buf["val"])
        adv = torch.zeros_like(rew); gae = torch.zeros(B)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else vals[t + 1]
            delta = rew[t] + cfg.gamma * nv - vals[t]
            gae = delta + cfg.gamma * cfg.lam * gae
            adv[t] = gae
        ret = (adv + vals).flatten()
        adv = adv.flatten()

        K = max(a.shape[1] for a in buf["act"])
        data = Batch.cat(buf["batch"])
        old_logp = torch.cat(buf["logp"])
        acts = torch.cat([torch.nn.functional.pad(a, (0, K - a.shape[1]), value=-1) for a in buf["act"]])

        actor.train(); critic.train()
        stats = {"pi": [], "v": [], "ent": [], "kl": [], "clipfrac": []}
        N = T * B
        for _ in range(cfg.epochs):
            for mb in torch.randperm(N).split(cfg.minibatch):
                mbatch = data[mb]
                v = critic(mbatch)
                v_loss = ((v - ret[mb]) ** 2).mean()
                opt_c.zero_grad(); v_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad)
                opt_c.step()
                stats["v"].append(v_loss.item())
                if it < cfg.critic_warmup:
                    continue
                _, logp, ent = actor.decode(mbatch, actions=acts[mb])
                a = adv[mb]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (logp - old_logp[mb]).exp()
                pi_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                loss = pi_loss - cfg.ent_coef * ent.mean()
                opt_a.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad)
                opt_a.step()
                with torch.no_grad():
                    stats["pi"].append(pi_loss.item()); stats["ent"].append(ent.mean().item())
                    stats["kl"].append((old_logp[mb] - logp).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
        rec = {"iter": it, "M": envs[0].M, "real": envs[0].oracle is None,
               "mean_reward": float(rew.mean()),
               "alpha": float(actor.alpha.detach()),
               **{k: float(np.mean(v)) if v else None for k, v in stats.items()}}
        log["ppo"].append(rec)
        if it % 10 == 0 or it == cfg.ppo_iters - 1:
            say(f"PPO {it:3d}  M={rec['M']:3d}{' real' if rec['real'] else ' syn '}  r={rec['mean_reward']:.3f}  v_loss={rec['v']:.4f}"
                + (f"  kl={rec['kl']:.4f}  clip={rec['clipfrac']:.2f}  ent={rec['ent']:.2f}" if rec['kl'] is not None else ""))
        if it >= cfg.critic_warmup and ((it + 1) % cfg.eval_every == 0 or it == cfg.ppo_iters - 1):
            validate("ppo", it)
            save_state(it)

    log["best"] = {k: v for k, v in best.items()}
    log["train_seconds"] = time.time() - t_start
    state_path.unlink(missing_ok=True)
    Path(cfg.log).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.log).write_text(json.dumps(log, indent=1))
    say(f"done. best checkpoint: {best}")
    return log


def save(actor, cfg, path, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": actor.state_dict(), "d": cfg.d, "layers": cfg.layers,
                "extra": extra or {}}, path)


def load_actor(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    actor = Actor(ck["d"], ck["layers"])
    actor.load_state_dict(ck["state_dict"])
    actor.eval()
    return actor, ck.get("extra", {})
