from dataclasses import dataclass, fields

import numpy as np
import torch

N_FEATURES = 13


def node_features(obs, M, C, freshness):
    """Per-sensor features + the two graphs the GNN passes messages over.

    Returns X (M, F), A_corr (M, M) static correlation graph,
    A_post (M, M) |posterior correlation| (who is currently redundant with
    whom), forced (M,) bool.
    """
    age = obs.age
    P = obs.P_hat
    var = np.clip(np.diag(P), 1e-6, None)
    gain = (P**2).sum(0) / (var + obs.obs_noise_var) / M  # mean variance drop if polled alone
    period = M / C
    forced = np.zeros(M, dtype=bool)
    forced[obs.forced] = True
    slack = np.clip((freshness - age) / freshness, 0, 1) if freshness > 0 else np.ones(M)

    X = np.stack([
        np.clip(age / period, 0, 6) / 6,
        obs.last_polled,
        var,
        np.log(var + 1e-3) / 7,
        gain,
        np.log(gain + 1e-4) / 9,
        forced.astype(float),
        slack,
        np.full(M, C / M),
        obs.rho_hat,                                     # per-sensor temporal correlation
        np.log(1 - obs.rho_hat**2 + 1e-5) / 12,         # how fast the sensor drifts
        obs.adj.sum(1) / max(1, M - 1),
        np.full(M, float(freshness > 0)),
    ], axis=1).astype(np.float32)

    d = np.sqrt(var)
    A_post = np.abs(P) / np.outer(d, d)
    np.fill_diagonal(A_post, 0.0)
    return X, obs.adj.astype(np.float32), np.clip(A_post, 0, 1).astype(np.float32), forced


@dataclass
class Batch:
    X: torch.Tensor         # (B, M, F) node features
    A_corr: torch.Tensor    # (B, M, M) static correlation graph
    A_post: torch.Tensor    # (B, M, M) |posterior correlation|
    conflict: torch.Tensor  # (B, M, M) bool interference
    forced: torch.Tensor    # (B, M) bool, already committed by the freshness guard
    n_pick: torch.Tensor    # (B,) free picks left after forced ones
    P: torch.Tensor         # (B, M, M) sink covariance conditioned on forced picks
    r: torch.Tensor         # (B,) observation noise variance

    def __getitem__(self, idx):
        return Batch(*(getattr(self, f.name)[idx] for f in fields(self)))

    @staticmethod
    def cat(batches):
        return Batch(*(torch.cat([getattr(b, f.name) for b in batches]) for f in fields(Batch)))


def condition_on(P, idx, r):
    """Kalman covariance update for noisy observations of sensors idx."""
    for j in idx:
        pj = P[:, j].copy()
        P = P - np.outer(pj, pj) / (pj[j] + r)
    return P


def featurize(envs, obs_list):
    """Stack observations from same-M envs into a Batch."""
    cols = {k: [] for k in ("X", "A1", "A2", "fz", "n", "P", "r")}
    for env, obs in zip(envs, obs_list):
        X, A1, A2, forced = node_features(obs, env.M, env.C, env.sc.freshness)
        cols["X"].append(X); cols["A1"].append(A1); cols["A2"].append(A2); cols["fz"].append(forced)
        cols["n"].append(max(0, env.C - len(obs.forced)))
        cols["P"].append(condition_on(obs.P_hat, obs.forced, obs.obs_noise_var))
        cols["r"].append(obs.obs_noise_var)
    f32 = lambda a: torch.as_tensor(np.stack(a), dtype=torch.float32)
    return Batch(
        X=f32(cols["X"]), A_corr=f32(cols["A1"]), A_post=f32(cols["A2"]),
        conflict=torch.as_tensor(np.stack([e.conflict for e in envs])),
        forced=torch.as_tensor(np.stack(cols["fz"])),
        n_pick=torch.as_tensor(cols["n"], dtype=torch.long),
        P=f32(cols["P"]), r=torch.as_tensor(cols["r"], dtype=torch.float32),
    )
