import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gnn_aoi.config import Scenario
from gnn_aoi.env import SchedulingEnv
from gnn_aoi.metrics import mean_ci

METRICS = ("mse", "rmse_units", "pvar", "mean_age", "max_age", "aoci")
TEST_SEED_BASE = 10_000  # test layouts are 10_000, 10_001, ...; training draws from [0, 10_000)


def run_episode(env, policy, T, burn=50):
    """Run one episode; return per-slot traces plus steady-state summary."""
    obs = env.reset()
    policy.reset(env)
    traces = {k: np.empty(T) for k in METRICS}
    polls = np.zeros(env.M, dtype=int)
    t_select = 0.0
    for t in range(T):
        t0 = time.perf_counter()
        picks = policy.select(obs)
        t_select += time.perf_counter() - t0
        polls[picks] += 1
        obs, _, info = env.step(picks)
        for k in METRICS:
            traces[k][t] = info[k]
    summary = {k: float(traces[k][burn:].mean()) for k in METRICS}
    summary["max_age"] = float(traces["max_age"][burn:].max())  # peak, not mean
    summary["ms_per_slot"] = 1000 * t_select / T
    summary["starved"] = int((polls == 0).sum())
    return traces, summary


class SyntheticEnvs:
    """Picklable env factory: held-out synthetic layouts."""

    def __init__(self, sc):
        self.sc = sc

    def __call__(self, k):
        return SchedulingEnv(self.sc, layout_seed=TEST_SEED_BASE + k)


class RealEnvs:
    """Picklable env factory: disjoint test-period windows of a real dataset
    (optionally a fixed sensor subset)."""

    def __init__(self, sc, dataset, n_windows, T, sensors=None):
        self.sc, self.dataset, self.sensors = sc, dataset, sensors
        self.n_windows, self.T = n_windows, T
        self._starts = None

    def starts(self):
        if self._starts is None:
            from gnn_aoi.datasets import load
            self._starts = load(self.dataset).test_windows(self.n_windows, self.T, burn=0)
        return self._starts

    def __call__(self, k):
        from gnn_aoi.datasets import load
        ds = load(self.dataset)
        return SchedulingEnv(self.sc, layout_seed=k, field=ds.field(self.starts()[k], self.sensors))


def _eval_one(args):
    make_env, k, policy, T, burn = args
    _, s = run_episode(make_env(k), policy, T, burn)
    return s


def benchmark(sc: Scenario, policies, n_layouts=20, T=400, burn=50, workers=8, envs=None):
    """Evaluate every policy on the same held-out instances: n_layouts
    synthetic layouts, or (envs=RealEnvs(...)) real-data test windows.

    Returns {policy_name: {"label", "per_layout": [summary...], metric: [mean, ci95]}}.
    """
    envs = envs or SyntheticEnvs(sc)
    if isinstance(envs, RealEnvs):
        n_layouts = len(envs.starts())
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for pol in policies:
            per = list(ex.map(_eval_one, [(envs, k, pol, T, burn) for k in range(n_layouts)]))
            res = {"label": pol.label, "per_layout": per}
            for k in per[0]:
                res[k] = mean_ci([p[k] for p in per])
            out[pol.name] = res
    return out
