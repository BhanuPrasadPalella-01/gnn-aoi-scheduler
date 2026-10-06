"""Every experiment in the report. Each writes results/data/<name>.json,
which the figures (and the future frontend) read."""
import json
import time
from pathlib import Path

import numpy as np

from gnn_aoi.config import Scenario
from gnn_aoi.env import SchedulingEnv
from gnn_aoi.evaluate import TEST_SEED_BASE, RealEnvs, benchmark, run_episode
from gnn_aoi.policies.baselines import BASELINES
from gnn_aoi.policies.gnn_policy import GNNPolicy
from gnn_aoi.simulator import CorrelatedField, fit_model

DATA = Path("results/data")
HEADLINE = Scenario(M=50, C=10, freshness=15)
CORE = ("max_age", "greedy_diverse", "kalman_greedy", "gnn_no_mp", "gnn", "oracle")

# real datasets: capacity = 1/8 of the sensors (Intel ~1/7.5), freshness = 3 M / C slots
REAL = {"intel_lab": {"C": 6, "T": 200, "windows": 12},
        "pems_bay": {"C": 40, "T": 288, "windows": 15}}
REAL_ALL = ("random", "max_age", "greedy_diverse", "max_variance", "kalman_greedy", "gnn")
REAL_CORE = ("max_age", "greedy_diverse", "kalman_greedy", "gnn_no_mp", "gnn")


ABLATION_CKPT = Path("checkpoints/ablation_no_mp.pt")


def policies(names=None, checkpoint="checkpoints/gnn_scheduler.pt"):
    names = names or (*BASELINES, "gnn")
    out = []
    for n in names:
        if n == "gnn":
            out.append(GNNPolicy(checkpoint))
        elif n == "gnn_no_mp":
            if ABLATION_CKPT.exists():
                out.append(GNNPolicy(str(ABLATION_CKPT), "gnn_no_mp", "Ablation: no message passing"))
        else:
            out.append(BASELINES[n]())
    return out


def _strip(res):
    """Drop per-layout detail except the metric we plot per layout."""
    return {k: {m: v for m, v in r.items() if m != "per_layout"} |
            {"mse_per_layout": [p["mse"] for p in r["per_layout"]],
             "rmse_units_per_layout": [p["rmse_units"] for p in r["per_layout"]]} for k, r in res.items()}


def _save(name, payload):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{name}.json").write_text(json.dumps(payload, indent=1))
    return payload


def _sweep(name, param, values, make, pols, n_layouts, T, envs=None):
    rows = []
    for v in values:
        t0 = time.time()
        sc = make(v)
        res = benchmark(sc, policies(pols), n_layouts, T, envs=envs(sc, v) if envs else None)
        rows.append({"value": v, "scenario": sc.to_dict(), "results": _strip(res)})
        print(f"  {name} {param}={v}  ({time.time() - t0:.0f}s)", flush=True)
    return _save(name, {"param": param, "rows": rows})


def main_benchmark(n_layouts=30, T=400):
    out = {}
    for key, sc in {"uniform": HEADLINE, "clustered": HEADLINE.with_(layout="clustered")}.items():
        names = (*BASELINES, "gnn", "gnn_no_mp")
        out[key] = {"scenario": sc.to_dict(), "results": _strip(benchmark(sc, policies(names), n_layouts, T))}
        print(f"  main/{key} done", flush=True)
    return _save("main", out)


def unconstrained(n_layouts=30, T=400):
    sc = HEADLINE.with_(freshness=0)
    return _save("unconstrained", {"scenario": sc.to_dict(),
                                   "results": _strip(benchmark(sc, policies(), n_layouts, T))})


def capacity_sweep(n_layouts=20, T=300):
    return _sweep("capacity", "C", [4, 6, 8, 10, 15, 20, 25],
                  lambda c: HEADLINE.with_(C=c, freshness=int(3 * HEADLINE.M / c)), CORE, n_layouts, T)


def correlation_sweep(n_layouts=20, T=300):
    return _sweep("correlation", "length_scale", [0.05, 0.1, 0.15, 0.2, 0.3, 0.45],
                  lambda ls: HEADLINE.with_(length_scale=ls), CORE, n_layouts, T)


def scale_sweep(n_layouts=12, T=250):
    """Training saw M in [10, 100]; 150 and 200 are zero-shot."""
    return _sweep("scale", "M", [20, 50, 100, 150, 200],
                  lambda m: HEADLINE.with_(M=m, C=m // 5), CORE, n_layouts, T)


def interference_sweep(n_layouts=20, T=300):
    return _sweep("interference", "radius", [0.0, 0.05, 0.1, 0.15, 0.2],
                  lambda r: HEADLINE.with_(interference_radius=r), CORE, n_layouts, T)


def pilot_sweep(n_layouts=20, T=300):
    """How long must the sink listen before it knows the correlation well
    enough to schedule with it?"""
    rows = []
    for n in [25, 50, 100, 200, 500, 1000, 2000]:
        sc = HEADLINE.with_(pilot_len=n)
        maes = []
        for k in range(n_layouts):
            f = CorrelatedField(sc, TEST_SEED_BASE + k)
            est = fit_model(f.pilot(n)).innovation_corr
            iu = np.triu_indices(sc.M, 1)
            maes.append(float(np.abs(est[iu] - f.ground_truth_correlation()[iu]).mean()))
        res = _strip(benchmark(sc, policies(("max_age", "kalman_greedy", "gnn", "oracle")), n_layouts, T))
        rows.append({"value": n, "corr_mae": float(np.mean(maes)), "results": res})
        print(f"  pilot n={n}", flush=True)
    return _save("pilot", {"param": "pilot_len", "rows": rows})


def _real_sc(name, C=None, M=None):
    from gnn_aoi.datasets import load
    M = M or load(name).M
    C = C or REAL[name]["C"]
    return Scenario(M=M, C=C, freshness=int(3 * M / C))


def real_benchmark():
    """Headline on real data: every scheduler on the same held-out
    test-period windows (sink fitted on the training period only)."""
    from gnn_aoi.datasets import INFO, load
    out = {}
    for name, cfg in REAL.items():
        sc = _real_sc(name)
        ds = load(name)
        res = benchmark(sc, policies((*REAL_ALL, "gnn_no_mp")), T=cfg["T"],
                        envs=RealEnvs(sc, name, cfg["windows"], cfg["T"]))
        out[name] = {"scenario": sc.to_dict(), "info": {k: v for k, v in INFO[name].items() if k != "weekly"},
                     "train_period": [str(ds.timestamps[0])[:10], str(ds.timestamps[ds.train_end - 1])[:10]],
                     "test_period": [str(ds.timestamps[ds.train_end])[:10], str(ds.timestamps[-1])[:10]],
                     "results": _strip(res)}
        print(f"  real/{name} done", flush=True)
    return _save("real_main", out)


def real_capacity():
    for name, Cs in {"intel_lab": [3, 4, 6, 9, 12, 15], "pems_bay": [10, 20, 40, 60, 80]}.items():
        cfg = REAL[name]
        _sweep(f"real_capacity_{name}", "C", Cs, lambda c, n=name: _real_sc(n, C=c), REAL_CORE, 0, cfg["T"],
               envs=lambda sc, v, n=name, c=cfg: RealEnvs(sc, n, c["windows"], c["T"]))


def real_scale():
    """PEMS-BAY sub-networks of growing size (nearest sensors around the
    network's centre), up to all 325 sensors."""
    from gnn_aoi.datasets import load
    ds = load("pems_bay")
    centre = np.argmin(np.linalg.norm(ds.positions - ds.positions.mean(0), axis=1))
    order = np.argsort(np.linalg.norm(ds.positions - ds.positions[centre], axis=1))
    cfg = REAL["pems_bay"]
    return _sweep("real_scale", "M", [25, 50, 100, 200, 325],
                  lambda m: _real_sc("pems_bay", C=max(3, m // 8), M=m), REAL_CORE, 0, cfg["T"],
                  envs=lambda sc, m: RealEnvs(sc, "pems_bay", cfg["windows"], cfg["T"],
                                              sensors=np.sort(order[:m])))


def poll_frequency(name="intel_lab", T=288):
    """Where does each scheduler spend the channel on the real lab floor?
    Polling share per sensor over one test day."""
    from gnn_aoi.live import LiveSession
    s = LiveSession(name, policies=("max_age", "kalman_greedy", "gnn"))
    meta = s.meta()
    for _ in s.run(T):
        pass
    out = {"meta": meta, "policies": {}}
    for pol in s.policies:
        st = s.stats[pol.name]
        out["policies"][pol.name] = {"label": pol.label, "share": (st["polls"] / T).tolist(),
                                     "rmse": float(np.sqrt(st["sq_err"] / T))}
    return _save("poll_frequency", out)


def replay(name="intel_lab", T=144, names=("max_age", "gnn")):
    """Slot-by-slot recording on real data for the animation: same day,
    same readings, different schedulers."""
    from gnn_aoi.datasets import load
    ds = load(name)
    sc = _real_sc(name)
    start = ds.test_windows(1, 288)[0] + 288  # second test day
    out = {"scenario": sc.to_dict(), "dataset": name, "policies": {}}
    for pol in policies(names):
        env = SchedulingEnv(sc, field=ds.field(start), record=True)
        out["positions"] = env.positions.round(4).tolist()
        out["corr_edges"] = [[int(i), int(j), round(float(env.adj[i, j]), 3)]
                             for i, j in zip(*np.nonzero(np.triu(env.adj, 1)))]
        out["units"] = env.field.units
        out["times"] = [str(t)[:16] for t in ds.timestamps[start:start + T]]
        _, summary = run_episode(env, pol, T, burn=0)
        out["policies"][pol.name] = {"label": pol.label, "summary": summary, "slots": env.trace}
    return _save("replay", out)


def live_exports():
    """Pre-computed live sessions for the frontend (one test day each)."""
    from gnn_aoi.live import export
    for name, slots in {"intel_lab": 288, "pems_bay": 144}.items():
        p = export(name, slots, policies=("max_age", "kalman_greedy", "gnn"))
        print(f"  wrote {p}", flush=True)


def significance():
    """Paired per-instance comparison of the GNN against every other policy
    (win rate + Wilcoxon signed-rank p-value) on the synthetic and real
    headline benchmarks."""
    from scipy.stats import wilcoxon
    blocks = {}
    for f in ("main", "real_main"):
        if (DATA / f"{f}.json").exists():
            blocks.update(json.loads((DATA / f"{f}.json").read_text()))
    out = {}
    for key, block in blocks.items():
        g = np.array(block["results"]["gnn"]["mse_per_layout"])
        out[key] = {}
        for name, r in block["results"].items():
            if name == "gnn":
                continue
            b = np.array(r["mse_per_layout"])
            out[key][name] = {"label": r["label"], "gnn_win_rate": float((g < b).mean()),
                              "wilcoxon_p": float(wilcoxon(g, b).pvalue),
                              "mean_rel_change_pct": float(((g - b) / b).mean() * 100)}
    return _save("significance", out)


ALL = {
    "real_main": real_benchmark, "real_capacity": real_capacity, "real_scale": real_scale,
    "main": main_benchmark, "significance": significance, "unconstrained": unconstrained,
    "capacity": capacity_sweep, "correlation": correlation_sweep, "scale": scale_sweep,
    "interference": interference_sweep, "pilot": pilot_sweep,
    "poll_frequency": poll_frequency, "replay": replay, "live": live_exports,
}
