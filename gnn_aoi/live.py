"""Live transmission engine: replays a real dataset slot by slot and runs
several schedulers side by side on identical readings.

This is the backend of the live demo. Every call to `step()` returns one
JSON-serialisable event describing the slot: the real readings, which
sensors each scheduler polled (in decision order, and which were forced by
the freshness guarantee), each scheduler's per-sensor priority (why it chose
them), and what the sink now believes about every sensor.

    session = LiveSession("intel_lab")
    meta = session.meta()          # sensors, positions, graph, settings
    for event in session.run(100): # or session.step() at your own pace
        ...
"""
import json
import time
from pathlib import Path

import numpy as np

from gnn_aoi.config import Scenario
from gnn_aoi.datasets import INFO, PROCESSED, load
from gnn_aoi.env import SchedulingEnv
from gnn_aoi.simulator import LinearGaussianModel, fit_model

DEFAULT_C = {"intel_lab": 6, "pems_bay": 40}
_MODELS = {}  # (dataset, sensors) -> sink model fitted on the training period


def _model(ds, sensors, field):
    """The sink's model for these sensors. For the full sensor set it is
    pre-fitted on the training period and shipped in data/processed/
    (fitting PEMS-BAY's 4 months x 325 sensors at request time needs a lot
    of memory); otherwise it is fitted once and cached."""
    key = (ds.name, tuple(sensors.tolist()))
    if key not in _MODELS:
        path = PROCESSED / f"{ds.name}_model.npz"
        if len(sensors) == ds.M and path.exists():
            z = np.load(path)
            _MODELS[key] = LinearGaussianModel(z["rho"], z["Q"], z["P0"])
        else:
            _MODELS[key] = fit_model(field.pilot())
    return _MODELS[key]


def prefit_models():
    """Write data/processed/<dataset>_model.npz (run once after building the caches)."""
    for name in DEFAULT_C:
        ds = load(name)
        m = fit_model(ds.field(ds.train_end).pilot())
        np.savez_compressed(PROCESSED / f"{name}_model.npz", rho=m.rho, Q=m.Q, P0=m.P0)
        print(f"  wrote {PROCESSED / f'{name}_model.npz'}")


def _r(a, nd=3):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


class LiveSession:
    def __init__(self, dataset="intel_lab", policies=("max_age", "gnn"), C=None, freshness=None,
                 start=None, sensors=None, checkpoint="checkpoints/gnn_scheduler.pt"):
        from gnn_aoi.experiments import policies as make_policies
        self.ds = load(dataset)
        self.sensors = np.arange(self.ds.M) if sensors is None else np.asarray(sensors)
        M = len(self.sensors)
        C = C or DEFAULT_C.get(dataset, max(1, M // 8))
        self.sc = Scenario(M=M, C=C, freshness=freshness if freshness is not None else int(3 * M / C))
        self.start = start if start is not None else self.ds.test_windows(1, 288)[0]
        self.policies = make_policies(policies, checkpoint)
        self.envs, self.obs, self.stats = {}, {}, {}
        for pol in self.policies:
            field = self.ds.field(self.start, self.sensors)
            env = SchedulingEnv(self.sc, field=field, record=True, model=_model(self.ds, self.sensors, field))
            self.envs[pol.name] = env
            self.obs[pol.name] = env.reset()
            pol.reset(env)
            self.stats[pol.name] = {"sq_err": 0.0, "age": 0.0, "polls": np.zeros(M, int)}
        self.slot = 0

    def meta(self):
        """Everything static: sensors, where they are, the learned
        correlation graph, and the scheduling settings."""
        env = next(iter(self.envs.values()))
        info = INFO[self.ds.name]
        adj = env.adj
        raw = self.ds.raw_positions[self.sensors]
        return {
            "dataset": self.ds.name, **{k: v for k, v in info.items() if k != "weekly"},
            "M": self.sc.M, "C": self.sc.C, "freshness": self.sc.freshness,
            "start_time": str(self.ds.timestamps[self.start])[:16].replace("T", " "),
            "sensors": [{"index": i, "id": int(self.ds.sensor_ids[s]),
                         "x": round(float(env.positions[i, 0]), 4), "y": round(float(env.positions[i, 1]), 4),
                         "raw_x": round(float(raw[i, 0]), 5), "raw_y": round(float(raw[i, 1]), 5)}
                        for i, s in enumerate(self.sensors)],
            "edges": [[int(i), int(j), round(float(adj[i, j]), 3)] for i, j in zip(*np.nonzero(np.triu(adj, 1)))],
            "policies": [{"name": p.name, "label": p.label} for p in self.policies],
            "value_range": _r(np.percentile(self.ds.values[self.ds.train_end::7, self.sensors], [2, 50, 98]), 2),
            "replay_until": str(self.ds.timestamps[int(self.ds.segment_end[self.start])])[:16].replace("T", " "),
        }

    def step(self):
        """Advance one slot for every scheduler; return the slot's event.
        Raises IndexError when the recording ends or hits an outage."""
        first = next(iter(self.envs.values()))
        if first.field.exhausted:
            raise IndexError("recording gap" if first.field.end < len(self.ds.values) - 1 else "end of recording")
        event = {"slot": self.slot, "time": first.field.timestamp(), "policies": {}}
        for pol in self.policies:
            env, obs, st = self.envs[pol.name], self.obs[pol.name], self.stats[pol.name]
            t0 = time.perf_counter()
            picks = list(pol.select(obs))
            ms = 1000 * (time.perf_counter() - t0)
            prio = pol.priority(obs)
            self.obs[pol.name], _, info = env.step(picks)
            rec = env.trace[-1]
            env.trace.clear()  # only the latest slot is needed; don't grow without bound
            st["sq_err"] += info["rmse_units"] ** 2
            st["age"] += info["mean_age"]
            st["polls"][picks] += 1
            n = self.slot + 1
            if "truth" not in event:
                event["truth"] = _r(rec["truth_units"], 2)
            event["policies"][pol.name] = {
                "polled": [int(i) for i in picks],          # decision order
                "forced": [int(i) for i in obs.forced],     # required by the freshness bound
                "priority": None if prio is None else _r(prio / (np.max(np.abs(prio)) + 1e-12), 3),
                "estimate": _r(rec["estimate_units"], 2),
                "uncertainty": _r(np.sqrt(rec["variance"]) * env.field.scale, 2),
                "age": rec["age"],
                "rmse": round(info["rmse_units"], 4),
                "running_rmse": round(float(np.sqrt(st["sq_err"] / n)), 4),
                "mean_age": round(info["mean_age"], 2),
                "running_mean_age": round(st["age"] / n, 2),
                "decision_ms": round(ms, 2),
            }
        self.slot += 1
        return event

    def run(self, n):
        for _ in range(n):
            yield self.step()


def export(dataset, slots=288, path=None, **kw):
    """Pre-compute a session to JSON so a frontend can replay it without
    running Python."""
    s = LiveSession(dataset, **kw)
    out = {"meta": s.meta(), "events": list(s.run(slots))}
    path = Path(path or f"results/data/live_{dataset}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    return path


def terminal(dataset="intel_lab", slots=120, speed=4.0, policies=("max_age", "gnn")):
    """A minimal live view in the terminal (the real demo is the frontend)."""
    s = LiveSession(dataset, policies=policies)
    m = s.meta()
    ids = [x["id"] for x in m["sensors"]]
    print(f"\n  {m['title']} · {m['M']} sensors · {m['C']} polls per {m['slot_minutes']}-min slot · "
          f"every sensor refreshed within {m['freshness']} slots\n")
    colors = {"max_age": "\033[38;5;208m", "gnn": "\033[38;5;33m"}
    for ev in s.run(slots):
        line = f"  {ev['time']}  "
        for name, p in ev["policies"].items():
            c = colors.get(name, "")
            polled = " ".join(f"{ids[i]:>3}" for i in sorted(p["polled"]))
            line += f"{c}{name:>8}\033[0m ▸ {polled}  err {p['running_rmse']:.3f}{m['units']}   "
        print(line, flush=True)
        time.sleep(1 / speed)
