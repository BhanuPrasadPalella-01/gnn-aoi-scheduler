"""Local web server for the live demo (standard library only).

    python -m gnn_aoi serve            # http://localhost:8000

Serves the frontend in frontend/, the figures in results/figures/, and a
small JSON API. The schedulers run here, live: every /api/step call
advances the real-data replay by one slot and runs each scheduler's
decision on the spot.

    GET  /health         liveness check (for the hosting platform)
    GET  /api/datasets   datasets, test days, capacity ranges
    POST /api/session    {dataset, C, policies, date, hour} -> {id, meta}
    POST /api/step       {id, n} -> {events: [...]}
    GET  /api/results    headline numbers for the results page
"""
import json
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
DATA = ROOT / "results" / "data"

C_RANGE = {"intel_lab": (2, 15), "pems_bay": (10, 80)}
MAX_SESSIONS = 6
_sessions = {}  # id -> {"session", "lock", "used"}
_lock = threading.Lock()


def datasets():
    """Dataset list for the page, read from metadata only: a dataset's
    readings are loaded the first time someone opens a session on it."""
    from gnn_aoi.datasets import INFO, outages_from, summary
    from gnn_aoi.live import DEFAULT_C
    out = []
    for name in ("intel_lab", "pems_bay"):
        meta = summary(name)
        ts = meta["timestamps"].astype("datetime64[m]")
        test = np.arange(meta["train_end"], len(ts))
        days = []
        for day in np.unique(ts[test].astype("datetime64[D]")):
            idx = test[ts[test].astype("datetime64[D]") == day]
            if len(idx) >= 12:
                first, last = str(ts[idx[0]])[11:16], str(ts[idx[-1]])[11:16]
                partial = len(idx) < 288 - 2
                days.append({"date": str(day), "start": int(idx[0]), "weekday": int((day.astype(int) + 3) % 7),
                             "note": f"data {first}–{last}" if partial else ""})
        info = {k: v for k, v in INFO[name].items() if k != "weekly"}
        out.append({"name": name, **info, "M": meta["M"], "default_C": DEFAULT_C[name],
                    "C_range": C_RANGE[name], "days": days,
                    "outages": outages_from(ts, INFO[name]["slot_minutes"])})
    return out


def results_summary():
    out = {}
    for f in ("real_main", "significance", "main", "unconstrained"):
        p = DATA / f"{f}.json"
        if p.exists():
            d = json.loads(p.read_text())
            if f in ("real_main", "main"):  # drop per-window arrays
                for block in d.values():
                    for r in block["results"].values():
                        r["n_windows"] = len(r.pop("mse_per_layout", []))
                        r.pop("rmse_units_per_layout", None)
            out[f] = d
    return out


def _evict():
    if len(_sessions) <= MAX_SESSIONS:
        return
    oldest = sorted(_sessions, key=lambda k: _sessions[k]["used"])[: len(_sessions) - MAX_SESSIONS]
    for k in oldest:
        _sessions.pop(k, None)


def new_session(body):
    from gnn_aoi.live import LiveSession
    name = body.get("dataset", "intel_lab")
    lo, hi = C_RANGE[name]
    C = int(np.clip(int(body.get("C") or 0) or {"intel_lab": 6, "pems_bay": 40}[name], lo, hi))
    policies = tuple(p for p in body.get("policies", ["max_age", "gnn"])
                     if p in ("random", "max_age", "greedy_diverse", "max_variance", "kalman_greedy", "gnn"))
    start = None
    if body.get("date"):
        from gnn_aoi.datasets import load
        ds = load(name)
        when = np.datetime64(f"{body['date']}T{int(body.get('hour', 0)):02d}:00")
        start = int(np.clip(np.searchsorted(ds.timestamps.astype("datetime64[m]"), when),
                            ds.train_end, len(ds.timestamps) - 2))
    s = LiveSession(name, policies=policies or ("max_age", "gnn"), C=C, start=start,
                    checkpoint=str(ROOT / "checkpoints" / "gnn_scheduler.pt"))
    sid = uuid.uuid4().hex[:12]
    with _lock:
        _sessions[sid] = {"session": s, "lock": threading.Lock(), "used": time.time()}
        _evict()
    return {"id": sid, "meta": s.meta()}


def step(body):
    entry = _sessions.get(body.get("id"))
    if entry is None:
        raise KeyError("unknown or expired session")
    n = int(np.clip(int(body.get("n", 1)), 1, 50))
    with entry["lock"]:
        entry["used"] = time.time()
        events, reason = [], None
        for _ in range(n):
            try:
                events.append(entry["session"].step())
            except IndexError as e:
                reason = str(e)
                break
    return {"events": events, "done": reason is not None, "reason": reason}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(FRONTEND), **kw)

    def log_message(self, fmt, *args):  # keep the console readable
        if "/api/step" not in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _json(self, obj, code=200):
        data = json.dumps(obj, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def translate_path(self, path):
        if path.startswith("/results/figures/"):
            return str(ROOT / path.lstrip("/").split("?")[0])
        return super().translate_path(path)

    def do_HEAD(self):
        if self.path.startswith("/health"):
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
            return
        return super().do_HEAD()

    def do_GET(self):
        if self.path.startswith("/health"):  # cheap liveness check for the host
            data = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        try:
            if self.path.startswith("/api/datasets"):
                return self._json(DATASETS)
            if self.path.startswith("/api/results"):
                return self._json(results_summary())
        except Exception as e:  # noqa: BLE001 - report to the page
            return self._json({"error": str(e)}, 500)
        return super().do_GET()

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            if self.path.startswith("/api/session"):
                return self._json(new_session(body))
            if self.path.startswith("/api/step"):
                return self._json(step(body))
            return self._json({"error": "not found"}, 404)
        except KeyError as e:
            return self._json({"error": str(e)}, 410)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)


DATASETS = None


def serve(port=None, host=None):
    """Local by default. On a host such as Render, PORT is set and the server
    listens on all interfaces."""
    import os
    import torch
    global DATASETS
    port = int(port or os.environ.get("PORT", 8000))
    host = host or ("0.0.0.0" if "PORT" in os.environ else "127.0.0.1")
    os.chdir(ROOT)  # data/ and checkpoints/ are relative to the project root
    torch.set_num_threads(2)
    DATASETS = datasets()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"\n  GNN-AoI live demo running at  http://{'localhost' if host == '127.0.0.1' else host}:{port}\n  (Ctrl+C to stop)\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
