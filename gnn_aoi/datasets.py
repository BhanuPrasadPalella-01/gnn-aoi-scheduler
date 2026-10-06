"""Real sensor datasets, replayed slot by slot.

intel_lab  54 Mica2Dot motes in the Intel Berkeley Research Lab (2004),
           temperature every 31 s. Binned to 5-minute slots; the clean
           period Feb 28 - Mar 19 (motes die after that).
           https://db.csail.mit.edu/labdata/labdata.html
pems_bay   325 Caltrans loop detectors around San Jose (Jan-Jun 2017),
           traffic speed every 5 minutes.
           https://zenodo.org/record/5146275

Raw files live in data/raw/ (see README for download commands); the first
load builds a compact cache in data/processed/.

Train/test split is chronological: the sink fits its model only on the
training period; every evaluation window comes from the later test period.

The sink also learns each sensor's typical daily profile (time of day, plus
weekday/weekend for traffic) from the training period, and tracks the
*deviation* from that profile. "Rush hour happens" is known in advance;
what the scheduler has to find out is how today differs.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAW = Path("data/raw")
SOURCES = {
    "intel_lab_data.txt.gz": "https://db.csail.mit.edu/labdata/data.txt.gz",                      # 34 MB
    "intel_mote_locs.txt": "https://db.csail.mit.edu/labdata/mote_locs.txt",
    "PEMS-BAY.csv": "https://zenodo.org/api/records/5146275/files/PEMS-BAY.csv/content",         # 86 MB
    "PEMS-BAY-META.csv": "https://zenodo.org/api/records/5146275/files/PEMS-BAY-META.csv/content",
}
PROCESSED = Path("data/processed")

INFO = {
    "intel_lab": {"title": "Intel Berkeley Lab", "quantity": "Temperature", "units": "°C",
                  "slot_minutes": 5, "where": "Indoor lab floor (40 m × 31 m)", "weekly": False},
    "pems_bay": {"title": "PEMS-BAY", "quantity": "Traffic speed", "units": "mph",
                 "slot_minutes": 5, "where": "San Jose freeway network", "weekly": True},
}


@dataclass
class RealDataset:
    name: str
    values: np.ndarray       # (T, M) readings in physical units
    timestamps: np.ndarray   # (T,) datetime64
    positions: np.ndarray    # (M, 2) in [0, 1], aspect ratio preserved
    raw_positions: np.ndarray  # (M, 2) metres (Intel) or lon/lat (PEMS)
    sensor_ids: np.ndarray   # (M,)
    train_end: int           # values[:train_end] is the training period

    @property
    def M(self):
        return self.values.shape[1]

    @property
    def info(self):
        return INFO[self.name]

    def __post_init__(self):
        self._fit_profile()
        # the recordings have outages (network offline); never replay across one
        ts = self.timestamps.astype("datetime64[m]")
        brk = np.diff(ts).astype(int) != self.info["slot_minutes"]
        ends = np.append(np.where(brk)[0], len(ts) - 1)
        self.segment_end = np.repeat(ends, np.diff(np.append(-1, ends)))  # last index of each slot's run

    def _profile_keys(self):
        ts = self.timestamps.astype("datetime64[m]")
        minute = (ts - ts.astype("datetime64[D]")).astype(int)
        key = minute // self.info["slot_minutes"]
        if self.info["weekly"]:
            weekday = (ts.astype("datetime64[D]").astype(int) + 3) % 7  # 0 = Monday
            key = key + (weekday >= 5) * (1440 // self.info["slot_minutes"])
        return key

    def _fit_profile(self, smooth=5):
        """Typical value per (time-of-day[, weekend]) bin, from training only,
        smoothed over neighbouring bins. Kept as a small (bins, M) table plus
        a per-slot bin index, not a full (T, M) copy, to stay light in memory."""
        keys = self._profile_keys()
        n = keys.max() + 1
        tr = slice(0, self.train_end)
        sums = np.zeros((n, self.M)); cnt = np.zeros(n)
        np.add.at(sums, keys[tr], self.values[tr])
        np.add.at(cnt, keys[tr], 1)
        prof = sums / np.maximum(cnt, 1)[:, None]
        prof[cnt == 0] = self.values[tr].mean(0)
        k = np.ones(smooth) / smooth
        per_day = 1440 // self.info["slot_minutes"]
        for block in range(n // per_day):  # circular smoothing within each day type
            b = prof[block * per_day:(block + 1) * per_day]
            padded = np.concatenate([b[-smooth:], b, b[:smooth]])
            prof[block * per_day:(block + 1) * per_day] = np.apply_along_axis(
                lambda c: np.convolve(c, k, "same"), 0, padded)[smooth:-smooth]
        self.profile_keys = keys
        self.profile_table = prof.astype(np.float32)
        sq = np.zeros(self.M)
        for a in range(0, self.train_end, 4096):  # chunked: no full-size temporaries
            b = min(a + 4096, self.train_end)
            r = self.values[a:b] - self.profile_table[keys[a:b]]
            sq += (r.astype(np.float64) ** 2).sum(0)
        mean_r = np.zeros(self.M)
        for a in range(0, self.train_end, 4096):
            b = min(a + 4096, self.train_end)
            mean_r += (self.values[a:b] - self.profile_table[keys[a:b]]).sum(0)
        mean_r /= self.train_end
        self.std = np.sqrt(sq / self.train_end - mean_r**2) + 1e-6

    def profile_at(self, t, idx=None):
        """Expected (typical) reading at slot t, for all sensors or a subset."""
        row = self.profile_table[self.profile_keys[t]]
        return row if idx is None else row[idx]

    def standardized(self):
        """Deviation from the daily profile, in units of its training std
        (float32, computed once and shared by every replay)."""
        if getattr(self, "_z", None) is None:
            z = np.empty_like(self.values)
            inv = (1.0 / self.std).astype(np.float32)
            for a in range(0, len(z), 4096):  # chunked: no full-size temporaries
                b = min(a + 4096, len(z))
                z[a:b] = (self.values[a:b] - self.profile_table[self.profile_keys[a:b]]) * inv
            self._z = z
        return self._z

    def outages(self):
        """(last slot before, first slot after) for every recording gap."""
        return outages_from(self.timestamps, self.info["slot_minutes"])

    def test_windows(self, n, length, burn=50):
        """Start indices of up to n disjoint windows in the test period (each
        window = burn-in + length slots, never crossing a recording gap),
        evenly spread."""
        span = length + burn
        T = len(self.values)
        starts, t = [], self.train_end
        while t + span <= T:
            if self.segment_end[t] >= t + span - 1:
                starts.append(t)
                t += span
            else:
                t = self.segment_end[t] + 1  # skip to after the gap
        starts = np.array(starts)
        if len(starts) > n:
            starts = starts[np.linspace(0, len(starts) - 1, n).round().astype(int)]
        return [int(s) for s in starts]

    def field(self, start, sensors=None, train_slice=None):
        """A TraceField replaying the test period from `start`.

        sensors: optional subset of sensor indices (e.g. for scaling studies).
        train_slice: optional slice of the training period to fit on.
        """
        idx = np.arange(self.M) if sensors is None else np.asarray(sensors)
        full = len(idx) == self.M and np.array_equal(idx, np.arange(self.M))
        Z = self.standardized() if full else self.standardized()[:, idx]  # full set: shared, no copy
        train = Z[train_slice or slice(0, self.train_end)]
        return TraceField(self, idx, train, Z, start)


class TraceField:
    """Same interface as simulator.CorrelatedField, but every reading is a
    real recorded value. Polling a sensor returns exactly what it measured."""

    def __init__(self, ds: RealDataset, idx, train, Z, start):
        self.ds = ds
        self.idx = idx
        self.M = len(idx)
        self.positions = _normalize(ds.raw_positions[idx]) if len(idx) < ds.M else ds.positions
        self._train = train
        self._Z = Z
        self.t0 = start
        self.t = start - 1
        self.end = int(ds.segment_end[start])  # replay stops before the next recording gap
        self.exhausted = False
        self.state = None
        self._all = len(idx) == ds.M
        self.scale = ds.std[idx]
        self.units = ds.info["units"]

    def true_model(self):
        return None  # reality has no known model

    def mark(self):
        pass

    def rewind(self):
        self.t = self.t0 - 1
        self.exhausted = False

    def pilot(self, n=None):
        return self._train

    def step(self):
        if self.t + 1 > self.end:
            # next slot is past a recording gap: hold the last reading and flag it,
            # so a live session can stop cleanly (evaluation windows never get here)
            self.exhausted = True
            return self.state.copy()
        self.t += 1
        self.state = self._Z[self.t].copy()
        return self.state.copy()

    def observe(self, idx):
        return self.state[np.asarray(idx, dtype=int)]

    def to_units(self, x):
        return np.asarray(x) * self.scale + self.ds.profile_at(self.t, None if self._all else self.idx)

    def timestamp(self):
        return str(self.ds.timestamps[self.t])[:16].replace("T", " ")


def download():
    """Fetch the raw files into data/raw/ (skips files already present)."""
    import urllib.request
    RAW.mkdir(parents=True, exist_ok=True)
    for fname, url in SOURCES.items():
        dest = RAW / fname
        if dest.exists():
            print(f"  have {dest}")
            continue
        print(f"  downloading {fname} ...", flush=True)
        urllib.request.urlretrieve(url, dest)


def outages_from(timestamps, slot_minutes):
    ts = timestamps.astype("datetime64[m]")
    idx = np.where(np.diff(ts).astype(int) != slot_minutes)[0]
    return [(str(ts[i])[:16].replace("T", " "), str(ts[i + 1])[:16].replace("T", " ")) for i in idx]


def _normalize(xy):
    lo = xy.min(0)
    span = (xy.max(0) - lo).max()
    out = (xy - lo) / span
    return out + (1 - out.max(0)) / 2  # centre the shorter axis


def _build_intel():
    import pandas as pd
    df = pd.read_csv(RAW / "intel_lab_data.txt.gz", sep=r"\s+", header=None,
                     names=["date", "time", "epoch", "mote", "temp", "hum", "light", "volt"],
                     on_bad_lines="skip")
    df = df.dropna(subset=["mote", "temp"])
    df["mote"] = df["mote"].astype(int)
    df = df[(df.mote >= 1) & (df.mote <= 54) & (df.temp > 5) & (df.temp < 45)]
    df["ts"] = pd.to_datetime(df.date + " " + df.time, format="mixed")
    df = df[(df.ts >= "2004-02-28 01:00") & (df.ts < "2004-03-19")]
    piv = df.pivot_table(index=pd.Grouper(key="ts", freq="5min"), columns="mote",
                         values="temp", aggfunc="mean")
    piv = piv.loc[:, piv.notna().mean() > 0.7]
    piv = piv.interpolate(limit=24, limit_direction="both")
    piv = piv.loc[:, piv.notna().all()]
    locs = pd.read_csv(RAW / "intel_mote_locs.txt", sep=r"\s+", header=None,
                       names=["mote", "x", "y"]).set_index("mote")
    ids = piv.columns.to_numpy()
    xy = locs.loc[ids, ["x", "y"]].to_numpy(float)
    xy[:, 1] = xy[:, 1].max() - xy[:, 1]  # file's origin is the top corner; flip so north is up
    return dict(values=piv.to_numpy(float), timestamps=piv.index.to_numpy(),
                raw_positions=xy, sensor_ids=ids, train_end=int(0.55 * len(piv)))


def _build_pems():
    import pandas as pd
    X = pd.read_csv(RAW / "PEMS-BAY.csv", index_col=0)
    X.index = pd.to_datetime(X.index)
    X = X.replace(0.0, np.nan).interpolate(limit_direction="both")
    meta = pd.read_csv(RAW / "PEMS-BAY-META.csv").set_index("sensor_id")
    ids = X.columns.astype(int).to_numpy()
    lat, lon = meta.loc[ids, "Latitude"].to_numpy(), meta.loc[ids, "Longitude"].to_numpy()
    xy = np.stack([lon * np.cos(np.deg2rad(lat.mean())), lat], 1)
    train_end = int(np.searchsorted(X.index.to_numpy(), np.datetime64("2017-05-01")))
    return dict(values=X.to_numpy(float), timestamps=X.index.to_numpy(),
                raw_positions=np.stack([lon, lat], 1), sensor_ids=ids, train_end=train_end,
                _xy=xy)


_LOADED = {}


def summary(name):
    """Timestamps, size and split of a dataset without loading its readings
    (the server lists datasets at start-up; readings load on first use)."""
    _build_cache(name)
    with np.load(PROCESSED / f"{name}.npz", allow_pickle=False) as z:
        return {"timestamps": z["timestamps"], "train_end": int(z["train_end"]),
                "M": int(z["sensor_ids"].shape[0])}


def load(name) -> RealDataset:
    """Load (and on first use, build) a dataset; cached per process."""
    if name not in _LOADED:
        _LOADED[name] = _load(name)
    return _LOADED[name]


def _build_cache(name):
    cache = PROCESSED / f"{name}.npz"
    if not cache.exists():
        build = {"intel_lab": _build_intel, "pems_bay": _build_pems}[name]
        d = build()
        xy = d.pop("_xy", d["raw_positions"])
        d["positions"] = _normalize(xy)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        d["values"] = d["values"].astype(np.float32)
        np.savez_compressed(cache, **d)
    return cache


def _load(name) -> RealDataset:
    z = np.load(_build_cache(name), allow_pickle=False)
    return RealDataset(name=name, values=z["values"].astype(np.float32), timestamps=z["timestamps"],
                       positions=z["positions"], raw_positions=z["raw_positions"],
                       sensor_ids=z["sensor_ids"], train_end=int(z["train_end"]))
