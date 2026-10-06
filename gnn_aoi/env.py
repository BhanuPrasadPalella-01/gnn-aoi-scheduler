from dataclasses import dataclass

import numpy as np

from gnn_aoi.config import Scenario
from gnn_aoi.estimator import KalmanSink
from gnn_aoi.graphs import correlation_adjacency, interference_adjacency
from gnn_aoi.metrics import aoci_proxy
from gnn_aoi.selection import feasible_top
from gnn_aoi.simulator import CorrelatedField, fit_model


@dataclass
class Observation:
    """What a scheduler may look at when choosing this slot's polls.

    Everything here is available to a real sink. `P_true` is privileged
    (it uses the true model) and only the oracle policy reads it; on real
    data there is no true model and it equals P_hat.
    """
    age: np.ndarray          # (M,) slots since last successful poll
    last_polled: np.ndarray  # (M,) 1 if polled in the previous slot
    P_hat: np.ndarray        # (M, M) sink's predicted posterior covariance
    corr_hat: np.ndarray     # (M, M) innovation correlation estimated from the pilot phase
    adj: np.ndarray          # (M, M) weighted correlation graph
    conflict: np.ndarray     # (M, M) bool interference graph
    C: int
    rho_hat: np.ndarray      # (M,) estimated per-sensor temporal correlation
    obs_noise_var: float
    forced: list             # sensors the freshness guard says MUST be polled now
    P_true: np.ndarray       # privileged


class SchedulingEnv:
    """One sensor field + one sink.

    Slot protocol: the field advances, the sink predicts, the policy sees an
    Observation and returns <= C sensor indices, those sensors report, the
    sink updates, and per-slot metrics are emitted.

    The field is either the simulator (CorrelatedField) or a replay of a real
    dataset (datasets.TraceField); both expose the same interface. The sink
    fits its model on the pilot phase (simulator) or the training period
    (real data) and never sees the evaluation readings in advance.

    Reward (used for RL) is minus the mean posterior variance under the TRUE
    model when one exists (simulator): how much the sink could know, given
    the schedule. It is deterministic in the schedule, so it is a
    low-variance training signal. On real data it falls back to the sink's
    own posterior variance.
    """

    def __init__(self, sc: Scenario, layout_seed: int = 0, noise_seed: int | None = None,
                 record: bool = False, field=None, model=None):
        self.sc = sc
        self.M, self.C = sc.M, sc.C
        self.layout_seed = layout_seed
        self.field = field if field is not None else CorrelatedField(sc, layout_seed, noise_seed)
        assert self.field.M == self.M

        self.model = model if model is not None else fit_model(self.field.pilot(sc.pilot_len))
        self.corr_hat = self.model.innovation_corr
        self.rho_hat = self.model.rho
        self.adj = correlation_adjacency(self.corr_hat, sc.corr_threshold)
        self.conflict = interference_adjacency(self.field.positions, sc.interference_radius)

        self.sink = KalmanSink(self.model, sc.obs_noise_var)
        true = self.field.true_model()
        self.oracle = KalmanSink(true, sc.obs_noise_var, track_mean=False) if true is not None else None
        self.record = record
        self.field.mark()
        self.reset()

    @property
    def positions(self):
        return self.field.positions

    def reset(self):
        self.field.rewind()
        self.sink.reset()
        if self.oracle is not None:
            self.oracle.reset()
        self.age = np.ones(self.M)
        self.last_polled = np.zeros(self.M)
        self.t = 0
        self.trace = [] if self.record else None
        self._begin_slot()
        return self.observe()

    def _begin_slot(self):
        self.x = self.field.step()
        self.sink.predict()
        if self.oracle is not None:
            self.oracle.predict()

    def forced_polls(self):
        """Sensors that would break the freshness bound if skipped this slot
        (oldest first, conflict-feasible, at most C)."""
        F = self.sc.freshness
        if F <= 0:
            return []
        due = np.where(self.age >= F, self.age, -np.inf)
        if not np.isfinite(due).any():
            return []
        return [i for i in feasible_top(due, self.C, self.conflict) if np.isfinite(due[i])]

    def observe(self):
        self._forced = self.forced_polls()
        return Observation(
            age=self.age.copy(), last_polled=self.last_polled.copy(),
            P_hat=self.sink.P.copy(), corr_hat=self.corr_hat, adj=self.adj,
            conflict=self.conflict, C=self.C, rho_hat=self.rho_hat,
            obs_noise_var=self.sc.obs_noise_var, forced=list(self._forced),
            P_true=(self.oracle or self.sink).P.copy(),
        )

    def step(self, picks):
        picks = np.asarray(sorted(set(int(i) for i in picks)), dtype=int)
        assert len(picks) <= self.C, "exceeds channel capacity"
        assert set(self._forced) <= set(picks.tolist()), "freshness guard violated"
        if len(picks) and self.conflict.any():
            assert not self.conflict[np.ix_(picks, picks)].any(), "interfering sensors polled together"

        self.age += 1
        self.age[picks] = 1
        self.last_polled = np.zeros(self.M)
        self.last_polled[picks] = 1

        y = self.field.observe(picks)
        self.sink.update(picks, y)
        if self.oracle is not None:
            self.oracle.update(picks)

        err = self.x - self.sink.m
        pvar = float((self.oracle or self.sink).variance.mean())
        info = {
            "mse": float((err**2).mean()),
            "rmse_units": float(np.sqrt(((err * self.field.scale) ** 2).mean())),
            "pvar": pvar,
            "sink_var": float(self.sink.variance.mean()),
            "mean_age": float(self.age.mean()),
            "max_age": float(self.age.max()),
            "aoci": float(aoci_proxy(self.age, self.adj).mean()),
        }
        if self.record:
            self.trace.append({
                "picks": picks.tolist(),
                "age": self.age.astype(int).tolist(),
                "truth": self.x.round(4).tolist(),
                "estimate": self.sink.m.round(4).tolist(),
                "truth_units": self.field.to_units(self.x).round(3).tolist(),
                "estimate_units": self.field.to_units(self.sink.m).round(3).tolist(),
                "variance": self.sink.variance.round(4).tolist(),
                **{k: round(v, 5) for k, v in info.items()},
            })
        self.t += 1
        self._begin_slot()
        return self.observe(), -pvar, info
