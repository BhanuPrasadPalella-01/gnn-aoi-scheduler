from pathlib import Path

import numpy as np
import pytest
import torch

from gnn_aoi.config import Scenario
from gnn_aoi.env import SchedulingEnv
from gnn_aoi.estimator import KalmanSink
from gnn_aoi.evaluate import run_episode
from gnn_aoi.gnn.features import featurize
from gnn_aoi.gnn.model import Actor
from gnn_aoi.graphs import estimate_correlation
from gnn_aoi.metrics import aoci_proxy
from gnn_aoi.policies.baselines import BASELINES, KalmanGreedyPolicy
from gnn_aoi.selection import greedy_variance_reduction
from gnn_aoi.simulator import CorrelatedField, LinearGaussianModel, fit_model


def test_layout_seed_controls_positions_and_noise_seed_does_not():
    sc = Scenario()
    a = CorrelatedField(sc, layout_seed=1, noise_seed=5)
    b = CorrelatedField(sc, layout_seed=1, noise_seed=6)
    c = CorrelatedField(sc, layout_seed=2, noise_seed=5)
    assert np.allclose(a.positions, b.positions)
    assert not np.allclose(a.positions, c.positions)
    assert not np.allclose(a.step(), b.step())


def test_field_is_stationary_with_target_covariance():
    f = CorrelatedField(Scenario(M=8, ar_rho=0.5), layout_seed=3)
    x = f.pilot(20_000)
    assert np.allclose(x.var(0), 1.0, atol=0.06)
    est = estimate_correlation(x)
    assert np.abs(est - f.ground_truth_correlation()).max() < 0.06


def test_env_graph_describes_its_own_sensors():
    """Regression test for the legacy bug: the correlation graph must be
    estimated on the same layout the scheduler runs on."""
    env = SchedulingEnv(Scenario(), layout_seed=10_000)
    iu = np.triu_indices(env.M, 1)
    r = np.corrcoef(env.corr_hat[iu], env.field.ground_truth_correlation()[iu])[0, 1]
    assert r > 0.95


def test_kalman_variance_is_calibrated():
    """With the true model, empirical MSE should match the filter's own
    posterior variance."""
    sc = Scenario(M=10, C=3)
    env = SchedulingEnv(sc, layout_seed=4)
    sink = KalmanSink(env.field.true_model(), sc.obs_noise_var)
    errs, pvars = [], []
    rng = np.random.default_rng(0)
    for _ in range(4000):
        x = env.field.step()
        sink.predict()
        idx = rng.choice(sc.M, sc.C, replace=False)
        sink.update(idx, env.field.observe(idx))
        errs.append(((x - sink.m) ** 2).mean())
        pvars.append(sink.variance.mean())
    assert np.mean(errs) == pytest.approx(np.mean(pvars), rel=0.08)


def test_polling_reduces_uncertainty():
    S = np.eye(3) * 0.5 + 0.5
    sink = KalmanSink(LinearGaussianModel(np.full(3, 0.9), 0.19 * S, S), 1e-3)
    sink.predict()
    before = sink.variance.copy()
    sink.update([0], np.array([0.3]))
    assert sink.variance[0] < 0.01 < before[0]
    assert np.all(sink.variance[1:] < before[1:])  # correlated neighbours learn too


def test_greedy_respects_capacity_conflicts_and_forced():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(12, 12))
    P = A @ A.T / 12 + np.eye(12) * 0.1
    conflict = np.zeros((12, 12), dtype=bool)
    conflict[0, 1] = conflict[1, 0] = True
    picks = greedy_variance_reduction(P, 4, 1e-3, conflict, start=[0])
    assert picks[0] == 0 and len(picks) == 4 and 1 not in picks and len(set(picks)) == 4


@pytest.mark.parametrize("name", list(BASELINES))
def test_every_baseline_keeps_freshness_and_interference(name):
    sc = Scenario(M=16, C=4, freshness=8, interference_radius=0.15)
    env = SchedulingEnv(sc, layout_seed=7)
    traces, summary = run_episode(env, BASELINES[name](), T=120, burn=0)
    assert summary["max_age"] <= 8  # env.step also asserts conflicts/forced/capacity


def test_aoci_proxy_matches_legacy_definition():
    rng = np.random.default_rng(1)
    age = rng.integers(1, 10, size=8).astype(float)
    adj = (rng.random((8, 8)) > 0.6).astype(float)
    adj = np.triu(adj, 1); adj = adj + adj.T
    legacy = age.copy()
    for i in range(8):
        nb = [age[j] for j in range(8) if adj[i, j] > 0]
        if nb:
            legacy[i] = min(age[i], min(nb))
    assert np.allclose(aoci_proxy(age, adj), legacy)


def test_untrained_actor_reproduces_kalman_greedy():
    """Residual design: f_theta = G = 0 at init => greedy decode == Kalman-greedy."""
    torch.manual_seed(0)
    actor = Actor()
    sc = Scenario(freshness=12)
    env = SchedulingEnv(sc, layout_seed=11)
    obs = env.reset()
    teacher = KalmanGreedyPolicy()
    for _ in range(40):
        picks, _, _ = actor.decode(featurize([env], [obs]), greedy=True)
        mine = sorted(obs.forced + [int(a) for a in picks[0] if a >= 0])
        assert mine == sorted(teacher.select(obs))
        obs, _, _ = env.step(mine)


def test_decode_logprob_is_consistent():
    torch.manual_seed(0)
    actor = Actor()
    envs = [SchedulingEnv(Scenario(M=12, C=3, interference_radius=0.2), layout_seed=s) for s in range(4)]
    batch = featurize(envs, [e.reset() for e in envs])
    picks, logp, _ = actor.decode(batch, generator=torch.Generator().manual_seed(1))
    _, logp2, _ = actor.decode(batch, actions=picks)
    assert torch.allclose(logp, logp2, atol=1e-5)


def test_fit_model_recovers_per_sensor_dynamics():
    sc = Scenario(M=10, ar_rho=0.85, rho_spread=0.1)
    f = CorrelatedField(sc, layout_seed=5)
    m = fit_model(f.pilot(20_000))
    assert np.abs(m.rho - f.rho).max() < 0.02
    assert np.abs(m.Q - f.true_model().Q).max() < 0.03


def test_heterogeneous_field_is_stationary_unit_variance():
    f = CorrelatedField(Scenario(M=8, ar_rho=0.8, rho_spread=0.15), layout_seed=2)
    assert np.allclose(np.diag(f.true_model().P0), 1.0)
    assert np.all(np.linalg.eigvalsh(f.true_model().P0) > 0)


real = pytest.mark.skipif(
    not all((Path("data/processed") / f"{n}.npz").exists() for n in ("intel_lab", "pems_bay")),
    reason="processed real datasets missing (python -m gnn_aoi download)")


@real
@pytest.mark.parametrize("name", ["intel_lab", "pems_bay"])
def test_real_replay_has_no_test_leakage_and_exact_readings(name):
    from gnn_aoi.datasets import load
    ds = load(name)
    starts = ds.test_windows(5, 100)
    assert min(starts) >= ds.train_end
    f = ds.field(starts[0])
    assert len(f.pilot()) == ds.train_end  # sink fits on training period only
    x = f.step()
    assert np.allclose(f.to_units(x), ds.values[starts[0]], atol=1e-3)  # float32 storage
    assert np.allclose(f.observe([0, 1]), x[:2])
    assert f.true_model() is None


def test_reset_replays_identical_readings():
    env = SchedulingEnv(Scenario(M=8, C=2), layout_seed=3)
    a = [env.x.copy()] + [env.step([0, 1])[2]["mse"] for _ in range(5)]
    env.reset()
    b = [env.x.copy()] + [env.step([0, 1])[2]["mse"] for _ in range(5)]
    assert np.allclose(a[0], b[0]) and np.allclose(a[1:], b[1:])


@real
def test_replay_never_crosses_a_recording_gap():
    from gnn_aoi.datasets import load
    from gnn_aoi.live import LiveSession
    ds = load("intel_lab")
    assert ds.outages(), "Intel Lab is known to have outages"
    for s in ds.test_windows(20, 200, burn=0):
        assert ds.segment_end[s] >= s + 199
    # a live session started just before an outage stops cleanly at it
    gap_start = int(np.where(np.diff(ds.timestamps.astype("datetime64[m]")).astype(int) != 5)[0][0])
    sess = LiveSession("intel_lab", policies=("max_age",), start=gap_start - 3)
    n = 0
    with pytest.raises(IndexError, match="gap"):
        for _ in range(10):
            sess.step()
            n += 1
    assert n == 4  # slots gap_start-3 .. gap_start, then stop
