import numpy as np

from gnn_aoi.config import Scenario


def make_layout(sc: Scenario, rng):
    """Sensor positions in the unit square."""
    if sc.layout == "uniform":
        return rng.uniform(0, 1, size=(sc.M, 2))
    if sc.layout == "clustered":
        centers = rng.uniform(0.15, 0.85, size=(sc.n_clusters, 2))
        which = rng.integers(0, sc.n_clusters, size=sc.M)
        pos = centers[which] + rng.normal(0, sc.cluster_std, size=(sc.M, 2))
        return np.clip(pos, 0, 1)
    raise ValueError(f"unknown layout {sc.layout!r}")


def rbf_covariance(positions, length_scale, jitter=1e-6):
    d2 = ((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * length_scale**2)) + jitter * np.eye(len(positions))


class LinearGaussianModel:
    """x_t = diag(rho) x_{t-1} + w_t,  w_t ~ N(0, Q).

    The model the sink's Kalman filter runs. For the simulator it is the
    truth; for real data it is fitted from the training period (fit_model).
    P0 is the stationary covariance (the sink's prior when it knows nothing).
    """

    def __init__(self, rho, Q, P0):
        self.rho = np.asarray(rho, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.P0 = np.asarray(P0, dtype=float)

    @property
    def innovation_corr(self):
        d = np.sqrt(np.clip(np.diag(self.Q), 1e-12, None))
        return self.Q / np.outer(d, d)


def fit_model(samples, rho_max=0.9995):
    """Fit per-sensor AR(1) dynamics with correlated innovations from a
    (T, M) block of standardised readings (least squares on one-step
    changes, then the residual covariance). Works for both the simulator's
    pilot phase and the training period of a real dataset.

    (Matching the autocorrelation at longer lags instead was tried on the
    real datasets and made the sink 4-17x overconfident; one-step is best.)
    """
    x0, x1 = samples[:-1], samples[1:]
    rho = np.clip((x0 * x1).sum(0) / (x0**2).sum(0), 0.0, rho_max)
    res = x1 - rho * x0
    Q = np.cov(res, rowvar=False) + 1e-6 * np.eye(samples.shape[1])
    P0 = np.cov(samples, rowvar=False) + 1e-6 * np.eye(samples.shape[1])
    return LinearGaussianModel(rho, Q, P0)


class CorrelatedField:
    """M spatially correlated sources, each a stationary AR(1) process with
    its own speed rho_i and spatially correlated innovations:

        x_t = diag(rho) x_{t-1} + D w_t,   w_t ~ N(0, Sigma),  D = diag(sqrt(1 - rho_i^2))

    Every source has unit variance. The stationary covariance is
    P_ij = Q_ij / (1 - rho_i rho_j), which is PSD by the Schur product theorem.
    With rho_spread = 0 this is the original single-rho model.

    The *layout* (sensor positions => Sigma, rho) and the *noise* use
    separate seeds. The legacy code used one seed for both, which meant the
    correlation graph built on seed 42 described a different set of sensors
    than the seed-123 scenario it was evaluated on.
    """

    units = "σ"
    scale = 1.0

    def __init__(self, sc: Scenario, layout_seed: int, noise_seed: int | None = None):
        self.sc = sc
        self.M = sc.M
        lrng = np.random.default_rng(layout_seed)
        self.positions = make_layout(sc, lrng)
        self.Sigma = rbf_covariance(self.positions, sc.length_scale)
        self.rho = np.clip(sc.ar_rho + lrng.uniform(-sc.rho_spread, sc.rho_spread, sc.M), 0.0, 0.999)
        D = np.sqrt(1 - self.rho**2)
        Q = self.Sigma * np.outer(D, D)
        P0 = Q / (1 - np.outer(self.rho, self.rho))
        self.model = LinearGaussianModel(self.rho, Q, P0)
        self._LQ = np.linalg.cholesky(Q + 1e-9 * np.eye(self.M))
        seed = layout_seed * 7919 + 17 if noise_seed is None else noise_seed
        self.rng = np.random.default_rng(seed)
        # separate stream so every policy sees the identical field trajectory
        self.obs_rng = np.random.default_rng(seed + 1)
        self.state = np.linalg.cholesky(P0 + 1e-9 * np.eye(self.M)) @ self.rng.standard_normal(self.M)

    def true_model(self):
        return self.model

    def mark(self):
        """Remember the current point so rewind() replays the exact same
        trajectory (every policy is evaluated on identical readings)."""
        import copy
        self._mark = (self.state.copy(), copy.deepcopy(self.rng), copy.deepcopy(self.obs_rng))

    def rewind(self):
        import copy
        state, rng, obs_rng = self._mark
        self.state, self.rng, self.obs_rng = state.copy(), copy.deepcopy(rng), copy.deepcopy(obs_rng)

    def to_units(self, x):
        return np.asarray(x)

    def step(self):
        self.state = self.rho * self.state + self._LQ @ self.rng.standard_normal(self.M)
        return self.state.copy()

    def observe(self, idx):
        idx = np.asarray(idx, dtype=int)
        noise = np.sqrt(self.sc.obs_noise_var) * self.obs_rng.standard_normal(len(idx))
        return self.state[idx] + noise

    def pilot(self, n):
        """n consecutive full snapshots (a pilot phase in which the sink
        hears every sensor) used to estimate the correlation structure."""
        return np.stack([self.step() for _ in range(n)])

    def ground_truth_correlation(self):
        d = np.sqrt(np.diag(self.Sigma))
        return self.Sigma / np.outer(d, d)
