import numpy as np


class KalmanSink:
    """The sink's view of the field: a Kalman filter for the linear-Gaussian
    model (see simulator.LinearGaussianModel)

        x_t = diag(rho) x_{t-1} + w_t,  w_t ~ N(0, Q)
        y_t = x_t[S] + n_t,             n_t ~ N(0, r I)

    The posterior covariance P is the natural *correlation-aware* staleness
    measure: P_ii is how uncertain the sink is about sensor i right now,
    given everything it heard from i AND from i's correlated neighbours.
    Note P evolves deterministically given the schedule, independent of the
    readings themselves.
    """

    def __init__(self, model, obs_noise_var, track_mean=True):
        self.model = model
        self.rho = model.rho
        self.Q = model.Q
        self.M = len(self.Q)
        self.r = obs_noise_var
        self.track_mean = track_mean
        self.reset()

    def reset(self):
        self.m = np.zeros(self.M)
        self.P = self.model.P0.copy()

    def predict(self):
        self.m = self.rho * self.m
        self.P = self.rho[:, None] * self.P * self.rho[None, :] + self.Q

    def update(self, idx, y=None):
        idx = np.asarray(idx, dtype=int)
        if len(idx) == 0:
            return
        PS = self.P[:, idx]
        Sm = PS[idx] + self.r * np.eye(len(idx))
        K = np.linalg.solve(Sm, PS.T).T  # (M, |S|)
        if self.track_mean and y is not None:
            self.m = self.m + K @ (y - self.m[idx])
        self.P = self.P - K @ PS.T
        self.P = 0.5 * (self.P + self.P.T)

    @property
    def variance(self):
        return np.clip(np.diag(self.P), 0.0, None)
