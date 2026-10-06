import numpy as np
from sklearn.gaussian_process.kernels import RBF


class CorrelatedSourceSimulator:
    """
    M sources with spatially correlated readings (RBF kernel over random
    2D positions) evolving over time via an AR(1) process, so readings
    are correlated both across sources (space) and across time.
    """
    def __init__(self, M, length_scale=0.3, ar_rho=0.9, seed=None):
        self.M = M
        self.ar_rho = ar_rho
        self.rng = np.random.default_rng(seed)

        self.positions = self.rng.uniform(0, 1, size=(M, 2))
        kernel = RBF(length_scale=length_scale)
        self.spatial_cov = kernel(self.positions) + 1e-6 * np.eye(M)

        self.state = np.zeros(M)

    def step(self):
        innovation = self.rng.multivariate_normal(
            mean=np.zeros(self.M), cov=self.spatial_cov
        )
        self.state = self.ar_rho * self.state + np.sqrt(1 - self.ar_rho**2) * innovation
        return self.state.copy()

    def ground_truth_correlation(self):
        """Normalized spatial covariance — used later (Step 6) to validate
        the learned correlation graph against this known ground truth."""
        d = np.sqrt(np.diag(self.spatial_cov))
        return self.spatial_cov / np.outer(d, d)
