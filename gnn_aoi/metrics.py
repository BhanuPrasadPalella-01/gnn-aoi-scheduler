import numpy as np


def aoci_proxy(age, adj):
    """Legacy graph-based 'age of correlated information': a source is only
    as stale as its freshest correlated neighbour (adj > 0). Kept for
    continuity with the mid-sem results; the Kalman posterior variance is
    the principled version of the same idea."""
    masked = np.where(adj > 0, age[None, :], np.inf)
    return np.minimum(age, masked.min(1))


def mean_ci(values, z=1.96):
    """Mean and 95% normal-approx confidence half-width."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(z * v.std(ddof=1) / np.sqrt(len(v)))
