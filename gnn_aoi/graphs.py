import numpy as np


def estimate_correlation(samples):
    """Sample correlation matrix from a (T, M) block of readings."""
    corr = np.corrcoef(samples, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def correlation_adjacency(corr, threshold):
    """Weighted adjacency |corr_ij| for pairs above threshold, zero diagonal."""
    A = np.abs(corr) * (np.abs(corr) >= threshold)
    np.fill_diagonal(A, 0.0)
    return A


def interference_adjacency(positions, radius):
    """Boolean conflict matrix: sensors within `radius` cannot be polled in the
    same slot. radius <= 0 disables interference."""
    M = len(positions)
    if radius <= 0:
        return np.zeros((M, M), dtype=bool)
    d = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    conflict = d <= radius
    np.fill_diagonal(conflict, False)
    return conflict

