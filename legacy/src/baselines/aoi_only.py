import numpy as np


def aoi_only_policy(age, capacity):
    """B1: correlation-agnostic. Poll the `capacity` sources with the
    highest current age (classic max-age / round-robin policy)."""
    return list(np.argsort(-age)[:capacity])
