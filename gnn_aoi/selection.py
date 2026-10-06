import numpy as np


def feasible_top(scores, C, conflict, start=()):
    """Highest-scoring sensors, skipping any that interfere with one already
    chosen. `start` are picks already committed (e.g. forced by the
    freshness guard). Ties are broken by index, so equal ages fall back to
    round robin."""
    picks = list(start)
    blocked = np.zeros(len(scores), dtype=bool)
    for i in picks:
        blocked[i] = True
        blocked |= conflict[i]
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        if len(picks) >= C:
            break
        if blocked[i]:
            continue
        picks.append(int(i))
        blocked[i] = True
        blocked |= conflict[i]
    return picks


def greedy_variance_reduction(P, C, r, conflict, start=()):
    """Pick sensors one at a time, each maximising the drop in trace(P)
    given the picks so far (the classic greedy for sensor selection; close
    to optimal within a slot because the objective is near-submodular).
    It is myopic across slots, which is where a learned policy can win."""
    P = P.copy()
    allowed = np.ones(len(P), dtype=bool)
    picks = []

    def take(j):
        nonlocal P
        picks.append(int(j))
        allowed[j] = False
        allowed[:] &= ~conflict[j]
        pj = P[:, j].copy()
        P -= np.outer(pj, pj) / (pj[j] + r)

    for j in start:
        take(j)
    while len(picks) < C and allowed.any():
        gain = (P**2).sum(0) / (np.diag(P) + r)
        take(int(np.argmax(np.where(allowed, gain, -np.inf))))
    return picks
