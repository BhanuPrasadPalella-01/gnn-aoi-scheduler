import numpy as np

from gnn_aoi.selection import feasible_top, greedy_variance_reduction


class Policy:
    """A scheduler. `select(obs)` returns the sensors to poll this slot; it
    must contain every sensor in `obs.forced` (freshness guard), respect
    `obs.conflict`, and have at most `obs.C` entries."""
    name = "policy"
    label = "Policy"

    def reset(self, env):
        pass

    def select(self, obs):
        raise NotImplementedError

    def priority(self, obs):
        """Per-sensor score behind this slot's choice (higher = more urgent),
        for explaining decisions in the live demo. None if not meaningful."""
        return None


class RandomPolicy(Policy):
    name, label = "random", "Random"

    def __init__(self, seed=0):
        self.seed = seed

    def reset(self, env):
        self.rng = np.random.default_rng(self.seed + env.layout_seed)

    def select(self, obs):
        return feasible_top(self.rng.random(len(obs.age)), obs.C, obs.conflict, obs.forced)


class MaxAgePolicy(Policy):
    """B1: correlation-agnostic max-age (round robin in steady state)."""
    name, label = "max_age", "Max-AoI (B1)"

    def select(self, obs):
        return feasible_top(obs.age, obs.C, obs.conflict, obs.forced)

    def priority(self, obs):
        return obs.age.astype(float)


class GreedyDiversePolicy(Policy):
    """Legacy v4 heuristic: take the stalest sensor, then discount every
    remaining candidate's age by its |correlation| with what was picked."""
    name, label = "greedy_diverse", "Corr-diverse (v4)"

    def select(self, obs):
        score = obs.age.astype(float).copy()
        allowed = np.ones(len(score), dtype=bool)
        picks = []

        def take(i):
            picks.append(int(i))
            allowed[i] = False
            allowed[:] &= ~obs.conflict[i]
            score[:] *= 1 - np.abs(obs.corr_hat[i])

        for i in obs.forced:
            take(i)
        while len(picks) < obs.C and allowed.any():
            take(np.argmax(np.where(allowed, score, -np.inf)))
        return picks


class MaxVariancePolicy(Policy):
    """Poll the sensors the sink is most uncertain about, with no
    within-slot redundancy check. Ablation of Kalman-greedy."""
    name, label = "max_variance", "Max-uncertainty"

    def select(self, obs):
        return feasible_top(np.diag(obs.P_hat), obs.C, obs.conflict, obs.forced)

    def priority(self, obs):
        return np.diag(obs.P_hat).copy()


class KalmanGreedyPolicy(Policy):
    """Greedy posterior-variance reduction using the sink's *estimated*
    covariance: the strongest one-step model-based scheduler a real sink
    can run."""
    name, label = "kalman_greedy", "Kalman-greedy"

    def select(self, obs):
        return greedy_variance_reduction(obs.P_hat, obs.C, obs.obs_noise_var, obs.conflict, obs.forced)

    def priority(self, obs):
        P = obs.P_hat
        return (P**2).sum(0) / (np.diag(P) + obs.obs_noise_var)  # variance removed if polled


class OraclePolicy(Policy):
    """Kalman-greedy with the TRUE covariance. Not deployable; reference."""
    name, label = "oracle", "Oracle-greedy (true Σ)"

    def select(self, obs):
        return greedy_variance_reduction(obs.P_true, obs.C, obs.obs_noise_var, obs.conflict, obs.forced)


BASELINES = {p.name: p for p in (RandomPolicy, MaxAgePolicy, GreedyDiversePolicy,
                                  MaxVariancePolicy, KalmanGreedyPolicy, OraclePolicy)}
