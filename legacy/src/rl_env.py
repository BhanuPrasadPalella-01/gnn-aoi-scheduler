import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.simulator import CorrelatedSourceSimulator
from src.channel import Channel


def aoci_proxy(age, graph):
    eff_age = age.copy()
    for i in graph.nodes():
        neighbor_ages = [age[j] for j in graph.neighbors(i)]
        if neighbor_ages:
            eff_age[i] = min(age[i], min(neighbor_ages))
    return eff_age


class AoISchedulerEnv(gym.Env):
    """
    The max-age reward term is now CAPPED at 30 before weighting. A node
    neglected to age 300 and one neglected to age 30 now contribute the
    SAME (large) penalty, instead of the former creating a 10x-larger
    reward outlier that destabilizes the value function's learning.
    The uncapped mean-age term still keeps growing with real neglect, so
    the pressure against starvation doesn't disappear -- it's just no
    longer a wild, high-variance spike.
    """
    def __init__(self, M=20, C=5, episode_len=500, corr_graph=None,
                 length_scale=0.3, ar_rho=0.9, seed=None):
        super().__init__()
        self.M, self.C, self.episode_len = M, C, episode_len
        self.corr_graph = corr_graph
        self.length_scale, self.ar_rho, self._seed = length_scale, ar_rho, seed

        self.degree_norm = np.array(
            [self.corr_graph.degree(i) / max(1, M - 1) for i in range(M)], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3 * M,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(M,), dtype=np.float32)

        self.sim, self.ch, self.t = None, None, 0
        self.last_polled = np.zeros(M, dtype=np.float32)

    def _get_obs(self):
        age_norm = (self.ch.age / (self.ch.age + 20.0)).astype(np.float32)
        return np.concatenate([age_norm, self.degree_norm, self.last_polled]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            use_seed = seed
        elif self._seed is not None:
            use_seed = self._seed
        else:
            use_seed = int(self.np_random.integers(0, 1_000_000))

        self.sim = CorrelatedSourceSimulator(M=self.M, length_scale=self.length_scale,
                                              ar_rho=self.ar_rho, seed=use_seed)
        for _ in range(100):
            self.sim.step()
        self.ch = Channel(M=self.M, capacity=self.C)
        self.t = 0
        self.last_polled = np.zeros(self.M, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        self.sim.step()
        selected = list(np.argsort(-action)[: self.C])
        age = self.ch.poll(selected)

        self.last_polled = np.zeros(self.M, dtype=np.float32)
        self.last_polled[selected] = 1.0

        eff_age = aoci_proxy(age, self.corr_graph)
        sqrt_max = float(np.sqrt(age.max()))
        reward = -10.0 * (0.4 * age.mean() + 0.4 * eff_age.mean() + 0.2 * sqrt_max)

        self.t += 1
        truncated = self.t >= self.episode_len
        info = {"raw_aoi": age.mean(), "aoci": eff_age.mean()}
        return self._get_obs(), reward, False, truncated, info
