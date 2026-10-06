from stable_baselines3.common.env_checker import check_env

from src.simulator import CorrelatedSourceSimulator
from src.graphs import CorrelationGraphBuilder
from src.rl_env import AoISchedulerEnv

M = 20
sim_warmup = CorrelatedSourceSimulator(M=M, length_scale=0.3, ar_rho=0.9, seed=42)
for _ in range(100):
    sim_warmup.step()
corr_builder = CorrelationGraphBuilder(M=M, window=1000)
for _ in range(1000):
    corr_builder.update(sim_warmup.step())
corr_graph = corr_builder.to_networkx(threshold=0.3)

env = AoISchedulerEnv(M=M, C=5, corr_graph=corr_graph, seed=1)
check_env(env, warn=True)
print("Environment passed check_env -- safe to train on.")

obs, _ = env.reset()
print(f"obs shape: {obs.shape}, action shape: {env.action_space.shape}")
for _ in range(3):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(f"reward={reward:.3f}, raw_aoi={info['raw_aoi']:.3f}, aoci={info['aoci']:.3f}")
