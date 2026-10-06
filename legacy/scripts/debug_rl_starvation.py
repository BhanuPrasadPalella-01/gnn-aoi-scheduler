import numpy as np
from stable_baselines3 import PPO

from src.simulator import CorrelatedSourceSimulator
from src.graphs import CorrelationGraphBuilder
from src.rl_env import AoISchedulerEnv

M, C = 20, 5
SCENARIO_SEED = 123
CORR_THRESHOLD = 0.3

sim_warmup = CorrelatedSourceSimulator(M=M, length_scale=0.3, ar_rho=0.9, seed=42)
for _ in range(100):
    sim_warmup.step()
corr_builder = CorrelationGraphBuilder(M=M, window=1000)
for _ in range(1000):
    corr_builder.update(sim_warmup.step())
corr_graph = corr_builder.to_networkx(threshold=CORR_THRESHOLD)

print("Correlation-graph degree per node:")
for i in range(M):
    print(f"  node {i:2d}: degree={corr_graph.degree(i)}")

model = PPO.load("checkpoints/ppo_scheduler")
env = AoISchedulerEnv(M=M, C=C, episode_len=600, corr_graph=corr_graph, seed=SCENARIO_SEED)
obs, _ = env.reset()

poll_counts = np.zeros(M, dtype=int)
sample_actions = []
for t in range(500):
    action, _ = model.predict(obs, deterministic=True)
    if t < 5 or t % 100 == 0:
        sample_actions.append((t, action.copy()))
    selected = list(np.argsort(-action)[:C])
    for i in selected:
        poll_counts[i] += 1
    obs, reward, term, trunc, info = env.step(action)

print("\nPoll count per node over 500 eval steps (should sum to 2500):")
for i in range(M):
    flag = "  <-- NEVER POLLED" if poll_counts[i] == 0 else ""
    print(f"  node {i:2d}: polled {poll_counts[i]:4d} times, degree={corr_graph.degree(i)}{flag}")

starved = [i for i in range(M) if poll_counts[i] == 0]
print(f"\nStarved node(s): {starved}")

print("\nSample raw action vectors (rank of each node's score, 1=highest priority):")
for t, a in sample_actions:
    ranks = np.argsort(np.argsort(-a)) + 1
    print(f"  t={t:3d}: " + " ".join(f"n{i}:{ranks[i]:2d}" for i in range(M)))
