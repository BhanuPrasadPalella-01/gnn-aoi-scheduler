import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy

from src.simulator import CorrelatedSourceSimulator
from src.channel import Channel
from src.baselines.aoi_only import aoi_only_policy
from src.graphs import CorrelationGraphBuilder
from src.rl_env import AoISchedulerEnv

M, C = 20, 5
EPISODE_LEN = 500
TOTAL_TIMESTEPS = 1_000_000
SCENARIO_SEED = 123  # fixed scenario used ONLY for evaluation
CORR_THRESHOLD = 0.3

sim_warmup = CorrelatedSourceSimulator(M=M, length_scale=0.3, ar_rho=0.9, seed=42)
for _ in range(100):
    sim_warmup.step()
corr_builder = CorrelationGraphBuilder(M=M, window=1000)
for _ in range(1000):
    corr_builder.update(sim_warmup.step())
corr_matrix = corr_builder.estimate()
corr_graph = corr_builder.to_networkx(threshold=CORR_THRESHOLD)


def aoci_proxy(age, graph):
    eff_age = age.copy()
    for i in graph.nodes():
        neighbor_ages = [age[j] for j in graph.neighbors(i)]
        if neighbor_ages:
            eff_age[i] = min(age[i], min(neighbor_ages))
    return eff_age


def correlation_aware_policy_v4(age, corr_matrix, capacity):
    M = len(age)
    remaining = age.astype(float).copy()
    candidates = list(range(M))
    selected = []
    for _ in range(capacity):
        i = max(candidates, key=lambda x: remaining[x])
        selected.append(i)
        candidates.remove(i)
        for j in candidates:
            remaining[j] *= (1 - abs(corr_matrix[i, j]))
    return selected


def make_env():
    return AoISchedulerEnv(M=M, C=C, episode_len=EPISODE_LEN, corr_graph=corr_graph, seed=None)

vec_env = make_vec_env(make_env, n_envs=4, monitor_dir="logs")

model = PPO(
    "MlpPolicy", vec_env,
    n_steps=500, batch_size=500, learning_rate=3e-4, gamma=0.99,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    device="cpu", verbose=1,
)
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save("checkpoints/ppo_scheduler")
print("Saved PPO checkpoint to checkpoints/ppo_scheduler.zip")

df = load_results("logs")
x, y = ts2xy(df, "timesteps")
window = 20
y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
x_smooth = x[window - 1:]

plt.figure(figsize=(8, 5))
plt.plot(x, y, alpha=0.25, color="gray", label="raw episode reward")
plt.plot(x_smooth, y_smooth, color="#45D6C0", label=f"{window}-episode moving average")
plt.xlabel("Training timesteps"); plt.ylabel("Episode reward"); plt.title("PPO training curve"); plt.legend()
plt.tight_layout()
plt.savefig("results_rl_training_curve.png", dpi=150)
print("Saved plot to results_rl_training_curve.png")

def run_baseline(policy_fn):
    sim = CorrelatedSourceSimulator(M=M, length_scale=0.3, ar_rho=0.9, seed=SCENARIO_SEED)
    ch = Channel(M=M, capacity=C)
    raw, aoci, maxage = [], [], []
    for t in range(500):
        sim.step()
        selected = policy_fn(ch.age)
        age = ch.poll(selected)
        raw.append(age.mean()); aoci.append(aoci_proxy(age, corr_graph).mean()); maxage.append(age.max())
    return np.array(raw), np.array(aoci), np.array(maxage)

b1_raw, b1_aoci, b1_max = run_baseline(lambda age: aoi_only_policy(age, C))
v4_raw, v4_aoci, v4_max = run_baseline(lambda age: correlation_aware_policy_v4(age, corr_matrix, C))

eval_env = AoISchedulerEnv(M=M, C=C, episode_len=600, corr_graph=corr_graph, seed=SCENARIO_SEED)
obs, _ = eval_env.reset()
ppo_raw, ppo_aoci, ppo_max = [], [], []
for t in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, info = eval_env.step(action)
    ppo_raw.append(info["raw_aoi"]); ppo_aoci.append(info["aoci"]); ppo_max.append(eval_env.ch.age.max())
ppo_raw, ppo_aoci, ppo_max = np.array(ppo_raw), np.array(ppo_aoci), np.array(ppo_max)

ss = lambda x: x[-100:].mean()
print(f"\n{'Policy':<24}{'Raw AoI':>10}{'AoCI':>10}{'Max age':>10}")
print(f"{'B1 (AoI-only)':<24}{ss(b1_raw):>10.3f}{ss(b1_aoci):>10.3f}{b1_max[-100:].max():>10.0f}")
print(f"{'v4 (greedy heuristic)':<24}{ss(v4_raw):>10.3f}{ss(v4_aoci):>10.3f}{v4_max[-100:].max():>10.0f}")
print(f"{'PPO (learned)':<24}{ss(ppo_raw):>10.3f}{ss(ppo_aoci):>10.3f}{ppo_max[-100:].max():>10.0f}")
print(f"\nPPO AoCI vs B1:  {(ss(b1_aoci)-ss(ppo_aoci))/ss(b1_aoci)*100:+.1f}%")
print(f"PPO AoCI vs v4:  {(ss(v4_aoci)-ss(ppo_aoci))/ss(v4_aoci)*100:+.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(b1_raw, label="B1", alpha=0.7); axes[0].plot(v4_raw, label="v4 heuristic", alpha=0.7); axes[0].plot(ppo_raw, label="PPO")
axes[0].set_title("Raw AoI"); axes[0].set_xlabel("Time slot"); axes[0].legend()
axes[1].plot(b1_aoci, label="B1", alpha=0.7); axes[1].plot(v4_aoci, label="v4 heuristic", alpha=0.7); axes[1].plot(ppo_aoci, label="PPO")
axes[1].set_title("AoCI"); axes[1].set_xlabel("Time slot"); axes[1].legend()
plt.tight_layout()
plt.savefig("results_rl_final_comparison.png", dpi=150)
print("Saved plot to results_rl_final_comparison.png")
