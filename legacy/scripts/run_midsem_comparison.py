import numpy as np
import matplotlib.pyplot as plt

from src.simulator import CorrelatedSourceSimulator
from src.channel import Channel
from src.baselines.aoi_only import aoi_only_policy
from src.graphs import CorrelationGraphBuilder

M = 20
C = 5
T = 500
CORR_THRESHOLD = 0.3

# Learn a stable correlation structure once (window=1000, per Step 5's finding)
sim_warmup = CorrelatedSourceSimulator(M=M, length_scale=0.3, ar_rho=0.9, seed=42)
for _ in range(100):
    sim_warmup.step()
corr_builder = CorrelationGraphBuilder(M=M, window=1000)
for _ in range(1000):
    corr_builder.update(sim_warmup.step())
corr_matrix = corr_builder.estimate()
corr_graph = corr_builder.to_networkx(threshold=CORR_THRESHOLD)


def correlation_aware_policy(age, corr_matrix, recently_polled, capacity, redundancy_weight=0.8, memory=3):
    """Score = age, discounted if a recently-polled neighbor is highly correlated
    (that neighbor's freshness already substitutes for this source's info)."""
    M = len(age)
    redundancy = np.zeros(M)
    recent = recently_polled[-memory:]
    if recent:
        for i in range(M):
            corrs = [abs(corr_matrix[i, j]) for j in recent if j != i]
            redundancy[i] = max(corrs) if corrs else 0.0
    score = age * (1 - redundancy_weight * redundancy)
    return list(np.argsort(-score)[:capacity])


def aoci_proxy(age, graph):
    """A source is only as 'stale' as its freshest correlated neighbor.
    Operationalizes the project's central idea directly."""
    eff_age = age.copy()
    for i in graph.nodes():
        neighbor_ages = [age[j] for j in graph.neighbors(i)]
        if neighbor_ages:
            eff_age[i] = min(age[i], min(neighbor_ages))
    return eff_age


def run_policy(use_corr):
    sim = CorrelatedSourceSimulator(M=M, length_scale=0.3, ar_rho=0.9, seed=123)
    ch = Channel(M=M, capacity=C)
    raw_trace, aoci_trace = [], []
    recently_polled = []
    for t in range(T):
        sim.step()
        selected = correlation_aware_policy(ch.age, corr_matrix, recently_polled, C) if use_corr else aoi_only_policy(ch.age, C)
        age = ch.poll(selected)
        raw_trace.append(age.mean())
        aoci_trace.append(aoci_proxy(age, corr_graph).mean())
        recently_polled = (recently_polled + list(selected))[-15:]
    return np.array(raw_trace), np.array(aoci_trace)

b1_raw, b1_aoci = run_policy(use_corr=False)
corr_raw, corr_aoci = run_policy(use_corr=True)
ss = lambda x: x[-100:].mean()

print(f"{'Policy':<28}{'Raw AoI':>10}{'AoCI (proxy)':>16}")
print(f"{'B1 (AoI-only)':<28}{ss(b1_raw):>10.3f}{ss(b1_aoci):>16.3f}")
print(f"{'Correlation-aware (ours)':<28}{ss(corr_raw):>10.3f}{ss(corr_aoci):>16.3f}")
print(f"\nAoCI improvement over B1: {(ss(b1_aoci)-ss(corr_aoci))/ss(b1_aoci)*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(b1_raw, label="B1: AoI-only"); axes[0].plot(corr_raw, label="Correlation-aware")
axes[0].set_title("Raw AoI"); axes[0].set_xlabel("Time slot"); axes[0].legend()
axes[1].plot(b1_aoci, label="B1: AoI-only"); axes[1].plot(corr_aoci, label="Correlation-aware")
axes[1].set_title("AoCI (correlation-adjusted)"); axes[1].set_xlabel("Time slot"); axes[1].legend()
plt.tight_layout()
plt.savefig("results_midsem_comparison.png", dpi=150)
print("Saved plot to results_midsem_comparison.png")
