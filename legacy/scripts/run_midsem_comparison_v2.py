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

sim_warmup = CorrelatedSourceSimulator(M=M, length_scale=0.3, ar_rho=0.9, seed=42)
for _ in range(100):
    sim_warmup.step()
corr_builder = CorrelationGraphBuilder(M=M, window=1000)
for _ in range(1000):
    corr_builder.update(sim_warmup.step())
corr_graph = corr_builder.to_networkx(threshold=CORR_THRESHOLD)


def aoci_proxy(age, graph):
    eff_age = age.copy()
    for i in graph.nodes():
        neighbor_ages = [age[j] for j in graph.neighbors(i)]
        if neighbor_ages:
            eff_age[i] = min(age[i], min(neighbor_ages))
    return eff_age


def correlation_aware_policy_v2(age, graph, capacity):
    """Directly poll whichever sources currently have the highest EFFECTIVE
    (AoCI) age -- aligned with the metric itself, not a proxy for it."""
    eff_age = aoci_proxy(age, graph)
    return list(np.argsort(-eff_age)[:capacity])


def run_policy(use_corr):
    sim = CorrelatedSourceSimulator(M=M, length_scale=0.3, ar_rho=0.9, seed=123)
    ch = Channel(M=M, capacity=C)
    raw_trace, aoci_trace = [], []
    for t in range(T):
        sim.step()
        selected = correlation_aware_policy_v2(ch.age, corr_graph, C) if use_corr else aoi_only_policy(ch.age, C)
        age = ch.poll(selected)
        raw_trace.append(age.mean())
        aoci_trace.append(aoci_proxy(age, corr_graph).mean())
    return np.array(raw_trace), np.array(aoci_trace)

b1_raw, b1_aoci = run_policy(use_corr=False)
corr_raw, corr_aoci = run_policy(use_corr=True)
ss = lambda x: x[-100:].mean()

print(f"{'Policy':<28}{'Raw AoI':>10}{'AoCI (proxy)':>16}")
print(f"{'B1 (AoI-only)':<28}{ss(b1_raw):>10.3f}{ss(b1_aoci):>16.3f}")
print(f"{'Correlation-aware v2 (ours)':<28}{ss(corr_raw):>10.3f}{ss(corr_aoci):>16.3f}")
print(f"\nAoCI improvement over B1: {(ss(b1_aoci)-ss(corr_aoci))/ss(b1_aoci)*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(b1_raw, label="B1: AoI-only"); axes[0].plot(corr_raw, label="Correlation-aware v2")
axes[0].set_title("Raw AoI"); axes[0].set_xlabel("Time slot"); axes[0].legend()
axes[1].plot(b1_aoci, label="B1: AoI-only"); axes[1].plot(corr_aoci, label="Correlation-aware v2")
axes[1].set_title("AoCI (correlation-adjusted)"); axes[1].set_xlabel("Time slot"); axes[1].legend()
plt.tight_layout()
plt.savefig("results_midsem_comparison_v2.png", dpi=150)
print("Saved plot to results_midsem_comparison_v2.png")
