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
    """Greedy diverse selection: pick the stalest source, then discount
    every remaining candidate by its correlation to what was just picked,
    so a second highly-redundant source isn't picked in the SAME slot.
    Raw age is never suppressed across slots -- only within-slot competition
    is affected -- so this can't starve a source the way v2 did."""
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


def run_policy(use_corr):
    sim = CorrelatedSourceSimulator(M=M, length_scale=0.3, ar_rho=0.9, seed=123)
    ch = Channel(M=M, capacity=C)
    raw_trace, aoci_trace, max_age_trace = [], [], []
    for t in range(T):
        sim.step()
        selected = correlation_aware_policy_v4(ch.age, corr_matrix, C) if use_corr else aoi_only_policy(ch.age, C)
        age = ch.poll(selected)
        raw_trace.append(age.mean())
        aoci_trace.append(aoci_proxy(age, corr_graph).mean())
        max_age_trace.append(age.max())
    return np.array(raw_trace), np.array(aoci_trace), np.array(max_age_trace)

b1_raw, b1_aoci, b1_max = run_policy(use_corr=False)
corr_raw, corr_aoci, corr_max = run_policy(use_corr=True)
ss = lambda x: x[-100:].mean()

print(f"{'Policy':<28}{'Raw AoI':>10}{'AoCI':>10}{'Max age (last 100)':>22}")
print(f"{'B1 (AoI-only)':<28}{ss(b1_raw):>10.3f}{ss(b1_aoci):>10.3f}{b1_max[-100:].max():>22.0f}")
print(f"{'Correlation-aware v4':<28}{ss(corr_raw):>10.3f}{ss(corr_aoci):>10.3f}{corr_max[-100:].max():>22.0f}")
print(f"\nAoCI improvement over B1: {(ss(b1_aoci)-ss(corr_aoci))/ss(b1_aoci)*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(b1_raw, label="B1: AoI-only"); axes[0].plot(corr_raw, label="Correlation-aware v4")
axes[0].set_title("Raw AoI"); axes[0].set_xlabel("Time slot"); axes[0].legend()
axes[1].plot(b1_aoci, label="B1: AoI-only"); axes[1].plot(corr_aoci, label="Correlation-aware v4")
axes[1].set_title("AoCI (correlation-adjusted)"); axes[1].set_xlabel("Time slot"); axes[1].legend()
plt.tight_layout()
plt.savefig("results_midsem_comparison_v4.png", dpi=150)
print("Saved plot to results_midsem_comparison_v4.png")
