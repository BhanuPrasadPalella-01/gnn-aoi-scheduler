import numpy as np
import matplotlib.pyplot as plt

from src.simulator import CorrelatedSourceSimulator
from src.channel import Channel
from src.baselines.aoi_only import aoi_only_policy

M = 20   # number of sources
C = 5    # channel capacity (max polls per slot)
T = 500  # simulation length in slots

sim = CorrelatedSourceSimulator(M=M, length_scale=0.3, ar_rho=0.9, seed=42)
ch = Channel(M=M, capacity=C)

mean_age_trace = []
for t in range(T):
    sim.step()  # readings generated each slot (not used by B1, but wired for later baselines)
    selected = aoi_only_policy(ch.age, C)
    age = ch.poll(selected)
    mean_age_trace.append(age.mean())

steady_state = np.mean(mean_age_trace[-100:])
print(f"Steady-state mean AoI (last 100 slots): {steady_state:.3f}")

plt.plot(mean_age_trace)
plt.xlabel("Time slot")
plt.ylabel("Mean AoI across sources")
plt.title(f"B1 (AoI-only) baseline — M={M}, C={C}")
plt.savefig("results_b1_mean_aoi.png", dpi=150)
print("Saved plot to results_b1_mean_aoi.png")
