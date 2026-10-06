import numpy as np
import matplotlib.pyplot as plt

from src.simulator import CorrelatedSourceSimulator
from src.graphs import CorrelationGraphBuilder

M = 20
sim = CorrelatedSourceSimulator(M=M, length_scale=0.3, ar_rho=0.9, seed=42)

# burn-in so the AR(1) process reaches stationarity before we start estimating
for _ in range(100):
    sim.step()

true_corr = sim.ground_truth_correlation()

results = {}
for window in [50, 200, 1000]:
    builder = CorrelationGraphBuilder(M=M, window=window)
    for _ in range(window):
        reading = sim.step()
        builder.update(reading)
    est_corr = builder.estimate()

    iu = np.triu_indices(M, k=1)
    true_vals = true_corr[iu]
    est_vals = est_corr[iu]

    mae = np.mean(np.abs(true_vals - est_vals))
    pearson = np.corrcoef(true_vals, est_vals)[0, 1]
    results[window] = (mae, pearson, true_vals, est_vals)
    print(f"window={window}: MAE={mae:.4f}, Pearson(est vs true)={pearson:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, window in zip(axes, [50, 200, 1000]):
    mae, pearson, true_vals, est_vals = results[window]
    ax.scatter(true_vals, est_vals, alpha=0.4, s=10)
    ax.plot([-0.5, 1], [-0.5, 1], 'r--', linewidth=1)
    ax.set_xlabel("True correlation (GP ground truth)")
    ax.set_ylabel("Estimated correlation")
    ax.set_title(f"window={window}\nMAE={mae:.3f}, r={pearson:.3f}")
    ax.set_xlim(-0.5, 1)
    ax.set_ylim(-0.5, 1)

plt.tight_layout()
plt.savefig("results_corr_validation.png", dpi=150)
print("Saved plot to results_corr_validation.png")
