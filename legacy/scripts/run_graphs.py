import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from src.simulator import CorrelatedSourceSimulator
from src.graphs import CorrelationGraphBuilder, InterferenceGraph

M = 20
sim = CorrelatedSourceSimulator(M=M, length_scale=0.3, ar_rho=0.9, seed=42)
corr_builder = CorrelationGraphBuilder(M=M, window=50)

for t in range(200):
    reading = sim.step()
    corr_builder.update(reading)

corr_G = corr_builder.to_networkx(threshold=0.3)
interf_G = InterferenceGraph(sim.positions, radius=0.25).graph

print(f"Correlation graph: {corr_G.number_of_nodes()} nodes, {corr_G.number_of_edges()} edges")
print(f"Interference graph: {interf_G.number_of_nodes()} nodes, {interf_G.number_of_edges()} edges")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
pos = {i: sim.positions[i] for i in range(M)}

nx.draw(corr_G, pos, ax=axes[0], with_labels=True, node_size=300, node_color="lightblue")
axes[0].set_title("Estimated correlation graph")

nx.draw(interf_G, pos, ax=axes[1], with_labels=True, node_size=300, node_color="salmon")
axes[1].set_title("Interference graph")

plt.tight_layout()
plt.savefig("results_graphs.png", dpi=150)
print("Saved plot to results_graphs.png")
