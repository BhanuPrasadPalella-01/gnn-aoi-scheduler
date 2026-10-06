import numpy as np
import networkx as nx
from collections import deque


class CorrelationGraphBuilder:
    """
    Maintains a rolling window of readings and estimates a correlation
    graph online. This is the piece that will later be validated against
    ground-truth GP covariance (Step 6), and eventually replaced/improved
    by the learned GNN.
    """
    def __init__(self, M, window=50):
        self.M = M
        self.window = window
        self.buffer = deque(maxlen=window)

    def update(self, reading):
        self.buffer.append(reading.copy())

    def estimate(self):
        """Returns an MxM normalized correlation matrix. Returns identity
        until enough samples have accumulated."""
        if len(self.buffer) < 2:
            return np.eye(self.M)
        data = np.array(self.buffer)  # shape (T, M)
        corr = np.corrcoef(data, rowvar=False)
        return np.nan_to_num(corr, nan=0.0)

    def to_networkx(self, threshold=0.3):
        corr = self.estimate()
        G = nx.Graph()
        G.add_nodes_from(range(self.M))
        for i in range(self.M):
            for j in range(i + 1, self.M):
                if abs(corr[i, j]) >= threshold:
                    G.add_edge(i, j, weight=corr[i, j])
        return G


class InterferenceGraph:
    """
    Static conflict graph: two sources conflict (cannot be polled in the
    same slot) if they are within `radius` of each other in physical
    space. Deliberately simple for now (Phase 3 says: start simple,
    extend later).
    """
    def __init__(self, positions, radius=0.25):
        self.M = positions.shape[0]
        self.positions = positions
        self.radius = radius
        self.graph = self._build()

    def _build(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.M))
        for i in range(self.M):
            for j in range(i + 1, self.M):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                if d <= self.radius:
                    G.add_edge(i, j)
        return G

    def conflicts_with(self, node):
        return set(self.graph.neighbors(node))
