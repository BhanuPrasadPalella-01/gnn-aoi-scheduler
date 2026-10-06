import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.simulator import CorrelatedSourceSimulator
from src.graphs import CorrelationGraphBuilder

M = 20
C = 5
device = torch.device("cpu")  # small graphs — CPU is plenty fast and avoids MPS overhead here

sim = CorrelatedSourceSimulator(M=M, length_scale=0.3, ar_rho=0.9, seed=42)
for _ in range(100):
    sim.step()  # burn-in

# Build a stable correlation graph (window=1000, per Step 5's finding)
corr_builder = CorrelationGraphBuilder(M=M, window=1000)
for _ in range(1000):
    corr_builder.update(sim.step())

corr_G = corr_builder.to_networkx(threshold=0.3)
edges = list(corr_G.edges())
edge_index = torch.tensor(edges + [(b, a) for a, b in edges], dtype=torch.long).t().contiguous()
print(f"Correlation graph for GNN: {corr_G.number_of_nodes()} nodes, {corr_G.number_of_edges()} edges")

# Build dataset: each sample = one time slot, with a random poll mask.
# x = [reading if polled else 0, polled flag]. Target = true reading for ALL nodes,
# but loss is only computed on the UN-polled ones.
rng = np.random.default_rng(0)
dataset = []
for _ in range(2000):
    reading = sim.step()
    polled = rng.choice(M, size=C, replace=False)
    polled_mask = np.zeros(M, dtype=bool)
    polled_mask[polled] = True

    x = np.stack([
        np.where(polled_mask, reading, 0.0),
        polled_mask.astype(np.float32),
    ], axis=1)

    dataset.append(Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(reading, dtype=torch.float32),
        mask=torch.tensor(~polled_mask),
    ))

split = int(0.8 * len(dataset))
train_loader = DataLoader(dataset[:split], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset[split:], batch_size=32)


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden=32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.head(h).squeeze(-1)


model = GraphSAGEEncoder().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def masked_mse(pred, y, mask):
    return F.mse_loss(pred[mask], y[mask])

for epoch in range(60):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        opt.zero_grad()
        pred = model(batch.x, batch.edge_index)
        loss = masked_mse(pred, batch.y, batch.mask)
        loss.backward()
        opt.step()
        total_loss += loss.item() * batch.num_graphs
    if epoch % 10 == 0 or epoch == 59:
        print(f"epoch {epoch}: train masked MSE = {total_loss/split:.4f}")

model.eval()
test_loss, baseline_loss = 0.0, 0.0
with torch.no_grad():
    for batch in test_loader:
        pred = model(batch.x, batch.edge_index)
        test_loss += masked_mse(pred, batch.y, batch.mask).item() * batch.num_graphs
        zero_pred = torch.zeros_like(batch.y)
        baseline_loss += masked_mse(zero_pred, batch.y, batch.mask).item() * batch.num_graphs

n_test = len(dataset) - split
print(f"\nTest masked MSE (GNN):           {test_loss/n_test:.4f}")
print(f"Test masked MSE (zero baseline): {baseline_loss/n_test:.4f}")

torch.save(model.state_dict(), "checkpoints/gnn_encoder_isolation.pt")
print("Saved checkpoint to checkpoints/gnn_encoder_isolation.pt")
