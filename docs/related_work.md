# Related Work Tracker

## Core papers (read in full)
- [ ] Tripathi & Modiano 2022 — https://arxiv.org/abs/2208.01738
- [ ] Zhao et al. 2021/2023 — https://arxiv.org/abs/2109.05536

## Supporting papers (skim abstract + method section)
- [ ] Ramakanth, Tripathi & Modiano 2023 — https://arxiv.org/abs/2312.16813
- [ ] Erbayat et al. 2024 — https://arxiv.org/abs/2404.08638
- [ ] Fidler et al. 2024 (2D-AoI) — https://arxiv.org/abs/2412.12789
- [ ] Zhang et al. 2024 (Graph Attention RL) — https://arxiv.org/abs/2404.18084
- [ ] He, Dán & Fodor 2018/2019 — search Google Scholar
- [ ] Zhou & Saad 2020 — search Google Scholar

## Positioning table (from project plan — refine as you read)
| Literature | What it does | What it doesn't do |
|---|---|---|
| AoCI (He/Dán/Fodor, Tripathi/Modiano, Ramakanth et al., Zhou/Saad, Erbayat et al., Fidler et al.) | Analytically models correlation-aware scheduling | Correlation is hand-modeled, never learned from a graph |
| GNN wireless scheduling (Zhao et al., Zhang et al.) | Learns topology-aware scheduling over interference/contention graphs | Graph = network topology, not inter-sensor statistical redundancy |
| **This project** | Learns the correlation graph from a pilot phase, measures staleness as the sink's Kalman posterior variance, and trains a GNN (residual over Kalman-greedy) with PPO under a hard AoI guarantee and optional interference graph | Gaussian AR(1) field only; no unreliable links or energy constraints yet |

## Notes (fill in after reading each paper)
### Tripathi & Modiano 2022
-

### Zhao et al. 2021/2023
-
