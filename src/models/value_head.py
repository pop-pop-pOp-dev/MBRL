from __future__ import annotations

import torch
from torch import nn

from src.models.state_encoder import GraphStateEncoder


class GraphValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        edge_dim: int | None = None,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GraphStateEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            heads=heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        _, graph_emb = self.encoder(node_x, edge_index, edge_attr=edge_attr)
        return self.mlp(graph_emb).squeeze(-1)
