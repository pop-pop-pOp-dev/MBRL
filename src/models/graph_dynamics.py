from __future__ import annotations

import torch
from torch import nn

from src.models.action_fusion import ActionFusion
from src.models.state_encoder import GraphStateEncoder
from src.models.temporal_model import TemporalGRU


class GraphDynamicsModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        max_actions: int,
        edge_dim: int | None = None,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GraphStateEncoder(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            heads=heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.action_fusion = ActionFusion(max_actions=max_actions, action_dim=action_dim)
        self.temporal = TemporalGRU(input_dim=hidden_dim + action_dim, hidden_dim=hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        node_emb, _ = self.encoder(state, edge_index, edge_attr=edge_attr)
        action_emb = self.action_fusion(actions)
        fused = torch.cat([node_emb, action_emb], dim=-1)
        temporal_out = self.temporal(fused.unsqueeze(1))
        return self.decoder(temporal_out)
