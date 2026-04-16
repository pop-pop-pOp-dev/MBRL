from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from src.models.state_encoder import GraphStateEncoder


@dataclass
class PolicyOutput:
    actions: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    logits: torch.Tensor


class MultiDiscretePolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        max_actions: int,
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
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_actions),
        )

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        node_emb, graph_emb = self.encoder(node_x, edge_index, edge_attr=edge_attr)
        global_ctx = graph_emb.expand(node_emb.size(0), -1)
        logits = self.head(torch.cat([node_emb, global_ctx], dim=-1))
        return logits.masked_fill(action_mask <= 0, -1e9)

    def sample(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        logits = self.forward(node_x, edge_index, edge_attr, action_mask)
        dist = Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(actions).sum()
        entropy = dist.entropy().mean()
        return PolicyOutput(actions=actions, log_prob=log_prob, entropy=entropy, logits=logits)

    def evaluate_actions(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(node_x, edge_index, edge_attr, action_mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions).sum(), dist.entropy().mean()
