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
        self._edge_cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor | None]] = {}

    def _batched_graph_inputs(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        num_nodes: int,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if batch_size <= 1:
            return edge_index, edge_attr
        cache_key = (batch_size, device)
        cached = self._edge_cache.get(cache_key)
        if cached is not None:
            return cached
        num_edges = edge_index.size(1)
        offsets = torch.arange(batch_size, device=device, dtype=edge_index.dtype).view(batch_size, 1, 1) * num_nodes
        expanded_edge_index = edge_index.unsqueeze(0) + offsets
        expanded_edge_index = expanded_edge_index.permute(1, 0, 2).reshape(2, batch_size * num_edges)
        expanded_edge_attr = None
        if edge_attr is not None:
            expanded_edge_attr = edge_attr.repeat(batch_size, 1)
        self._edge_cache[cache_key] = (expanded_edge_index, expanded_edge_attr)
        return expanded_edge_index, expanded_edge_attr

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        batched = node_x.dim() == 3
        if batched:
            batch_size, num_nodes, feat_dim = node_x.shape
            flat_node_x = node_x.reshape(batch_size * num_nodes, feat_dim)
            flat_action_mask = action_mask.reshape(batch_size * num_nodes, -1)
            expanded_edge_index, expanded_edge_attr = self._batched_graph_inputs(
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                batch_size=batch_size,
                device=node_x.device,
            )
            batch_index = torch.arange(batch_size, device=node_x.device).repeat_interleave(num_nodes)
            node_emb, graph_emb = self.encoder(flat_node_x, expanded_edge_index, edge_attr=expanded_edge_attr, batch=batch_index)
            global_ctx = graph_emb[batch_index]
            logits = self.head(torch.cat([node_emb, global_ctx], dim=-1))
            logits = logits.masked_fill(flat_action_mask <= 0, -1e9)
            return logits.reshape(batch_size, num_nodes, -1)
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
