from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool


class GraphStateEncoder(nn.Module):
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
        self.use_edge_attr = edge_dim is not None and int(edge_dim) > 0
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = float(dropout)
        last_dim = input_dim
        for _ in range(max(num_layers - 1, 1)):
            out_dim = hidden_dim
            self.layers.append(
                GATConv(
                    last_dim,
                    out_dim,
                    heads=heads,
                    concat=True,
                    dropout=self.dropout,
                    edge_dim=int(edge_dim) if self.use_edge_attr else None,
                )
            )
            self.norms.append(nn.LayerNorm(out_dim * heads))
            last_dim = out_dim * heads
        self.final = GATConv(
            last_dim,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=self.dropout,
            edge_dim=int(edge_dim) if self.use_edge_attr else None,
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _conv(self, layer: GATConv, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        if self.use_edge_attr:
            return layer(x, edge_index, edge_attr=edge_attr)
        return layer(x, edge_index)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch is None:
            batch = torch.zeros(node_x.size(0), dtype=torch.long, device=node_x.device)
        x = node_x
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = self._conv(layer, x, edge_index, edge_attr)
            x = norm(x)
            x = torch.relu(x)
            if residual.shape == x.shape:
                x = x + residual
        x = self._conv(self.final, x, edge_index, edge_attr)
        x = torch.relu(self.final_norm(x))
        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return x, pooled
