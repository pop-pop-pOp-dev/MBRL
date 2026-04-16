from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn

from src.models.graph_dynamics import GraphDynamicsModel


class DynamicsEnsemble(nn.Module):
    def __init__(self, models: Iterable[GraphDynamicsModel]):
        super().__init__()
        self.models = nn.ModuleList(list(models))
        if len(self.models) == 0:
            raise ValueError('DynamicsEnsemble requires at least one model.')

    def forward(
        self,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        actions: torch.Tensor,
    ) -> List[torch.Tensor]:
        return [model(state, edge_index, edge_attr, actions) for model in self.models]

    def predict_mean_var(
        self,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds = torch.stack(self.forward(state, edge_index, edge_attr, actions), dim=0)
        return preds.mean(dim=0), preds.var(dim=0, unbiased=False)
