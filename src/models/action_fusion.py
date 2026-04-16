from __future__ import annotations

import torch
from torch import nn


class ActionFusion(nn.Module):
    def __init__(self, max_actions: int, action_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_actions, action_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.embedding(actions.long())
