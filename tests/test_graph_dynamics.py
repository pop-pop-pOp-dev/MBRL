from __future__ import annotations

import torch

from src.models.graph_dynamics import GraphDynamicsModel



def test_graph_dynamics_shapes():
    model = GraphDynamicsModel(state_dim=6, hidden_dim=8, action_dim=4, max_actions=5, edge_dim=3)
    state = torch.randn(3, 6)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_attr = torch.randn(3, 3)
    actions = torch.tensor([0, 1, 2], dtype=torch.long)
    pred = model(state, edge_index, edge_attr, actions)
    assert pred.shape == state.shape
