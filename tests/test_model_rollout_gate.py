from __future__ import annotations

import torch

from src.models.graph_dynamics import GraphDynamicsModel
from src.models.model_rollout import rollout_model
from src.models.uncertainty_ensemble import DynamicsEnsemble



def test_rollout_gate_stops_on_high_uncertainty():
    models = [GraphDynamicsModel(state_dim=4, hidden_dim=8, action_dim=4, max_actions=3, edge_dim=2) for _ in range(2)]
    ensemble = DynamicsEnsemble(models)
    state = torch.randn(2, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.randn(2, 2)
    action_mask = torch.ones(2, 3)
    transitions = rollout_model(
        ensemble,
        state,
        edge_index,
        edge_attr,
        action_mask,
        horizon=3,
        policy_fn=lambda s, e, ea, m: torch.zeros(s.size(0), dtype=torch.long),
        reward_fn=lambda s, n, a: 0.0,
        uncertainty_threshold=-1.0,
        lambda_uncertainty=0.1,
    )
    assert transitions == []
