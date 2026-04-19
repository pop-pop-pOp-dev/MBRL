from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass

from src.env.observation import ObservationSpec
from src.models.graph_dynamics import GraphDynamicsModel
from src.models.model_rollout import rollout_model
from src.models.uncertainty_ensemble import DynamicsEnsemble



def test_rollout_gate_stops_on_high_uncertainty():
    spec = ObservationSpec(dynamic_dim=4, stack_k=1, static_dim=0)
    models = [GraphDynamicsModel(state_dim=4, hidden_dim=8, action_dim=4, max_actions=3, observation_spec=spec, edge_dim=2) for _ in range(2)]
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
        action_mask_fn=lambda s: torch.ones(s.size(0), 3),
        reward_fn=lambda s, n, a: (0.0, {}),
        uncertainty_threshold=-1.0,
        lambda_uncertainty=0.1,
        uncertainty_mode='threshold_ranked',
        pessimism_coef=0.1,
    )
    assert transitions == []


@dataclass
class _FakeEnsemble:
    sigmas: list[float]
    step: int = 0

    def predict_mean_var(self, history_states, history_actions, edge_index, edge_attr):
        del history_actions, edge_index, edge_attr
        sigma = float(self.sigmas[min(self.step, len(self.sigmas) - 1)])
        self.step += 1
        mean_state = history_states[-1] + 1.0
        var_state = torch.full_like(mean_state, sigma)
        return mean_state, var_state


def test_ranked_rollout_keeps_lowest_uncertainty_transitions():
    ensemble = _FakeEnsemble(sigmas=[0.04, 0.01, 0.03])
    state = torch.zeros(2, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    action_mask = torch.ones(2, 3)
    transitions = rollout_model(
        ensemble,
        state,
        edge_index,
        None,
        action_mask,
        horizon=3,
        policy_fn=lambda s, e, ea, m: torch.zeros(s.size(0), dtype=torch.long),
        action_mask_fn=lambda s: torch.ones(s.size(0), 3),
        reward_fn=lambda s, n, a: (0.0, {}),
        uncertainty_threshold=0.1,
        lambda_uncertainty=0.0,
        uncertainty_mode='threshold_ranked',
        uncertainty_keep_topk=2,
        uncertainty_rank_metric='uncertainty',
        pessimism_coef=0.0,
    )
    assert len(transitions) == 2
    kept_uncertainties = sorted(float(item['uncertainty']) for item in transitions)
    assert np.allclose(kept_uncertainties, [0.01, 0.03], atol=1e-6)
