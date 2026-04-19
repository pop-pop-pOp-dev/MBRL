from __future__ import annotations

import torch
from dataclasses import dataclass

from src.env.observation import ObservationSpec
from src.models.graph_dynamics import GraphDynamicsModel
from src.models.policy_head import MultiDiscretePolicy
from src.models.uncertainty_ensemble import DynamicsEnsemble
from src.models.value_head import GraphValueHead
from src.training.decision_selector import build_action_plans, select_action_with_world_model



def test_decision_selector_returns_valid_action():
    device = torch.device('cpu')
    spec = ObservationSpec(dynamic_dim=5, stack_k=1, static_dim=0)
    policy = MultiDiscretePolicy(input_dim=5, hidden_dim=8, max_actions=4, edge_dim=2).to(device)
    value = GraphValueHead(input_dim=5, hidden_dim=8, edge_dim=2).to(device)
    models = [GraphDynamicsModel(state_dim=5, hidden_dim=8, action_dim=4, max_actions=4, observation_spec=spec, edge_dim=2) for _ in range(2)]
    ensemble = DynamicsEnsemble(models).to(device)
    node_x = torch.randn(3, 5, device=device)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long, device=device)
    edge_attr = torch.randn(3, 2, device=device)
    action_mask = torch.ones(3, 4, device=device)
    selection = select_action_with_world_model(
        policy=policy,
        value_net=value,
        ensemble=ensemble,
        node_x=node_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        action_mask=action_mask,
        observation_spec=spec,
        reward_fn=lambda prev_state, next_state, actions: (float(torch.mean(prev_state[:, 0] - next_state[:, 0]).item()), {}),
        action_mask_fn=lambda state: torch.ones(state.size(0), 4, device=state.device),
        cfg={'mode': 'sequence_shooting', 'candidate_count': 3, 'plan_count': 3, 'include_greedy': True, 'horizon': 2, 'discount': 0.99, 'uncertainty_coef': 0.1, 'pessimism_coef': 0.1, 'future_action_mode': 'greedy'},
    )
    assert selection.action.shape == (3,)
    assert selection.candidate_count >= 1
    assert selection.plan_count >= 1


@dataclass
class _FakePolicyOutput:
    actions: torch.Tensor


class _StateAwarePolicy:
    def forward(self, node_x, edge_index, edge_attr, action_mask):
        del edge_index, edge_attr
        logits = torch.full((node_x.size(0), action_mask.size(1)), -1e9, device=node_x.device)
        preferred = 0 if float(node_x[0, 0].item()) < 0.5 else 1
        logits[:, preferred] = 5.0
        return logits.masked_fill(action_mask <= 0, -1e9)

    def sample(self, node_x, edge_index, edge_attr, action_mask, deterministic=False):
        del deterministic
        logits = self.forward(node_x, edge_index, edge_attr, action_mask)
        return _FakePolicyOutput(actions=torch.argmax(logits, dim=-1))


class _ShiftEnsemble:
    def predict_mean_var(self, history_states, history_actions, edge_index, edge_attr):
        del history_actions, edge_index, edge_attr
        mean_state = history_states[-1] + 1.0
        var_state = torch.full_like(mean_state, 1.0e-4)
        return mean_state, var_state


def test_sequence_shooting_builds_future_actions_from_imagined_state():
    device = torch.device('cpu')
    policy = _StateAwarePolicy()
    ensemble = _ShiftEnsemble()
    node_x = torch.zeros(2, 5, device=device)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    action_mask = torch.ones(2, 2, device=device)
    plans = build_action_plans(
        policy=policy,
        ensemble=ensemble,
        node_x=node_x,
        edge_index=edge_index,
        edge_attr=None,
        action_mask=action_mask,
        action_mask_fn=lambda state: torch.ones(state.size(0), 2, device=state.device),
        cfg={'mode': 'sequence_shooting', 'candidate_count': 1, 'plan_count': 1, 'include_greedy': True, 'horizon': 2, 'future_action_mode': 'greedy', 'pessimism_coef': 0.0},
    )
    assert len(plans) == 1
    assert [int(x) for x in plans[0][0].tolist()] == [0, 0]
    assert [int(x) for x in plans[0][1].tolist()] == [1, 1]
