from __future__ import annotations

from typing import Callable, Dict, List

import torch

from src.models.uncertainty_ensemble import DynamicsEnsemble


@torch.no_grad()
def rollout_model(
    ensemble: DynamicsEnsemble,
    state: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    action_mask: torch.Tensor,
    horizon: int,
    policy_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor], torch.Tensor],
    reward_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], float],
    uncertainty_threshold: float,
    lambda_uncertainty: float,
) -> List[Dict[str, torch.Tensor | float]]:
    transitions: List[Dict[str, torch.Tensor | float]] = []
    current_state = state
    current_mask = action_mask
    for _ in range(int(horizon)):
        actions = policy_fn(current_state, edge_index, edge_attr, current_mask)
        mean_state, var_state = ensemble.predict_mean_var(current_state, edge_index, edge_attr, actions)
        sigma = float(var_state.mean().item())
        if sigma > float(uncertainty_threshold):
            break
        reward = float(reward_fn(current_state, mean_state, actions)) - float(lambda_uncertainty) * sigma
        transitions.append(
            {
                'state': current_state,
                'action': actions,
                'next_state': mean_state,
                'uncertainty': sigma,
                'reward': reward,
            }
        )
        current_state = mean_state
    return transitions
