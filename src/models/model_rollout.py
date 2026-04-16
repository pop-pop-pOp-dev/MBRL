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
    action_mask_fn: Callable[[torch.Tensor], torch.Tensor],
    reward_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[float, Dict[str, float]]],
    uncertainty_threshold: float,
    lambda_uncertainty: float,
) -> List[Dict[str, torch.Tensor | float | Dict[str, float]]]:
    transitions: List[Dict[str, torch.Tensor | float | Dict[str, float]]] = []
    history_states = [state]
    history_actions = []
    current_state = state
    current_mask = action_mask
    for _ in range(int(horizon)):
        actions = policy_fn(current_state, edge_index, edge_attr, current_mask)
        history_actions.append(actions)
        mean_state, var_state = ensemble.predict_mean_var(torch.stack(history_states, dim=0), torch.stack(history_actions, dim=0), edge_index, edge_attr)
        sigma = float(var_state.mean().item())
        if sigma > float(uncertainty_threshold):
            break
        reward, reward_terms = reward_fn(current_state, mean_state, actions)
        reward = float(reward) - float(lambda_uncertainty) * sigma
        next_mask = action_mask_fn(mean_state)
        transitions.append(
            {
                'state': current_state,
                'action': actions,
                'next_state': mean_state,
                'action_mask': current_mask,
                'next_action_mask': next_mask,
                'uncertainty': sigma,
                'reward': reward,
                'reward_terms': reward_terms,
            }
        )
        current_state = mean_state
        current_mask = next_mask
        history_states.append(mean_state)
    return transitions
