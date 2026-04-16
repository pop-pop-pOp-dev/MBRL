from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch.distributions import Categorical

from src.models.policy_head import MultiDiscretePolicy
from src.models.uncertainty_ensemble import DynamicsEnsemble
from src.models.value_head import GraphValueHead


@dataclass
class DecisionSelection:
    action: torch.Tensor
    log_prob: torch.Tensor
    value: float
    candidate_count: int
    plan_count: int
    selected_index: int
    selected_score: float
    selected_uncertainty: float
    score_gap: float


@torch.no_grad()
def sample_candidate_actions(
    policy: MultiDiscretePolicy,
    node_x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    action_mask: torch.Tensor,
    candidate_count: int,
    include_greedy: bool = True,
) -> List[torch.Tensor]:
    logits = policy.forward(node_x, edge_index, edge_attr, action_mask)
    dist = Categorical(logits=logits)
    candidates: List[torch.Tensor] = []
    seen = set()

    def _add(action: torch.Tensor) -> None:
        key = tuple(int(x) for x in action.detach().cpu().tolist())
        if key in seen:
            return
        seen.add(key)
        candidates.append(action.detach().clone())

    if include_greedy:
        _add(torch.argmax(logits, dim=-1))
    max_trials = max(int(candidate_count) * 4, 4)
    for _ in range(max_trials):
        _add(dist.sample())
        if len(candidates) >= int(candidate_count):
            break
    return candidates[: max(int(candidate_count), 1)]


@torch.no_grad()
def _rollout_action_plan(
    policy: MultiDiscretePolicy,
    value_net: GraphValueHead,
    ensemble: DynamicsEnsemble,
    state: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    action_mask: torch.Tensor,
    action_plan: Sequence[torch.Tensor],
    observation_spec,
    reward_fn,
    action_mask_fn,
    cfg: Dict,
) -> tuple[float, float, torch.Tensor]:
    gamma = float(cfg.get('discount', 0.99))
    uncertainty_coef = float(cfg.get('uncertainty_coef', 0.0))
    pessimism_coef = float(cfg.get('pessimism_coef', 0.0))
    future_action_mode = str(cfg.get('future_action_mode', 'greedy'))

    current_state = state
    current_mask = action_mask
    history_states = [state]
    history_actions = []
    total_score = 0.0
    total_uncertainty = 0.0
    discount = 1.0
    plan = list(action_plan)
    if len(plan) == 0:
        raise ValueError('action_plan must contain at least one action.')

    for step_idx, current_action in enumerate(plan):
        history_actions.append(current_action)
        mean_state, var_state = ensemble.predict_mean_var(
            torch.stack(history_states, dim=0),
            torch.stack(history_actions, dim=0),
            edge_index,
            edge_attr,
        )
        sigma = float(var_state.mean().item())
        std_state = torch.sqrt(torch.clamp(var_state, min=1e-8))
        next_state = mean_state - pessimism_coef * std_state
        reward, _ = reward_fn(current_state, next_state, current_action)
        total_score += discount * (float(reward) - uncertainty_coef * sigma)
        total_uncertainty += sigma
        current_state = next_state
        current_mask = action_mask_fn(current_state)
        history_states.append(current_state)
        discount *= gamma

    bootstrap = float(value_net(current_state, edge_index, edge_attr).item())
    total_score += discount * bootstrap
    return total_score, total_uncertainty / float(max(len(plan), 1)), current_state


@torch.no_grad()
def score_candidate_action(
    policy: MultiDiscretePolicy,
    value_net: GraphValueHead,
    ensemble: DynamicsEnsemble,
    state: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    action_mask: torch.Tensor,
    first_action: torch.Tensor,
    observation_spec,
    reward_fn,
    action_mask_fn,
    cfg: Dict,
) -> tuple[float, float, torch.Tensor]:
    horizon = int(cfg.get('horizon', 3))
    future_action_mode = str(cfg.get('future_action_mode', 'greedy'))
    current_state = state
    current_mask = action_mask
    plan: List[torch.Tensor] = [first_action]
    for _ in range(max(horizon - 1, 0)):
        if future_action_mode == 'sample':
            next_action = policy.sample(current_state, edge_index, edge_attr, current_mask, deterministic=False).actions
        else:
            next_action = policy.sample(current_state, edge_index, edge_attr, current_mask, deterministic=True).actions
        plan.append(next_action)
        score, avg_uncertainty, current_state = _rollout_action_plan(
            policy=policy,
            value_net=value_net,
            ensemble=ensemble,
            state=state,
            edge_index=edge_index,
            edge_attr=edge_attr,
            action_mask=action_mask,
            action_plan=plan,
            observation_spec=observation_spec,
            reward_fn=reward_fn,
            action_mask_fn=action_mask_fn,
            cfg=cfg,
        )
        current_mask = action_mask_fn(current_state)
    return _rollout_action_plan(
        policy=policy,
        value_net=value_net,
        ensemble=ensemble,
        state=state,
        edge_index=edge_index,
        edge_attr=edge_attr,
        action_mask=action_mask,
        action_plan=plan,
        observation_spec=observation_spec,
        reward_fn=reward_fn,
        action_mask_fn=action_mask_fn,
        cfg=cfg,
    )


@torch.no_grad()
def build_action_plans(
    policy: MultiDiscretePolicy,
    node_x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    action_mask: torch.Tensor,
    cfg: Dict,
) -> List[List[torch.Tensor]]:
    mode = str(cfg.get('mode', 'first_action_rerank'))
    candidate_count = max(int(cfg.get('candidate_count', 4)), 1)
    horizon = max(int(cfg.get('horizon', 3)), 1)
    future_action_mode = str(cfg.get('future_action_mode', 'greedy'))
    if mode != 'sequence_shooting':
        first_actions = sample_candidate_actions(
            policy,
            node_x,
            edge_index,
            edge_attr,
            action_mask,
            candidate_count=candidate_count,
            include_greedy=bool(cfg.get('include_greedy', True)),
        )
        return [[action] for action in first_actions]

    plan_count = max(int(cfg.get('plan_count', candidate_count)), 1)
    first_actions = sample_candidate_actions(
        policy,
        node_x,
        edge_index,
        edge_attr,
        action_mask,
        candidate_count=plan_count,
        include_greedy=bool(cfg.get('include_greedy', True)),
    )
    plans: List[List[torch.Tensor]] = []
    for first_action in first_actions:
        plan = [first_action]
        current_state = node_x
        current_mask = action_mask
        for _ in range(max(horizon - 1, 0)):
            if future_action_mode == 'sample':
                next_action = policy.sample(current_state, edge_index, edge_attr, current_mask, deterministic=False).actions
            else:
                next_action = policy.sample(current_state, edge_index, edge_attr, current_mask, deterministic=True).actions
            plan.append(next_action)
        plans.append(plan)
    return plans


@torch.no_grad()
def select_action_with_world_model(
    policy: MultiDiscretePolicy,
    value_net: GraphValueHead,
    ensemble: DynamicsEnsemble,
    node_x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    action_mask: torch.Tensor,
    observation_spec,
    reward_fn,
    action_mask_fn,
    cfg: Dict,
) -> DecisionSelection:
    plans = build_action_plans(
        policy=policy,
        node_x=node_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        action_mask=action_mask,
        cfg=cfg,
    )
    scores: List[float] = []
    uncertainties: List[float] = []
    for plan in plans:
        if str(cfg.get('mode', 'first_action_rerank')) == 'sequence_shooting':
            score, avg_uncertainty, _ = _rollout_action_plan(
                policy=policy,
                value_net=value_net,
                ensemble=ensemble,
                state=node_x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                action_mask=action_mask,
                action_plan=plan,
                observation_spec=observation_spec,
                reward_fn=reward_fn,
                action_mask_fn=action_mask_fn,
                cfg=cfg,
            )
        else:
            score, avg_uncertainty, _ = score_candidate_action(
                policy=policy,
                value_net=value_net,
                ensemble=ensemble,
                state=node_x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                action_mask=action_mask,
                first_action=plan[0],
                observation_spec=observation_spec,
                reward_fn=reward_fn,
                action_mask_fn=action_mask_fn,
                cfg=cfg,
            )
        scores.append(float(score))
        uncertainties.append(float(avg_uncertainty))
    best_idx = int(max(range(len(plans)), key=lambda idx: scores[idx]))
    ordered_scores = sorted(scores, reverse=True)
    gap = float(ordered_scores[0] - ordered_scores[1]) if len(ordered_scores) > 1 else 0.0
    best_plan = plans[best_idx]
    best_action = best_plan[0]
    log_prob, _ = policy.evaluate_actions(node_x, edge_index, edge_attr, action_mask, best_action)
    value = float(value_net(node_x, edge_index, edge_attr).item())
    return DecisionSelection(
        action=best_action,
        log_prob=log_prob,
        value=value,
        candidate_count=len(plans),
        plan_count=len(plans),
        selected_index=best_idx,
        selected_score=float(scores[best_idx]),
        selected_uncertainty=float(uncertainties[best_idx]),
        score_gap=gap,
    )
