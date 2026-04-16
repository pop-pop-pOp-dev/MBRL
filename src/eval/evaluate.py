from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from src.env.cityflow_signal_env import CityFlowSignalEnv
from src.env.reward import compute_synthetic_reward_from_states
from src.models.policy_head import MultiDiscretePolicy
from src.models.uncertainty_ensemble import DynamicsEnsemble
from src.models.value_head import GraphValueHead
from src.training.decision_selector import select_action_with_world_model


def _build_action_mask_from_state(state: torch.Tensor, action_space_n: int, observation_spec) -> torch.Tensor:
    mask = torch.ones(state.size(0), action_space_n, device=state.device)
    latest = observation_spec.latest_dynamic(state)
    remaining_green = latest[:, 4]
    hold_only = remaining_green > 0.01
    mask[hold_only, 1:] = 0.0
    return mask


def _make_synthetic_reward_fn(cfg: Dict[str, Any], observation_spec):
    reward_cfg = cfg.get('env', {}).get('reward', {})

    def _reward(prev_state: torch.Tensor, next_state: torch.Tensor, actions: torch.Tensor):
        return compute_synthetic_reward_from_states(prev_state, next_state, actions, observation_spec, reward_cfg)

    return _reward



def evaluate_policy(
    cfg: Dict[str, Any],
    policy: MultiDiscretePolicy,
    value_net: GraphValueHead | None = None,
    ensemble: DynamicsEnsemble | None = None,
    episodes: int = 3,
) -> Dict[str, float]:
    env = CityFlowSignalEnv(cfg)
    device = next(policy.parameters()).device
    delay_list: List[float] = []
    queue_list: List[float] = []
    throughput_list: List[float] = []
    reward_list: List[float] = []
    decision_used = bool(cfg.get('decision', {}).get('eval_enabled', False)) and ensemble is not None and value_net is not None
    for _ in range(int(episodes)):
        obs, info = env.reset()
        observation_spec = info['observation_spec']
        reward_fn = _make_synthetic_reward_fn(cfg, observation_spec)
        done = False
        ep_reward = 0.0
        delays = []
        queues = []
        throughputs = []
        while not done:
            node_x = torch.tensor(obs['node_features'], dtype=torch.float32, device=device)
            edge_index = torch.tensor(obs['edge_index'], dtype=torch.long, device=device)
            edge_attr = torch.tensor(obs['edge_attr'], dtype=torch.float32, device=device)
            action_mask = torch.tensor(obs['action_mask'], dtype=torch.float32, device=device)
            with torch.no_grad():
                if decision_used:
                    selection = select_action_with_world_model(
                        policy=policy,
                        value_net=value_net,
                        ensemble=ensemble,
                        node_x=node_x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        action_mask=action_mask,
                        observation_spec=observation_spec,
                        reward_fn=reward_fn,
                        action_mask_fn=lambda state: _build_action_mask_from_state(state, action_mask.size(1), observation_spec),
                        cfg=dict(cfg.get('decision', {})),
                    )
                    action_tensor = selection.action
                else:
                    action_tensor = policy.sample(node_x, edge_index, edge_attr, action_mask, deterministic=True).actions
            obs, reward, terminated, truncated, info = env.step(action_tensor.detach().cpu().numpy())
            delays.append(float(info.get('delay', 0.0)))
            queues.append(float(info.get('queue', 0.0)))
            throughputs.append(float(info.get('throughput', 0.0)))
            ep_reward += float(reward)
            done = bool(terminated or truncated)
        delay_list.append(float(np.mean(delays)) if delays else 0.0)
        queue_list.append(float(np.mean(queues)) if queues else 0.0)
        throughput_list.append(float(np.mean(throughputs)) if throughputs else 0.0)
        reward_list.append(ep_reward)
    return {
        'avg_delay': float(np.mean(delay_list)),
        'avg_queue': float(np.mean(queue_list)),
        'avg_throughput': float(np.mean(throughput_list)),
        'avg_reward': float(np.mean(reward_list)),
        'decision_guidance_eval': 1.0 if decision_used else 0.0,
    }
