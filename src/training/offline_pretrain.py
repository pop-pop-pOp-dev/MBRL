from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.baselines.fixed_time import fixed_time_action
from src.baselines.max_pressure import max_pressure_action
from src.baselines.random_phase import random_phase_action
from src.data.offline_dataset import Transition, TransitionDataset
from src.models.policy_head import MultiDiscretePolicy



def _select_behavior(obs: dict, episode: int, step: int, cfg: Dict) -> np.ndarray:
    mix = cfg.get('training', {}).get('offline_policy_mix', {'fixed_time': 0.35, 'max_pressure': 0.35, 'random_phase': 0.30})
    epsilon = float(cfg.get('training', {}).get('offline_epsilon', 0.1))
    choices = [
        ('fixed_time', float(mix.get('fixed_time', 0.35))),
        ('max_pressure', float(mix.get('max_pressure', 0.35))),
        ('random_phase', float(mix.get('random_phase', 0.30))),
    ]
    labels = [label for label, _ in choices]
    probs = np.asarray([weight for _, weight in choices], dtype=np.float32)
    probs = probs / max(probs.sum(), 1e-6)
    label = str(np.random.choice(labels, p=probs))
    if label == 'fixed_time':
        action = fixed_time_action(obs['action_mask'], step=step)
    elif label == 'max_pressure':
        action = max_pressure_action(obs['node_features'], obs['action_mask'])
    else:
        action = random_phase_action(obs['action_mask'])
    if np.random.rand() < epsilon:
        action = random_phase_action(obs['action_mask'])
    return action



def collect_offline_transitions(env, num_episodes: int, cfg: Dict | None = None) -> List[Transition]:
    cfg = cfg or {}
    transitions: List[Transition] = []
    for episode in range(int(num_episodes)):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done:
            action = _select_behavior(obs, episode=episode, step=step, cfg=cfg)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            transitions.append(
                Transition(
                    state=obs['node_features'],
                    action=action,
                    reward=float(reward),
                    next_state=next_obs['node_features'],
                    done=float(terminated or truncated),
                    action_mask=obs['action_mask'],
                    next_action_mask=next_obs['action_mask'],
                    source='offline',
                )
            )
            obs = next_obs
            done = bool(terminated or truncated)
            step += 1
    return transitions



def behavior_clone_policy(
    policy: MultiDiscretePolicy,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    transitions: List[Transition],
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    if not transitions:
        return {'bc_loss': 0.0}
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    loader = DataLoader(TransitionDataset(transitions), batch_size=batch_size, shuffle=True)
    last_loss = 0.0
    policy.train()
    for _ in range(int(epochs)):
        for batch in loader:
            loss = 0.0
            for state, action_mask, action in zip(batch['state'], batch['action_mask'], batch['action']):
                state = state.to(device)
                action_mask = action_mask.to(device)
                action = action.to(device)
                log_prob, _ = policy.evaluate_actions(state, edge_index, edge_attr, action_mask, action)
                loss = loss - log_prob
            loss = loss / float(max(len(batch['state']), 1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
    return {'bc_loss': last_loss}
