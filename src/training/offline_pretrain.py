from __future__ import annotations

import os
from datetime import datetime
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
        print(f"[offline] Collecting episode {episode+1}/{num_episodes}...", flush=True)
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
    cfg: Dict | None,
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
    cfg = cfg or {}
    train_cfg = cfg.get('training', {})
    bc_batch_size = int(train_cfg.get('bc_batch_size', max(int(batch_size), 256)))
    num_workers = int(train_cfg.get('bc_num_workers', min(8, os.cpu_count() or 1)))
    num_workers = max(0, num_workers)
    prefetch_factor = int(train_cfg.get('bc_prefetch_factor', 4))
    pin_memory = bool(train_cfg.get('bc_pin_memory', device.type == 'cuda'))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    loader_kwargs = {
        'batch_size': bc_batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = max(2, prefetch_factor)
    loader = DataLoader(TransitionDataset(transitions), **loader_kwargs)
    last_loss = 0.0
    policy.train()
    print(
        f"[bc] start time={datetime.now().isoformat()} samples={len(transitions)} "
        f"batch_size={bc_batch_size} num_workers={num_workers} epochs={int(epochs)}",
        flush=True,
    )
    start_time = datetime.now()
    for epoch_idx in range(int(epochs)):
        epoch_start = datetime.now()
        epoch_loss = 0.0
        epoch_batches = 0
        for batch in loader:
            state = batch['state'].to(device, non_blocking=pin_memory)
            action_mask = batch['action_mask'].to(device, non_blocking=pin_memory)
            action = batch['action'].to(device, non_blocking=pin_memory)
            log_prob, _ = policy.evaluate_actions(state, edge_index, edge_attr, action_mask, action)
            loss = -log_prob / float(max(state.size(0), 1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
            epoch_loss += last_loss
            epoch_batches += 1
        epoch_elapsed = (datetime.now() - epoch_start).total_seconds()
        total_elapsed = (datetime.now() - start_time).total_seconds()
        remaining_epochs = max(int(epochs) - (epoch_idx + 1), 0)
        eta_seconds = remaining_epochs * epoch_elapsed
        print(
            f"[bc] epoch={epoch_idx + 1}/{int(epochs)} batches={epoch_batches} "
            f"loss={epoch_loss / float(max(epoch_batches, 1)):.6f} "
            f"elapsed_sec={epoch_elapsed:.2f} total_elapsed_sec={total_elapsed:.2f} eta_sec={eta_seconds:.2f}",
            flush=True,
        )
    print(
        f"[bc] finished time={datetime.now().isoformat()} elapsed_sec={(datetime.now() - start_time).total_seconds():.2f}",
        flush=True,
    )
    return {'bc_loss': last_loss}
