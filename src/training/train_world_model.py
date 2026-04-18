from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import math
import torch
from torch.utils.data import DataLoader

from src.data.offline_dataset import MultiStepTransitionDataset, Transition, build_multistep_windows
from src.models.dynamics_loss import multi_step_loss, one_step_loss, prior_penalty
from src.models.graph_dynamics import GraphDynamicsModel
from src.models.uncertainty_ensemble import DynamicsEnsemble



def build_ensemble(cfg: Dict[str, Any], state_dim: int, max_actions: int, edge_dim: int | None, observation_spec):
    model_cfg = cfg.get('model', {})
    dyn_cfg = cfg.get('dynamics', {})
    models = [
        GraphDynamicsModel(
            state_dim=state_dim,
            hidden_dim=int(dyn_cfg.get('hidden_dim', model_cfg.get('hidden_dim', 128))),
            action_dim=int(dyn_cfg.get('action_dim', 32)),
            max_actions=int(max_actions),
            observation_spec=observation_spec,
            edge_dim=edge_dim,
            heads=int(model_cfg.get('gat_heads', 4)),
            num_layers=int(model_cfg.get('gat_layers', 3)),
            dropout=float(model_cfg.get('dropout', 0.1)),
        )
        for _ in range(int(dyn_cfg.get('ensemble_size', 5)))
    ]
    return DynamicsEnsemble(models)



def _scheduled_teacher_forcing(epoch_idx: int, total_epochs: int, cfg: Dict[str, Any]) -> float:
    dyn_cfg = cfg.get('dynamics', {})
    schedule = str(dyn_cfg.get('rollout_schedule', 'linear'))
    base = float(dyn_cfg.get('teacher_forcing_ratio', 1.0))
    free_ratio = float(dyn_cfg.get('free_rollout_ratio', 0.5))
    if total_epochs <= 1:
        return max(0.0, min(1.0, base))
    progress = float(epoch_idx) / float(max(total_epochs - 1, 1))
    if schedule == 'constant':
        value = base
    elif schedule == 'piecewise':
        value = base if progress < 0.5 else max(0.0, base * (1.0 - free_ratio))
    elif schedule == 'cosine':
        value = base * (1.0 - free_ratio * 0.5 * (1.0 - math.cos(math.pi * progress)))
    else:
        value = base * (1.0 - progress * free_ratio)
    return max(0.0, min(1.0, value))



def _effective_rollout_horizon(epoch_idx: int, total_epochs: int, cfg: Dict[str, Any]) -> int:
    dyn_cfg = cfg.get('dynamics', {})
    target_horizon = int(dyn_cfg.get('train_horizon', dyn_cfg.get('horizon', 5)))
    schedule = str(dyn_cfg.get('rollout_schedule', 'linear'))
    if target_horizon <= 1 or total_epochs <= 1:
        return target_horizon
    progress = float(epoch_idx) / float(max(total_epochs - 1, 1))
    if schedule == 'constant':
        return target_horizon
    if schedule == 'piecewise':
        return 1 if progress < 0.5 else target_horizon
    if schedule == 'cosine':
        return max(1, int(round(1 + (target_horizon - 1) * 0.5 * (1.0 - math.cos(math.pi * progress)))))
    return max(1, int(round(1 + (target_horizon - 1) * progress)))



def train_world_model(
    cfg: Dict[str, Any],
    ensemble: DynamicsEnsemble,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    transitions: List[Transition],
    device: torch.device,
) -> Dict[str, float]:
    max_horizon = int(cfg.get('dynamics', {}).get('train_horizon', cfg.get('dynamics', {}).get('horizon', 5)))
    windows = build_multistep_windows(transitions, horizon=max_horizon)
    if not windows:
        return {'world_model_loss': 0.0, 'one_step_loss': 0.0, 'multistep_loss': 0.0, 'prior_penalty': 0.0, 'rollout_horizon': float(max_horizon), 'teacher_forcing_ratio': 1.0}
    train_cfg = cfg.get('training', {})
    world_batch_size = int(train_cfg.get('world_model_batch_size', max(int(train_cfg.get('batch_size', 64)), 256)))
    num_workers = int(train_cfg.get('world_model_num_workers', min(8, os.cpu_count() or 1)))
    num_workers = max(0, num_workers)
    prefetch_factor = int(train_cfg.get('world_model_prefetch_factor', 4))
    pin_memory = bool(train_cfg.get('world_model_pin_memory', device.type == 'cuda'))
    loader_kwargs: Dict[str, Any] = {
        'batch_size': world_batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = max(2, prefetch_factor)
    loader = DataLoader(MultiStepTransitionDataset(windows), **loader_kwargs)
    optimizers = [torch.optim.Adam(model.parameters(), lr=3e-4) for model in ensemble.models]
    lambda_phase = float(cfg.get('dynamics', {}).get('lambda_phase', 2.0))
    lambda_multistep = float(cfg.get('dynamics', {}).get('lambda_multistep', 0.5))
    prior_weight = float(cfg.get('dynamics', {}).get('prior_weight', 0.2))
    phase_slice = ensemble.models[0].observation_spec.latest_dynamic_slice
    last_stats = {'world_model_loss': 0.0, 'one_step_loss': 0.0, 'multistep_loss': 0.0, 'prior_penalty': 0.0, 'rollout_horizon': float(max_horizon), 'teacher_forcing_ratio': 1.0}
    ensemble.to(device)
    total_epochs = int(train_cfg.get('world_model_epochs', 20))
    print(
        f"[world_model] start time={datetime.now().isoformat()} "
        f"windows={len(windows)} batch_size={world_batch_size} num_workers={num_workers} "
        f"epochs={total_epochs} ensemble_size={len(ensemble.models)}",
        flush=True,
    )
    global_start = datetime.now()
    for epoch_idx in range(total_epochs):
        epoch_start = datetime.now()
        teacher_ratio = _scheduled_teacher_forcing(epoch_idx, total_epochs, cfg)
        effective_horizon = _effective_rollout_horizon(epoch_idx, total_epochs, cfg)
        epoch_loss = 0.0
        epoch_one = 0.0
        epoch_multi = 0.0
        epoch_prior = 0.0
        epoch_batches = 0
        for batch in loader:
            states = batch['states'].transpose(0, 1).to(device, non_blocking=pin_memory)
            actions = batch['actions'].transpose(0, 1).to(device, non_blocking=pin_memory)
            horizon = min(int(actions.size(0)), int(effective_horizon))
            seq_states = states[: horizon + 1]
            seq_actions = actions[:horizon]
            targets = [seq_states[idx + 1] for idx in range(horizon)]
            for model, optimizer in zip(ensemble.models, optimizers):
                predictions = model.rollout_sequence(
                    seq_states,
                    seq_actions,
                    edge_index,
                    edge_attr,
                    teacher_forcing_ratio=teacher_ratio,
                )
                one = one_step_loss(predictions[0], targets[0], phase_slice=phase_slice, lambda_phase=lambda_phase)
                multi = multi_step_loss(predictions, targets, lambda_multistep=lambda_multistep, phase_slice=phase_slice, lambda_phase=lambda_phase)
                prior = 0.0
                for step_idx, pred in enumerate(predictions):
                    prior = prior + prior_penalty(pred, seq_states[step_idx], seq_actions[step_idx], model.observation_spec)
                prior = prior / float(max(len(predictions), 1))
                batch_loss = one + multi + float(prior_weight) * prior
                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                optimizer.step()
                last_stats = {
                    'world_model_loss': float(batch_loss.detach().cpu().item()),
                    'one_step_loss': float(one.detach().cpu().item()),
                    'multistep_loss': float(multi.detach().cpu().item()),
                    'prior_penalty': float((float(prior_weight) * prior).detach().cpu().item()),
                    'rollout_horizon': float(effective_horizon),
                    'teacher_forcing_ratio': float(teacher_ratio),
                }
                epoch_loss += last_stats['world_model_loss']
                epoch_one += last_stats['one_step_loss']
                epoch_multi += last_stats['multistep_loss']
                epoch_prior += last_stats['prior_penalty']
                epoch_batches += 1
        epoch_elapsed = (datetime.now() - epoch_start).total_seconds()
        mean_scale = float(max(epoch_batches, 1))
        total_elapsed = (datetime.now() - global_start).total_seconds()
        remaining_epochs = max(total_epochs - (epoch_idx + 1), 0)
        eta_seconds = remaining_epochs * epoch_elapsed
        print(
            f"[world_model] epoch={epoch_idx + 1}/{total_epochs} "
            f"batches={epoch_batches} horizon={effective_horizon} teacher_forcing={teacher_ratio:.3f} "
            f"loss={epoch_loss / mean_scale:.6f} one={epoch_one / mean_scale:.6f} "
            f"multi={epoch_multi / mean_scale:.6f} prior={epoch_prior / mean_scale:.6f} "
            f"elapsed_sec={epoch_elapsed:.2f} total_elapsed_sec={total_elapsed:.2f} eta_sec={eta_seconds:.2f}",
            flush=True,
        )
    print(
        f"[world_model] finished time={datetime.now().isoformat()} "
        f"elapsed_sec={(datetime.now() - global_start).total_seconds():.2f}",
        flush=True,
    )
    return last_stats



def save_ensemble(ensemble: DynamicsEnsemble, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'models': [model.state_dict() for model in ensemble.models]}, path)
