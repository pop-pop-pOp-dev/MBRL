from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

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
    base = float(dyn_cfg.get('teacher_forcing_ratio', 1.0))
    free_ratio = float(dyn_cfg.get('free_rollout_ratio', 0.5))
    if total_epochs <= 1:
        return max(0.0, min(1.0, base))
    progress = float(epoch_idx) / float(max(total_epochs - 1, 1))
    return max(0.0, min(1.0, base * (1.0 - progress * free_ratio)))



def train_world_model(
    cfg: Dict[str, Any],
    ensemble: DynamicsEnsemble,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    transitions: List[Transition],
    device: torch.device,
) -> Dict[str, float]:
    horizon = int(cfg.get('dynamics', {}).get('train_horizon', cfg.get('dynamics', {}).get('horizon', 5)))
    windows = build_multistep_windows(transitions, horizon=horizon)
    if not windows:
        return {'world_model_loss': 0.0, 'one_step_loss': 0.0, 'multistep_loss': 0.0, 'prior_penalty': 0.0, 'rollout_horizon': float(horizon)}
    loader = DataLoader(MultiStepTransitionDataset(windows), batch_size=int(cfg.get('training', {}).get('batch_size', 64)), shuffle=True)
    optimizers = [torch.optim.Adam(model.parameters(), lr=3e-4) for model in ensemble.models]
    lambda_phase = float(cfg.get('dynamics', {}).get('lambda_phase', 2.0))
    lambda_multistep = float(cfg.get('dynamics', {}).get('lambda_multistep', 0.5))
    prior_weight = float(cfg.get('dynamics', {}).get('prior_weight', 0.2))
    phase_slice = ensemble.models[0].observation_spec.latest_dynamic_slice
    last_stats = {'world_model_loss': 0.0, 'one_step_loss': 0.0, 'multistep_loss': 0.0, 'prior_penalty': 0.0, 'rollout_horizon': float(horizon)}
    ensemble.to(device)
    total_epochs = int(cfg.get('training', {}).get('world_model_epochs', 20))
    for epoch_idx in range(total_epochs):
        teacher_ratio = _scheduled_teacher_forcing(epoch_idx, total_epochs, cfg)
        for batch in loader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            for model, optimizer in zip(ensemble.models, optimizers):
                batch_loss = 0.0
                batch_one = 0.0
                batch_multi = 0.0
                batch_prior = 0.0
                for seq_states, seq_actions in zip(states, actions):
                    predictions = model.rollout_sequence(seq_states, seq_actions, edge_index, edge_attr, teacher_forcing_ratio=teacher_ratio)
                    targets = [seq_states[idx + 1] for idx in range(seq_actions.size(0))]
                    one = one_step_loss(predictions[0], targets[0], phase_slice=phase_slice, lambda_phase=lambda_phase)
                    multi = multi_step_loss(predictions, targets, lambda_multistep=lambda_multistep, phase_slice=phase_slice, lambda_phase=lambda_phase)
                    prior = 0.0
                    for step_idx, pred in enumerate(predictions):
                        prior = prior + prior_penalty(pred, seq_states[step_idx], seq_actions[step_idx], model.observation_spec)
                    prior = prior / float(max(len(predictions), 1))
                    loss = one + multi + float(prior_weight) * prior
                    batch_loss = batch_loss + loss
                    batch_one = batch_one + one.detach()
                    batch_multi = batch_multi + multi.detach()
                    batch_prior = batch_prior + (float(prior_weight) * prior).detach()
                scale = float(max(states.size(0), 1))
                batch_loss = batch_loss / scale
                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                optimizer.step()
                last_stats = {
                    'world_model_loss': float(batch_loss.detach().cpu().item()),
                    'one_step_loss': float((batch_one / scale).cpu().item()),
                    'multistep_loss': float((batch_multi / scale).cpu().item()),
                    'prior_penalty': float((batch_prior / scale).cpu().item()),
                    'rollout_horizon': float(horizon),
                }
    return last_stats



def save_ensemble(ensemble: DynamicsEnsemble, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'models': [model.state_dict() for model in ensemble.models]}, path)
