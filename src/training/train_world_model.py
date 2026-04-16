from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from src.data.offline_dataset import Transition, TransitionDataset
from src.models.dynamics_loss import one_step_loss
from src.models.graph_dynamics import GraphDynamicsModel
from src.models.uncertainty_ensemble import DynamicsEnsemble



def build_ensemble(cfg: Dict[str, Any], state_dim: int, max_actions: int, edge_dim: int | None):
    model_cfg = cfg.get('model', {})
    dyn_cfg = cfg.get('dynamics', {})
    models = [
        GraphDynamicsModel(
            state_dim=state_dim,
            hidden_dim=int(dyn_cfg.get('hidden_dim', model_cfg.get('hidden_dim', 128))),
            action_dim=int(dyn_cfg.get('action_dim', 32)),
            max_actions=int(max_actions),
            edge_dim=edge_dim,
            heads=int(model_cfg.get('gat_heads', 4)),
            num_layers=int(model_cfg.get('gat_layers', 3)),
            dropout=float(model_cfg.get('dropout', 0.1)),
        )
        for _ in range(int(dyn_cfg.get('ensemble_size', 5)))
    ]
    return DynamicsEnsemble(models)



def train_world_model(
    cfg: Dict[str, Any],
    ensemble: DynamicsEnsemble,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    transitions: List[Transition],
    device: torch.device,
) -> Dict[str, float]:
    if not transitions:
        return {'world_model_loss': 0.0}
    loader = DataLoader(TransitionDataset(transitions), batch_size=int(cfg.get('training', {}).get('batch_size', 64)), shuffle=True)
    optimizers = [torch.optim.Adam(model.parameters(), lr=3e-4) for model in ensemble.models]
    lambda_phase = float(cfg.get('dynamics', {}).get('lambda_phase', 2.0))
    phase_slice = slice(3, 5)
    last_loss = 0.0
    ensemble.to(device)
    for _ in range(int(cfg.get('training', {}).get('world_model_epochs', 20))):
        for batch in loader:
            for model, optimizer in zip(ensemble.models, optimizers):
                loss = 0.0
                for state, action, next_state in zip(batch['state'], batch['action'], batch['next_state']):
                    pred = model(state.to(device), edge_index, edge_attr, action.to(device))
                    loss = loss + one_step_loss(pred, next_state.to(device), phase_slice=phase_slice, lambda_phase=lambda_phase)
                loss = loss / float(max(len(batch['state']), 1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu().item())
    return {'world_model_loss': last_loss}



def save_ensemble(ensemble: DynamicsEnsemble, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'models': [model.state_dict() for model in ensemble.models]}, path)
