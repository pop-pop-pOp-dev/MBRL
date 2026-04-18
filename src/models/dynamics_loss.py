from __future__ import annotations

from typing import Sequence

import torch

from src.env.observation import ObservationSpec



def one_step_loss(pred: torch.Tensor, target: torch.Tensor, phase_slice: slice | None = None, lambda_phase: float = 1.0) -> torch.Tensor:
    weights = torch.ones_like(target)
    if phase_slice is not None:
        weights[..., phase_slice] = float(lambda_phase)
    return ((pred - target) ** 2 * weights).mean()



def multi_step_loss(predictions: Sequence[torch.Tensor], targets: Sequence[torch.Tensor], lambda_multistep: float = 0.5, phase_slice: slice | None = None, lambda_phase: float = 1.0) -> torch.Tensor:
    if len(predictions) != len(targets):
        raise ValueError('Predictions and targets must have the same length.')
    if len(predictions) == 0:
        return torch.tensor(0.0)
    total = None
    for pred, target in zip(predictions, targets):
        step_loss = one_step_loss(pred, target, phase_slice=phase_slice, lambda_phase=lambda_phase)
        total = step_loss if total is None else total + step_loss
    return float(lambda_multistep) * total / max(len(predictions), 1)



def prior_penalty(
    pred_state: torch.Tensor,
    prev_state: torch.Tensor,
    actions: torch.Tensor,
    spec: ObservationSpec,
    hold_weight: float = 1.0,
    range_weight: float = 1.0,
) -> torch.Tensor:
    latest = spec.latest_dynamic(pred_state)
    prev_latest = spec.latest_dynamic(prev_state)
    queue = latest[..., 0]
    vehicle_count = latest[..., 1]
    speed = latest[..., 2]
    phase = latest[..., 3]
    remaining = latest[..., 4]
    prev_phase = prev_latest[..., 3]
    prev_remaining = prev_latest[..., 4]
    hold_mask = (actions == 0).float()
    non_negative = torch.relu(-queue).mean() + torch.relu(-vehicle_count).mean() + torch.relu(-remaining).mean()
    bounded = torch.relu(-speed).mean() + torch.relu(speed - 1.0).mean() + torch.relu(-phase).mean() + torch.relu(phase - 1.0).mean()
    hold_consistency = (torch.abs(phase - prev_phase) * hold_mask).mean()
    remaining_consistency = (torch.relu(remaining - prev_remaining) * hold_mask).mean()
    return float(range_weight) * (non_negative + bounded) + float(hold_weight) * (hold_consistency + remaining_consistency)
