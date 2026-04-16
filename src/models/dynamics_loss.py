from __future__ import annotations

from typing import Sequence

import torch


def one_step_loss(pred: torch.Tensor, target: torch.Tensor, phase_slice: slice | None = None, lambda_phase: float = 1.0) -> torch.Tensor:
    weights = torch.ones_like(target)
    if phase_slice is not None:
        weights[:, phase_slice] = float(lambda_phase)
    return ((pred - target) ** 2 * weights).mean()


def multi_step_loss(predictions: Sequence[torch.Tensor], targets: Sequence[torch.Tensor], lambda_multistep: float = 0.5, phase_slice: slice | None = None, lambda_phase: float = 1.0) -> torch.Tensor:
    if len(predictions) != len(targets):
        raise ValueError('Predictions and targets must have the same length.')
    total = 0.0
    for pred, target in zip(predictions, targets):
        total = total + one_step_loss(pred, target, phase_slice=phase_slice, lambda_phase=lambda_phase)
    return lambda_multistep * total / max(len(predictions), 1)
