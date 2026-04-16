from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: float
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    uncertainty: float = 0.0


class TransitionDataset(Dataset):
    def __init__(self, transitions: Sequence[Transition]):
        self.transitions = list(transitions)

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> dict:
        item = self.transitions[index]
        return {
            'state': torch.tensor(item.state, dtype=torch.float32),
            'action': torch.tensor(item.action, dtype=torch.long),
            'reward': torch.tensor(item.reward, dtype=torch.float32),
            'next_state': torch.tensor(item.next_state, dtype=torch.float32),
            'done': torch.tensor(item.done, dtype=torch.float32),
            'action_mask': torch.tensor(item.action_mask, dtype=torch.float32),
            'next_action_mask': torch.tensor(item.next_action_mask, dtype=torch.float32),
            'uncertainty': torch.tensor(item.uncertainty, dtype=torch.float32),
        }


def build_multistep_windows(transitions: Sequence[Transition], horizon: int) -> List[List[Transition]]:
    windows: List[List[Transition]] = []
    if horizon <= 0:
        return windows
    current: List[Transition] = []
    for item in transitions:
        current.append(item)
        if len(current) >= horizon:
            windows.append(current[-horizon:])
        if item.done > 0.5:
            current = []
    return windows
