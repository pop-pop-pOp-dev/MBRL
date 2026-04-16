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
    source: str = 'unknown'


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


class MultiStepTransitionDataset(Dataset):
    def __init__(self, windows: Sequence[Sequence[Transition]]):
        self.windows = [list(window) for window in windows if len(window) > 0]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        window = self.windows[index]
        states = [window[0].state] + [step.next_state for step in window]
        actions = [step.action for step in window]
        action_masks = [window[0].action_mask] + [step.next_action_mask for step in window]
        rewards = [step.reward for step in window]
        dones = [step.done for step in window]
        return {
            'states': torch.tensor(np.stack(states, axis=0), dtype=torch.float32),
            'actions': torch.tensor(np.stack(actions, axis=0), dtype=torch.long),
            'action_masks': torch.tensor(np.stack(action_masks, axis=0), dtype=torch.float32),
            'rewards': torch.tensor(np.asarray(rewards, dtype=np.float32), dtype=torch.float32),
            'dones': torch.tensor(np.asarray(dones, dtype=np.float32), dtype=torch.float32),
        }


def build_multistep_windows(transitions: Sequence[Transition], horizon: int) -> List[List[Transition]]:
    windows: List[List[Transition]] = []
    if horizon <= 0:
        return windows
    current: List[Transition] = []
    for item in transitions:
        current.append(item)
        if len(current) >= horizon:
            candidate = current[-horizon:]
            if sum(step.done > 0.5 for step in candidate[:-1]) == 0:
                windows.append(candidate)
        if item.done > 0.5:
            current = []
    return windows
