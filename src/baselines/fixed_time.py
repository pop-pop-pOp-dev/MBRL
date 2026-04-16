from __future__ import annotations

import numpy as np


def fixed_time_action(action_mask: np.ndarray, step: int = 0) -> np.ndarray:
    actions = np.zeros(action_mask.shape[0], dtype=np.int64)
    for idx in range(action_mask.shape[0]):
        valid = np.flatnonzero(action_mask[idx] > 0)
        if valid.size <= 1:
            actions[idx] = 0
        else:
            actions[idx] = int(valid[(step % (valid.size - 1)) + 1])
    return actions
