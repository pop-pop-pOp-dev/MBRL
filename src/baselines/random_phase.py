from __future__ import annotations

import numpy as np



def random_phase_action(action_mask: np.ndarray) -> np.ndarray:
    actions = np.zeros(action_mask.shape[0], dtype=np.int64)
    for idx in range(action_mask.shape[0]):
        valid = np.flatnonzero(action_mask[idx] > 0)
        actions[idx] = int(np.random.choice(valid)) if valid.size > 0 else 0
    return actions
