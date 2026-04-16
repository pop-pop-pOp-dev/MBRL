from __future__ import annotations

import numpy as np


def max_pressure_action(node_features: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    actions = np.zeros(action_mask.shape[0], dtype=np.int64)
    queue_proxy = node_features[:, 0]
    for idx in range(action_mask.shape[0]):
        valid = np.flatnonzero(action_mask[idx] > 0)
        if valid.size <= 1:
            actions[idx] = 0
            continue
        actions[idx] = int(valid[1 + (int(queue_proxy[idx] * 1000) % (valid.size - 1))])
    return actions
