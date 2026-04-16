from __future__ import annotations

from typing import Dict

import numpy as np


def compute_reward(queue: np.ndarray, speed: np.ndarray, switches: np.ndarray, throughput: float, cfg: Dict[str, float]) -> tuple[float, Dict[str, float]]:
    alpha = float(cfg.get('alpha_delay', 1.0))
    beta = float(cfg.get('beta_queue', 0.5))
    gamma = float(cfg.get('gamma_switch', 0.05))
    eta = float(cfg.get('eta_throughput', 0.2))
    zeta = float(cfg.get('zeta_unfairness', 0.0))
    threshold = float(cfg.get('speed_threshold', 0.1))

    delay = float(np.mean(np.maximum(1.0 - speed, 0.0)))
    total_queue = float(np.mean(queue))
    switch_cost = float(np.mean(switches))
    fairness = float(np.max(queue) - np.min(queue)) if queue.size > 0 else 0.0
    reward = -alpha * delay - beta * total_queue - gamma * switch_cost + eta * float(throughput) - zeta * fairness
    return reward, {
        'delay': delay,
        'queue': total_queue,
        'phase_switch': switch_cost,
        'throughput': float(throughput),
        'unfairness': fairness,
        'stopped_ratio': float(np.mean(speed <= threshold)) if speed.size > 0 else 0.0,
    }
