from __future__ import annotations

from typing import Dict

import numpy as np

from src.env.observation import IntersectionMetrics, ObservationSpec



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



def compute_reward_from_metrics(metrics: IntersectionMetrics, switches: np.ndarray, throughput: float, cfg: Dict[str, float]) -> tuple[float, Dict[str, float]]:
    return compute_reward(metrics.queue, metrics.average_speed, switches, throughput, cfg)



def compute_synthetic_reward_from_states(
    prev_state,
    next_state,
    actions,
    spec: ObservationSpec,
    cfg: Dict[str, float],
) -> tuple[float, Dict[str, float]]:
    prev_metrics = spec.metrics_from_state(prev_state)
    next_metrics = spec.metrics_from_state(next_state)
    switches = (np.abs(next_metrics.current_phase - prev_metrics.current_phase) > 1e-4).astype(np.float32)
    throughput = float(np.maximum(prev_metrics.vehicle_count - next_metrics.vehicle_count, 0.0).mean())
    del actions
    return compute_reward_from_metrics(next_metrics, switches, throughput, cfg)
