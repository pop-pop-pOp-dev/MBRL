from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.data.offline_dataset import Transition



def transition_priority_score(transition: Transition, alpha: float, beta: float, metrics: Sequence[str]) -> float:
    score = 0.0
    if 'uncertainty' in metrics:
        # Prefer lower-uncertainty imagined samples during conservative replay selection.
        score -= float(alpha) * float(transition.uncertainty)
    if 'reward' in metrics:
        score += float(beta) * abs(float(transition.reward))
    if 'queue_proxy' in metrics:
        latest = np.asarray(transition.next_state, dtype=np.float32)
        score += float(np.mean(latest[:, 0])) if latest.ndim == 2 and latest.shape[1] > 0 else 0.0
    return float(score)



def coverage_signature(transition: Transition, bins: Dict[str, int]) -> Tuple[int, int, int]:
    state = np.asarray(transition.next_state, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] < 5:
        return (0, 0, 0)
    queue_mean = float(np.mean(state[:, 0]))
    phase_mean = float(np.mean(state[:, 3]))
    uncertainty = float(transition.uncertainty)
    q_bin = min(int(queue_mean * max(int(bins.get('queue', 4)), 1)), max(int(bins.get('queue', 4)) - 1, 0))
    p_bin = min(int(phase_mean * max(int(bins.get('phase', 4)), 1)), max(int(bins.get('phase', 4)) - 1, 0))
    u_bin = min(int(uncertainty * max(int(bins.get('uncertainty', 4)), 1)), max(int(bins.get('uncertainty', 4)) - 1, 0))
    return (q_bin, p_bin, u_bin)



def rank_model_samples(
    transitions: Sequence[Transition],
    alpha: float,
    beta: float,
    metrics: Sequence[str],
) -> List[Tuple[float, Transition]]:
    ranked = [(transition_priority_score(item, alpha=alpha, beta=beta, metrics=metrics), item) for item in transitions]
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return ranked



def rebalance_by_coverage(
    ranked_transitions: Sequence[Tuple[float, Transition]],
    keep_count: int,
    bins: Dict[str, int],
    min_fraction: float,
) -> List[Transition]:
    keep_count = max(int(keep_count), 0)
    if keep_count == 0 or not ranked_transitions:
        return []
    grouped = defaultdict(list)
    for score, transition in ranked_transitions:
        grouped[coverage_signature(transition, bins)].append((score, transition))
    min_per_bucket = max(int(np.ceil(float(min_fraction) * keep_count)), 1)
    selected: List[Transition] = []
    for bucket_items in grouped.values():
        for _, item in bucket_items[:min_per_bucket]:
            if len(selected) < keep_count:
                selected.append(item)
    if len(selected) < keep_count:
        seen = {id(item) for item in selected}
        for _, item in ranked_transitions:
            if id(item) in seen:
                continue
            selected.append(item)
            if len(selected) >= keep_count:
                break
    return selected[:keep_count]



def select_model_samples(
    transitions: Sequence[Transition],
    keep_count: int,
    alpha: float,
    beta: float,
    metrics: Sequence[str],
    coverage_enabled: bool,
    bins: Dict[str, int],
    min_fraction: float,
) -> List[Transition]:
    ranked = rank_model_samples(transitions, alpha=alpha, beta=beta, metrics=metrics)
    if not coverage_enabled:
        return [item for _, item in ranked[:keep_count]]
    return rebalance_by_coverage(ranked, keep_count=keep_count, bins=bins, min_fraction=min_fraction)
