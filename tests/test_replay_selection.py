from __future__ import annotations

import numpy as np

from src.data.offline_dataset import Transition
from src.training.replay_buffer import SplitReplayBuffer
from src.training.sample_selection import select_model_samples, transition_priority_score



def _make_transition(value: float, uncertainty: float) -> Transition:
    state = np.full((2, 5), value, dtype=np.float32)
    next_state = np.full((2, 5), value + 0.1, dtype=np.float32)
    return Transition(
        state=state,
        action=np.zeros(2, dtype=np.int64),
        reward=float(value),
        next_state=next_state,
        done=0.0,
        action_mask=np.ones((2, 3), dtype=np.float32),
        next_action_mask=np.ones((2, 3), dtype=np.float32),
        uncertainty=float(uncertainty),
        source='model',
    )



def test_split_replay_buffer_ratio_sampling():
    replay = SplitReplayBuffer(real_capacity=10, model_capacity=10)
    for idx in range(6):
        replay.add_real(_make_transition(float(idx), 0.0))
    for idx in range(4):
        replay.add_model(_make_transition(float(idx), 0.5))
    real_items, model_items = replay.sample_mixed_by_ratio(total_count=5, real_ratio=0.6, model_ratio=0.4)
    assert len(real_items) == 3
    assert len(model_items) == 2



def test_priority_and_coverage_selection_runs():
    transitions = [_make_transition(float(idx) / 10.0, uncertainty=float(idx) / 10.0) for idx in range(6)]
    selected = select_model_samples(
        transitions=transitions,
        keep_count=3,
        alpha=1.0,
        beta=0.5,
        metrics=['uncertainty', 'reward', 'queue_proxy'],
        coverage_enabled=True,
        bins={'queue': 3, 'phase': 3, 'uncertainty': 3},
        min_fraction=0.1,
    )
    assert len(selected) == 3


def test_priority_score_penalizes_high_uncertainty():
    low_uncertainty = _make_transition(0.5, uncertainty=0.1)
    high_uncertainty = _make_transition(0.5, uncertainty=0.9)
    low_score = transition_priority_score(low_uncertainty, alpha=1.0, beta=0.0, metrics=['uncertainty'])
    high_score = transition_priority_score(high_uncertainty, alpha=1.0, beta=0.0, metrics=['uncertainty'])
    assert low_score > high_score
