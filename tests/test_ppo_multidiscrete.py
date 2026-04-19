from __future__ import annotations

import numpy as np
import torch

from src.models.policy_head import MultiDiscretePolicy
from src.models.value_head import GraphValueHead
from src.rl.ppo_multidiscrete import TrajectoryStep, compute_targets, update_ppo
from src.training.train_mbrl_ppo import _effective_model_ratio



def test_ppo_update_runs():
    device = torch.device('cpu')
    policy = MultiDiscretePolicy(input_dim=5, hidden_dim=8, max_actions=4, edge_dim=2).to(device)
    value = GraphValueHead(input_dim=5, hidden_dim=8, edge_dim=2).to(device)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    edge_attr = torch.randn(2, 2, device=device)
    next_state = np.random.randn(2, 5).astype(np.float32)
    trajectory = [
        TrajectoryStep(
            state=np.random.randn(2, 5).astype(np.float32),
            action_mask=np.ones((2, 4), dtype=np.float32),
            action=np.array([0, 1], dtype=np.int64),
            reward=1.0,
            done=0.0,
            log_prob=-0.5,
            value=0.1,
            next_state=next_state,
            next_action_mask=np.ones((2, 4), dtype=np.float32),
            next_value=0.2,
        ),
        TrajectoryStep(
            state=np.random.randn(2, 5).astype(np.float32),
            action_mask=np.ones((2, 4), dtype=np.float32),
            action=np.array([1, 2], dtype=np.int64),
            reward=0.5,
            done=1.0,
            log_prob=-0.3,
            value=0.0,
            next_state=np.random.randn(2, 5).astype(np.float32),
            next_action_mask=np.ones((2, 4), dtype=np.float32),
            next_value=0.0,
        ),
    ]
    policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    value_opt = torch.optim.Adam(value.parameters(), lr=1e-3)
    stats = update_ppo(
        policy,
        value,
        edge_index,
        edge_attr,
        trajectory,
        {'gamma': 0.99, 'gae_lambda': 0.95, 'clip_ratio': 0.2, 'entropy_coef': 0.01, 'value_coef': 0.5, 'grad_clip': 1.0, 'epochs': 1, 'minibatch_size': 2},
        policy_opt,
        value_opt,
        device,
    )
    assert 'policy_loss' in stats


def test_compute_targets_uses_discounted_returns():
    steps = [
        TrajectoryStep(
            state=np.zeros((2, 5), dtype=np.float32),
            action_mask=np.ones((2, 4), dtype=np.float32),
            action=np.array([0, 1], dtype=np.int64),
            reward=1.0,
            done=0.0,
            log_prob=0.0,
            value=0.5,
            next_state=np.zeros((2, 5), dtype=np.float32),
            next_action_mask=np.ones((2, 4), dtype=np.float32),
            next_value=0.4,
        ),
        TrajectoryStep(
            state=np.zeros((2, 5), dtype=np.float32),
            action_mask=np.ones((2, 4), dtype=np.float32),
            action=np.array([1, 2], dtype=np.int64),
            reward=2.0,
            done=1.0,
            log_prob=0.0,
            value=0.4,
            next_state=np.zeros((2, 5), dtype=np.float32),
            next_action_mask=np.ones((2, 4), dtype=np.float32),
            next_value=0.0,
        ),
    ]
    advantages, returns = compute_targets(steps, gamma=0.9, gae_lambda=1.0)
    assert advantages.shape == (2,)
    assert np.allclose(returns, np.asarray([2.8, 2.0], dtype=np.float32), atol=1e-5)


def test_effective_model_ratio_respects_warmup_and_ramp():
    train_cfg = {
        'model_ratio': 0.06,
        'model_warmup_updates': 2,
        'model_ratio_ramp_updates': 3,
    }
    assert _effective_model_ratio(train_cfg, update_idx=0, augmentation_enabled=True) == 0.0
    assert _effective_model_ratio(train_cfg, update_idx=1, augmentation_enabled=True) == 0.0
    assert np.isclose(_effective_model_ratio(train_cfg, update_idx=2, augmentation_enabled=True), 0.02)
    assert np.isclose(_effective_model_ratio(train_cfg, update_idx=3, augmentation_enabled=True), 0.04)
    assert np.isclose(_effective_model_ratio(train_cfg, update_idx=4, augmentation_enabled=True), 0.06)
