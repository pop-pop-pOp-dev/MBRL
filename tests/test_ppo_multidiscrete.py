from __future__ import annotations

import numpy as np
import torch

from src.models.policy_head import MultiDiscretePolicy
from src.models.value_head import GraphValueHead
from src.rl.ppo_multidiscrete import TrajectoryStep, update_ppo



def test_ppo_update_runs():
    device = torch.device('cpu')
    policy = MultiDiscretePolicy(input_dim=5, hidden_dim=8, max_actions=4, edge_dim=2).to(device)
    value = GraphValueHead(input_dim=5, hidden_dim=8, edge_dim=2).to(device)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    edge_attr = torch.randn(2, 2, device=device)
    trajectory = [
        TrajectoryStep(
            state=np.random.randn(2, 5).astype(np.float32),
            action_mask=np.ones((2, 4), dtype=np.float32),
            action=np.array([0, 1], dtype=np.int64),
            reward=1.0,
            done=0.0,
            log_prob=-0.5,
            value=0.1,
        ),
        TrajectoryStep(
            state=np.random.randn(2, 5).astype(np.float32),
            action_mask=np.ones((2, 4), dtype=np.float32),
            action=np.array([1, 2], dtype=np.int64),
            reward=0.5,
            done=1.0,
            log_prob=-0.3,
            value=0.0,
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
