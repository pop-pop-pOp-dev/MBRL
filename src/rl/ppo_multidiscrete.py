from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from src.models.policy_head import MultiDiscretePolicy
from src.models.value_head import GraphValueHead


@dataclass
class TrajectoryStep:
    state: np.ndarray
    action_mask: np.ndarray
    action: np.ndarray
    reward: float
    done: float
    log_prob: float
    value: float



def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for idx in reversed(range(len(rewards))):
        delta = rewards[idx] + gamma * values[idx + 1] * (1.0 - dones[idx]) - values[idx]
        gae = delta + gamma * gae_lambda * (1.0 - dones[idx]) * gae
        advantages[idx] = gae
    returns = advantages + values[:-1]
    return advantages, returns



def update_ppo(
    policy: MultiDiscretePolicy,
    value_net: GraphValueHead,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    trajectory: List[TrajectoryStep],
    cfg: Dict[str, float],
    policy_opt: torch.optim.Optimizer,
    value_opt: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    rewards = np.asarray([step.reward for step in trajectory], dtype=np.float32)
    dones = np.asarray([step.done for step in trajectory], dtype=np.float32)
    old_log_probs = np.asarray([step.log_prob for step in trajectory], dtype=np.float32)
    values = np.asarray([step.value for step in trajectory] + [trajectory[-1].value], dtype=np.float32)
    advantages, returns = compute_gae(rewards, values, dones, gamma=float(cfg['gamma']), gae_lambda=float(cfg['gae_lambda']))
    advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-6)

    clip_ratio = float(cfg['clip_ratio'])
    entropy_coef = float(cfg['entropy_coef'])
    value_coef = float(cfg['value_coef'])
    grad_clip = float(cfg['grad_clip'])
    epochs = int(cfg['epochs'])
    minibatch = int(cfg['minibatch_size'])
    indices = np.arange(len(trajectory))
    last_stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), minibatch):
            batch_ids = indices[start:start + minibatch]
            if len(batch_ids) == 0:
                continue
            policy_loss = 0.0
            value_loss = 0.0
            entropy = 0.0
            for idx in batch_ids:
                step = trajectory[int(idx)]
                node_x = torch.tensor(step.state, dtype=torch.float32, device=device)
                action_mask = torch.tensor(step.action_mask, dtype=torch.float32, device=device)
                action = torch.tensor(step.action, dtype=torch.long, device=device)
                new_log_prob, ent = policy.evaluate_actions(node_x, edge_index, edge_attr, action_mask, action)
                ratio = torch.exp(new_log_prob - torch.tensor(old_log_probs[int(idx)], dtype=torch.float32, device=device))
                adv = torch.tensor(float(advantages[int(idx)]), dtype=torch.float32, device=device)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
                policy_loss = policy_loss - torch.min(unclipped, clipped)
                target_value = torch.tensor(float(returns[int(idx)]), dtype=torch.float32, device=device)
                pred_value = value_net(node_x, edge_index, edge_attr).squeeze(0)
                value_loss = value_loss + F.mse_loss(pred_value, target_value)
                entropy = entropy + ent
            scale = float(len(batch_ids))
            policy_loss = policy_loss / scale
            value_loss = value_loss / scale
            entropy = entropy / scale
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            policy_opt.zero_grad(set_to_none=True)
            value_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), grad_clip)
            policy_opt.step()
            value_opt.step()
            last_stats = {
                'policy_loss': float(policy_loss.detach().cpu().item()),
                'value_loss': float(value_loss.detach().cpu().item()),
                'entropy': float(entropy.detach().cpu().item()),
            }
    return last_stats
