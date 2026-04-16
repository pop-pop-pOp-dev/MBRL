from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from src.env.cityflow_signal_env import CityFlowSignalEnv
from src.models.policy_head import MultiDiscretePolicy
from src.utils.seed import seed_everything



def evaluate_policy(cfg: Dict[str, Any], policy: MultiDiscretePolicy, episodes: int = 3) -> Dict[str, float]:
    seed_everything(int(cfg.get('seed', 42)))
    env = CityFlowSignalEnv(cfg)
    device = next(policy.parameters()).device
    delay_list: List[float] = []
    queue_list: List[float] = []
    throughput_list: List[float] = []
    reward_list: List[float] = []
    for _ in range(int(episodes)):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        delays = []
        queues = []
        throughputs = []
        while not done:
            node_x = torch.tensor(obs['node_features'], dtype=torch.float32, device=device)
            edge_index = torch.tensor(obs['edge_index'], dtype=torch.long, device=device)
            edge_attr = torch.tensor(obs['edge_attr'], dtype=torch.float32, device=device)
            action_mask = torch.tensor(obs['action_mask'], dtype=torch.float32, device=device)
            with torch.no_grad():
                out = policy.sample(node_x, edge_index, edge_attr, action_mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(out.actions.detach().cpu().numpy())
            delays.append(float(info.get('delay', 0.0)))
            queues.append(float(info.get('queue', 0.0)))
            throughputs.append(float(info.get('throughput', 0.0)))
            ep_reward += float(reward)
            done = bool(terminated or truncated)
        delay_list.append(float(np.mean(delays)) if delays else 0.0)
        queue_list.append(float(np.mean(queues)) if queues else 0.0)
        throughput_list.append(float(np.mean(throughputs)) if throughputs else 0.0)
        reward_list.append(ep_reward)
    return {
        'avg_delay': float(np.mean(delay_list)),
        'avg_queue': float(np.mean(queue_list)),
        'avg_throughput': float(np.mean(throughput_list)),
        'avg_reward': float(np.mean(reward_list)),
    }
