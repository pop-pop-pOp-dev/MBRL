from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.env.cityflow_signal_env import CityFlowSignalEnv
from src.eval.evaluate import evaluate_policy
from src.models.policy_head import MultiDiscretePolicy
from src.utils.config import load_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default='model_last.pt')
    args = parser.parse_args()
    cfg = load_config(args.config)
    env = CityFlowSignalEnv(cfg)
    obs, _ = env.reset()
    model_cfg = cfg.get('model', {})
    policy = MultiDiscretePolicy(
        input_dim=int(obs['node_features'].shape[1]),
        hidden_dim=int(model_cfg.get('hidden_dim', 128)),
        max_actions=int(obs['action_mask'].shape[1]),
        edge_dim=int(obs['edge_attr'].shape[1]),
        heads=int(model_cfg.get('gat_heads', 4)),
        num_layers=int(model_cfg.get('gat_layers', 3)),
        dropout=float(model_cfg.get('dropout', 0.1)),
    )
    ckpt_path = Path(cfg.get('output_dir', './outputs')) / args.checkpoint
    payload = torch.load(ckpt_path, map_location='cpu')
    policy.load_state_dict(payload['policy'])
    print(evaluate_policy(cfg, policy))
