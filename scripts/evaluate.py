from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.env.cityflow_signal_env import CityFlowSignalEnv
from src.eval.evaluate import evaluate_policy
from src.models.policy_head import MultiDiscretePolicy
from src.models.value_head import GraphValueHead
from src.training.train_world_model import build_ensemble
from src.utils.config import load_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default='model_last.pt')
    args = parser.parse_args()
    cfg = load_config(args.config)
    env = CityFlowSignalEnv(cfg)
    obs, info = env.reset()
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
    value_net = GraphValueHead(
        input_dim=int(obs['node_features'].shape[1]),
        hidden_dim=int(model_cfg.get('hidden_dim', 128)),
        edge_dim=int(obs['edge_attr'].shape[1]),
        heads=int(model_cfg.get('gat_heads', 4)),
        num_layers=int(model_cfg.get('gat_layers', 3)),
        dropout=float(model_cfg.get('dropout', 0.1)),
    )
    ckpt_path = Path(cfg.get('output_dir', './outputs')) / args.checkpoint
    payload = torch.load(ckpt_path, map_location='cpu')
    policy.load_state_dict(payload['policy'])
    value_net.load_state_dict(payload['value'])
    ensemble = None
    dynamics_path = Path(cfg.get('output_dir', './outputs')) / 'dynamics_ensemble.pt'
    if dynamics_path.exists() and bool(cfg.get('decision', {}).get('eval_enabled', False)):
        ensemble = build_ensemble(
            cfg,
            state_dim=int(obs['node_features'].shape[1]),
            max_actions=int(obs['action_mask'].shape[1]),
            edge_dim=int(obs['edge_attr'].shape[1]),
            observation_spec=info['observation_spec'],
        )
        ensemble_payload = torch.load(dynamics_path, map_location='cpu')
        for model, state_dict in zip(ensemble.models, ensemble_payload['models']):
            model.load_state_dict(state_dict)
    print(evaluate_policy(cfg, policy, value_net=value_net, ensemble=ensemble))
