from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.env.cityflow_signal_env import CityFlowSignalEnv
from src.training.offline_pretrain import collect_offline_transitions
from src.training.train_world_model import build_ensemble, save_ensemble, train_world_model
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.runtime_log import init_run_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--log-file', default='')
    args = parser.parse_args()
    cfg = load_config(args.config)
    init_run_log(cfg, run_name='train_world_model', log_file=args.log_file or None)
    env = CityFlowSignalEnv(cfg)
    obs, info = env.reset()
    device = resolve_device(cfg)
    edge_index = torch.tensor(obs['edge_index'], dtype=torch.long, device=device)
    edge_attr = torch.tensor(obs['edge_attr'], dtype=torch.float32, device=device)
    ensemble = build_ensemble(
        cfg,
        state_dim=int(obs['node_features'].shape[1]),
        max_actions=int(obs['action_mask'].shape[1]),
        edge_dim=int(obs['edge_attr'].shape[1]),
        observation_spec=info['observation_spec'],
    )
    transitions = collect_offline_transitions(env, num_episodes=int(cfg.get('training', {}).get('collect_episodes', 20)), cfg=cfg)
    stats = train_world_model(cfg, ensemble, edge_index, edge_attr, transitions, device=device)
    out_dir = Path(cfg.get('output_dir', './outputs'))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_ensemble(ensemble, out_dir / 'dynamics_ensemble.pt')
    print(stats)
