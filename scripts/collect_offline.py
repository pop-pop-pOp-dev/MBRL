from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.env.cityflow_signal_env import CityFlowSignalEnv
from src.training.offline_pretrain import collect_offline_transitions
from src.utils.config import load_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    env = CityFlowSignalEnv(cfg)
    transitions = collect_offline_transitions(env, num_episodes=int(cfg.get('training', {}).get('collect_episodes', 20)))
    out_dir = Path(cfg.get('output_dir', './outputs'))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(transitions, out_dir / 'offline_transitions.pt')
    print(f'saved {len(transitions)} transitions to {out_dir / "offline_transitions.pt"}')
