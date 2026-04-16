from __future__ import annotations

import argparse

from src.training.train_mbrl_ppo import train_mbrl_ppo
from src.utils.config import load_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = train_mbrl_ppo(cfg)
    print(result)
