from __future__ import annotations

import argparse

from src.baselines.ppo_real_only import train_real_only
from src.utils.config import load_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = train_real_only(cfg)
    print(result)
