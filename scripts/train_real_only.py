from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.ppo_real_only import train_real_only
from src.utils.config import load_config
from src.utils.runtime_log import init_run_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--log-file', default='')
    args = parser.parse_args()
    cfg = load_config(args.config)
    init_run_log(cfg, run_name='train_real_only', log_file=args.log_file or None)
    result = train_real_only(cfg)
    print(result)
