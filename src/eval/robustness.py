from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from src.eval.evaluate import evaluate_policy


def evaluate_robustness(cfg: Dict[str, Any], policy) -> Dict[str, float]:
    cfg_alt = deepcopy(cfg)
    reward_cfg = cfg_alt.setdefault('env', {}).setdefault('reward', {})
    reward_cfg['beta_queue'] = float(reward_cfg.get('beta_queue', 0.5)) * 1.2
    return evaluate_policy(cfg_alt, policy, episodes=2)
