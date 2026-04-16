from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from src.training.train_mbrl_ppo import train_mbrl_ppo


def train_real_only(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(cfg)
    cfg.setdefault('training', {})['model_ratio'] = 0.0
    return train_mbrl_ppo(cfg)
