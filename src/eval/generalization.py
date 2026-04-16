from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from src.eval.evaluate import evaluate_policy


def evaluate_generalization(cfg: Dict[str, Any], policy) -> Dict[str, float]:
    cfg_alt = deepcopy(cfg)
    cfg_alt.setdefault('data', {})['roadnetsz_variant'] = 'fuhua_4089'
    return evaluate_policy(cfg_alt, policy, episodes=2)
