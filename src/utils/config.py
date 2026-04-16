from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    base_ref = cfg.pop('_base_', None)
    if not base_ref:
        return cfg
    base_path = (path.parent / str(base_ref)).resolve()
    base_cfg = load_config(base_path)
    return _deep_update(base_cfg, cfg)
