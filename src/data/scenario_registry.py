from __future__ import annotations

from pathlib import Path
from typing import Dict
from urllib.request import urlretrieve


ROADNETSZ_BASE_URL = 'https://github.com/zhuliwen/RoadnetSZ/raw/main/data_cityflow/'
ROADNETSZ_VARIANTS = {
    'fuhua_2490': {
        'roadnet_file': 'roadnet_1_33.json',
        'flow_file': 'anon_1_33_fuhua_24hto1w_2490.json',
    },
    'fuhua_4089': {
        'roadnet_file': 'roadnet_1_33.json',
        'flow_file': 'anon_1_33_fuhua_4_27_24hto1w_4089.json',
    },
}


def ensure_builtin_scenario(data_dir: str, variant: str = 'fuhua_2490') -> Dict[str, str]:
    if variant not in ROADNETSZ_VARIANTS:
        raise ValueError(f'Unknown RoadnetSZ variant: {variant}')
    spec = ROADNETSZ_VARIANTS[variant]
    base = Path(data_dir) / 'RoadnetSZ'
    base.mkdir(parents=True, exist_ok=True)
    roadnet_path = base / spec['roadnet_file']
    flow_path = base / spec['flow_file']
    if not roadnet_path.exists():
        urlretrieve(f"{ROADNETSZ_BASE_URL}{spec['roadnet_file']}", roadnet_path)
    if not flow_path.exists():
        urlretrieve(f"{ROADNETSZ_BASE_URL}{spec['flow_file']}", flow_path)
    config_path = base / f'{variant}_engine_config.json'
    return {
        'roadnet_path': str(roadnet_path),
        'flow_path': str(flow_path),
        'config_path': str(config_path),
    }
