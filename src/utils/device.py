from __future__ import annotations

from typing import Any, Dict

import torch


def resolve_device(cfg: Dict[str, Any]) -> torch.device:
    requested = str(cfg.get('device', 'cpu'))
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Config requests CUDA, but torch.cuda.is_available() is False.')
    device = torch.device(requested)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
    print(
        f"[device] requested={requested} actual={device} "
        f"cuda_available={torch.cuda.is_available()} torch_cuda_version={torch.version.cuda}",
        flush=True,
    )
    if device.type == 'cuda':
        print(f"[device] gpu={torch.cuda.get_device_name(device)}", flush=True)
    return device
