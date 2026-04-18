from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import sys


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self._streams)


def init_run_log(cfg: Dict[str, Any], run_name: str, log_file: str | None = None) -> Path:
    output_dir = Path(cfg.get('output_dir', './outputs'))
    default_logs_dir = output_dir / 'logs'
    target = Path(log_file) if log_file else (default_logs_dir / f'{run_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    target.parent.mkdir(parents=True, exist_ok=True)

    log_handle = target.open('a', encoding='utf-8', buffering=1)
    sys.stdout = _TeeStream(sys.__stdout__, log_handle)
    sys.stderr = _TeeStream(sys.__stderr__, log_handle)
    print(f'[log] writing combined stdout/stderr to: {target.resolve()}', flush=True)
    return target
