from __future__ import annotations

import random
from collections import deque
from typing import Deque, Iterable, List, Sequence, TypeVar

T = TypeVar('T')


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer: Deque[T] = deque(maxlen=self.capacity)

    def add(self, item: T) -> None:
        self.buffer.append(item)

    def extend(self, items: Iterable[T]) -> None:
        for item in items:
            self.add(item)

    def sample(self, batch_size: int) -> List[T]:
        batch_size = min(int(batch_size), len(self.buffer))
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
