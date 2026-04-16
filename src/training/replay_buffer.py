from __future__ import annotations

import random
from collections import deque
from typing import Deque, Iterable, List, TypeVar

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


class SplitReplayBuffer:
    def __init__(self, real_capacity: int, model_capacity: int):
        self.real_buffer = ReplayBuffer(real_capacity)
        self.model_buffer = ReplayBuffer(model_capacity)

    def add_real(self, item: T) -> None:
        self.real_buffer.add(item)

    def add_model(self, item: T) -> None:
        self.model_buffer.add(item)

    def sample_mixed(self, real_count: int, model_count: int) -> List[T]:
        return self.real_buffer.sample(real_count) + self.model_buffer.sample(model_count)

    def __len__(self) -> int:
        return len(self.real_buffer) + len(self.model_buffer)
