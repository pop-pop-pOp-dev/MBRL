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

    def latest(self, batch_size: int) -> List[T]:
        batch_size = min(int(batch_size), len(self.buffer))
        if batch_size <= 0:
            return []
        return list(self.buffer)[-batch_size:]

    def all(self) -> List[T]:
        return list(self.buffer)

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

    def sample_real(self, batch_size: int, strategy: str = 'random') -> List[T]:
        if strategy == 'latest':
            return self.real_buffer.latest(batch_size)
        return self.real_buffer.sample(batch_size)

    def sample_model(self, batch_size: int, strategy: str = 'random') -> List[T]:
        if strategy == 'latest':
            return self.model_buffer.latest(batch_size)
        return self.model_buffer.sample(batch_size)

    def sample_mixed_by_ratio(
        self,
        total_count: int,
        real_ratio: float,
        model_ratio: float,
        real_strategy: str = 'random',
        model_strategy: str = 'random',
    ) -> tuple[List[T], List[T]]:
        total_count = max(int(total_count), 0)
        if total_count == 0:
            return [], []
        ratio_sum = max(float(real_ratio) + float(model_ratio), 1e-6)
        desired_real = int(total_count * float(real_ratio) / ratio_sum)
        desired_model = total_count - desired_real

        available_real = len(self.real_buffer)
        available_model = len(self.model_buffer)
        actual_real = min(desired_real, available_real)
        actual_model = min(desired_model, available_model)
        shortfall = total_count - actual_real - actual_model

        if shortfall > 0:
            real_extra = min(shortfall, max(available_real - actual_real, 0))
            actual_real += real_extra
            shortfall -= real_extra
        if shortfall > 0:
            model_extra = min(shortfall, max(available_model - actual_model, 0))
            actual_model += model_extra

        real_items = self.sample_real(actual_real, strategy=real_strategy)
        model_items = self.sample_model(actual_model, strategy=model_strategy)
        return real_items[:actual_real], model_items[:actual_model]

    def __len__(self) -> int:
        return len(self.real_buffer) + len(self.model_buffer)
