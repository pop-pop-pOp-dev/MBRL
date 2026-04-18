from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List

import numpy as np
import torch

from src.data.cityflow_parser import RoadGraph
from src.data.roadnet_features import build_static_node_features


@dataclass
class IntersectionMetrics:
    queue: np.ndarray
    vehicle_count: np.ndarray
    average_speed: np.ndarray
    current_phase: np.ndarray
    remaining_green: np.ndarray


@dataclass(frozen=True)
class ObservationSpec:
    dynamic_dim: int
    stack_k: int
    static_dim: int

    @property
    def feature_dim(self) -> int:
        return self.dynamic_dim * self.stack_k + self.static_dim

    @property
    def stacked_dynamic_dim(self) -> int:
        return self.dynamic_dim * self.stack_k

    @property
    def latest_dynamic_slice(self) -> slice:
        start = (self.stack_k - 1) * self.dynamic_dim
        return slice(start, start + self.dynamic_dim)

    @property
    def static_slice(self) -> slice:
        return slice(self.stacked_dynamic_dim, self.feature_dim)

    def split_state(self, state: np.ndarray | torch.Tensor):
        return state[..., : self.stacked_dynamic_dim], state[..., self.stacked_dynamic_dim : self.feature_dim]

    def latest_dynamic(self, state: np.ndarray | torch.Tensor):
        return state[..., self.latest_dynamic_slice]

    def previous_dynamic(self, state: np.ndarray | torch.Tensor):
        if self.stack_k <= 1:
            return self.latest_dynamic(state)
        start = (self.stack_k - 2) * self.dynamic_dim
        return state[..., start : start + self.dynamic_dim]

    def compose_next_state(self, previous_state: np.ndarray | torch.Tensor, next_dynamic: np.ndarray | torch.Tensor):
        stacked_dynamic, static_features = self.split_state(previous_state)
        if isinstance(previous_state, torch.Tensor):
            leading_shape = previous_state.shape[:-1]
            history = stacked_dynamic.reshape(*leading_shape, self.stack_k, self.dynamic_dim)
            shifted = torch.cat([history[..., 1:, :], next_dynamic.unsqueeze(-2)], dim=-2)
            return torch.cat([shifted.reshape(*leading_shape, -1), static_features], dim=-1)
        leading_shape = previous_state.shape[:-1]
        history = stacked_dynamic.reshape(*leading_shape, self.stack_k, self.dynamic_dim)
        shifted = np.concatenate([history[..., 1:, :], np.expand_dims(next_dynamic, axis=-2)], axis=-2)
        return np.concatenate([shifted.reshape(*leading_shape, -1), static_features], axis=-1)

    def metrics_from_state(self, state: np.ndarray | torch.Tensor) -> IntersectionMetrics:
        latest = self.latest_dynamic(state)
        if isinstance(latest, torch.Tensor):
            latest = latest.detach().cpu().numpy()
        return IntersectionMetrics(
            queue=np.asarray(latest[:, 0], dtype=np.float32),
            vehicle_count=np.asarray(latest[:, 1], dtype=np.float32),
            average_speed=np.asarray(latest[:, 2], dtype=np.float32),
            current_phase=np.asarray(latest[:, 3], dtype=np.float32),
            remaining_green=np.asarray(latest[:, 4], dtype=np.float32),
        )


class ObservationBuilder:
    def __init__(self, graph: RoadGraph, stack_k: int = 4, normalize: bool = True):
        self.graph = graph
        self.stack_k = int(stack_k)
        self.normalize = bool(normalize)
        self.static_features = build_static_node_features(graph)
        self.dynamic_dim = 5
        self.spec = ObservationSpec(dynamic_dim=self.dynamic_dim, stack_k=self.stack_k, static_dim=int(self.static_features.shape[1]))
        self.history: Deque[np.ndarray] = deque(maxlen=self.stack_k)
        self.last_metrics = IntersectionMetrics(
            queue=np.zeros(graph.num_nodes, dtype=np.float32),
            vehicle_count=np.zeros(graph.num_nodes, dtype=np.float32),
            average_speed=np.zeros(graph.num_nodes, dtype=np.float32),
            current_phase=np.zeros(graph.num_nodes, dtype=np.float32),
            remaining_green=np.zeros(graph.num_nodes, dtype=np.float32),
        )

    @property
    def feature_dim(self) -> int:
        return self.spec.feature_dim

    def reset(self) -> None:
        self.history.clear()
        self.last_metrics = IntersectionMetrics(
            queue=np.zeros(self.graph.num_nodes, dtype=np.float32),
            vehicle_count=np.zeros(self.graph.num_nodes, dtype=np.float32),
            average_speed=np.zeros(self.graph.num_nodes, dtype=np.float32),
            current_phase=np.zeros(self.graph.num_nodes, dtype=np.float32),
            remaining_green=np.zeros(self.graph.num_nodes, dtype=np.float32),
        )

    def _aggregate(
        self,
        lane_waiting: Dict[str, float],
        lane_vehicle_count: Dict[str, float],
        lane_speeds: Dict[str, float],
        current_phases: np.ndarray,
        remaining_green: np.ndarray,
    ) -> np.ndarray:
        rows: List[List[float]] = []
        for item in self.graph.intersections:
            queues = [float(lane_waiting.get(lane_id, 0.0)) for lane_id in item.incoming_lanes]
            counts = [float(lane_vehicle_count.get(lane_id, 0.0)) for lane_id in item.incoming_lanes]
            speeds = [float(lane_speeds.get(lane_id, 0.0)) for lane_id in item.incoming_lanes]
            rows.append(
                [
                    float(np.sum(queues)),
                    float(np.sum(counts)),
                    float(np.mean(speeds)) if speeds else 0.0,
                    float(current_phases[item.index]),
                    float(remaining_green[item.index]),
                ]
            )
        dynamic = np.asarray(rows, dtype=np.float32)
        self.last_metrics = IntersectionMetrics(
            queue=dynamic[:, 0].copy(),
            vehicle_count=dynamic[:, 1].copy(),
            average_speed=dynamic[:, 2].copy(),
            current_phase=dynamic[:, 3].copy(),
            remaining_green=dynamic[:, 4].copy(),
        )
        if self.normalize:
            denom = np.maximum(dynamic.max(axis=0, keepdims=True), 1.0)
            dynamic = dynamic / denom
        return dynamic

    def build(
        self,
        lane_waiting: Dict[str, float],
        lane_vehicle_count: Dict[str, float],
        lane_speeds: Dict[str, float],
        current_phases: np.ndarray,
        remaining_green: np.ndarray,
    ) -> np.ndarray:
        current_dynamic = self._aggregate(lane_waiting, lane_vehicle_count, lane_speeds, current_phases, remaining_green)
        self.history.append(current_dynamic)
        while len(self.history) < self.stack_k:
            self.history.appendleft(np.zeros_like(current_dynamic))
        stacked_dynamic = np.concatenate(list(self.history), axis=1)
        return np.concatenate([stacked_dynamic, self.static_features], axis=1).astype(np.float32)
