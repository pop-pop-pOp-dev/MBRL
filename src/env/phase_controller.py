from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.data.cityflow_parser import RoadGraph


@dataclass
class PhaseState:
    current_phase: int
    elapsed: float
    last_switch_time: float = 0.0


class PhaseController:
    def __init__(self, graph: RoadGraph, min_green: float, yellow_time: float, all_red_time: float):
        self.graph = graph
        self.min_green = float(min_green)
        self.yellow_time = float(yellow_time)
        self.all_red_time = float(all_red_time)
        self.states: Dict[int, PhaseState] = {
            item.index: PhaseState(current_phase=0, elapsed=0.0, last_switch_time=0.0)
            for item in graph.intersections
        }

    def reset(self) -> None:
        for idx in self.states:
            self.states[idx] = PhaseState(current_phase=0, elapsed=0.0, last_switch_time=0.0)

    def step_time(self, dt: float) -> None:
        for state in self.states.values():
            state.elapsed += float(dt)

    def build_action_mask(self) -> np.ndarray:
        max_phases = max(self.graph.max_phase_count, 1)
        mask = np.zeros((self.graph.num_nodes, max_phases + 1), dtype=np.float32)
        for item in self.graph.intersections:
            state = self.states[item.index]
            mask[item.index, 0] = 1.0
            if state.elapsed < self.min_green:
                continue
            for phase_idx in range(item.phase_count):
                mask[item.index, phase_idx + 1] = 1.0
        return mask

    def apply_actions(self, actions: np.ndarray, current_time: float) -> np.ndarray:
        switches = np.zeros(self.graph.num_nodes, dtype=np.float32)
        for item in self.graph.intersections:
            action_value = int(actions[item.index])
            state = self.states[item.index]
            if action_value <= 0:
                continue
            target_phase = action_value - 1
            if target_phase >= item.phase_count:
                continue
            if state.elapsed < self.min_green:
                continue
            if target_phase != state.current_phase:
                state.current_phase = target_phase
                state.elapsed = 0.0
                state.last_switch_time = float(current_time)
                switches[item.index] = 1.0
        return switches

    def current_phases(self) -> np.ndarray:
        return np.asarray([self.states[idx].current_phase for idx in range(self.graph.num_nodes)], dtype=np.int64)

    def remaining_green(self) -> np.ndarray:
        remaining: List[float] = []
        for idx in range(self.graph.num_nodes):
            state = self.states[idx]
            remaining.append(max(self.min_green - state.elapsed, 0.0))
        return np.asarray(remaining, dtype=np.float32)
