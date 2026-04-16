from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from src.data.cityflow_parser import RoadGraph, parse_cityflow_roadnet
from src.data.scenario_registry import ensure_builtin_scenario
from src.env.observation import ObservationBuilder
from src.env.phase_controller import PhaseController
from src.env.reward import compute_reward

try:
    import cityflow  # type: ignore
except Exception:
    cityflow = None


class CityFlowSignalEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.data_cfg = cfg.get('data', {})
        self.env_cfg = cfg.get('env', {})
        self.output_dir = Path(cfg.get('output_dir', './outputs'))
        self.resolved = self._resolve_paths()
        self.graph: RoadGraph = parse_cityflow_roadnet(self.resolved['roadnet_path'])
        self.phase_controller = PhaseController(
            self.graph,
            min_green=float(self.env_cfg.get('min_green', 10)),
            yellow_time=float(self.env_cfg.get('yellow_time', 3)),
            all_red_time=float(self.env_cfg.get('all_red_time', 1)),
        )
        self.observation_builder = ObservationBuilder(
            self.graph,
            stack_k=int(self.env_cfg.get('stack_k', 4)),
            normalize=bool(self.env_cfg.get('normalize_obs', True)),
        )
        self.engine = None
        self.current_step = 0
        self.control_interval = int(self.env_cfg.get('control_interval', 5))
        self.episode_horizon = int(self.env_cfg.get('episode_horizon', 360))
        self._last_finished_vehicle_count = 0.0
        max_phases = max(self.graph.max_phase_count, 1)
        self.action_space = gym.spaces.MultiDiscrete(np.full(self.graph.num_nodes, max_phases + 1, dtype=np.int64))
        self.observation_space = gym.spaces.Dict(
            {
                'node_features': gym.spaces.Box(low=0.0, high=1.0, shape=(self.graph.num_nodes, self.observation_builder.feature_dim), dtype=np.float32),
                'edge_index': gym.spaces.Box(low=0, high=max(self.graph.num_nodes - 1, 0), shape=self.graph.edge_index.shape, dtype=np.int64),
                'edge_attr': gym.spaces.Box(low=0.0, high=1.0, shape=self.graph.edge_features.shape, dtype=np.float32),
                'action_mask': gym.spaces.Box(low=0.0, high=1.0, shape=(self.graph.num_nodes, max_phases + 1), dtype=np.float32),
            }
        )

    def _resolve_paths(self) -> Dict[str, str]:
        scenario = str(self.data_cfg.get('scenario', 'roadnetsz_fuhua'))
        if scenario == 'roadnetsz_fuhua':
            resolved = ensure_builtin_scenario(str(self.data_cfg.get('data_dir', './data')), str(self.data_cfg.get('roadnetsz_variant', 'fuhua_2490')))
        else:
            resolved = {
                'roadnet_path': str(self.data_cfg['custom_roadnet_path']),
                'flow_path': str(self.data_cfg['custom_flow_path']),
                'config_path': str(self.data_cfg.get('custom_engine_config_path') or ''),
            }
        if not resolved.get('config_path'):
            resolved['config_path'] = self._write_engine_config(resolved['roadnet_path'], resolved['flow_path'])
        return resolved

    def _write_engine_config(self, roadnet_path: str, flow_path: str) -> str:
        base_dir = Path(roadnet_path).resolve().parent
        payload = {
            'interval': 1.0,
            'seed': int(self.cfg.get('seed', 42)),
            'dir': str(base_dir),
            'roadnetFile': Path(roadnet_path).name,
            'flowFile': Path(flow_path).name,
            'rlTrafficLight': True,
            'saveReplay': False,
            'laneChange': False,
        }
        config_path = base_dir / 'generated_engine_config.json'
        config_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return str(config_path)

    def _ensure_engine(self) -> None:
        if cityflow is None:
            raise ImportError('cityflow is required to run CityFlowSignalEnv. Install it with pip install cityflow.')
        if self.engine is None:
            self.engine = cityflow.Engine(self.resolved['config_path'], thread_num=1)

    def _lane_waiting(self) -> Dict[str, float]:
        return {str(k): float(v) for k, v in self.engine.get_lane_waiting_vehicle_count().items()}

    def _lane_vehicle_count(self) -> Dict[str, float]:
        return {str(k): float(v) for k, v in self.engine.get_lane_vehicle_count().items()}

    def _lane_speeds(self) -> Dict[str, float]:
        lane_vehicles = self.engine.get_lane_vehicles()
        vehicle_speed = self.engine.get_vehicle_speed()
        out: Dict[str, float] = {}
        for lane_id, vehicle_ids in lane_vehicles.items():
            speeds = [float(vehicle_speed.get(vehicle_id, 0.0)) for vehicle_id in vehicle_ids]
            out[str(lane_id)] = float(np.mean(speeds)) if speeds else 0.0
        return out

    def _throughput_value(self) -> float:
        if hasattr(self.engine, 'get_finished_vehicle_count'):
            finished = float(self.engine.get_finished_vehicle_count())
            throughput = max(finished - self._last_finished_vehicle_count, 0.0)
            self._last_finished_vehicle_count = finished
            return throughput
        current_count = float(np.sum(list(self.engine.get_lane_vehicle_count().values())))
        throughput = max(self._last_finished_vehicle_count - current_count, 0.0)
        self._last_finished_vehicle_count = current_count
        return throughput

    def _build_obs(self) -> Dict[str, np.ndarray]:
        phases = self.phase_controller.current_phases()
        remaining = self.phase_controller.remaining_green()
        node_features = self.observation_builder.build(
            self._lane_waiting(),
            self._lane_vehicle_count(),
            self._lane_speeds(),
            phases,
            remaining,
        )
        return {
            'node_features': node_features.astype(np.float32),
            'edge_index': self.graph.edge_index.astype(np.int64),
            'edge_attr': self.graph.edge_features.astype(np.float32),
            'action_mask': self.phase_controller.build_action_mask().astype(np.float32),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del seed, options
        self._ensure_engine()
        self.engine.reset()
        self.current_step = 0
        self._last_finished_vehicle_count = 0.0
        self.phase_controller.reset()
        self.observation_builder.reset()
        obs = self._build_obs()
        return obs, {'resolved_paths': self.resolved}

    def step(self, action: np.ndarray):
        self._ensure_engine()
        action = np.asarray(action, dtype=np.int64).reshape(self.graph.num_nodes)
        current_time = float(self.engine.get_current_time())
        switches = self.phase_controller.apply_actions(action, current_time=current_time)
        for inter in self.graph.intersections:
            if inter.is_virtual:
                continue
            if self.phase_controller.states[inter.index].current_phase < inter.phase_count:
                self.engine.set_tl_phase(inter.intersection_id, int(self.phase_controller.states[inter.index].current_phase))
        for _ in range(self.control_interval):
            self.engine.next_step()
            self.phase_controller.step_time(1.0)
        self.current_step += 1
        obs = self._build_obs()
        metrics = self.observation_builder.last_metrics
        throughput = self._throughput_value()
        reward, terms = compute_reward(metrics.queue, metrics.average_speed, switches, throughput, self.env_cfg.get('reward', {}))
        terminated = self.current_step >= self.episode_horizon
        info = dict(terms)
        info['current_time'] = float(self.engine.get_current_time())
        info['queue_mean'] = float(np.mean(metrics.queue))
        info['speed_mean'] = float(np.mean(metrics.average_speed))
        info['vehicle_count_mean'] = float(np.mean(metrics.vehicle_count))
        return obs, float(reward), bool(terminated), False, info
