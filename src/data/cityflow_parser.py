from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class LaneMeta:
    lane_id: str
    road_id: str
    index: int
    max_speed: float
    length: float


@dataclass(frozen=True)
class PhaseMeta:
    phase_index: int
    available_road_links: Tuple[int, ...]
    time: float = 0.0


@dataclass
class IntersectionMeta:
    intersection_id: str
    index: int
    incoming_lanes: List[str]
    outgoing_lanes: List[str]
    phases: List[PhaseMeta]
    phase_count: int
    is_virtual: bool
    road_links: List[dict] = field(default_factory=list)
    neighbors: List[int] = field(default_factory=list)


@dataclass
class RoadGraph:
    intersections: List[IntersectionMeta]
    lanes: Dict[str, LaneMeta]
    roads: Dict[str, dict]
    edge_index: np.ndarray
    edge_features: np.ndarray
    lane_to_intersection: Dict[str, int]
    max_phase_count: int
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.intersections)



def _lane_length(road: dict) -> float:
    points = road.get('points') or []
    if len(points) >= 2:
        total = 0.0
        for idx in range(len(points) - 1):
            dx = float(points[idx + 1].get('x', 0.0)) - float(points[idx].get('x', 0.0))
            dy = float(points[idx + 1].get('y', 0.0)) - float(points[idx].get('y', 0.0))
            total += float((dx * dx + dy * dy) ** 0.5)
        if total > 0:
            return total
    return float(road.get('length', 1.0) or 1.0)



def _lane_speed(lane: dict, road: dict) -> float:
    value = lane.get('maxSpeed', road.get('maxSpeed', 13.89))
    try:
        return max(float(value), 1e-6)
    except Exception:
        return 13.89



def parse_cityflow_roadnet(roadnet_path: str | Path) -> RoadGraph:
    payload = json.loads(Path(roadnet_path).read_text(encoding='utf-8'))
    intersections_raw = payload.get('intersections') or []
    roads_raw = payload.get('roads') or []
    if not intersections_raw or not roads_raw:
        raise ValueError(f'Invalid CityFlow roadnet: {roadnet_path}')

    intersections: List[IntersectionMeta] = []
    inter_id_to_idx: Dict[str, int] = {}
    for idx, item in enumerate(intersections_raw):
        inter_id = str(item['id'])
        inter_id_to_idx[inter_id] = idx
        phases_raw = (item.get('trafficLight') or {}).get('lightphases') or []
        phases = [
            PhaseMeta(
                phase_index=int(i),
                available_road_links=tuple(int(x) for x in (phase.get('availableRoadLinks') or [])),
                time=float(phase.get('time', 0.0) or 0.0),
            )
            for i, phase in enumerate(phases_raw)
        ]
        intersections.append(
            IntersectionMeta(
                intersection_id=inter_id,
                index=idx,
                incoming_lanes=[],
                outgoing_lanes=[],
                phases=phases,
                phase_count=len(phases),
                is_virtual=bool(item.get('virtual', False)),
                road_links=list(item.get('roadLinks') or []),
            )
        )

    roads: Dict[str, dict] = {}
    lanes: Dict[str, LaneMeta] = {}
    lane_to_intersection: Dict[str, int] = {}
    edges: List[Tuple[int, int]] = []
    edge_features: List[List[float]] = []

    for road in roads_raw:
        road_id = str(road['id'])
        roads[road_id] = road
        start = str(road['startIntersection'])
        end = str(road['endIntersection'])
        if start not in inter_id_to_idx or end not in inter_id_to_idx:
            continue
        src_idx = inter_id_to_idx[start]
        dst_idx = inter_id_to_idx[end]
        edges.append((src_idx, dst_idx))
        intersections[src_idx].neighbors.append(dst_idx)

        lane_length = _lane_length(road)
        road_lane_ids: List[str] = []
        speed_values: List[float] = []
        for lane_idx, lane in enumerate(road.get('lanes') or [{}]):
            lane_id = f'{road_id}_{lane_idx}'
            max_speed = _lane_speed(lane, road)
            speed_values.append(max_speed)
            lanes[lane_id] = LaneMeta(
                lane_id=lane_id,
                road_id=road_id,
                index=lane_idx,
                max_speed=max_speed,
                length=lane_length,
            )
            road_lane_ids.append(lane_id)
        lane_count = max(len(road_lane_ids), 1)
        avg_speed = float(np.mean(speed_values)) if speed_values else _lane_speed({}, road)
        edge_features.append([float(lane_count), float(lane_length), float(avg_speed)])
        intersections[dst_idx].incoming_lanes.extend(road_lane_ids)
        intersections[src_idx].outgoing_lanes.extend(road_lane_ids)
        for lane_id in road_lane_ids:
            lane_to_intersection[lane_id] = dst_idx

    if not edges:
        raise ValueError(f'No road edges found in {roadnet_path}')

    edge_index = np.asarray(edges, dtype=np.int64).T
    edge_attr = np.asarray(edge_features, dtype=np.float32)
    edge_denom = np.maximum(edge_attr.max(axis=0, keepdims=True), 1.0)
    edge_attr = edge_attr / edge_denom
    max_phase_count = max((item.phase_count for item in intersections), default=1)
    return RoadGraph(
        intersections=intersections,
        lanes=lanes,
        roads=roads,
        edge_index=edge_index,
        edge_features=edge_attr,
        lane_to_intersection=lane_to_intersection,
        max_phase_count=max_phase_count,
        metadata={
            'roadnet_path': str(roadnet_path),
            'num_intersections': len(intersections),
            'num_lanes': len(lanes),
            'num_roads': len(roads),
            'edge_feature_names': ['lane_count_norm', 'lane_length_norm', 'avg_speed_norm'],
        },
    )



def parse_flow_routes(flow_path: str | Path) -> List[dict]:
    payload = json.loads(Path(flow_path).read_text(encoding='utf-8'))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    for key in ('flow', 'flows', 'vehicle', 'vehicles', 'carFlow'):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    raise ValueError(f'Unsupported flow format: {flow_path}')
