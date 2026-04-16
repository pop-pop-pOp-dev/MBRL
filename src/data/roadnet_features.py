from __future__ import annotations

from typing import Dict

import numpy as np

from src.data.cityflow_parser import RoadGraph


def build_static_node_features(graph: RoadGraph) -> np.ndarray:
    features = []
    in_degree = np.zeros(graph.num_nodes, dtype=np.float32)
    out_degree = np.zeros(graph.num_nodes, dtype=np.float32)
    for src, dst in graph.edge_index.T:
        out_degree[int(src)] += 1.0
        in_degree[int(dst)] += 1.0
    for item in graph.intersections:
        features.append(
            [
                in_degree[item.index],
                out_degree[item.index],
                float(len(item.incoming_lanes)),
                float(len(item.outgoing_lanes)),
                float(item.phase_count),
                1.0 if item.is_virtual else 0.0,
            ]
        )
    arr = np.asarray(features, dtype=np.float32)
    denom = np.maximum(arr.max(axis=0, keepdims=True), 1.0)
    return arr / denom


def build_phase_lookup(graph: RoadGraph) -> Dict[int, int]:
    return {item.index: item.phase_count for item in graph.intersections}
