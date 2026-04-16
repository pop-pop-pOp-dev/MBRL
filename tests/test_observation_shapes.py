from __future__ import annotations

import json
import numpy as np

from src.data.cityflow_parser import parse_cityflow_roadnet
from src.env.observation import ObservationBuilder



def test_observation_builder_stack_shape(tmp_path):
    roadnet = {
        'intersections': [
            {'id': 'i0', 'virtual': False, 'roadLinks': [], 'trafficLight': {'lightphases': [{'availableRoadLinks': [0]}]}},
            {'id': 'i1', 'virtual': False, 'roadLinks': [], 'trafficLight': {'lightphases': [{'availableRoadLinks': [0]}]}},
        ],
        'roads': [
            {'id': 'r0', 'startIntersection': 'i0', 'endIntersection': 'i1', 'lanes': [{}, {}], 'points': [{'x': 0, 'y': 0}, {'x': 1, 'y': 0}]},
        ],
    }
    path = tmp_path / 'roadnet.json'
    path.write_text(json.dumps(roadnet), encoding='utf-8')
    graph = parse_cityflow_roadnet(path)
    builder = ObservationBuilder(graph, stack_k=3, normalize=True)
    obs = builder.build({'r0_0': 1, 'r0_1': 2}, {'r0_0': 3, 'r0_1': 4}, {'r0_0': 5, 'r0_1': 6}, np.array([0, 0]), np.array([0, 0]))
    assert obs.shape[0] == graph.num_nodes
    assert obs.shape[1] == builder.feature_dim
