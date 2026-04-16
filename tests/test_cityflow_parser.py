from __future__ import annotations

import json

from src.data.cityflow_parser import parse_cityflow_roadnet



def test_parse_cityflow_roadnet(tmp_path):
    roadnet = {
        'intersections': [
            {'id': 'i0', 'virtual': False, 'roadLinks': [], 'trafficLight': {'lightphases': [{'availableRoadLinks': [0], 'time': 30}]}},
            {'id': 'i1', 'virtual': False, 'roadLinks': [], 'trafficLight': {'lightphases': [{'availableRoadLinks': [0], 'time': 30}, {'availableRoadLinks': [1], 'time': 30}]}},
        ],
        'roads': [
            {'id': 'r0', 'startIntersection': 'i0', 'endIntersection': 'i1', 'lanes': [{'maxSpeed': 10.0}], 'points': [{'x': 0, 'y': 0}, {'x': 1, 'y': 0}]},
        ],
    }
    path = tmp_path / 'roadnet.json'
    path.write_text(json.dumps(roadnet), encoding='utf-8')
    graph = parse_cityflow_roadnet(path)
    assert graph.num_nodes == 2
    assert graph.edge_index.shape == (2, 1)
    assert graph.edge_features.shape == (1, 3)
    assert graph.max_phase_count == 2
    assert len(graph.lanes) == 1
