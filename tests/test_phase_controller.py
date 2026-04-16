from __future__ import annotations

import json
import numpy as np

from src.data.cityflow_parser import parse_cityflow_roadnet
from src.env.phase_controller import PhaseController


def test_phase_controller_respects_min_green(tmp_path):
    roadnet = {
        'intersections': [
            {'id': 'i0', 'virtual': False, 'roadLinks': [], 'trafficLight': {'lightphases': [{'availableRoadLinks': [0]}, {'availableRoadLinks': [1]}]}},
        ],
        'roads': [
            {'id': 'r0', 'startIntersection': 'i0', 'endIntersection': 'i0', 'lanes': [{}], 'points': [{'x': 0, 'y': 0}, {'x': 1, 'y': 0}]},
        ],
    }
    path = tmp_path / 'roadnet.json'
    path.write_text(json.dumps(roadnet), encoding='utf-8')
    graph = parse_cityflow_roadnet(path)
    controller = PhaseController(graph, min_green=10, yellow_time=3, all_red_time=1)
    mask = controller.build_action_mask()
    assert np.all(mask[:, 1:] == 0)
    controller.step_time(10)
    mask = controller.build_action_mask()
    assert mask[0, 1] == 1.0
    switches = controller.apply_actions(np.array([2]), current_time=10)
    assert switches[0] == 1.0
