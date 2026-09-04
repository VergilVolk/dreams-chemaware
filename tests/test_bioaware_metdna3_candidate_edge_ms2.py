import numpy as np
import pandas as pd

from tasks.audit_bioaware_metdna3_candidate_edge_ms2 import (
    best_bottleneck_for_identity_path,
    enumerate_shortest_paths,
)
from tasks.build_bioaware_metdna3_feature_ms2_cache import NodeIndex


def test_node_index_uses_normalized_mass_rt_distance() -> None:
    nodes = pd.DataFrame({
        "feature_node": [1, 2], "polarity": ["positive", "positive"],
        "mz": [100.0000, 100.0008], "rt_sec": [50.0, 40.0],
    })
    index = NodeIndex(nodes, ppm=15, rt_sec=25)
    assert index.nearest("positive", 100.0007, 40.1)[0] == 2
    assert index.nearest("negative", 100.0007, 40.1) is None


def test_shortest_paths_require_observed_intermediates() -> None:
    graph = {"C": {"I", "X"}, "I": {"C", "S"}, "X": {"C", "S"}, "S": {"I", "X"}}
    paths, truncated = enumerate_shortest_paths("C", graph, {"S"}, {"C", "I"}, 3)
    assert paths == [("C", "I", "S")]
    assert not truncated


def test_bottleneck_uses_weakest_edge(monkeypatch) -> None:
    tensors = {1: np.array([[1.0]]), 2: np.array([[2.0]])}
    scores = {(-1, 1): 0.8, (1, 2): 0.4}

    def fake(left, right):
        left_key = -1 if float(left[0, 0]) == 0 else int(left[0, 0])
        right_key = int(right[0, 0])
        return scores[(left_key, right_key)]

    monkeypatch.setattr("tasks.audit_bioaware_metdna3_candidate_edge_ms2.metdna3_reverse_dot", fake)
    score, combinations = best_bottleneck_for_identity_path(
        ("C", "I", "S"), np.array([[0.0]]), {"I": [1], "S": [2]}, tensors, {}
    )
    assert score == 0.4
    assert combinations == 2
