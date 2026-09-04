import numpy as np
import pandas as pd

from tasks.evaluate_bioaware_kgmn200std_hidden_seed import (
    adjacency,
    identity_cluster_bootstrap,
    shortest_paths,
)


def test_shortest_paths_stops_at_first_reached_depth() -> None:
    edges = pd.DataFrame({
        "ik14_a": ["A", "B", "A"],
        "ik14_b": ["B", "S", "S"],
        "minimum_step": [0, 0, 1],
    })
    graph0 = adjacency(edges, 0)
    assert shortest_paths("A", graph0, {"S"}, {"A", "B", "S"}) == [("A", "B", "S")]
    graph1 = adjacency(edges, 1)
    assert shortest_paths("A", graph1, {"S"}, {"A", "B", "S"}) == [("A", "S")]


def test_hidden_identity_not_reached_without_seed_membership() -> None:
    graph = {"A": {"H"}, "H": {"A"}}
    assert shortest_paths("A", graph, set(), {"A", "H"}) == []


def test_identity_bootstrap_clusters_repeats() -> None:
    frame = pd.DataFrame({
        "truth_candidate_id": ["A", "A", "B", "B"],
        "delta": [1, 1, 0, 0],
    })
    result = identity_cluster_bootstrap(frame, 100, 1)
    assert np.isclose(result["mean"], 0.5)
    assert result["clusters"] == 2
