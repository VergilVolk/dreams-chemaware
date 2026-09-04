from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tasks" / "audit_netid_dreams_edge_signal.py"
SPEC = importlib.util.spec_from_file_location("audit_netid_dreams_edge_signal", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_feature_centroid_is_normalized_and_metadata_is_aggregated() -> None:
    vectors = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    ids = np.array([10, 10, 20])
    result = MODULE.aggregate_feature_embeddings(
        vectors,
        ids,
        np.array([100.0, 102.0, 200.0]),
        np.array([1.0, 3.0, 4.0]),
        np.array([5, 7, 9]),
    )
    unique, centroids, mz, rt, counts = result
    assert unique.tolist() == [10, 20]
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0)
    assert mz.tolist() == [101.0, 200.0]
    assert rt.tolist() == [2.0, 4.0]
    assert counts.tolist() == [7, 9]


def test_matched_decoys_never_include_positive_edges() -> None:
    ids = np.array([10, 20, 30, 40, 50])
    positives = np.array([[10, 20], [30, 40]])
    decoys = MODULE.build_matched_decoys(
        ids,
        np.array([100, 101, 200, 201, 300], dtype=float),
        np.array([1, 1.1, 2, 2.1, 3], dtype=float),
        np.array([5, 6, 7, 8, 9]),
        np.array([1, 1, 1, 1, 0]),
        positives,
        controls_per_edge=2,
    )
    forbidden = {tuple(sorted(pair)) for pair in positives.tolist()}
    observed = {tuple(sorted(pair)) for pair in decoys.reshape(-1, 2).tolist()}
    assert forbidden.isdisjoint(observed)


def test_exact_topk_matches_full_lexsort_including_ties() -> None:
    rng = np.random.default_rng(11)
    candidates = np.stack([np.arange(100), np.arange(100)[::-1]], axis=1)
    distance = rng.integers(0, 8, size=100).astype(float)
    expected = np.lexsort((candidates[:, 1], candidates[:, 0], distance))[:7]
    observed = MODULE.exact_topk_indices(distance, candidates, 7)
    assert observed.tolist() == expected.tolist()


def test_cluster_bootstrap_reports_cluster_count() -> None:
    result = MODULE.cluster_bootstrap(
        np.array([0.1, 0.2, -0.1]), np.array([1, 1, 2]), repeats=100, seed=7
    )
    assert result["clusters"] == 2
    assert result["resamples"] == 100
