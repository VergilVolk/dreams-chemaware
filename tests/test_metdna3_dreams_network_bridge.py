from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "metdna3_bridge", ROOT / "tasks/build_metdna3_dreams_ms2_network.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_exact_edges_are_scored_in_author_order_and_reversal_is_symmetric():
    edges = MODULE.validate_edges(pd.DataFrame({
        "edge_index": [0, 1], "from": ["a", "c"], "to": ["b", "a"]
    }))
    names = np.asarray(["a", "b", "c"])
    embeddings = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    score = MODULE.score_edges(edges, names, embeddings)
    reversed_edges = MODULE.validate_edges(pd.DataFrame({
        "edge_index": [0, 1], "from": ["b", "a"], "to": ["a", "c"]
    }))
    reversed_score = MODULE.score_edges(reversed_edges, names, embeddings)
    np.testing.assert_allclose(score, [0.8, 0.0], atol=1e-7)
    np.testing.assert_allclose(score, reversed_score, atol=1e-7)


def test_duplicate_undirected_edge_fails_closed():
    with pytest.raises(RuntimeError, match="duplicate undirected"):
        MODULE.validate_edges(pd.DataFrame({"from": ["a", "b"], "to": ["b", "a"]}))


def test_truth_or_outcome_columns_are_forbidden():
    with pytest.raises(RuntimeError, match="forbidden"):
        MODULE.validate_edges(pd.DataFrame({
            "from": ["a"], "to": ["b"], "truth_label": [1]
        }))


def test_missing_feature_fails_closed():
    edges = MODULE.validate_edges(pd.DataFrame({"from": ["a"], "to": ["b"]}))
    with pytest.raises(RuntimeError, match="misses 1"):
        MODULE.score_edges(edges, np.asarray(["a"]), np.asarray([[1.0, 0.0]]))
