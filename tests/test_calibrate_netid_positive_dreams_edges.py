from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tasks" / "calibrate_netid_positive_dreams_edges.py"
SPEC = importlib.util.spec_from_file_location("calibrate_netid_positive_dreams_edges", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_balanced_component_folds_is_deterministic_and_complete() -> None:
    components = np.array([10, 20, 30, 40])
    sizes = np.array([100, 20, 10, 5])
    first = MODULE.balanced_component_folds(components, sizes, folds=3)
    second = MODULE.balanced_component_folds(components, sizes, folds=3)
    assert first == second
    assert set(first) == set(components)
    assert first[10] == 0


def test_fixed_fdr_threshold_controls_training_proxy() -> None:
    positive = np.array([0.9, 0.8, 0.7, 0.6])
    decoy = np.array([0.75, 0.2, 0.1, 0.05, 0.01, 0.0])
    result = MODULE.fixed_fdr_threshold(positive, decoy, controls_per_edge=2, target_fdr=0.1)
    assert result["estimated_fdr"] <= 0.1
    assert result["selected_positive"] >= 2


def test_component_isolated_decoys_exclude_held_features() -> None:
    metadata = {
        "feature_ids": np.array([1, 2, 3, 4, 5]),
        "vectors": np.eye(5, dtype=np.float32),
        "mz": np.array([100, 101, 200, 201, 300], dtype=float),
        "rt": np.array([1, 1.1, 2, 2.1, 3], dtype=float),
        "counts": np.array([5, 6, 7, 8, 9]),
    }
    result = MODULE.build_component_isolated_decoy_similarities(
        metadata,
        np.array([[1, 2]]),
        np.array([[1, 2], [3, 4]]),
        held_features={3, 4},
        degree_by_feature={1: 1, 2: 1, 3: 1, 4: 1},
        controls_per_edge=2,
    )
    assert result.shape == (1, 2)
