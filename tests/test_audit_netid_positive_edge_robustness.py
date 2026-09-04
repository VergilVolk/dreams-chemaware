from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tasks" / "audit_netid_positive_edge_robustness.py"
SPEC = importlib.util.spec_from_file_location("audit_netid_positive_edge_robustness", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_matched_randomization_detects_large_fixed_effect() -> None:
    scores = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (200, 1))
    result = MODULE.matched_randomization(scores, repeats=999, seed=3)
    assert result["observed_delta"] == 1.0
    assert result["one_sided_empirical_p"] <= 0.01


def test_equal_component_bootstrap_does_not_weight_large_cluster() -> None:
    delta = np.array([1.0] * 100 + [-1.0])
    clusters = np.array([1] * 100 + [2])
    result = MODULE.cluster_bootstrap(delta, clusters, repeats=200, seed=4, equal_component_weight=True)
    assert np.isclose(result["mean"], 0.0)
