from tasks.summarize_bioaware_metdna3_candidate_edge_decision import formula_bootstrap

import numpy as np


def test_formula_bootstrap_preserves_clustered_nonnegative_effect() -> None:
    result = formula_bootstrap(
        np.array([1.0, 1.0, 0.0, 0.0]), np.array(["A", "A", "B", "C"]), 7
    )
    assert result["mean"] == 0.5
    assert result["formulas"] == 3
    assert result["ci_low"] >= 0
