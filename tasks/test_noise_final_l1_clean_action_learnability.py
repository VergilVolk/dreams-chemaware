"""Fast tests for L1 clean-input action learnability."""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from audit_noise_final_l1_clean_action_learnability import (
    PRIMARY_POLICY, arguments, cell_only_predict, formula_equal_weights, random_projection,
    select_policy, spectrum_descriptors,
)


def main() -> None:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        parsed = arguments()
    finally:
        sys.argv = original_argv
    assert parsed.n_highest_peaks == 100
    projection_a = random_projection(8, 3, 7)
    projection_b = random_projection(8, 3, 7)
    assert projection_a.shape == (8, 3) and np.array_equal(projection_a, projection_b)
    assert np.allclose(np.linalg.norm(projection_a, axis=0), 1.0)
    descriptors = spectrum_descriptors(
        np.asarray([10.0, 20.0, 30.0, 0.0]),
        np.asarray([1.0, 0.5, 0.25, 0.0]), 50.0,
    )
    assert descriptors.shape == (36,) and np.all(np.isfinite(descriptors))
    weights = formula_equal_weights(np.asarray(["A", "A", "B"]))
    assert np.isclose(weights[:2].sum(), weights[2])
    train = pd.DataFrame({
        "selector": ["x", "x", "y"], "attenuation": [0.5, 0.5, 1.0], "step": [1, 1, 2],
        "paired_advantage": [0.02, 0.00, -0.01],
        "advantage_label": ["positive", "neutral", "harmful"],
        "query_formula": ["A", "B", "C"],
    })
    test = pd.DataFrame({"selector": ["x", "y"], "attenuation": [0.5, 1.0], "step": [1, 2]})
    gain, positive, harmful = cell_only_predict(train, test)
    assert np.allclose(gain, [0.01, -0.01])
    assert np.allclose(positive, [0.5, 0.0]) and np.allclose(harmful, [0.0, 1.0])
    policy = pd.DataFrame({
        "query_index": [0, 0, 1], "query_ik14": ["i0", "i0", "i1"],
        "query_formula": ["f0", "f0", "f1"], "has_near": [True, True, False],
        "baseline_rank": [2, 2, 1], "selector": ["x", "y", "x"],
        "attenuation": [0.5, 1.0, 0.5], "step": [1, 2, 1],
        "target_rank": [1, 2, 2], "target_margin": [0.1, 0.0, -0.1],
        "paired_advantage": [0.03, 0.01, -0.02],
        "advantage_label": ["positive", "positive", "harmful"],
        "transition": ["corrected", "persistent_wrong", "introduced"],
        "target_changes_top": [True, False, True],
        "baseline_top_ik14": ["w0", "w0", "i1"],
        "target_top_ik14": ["i0", "w0", "w1"],
        "clean_pred_gain": [0.03, 0.015, -0.02],
        "clean_p_positive": [0.9, 0.8, 0.1],
        "clean_p_harmful": [0.01, 0.02, 0.9],
    })
    summary, per_query = select_policy(policy, "clean", PRIMARY_POLICY, 100, 13)
    assert len(per_query) == 2 and summary["selected_queries"] == 1
    assert summary["corrected"] == 1 and summary["introduced"] == 0
    print("[test_noise_final_l1_clean_action_learnability] PASS", flush=True)


if __name__ == "__main__":
    main()
