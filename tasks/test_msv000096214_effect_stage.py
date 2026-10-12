from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_msv000096214_effect_stage import (
    ALTERNATIVES,
    DOUBLES,
    TT7,
    WT,
    exact_t2,
    prepare_log2,
)


def synthetic_table(seed: int = 7) -> tuple[pd.DataFrame, dict[str, str]]:
    rng = np.random.default_rng(seed)
    features = 120
    effect = np.r_[np.full(60, 2.0), np.full(60, -2.0)]
    patterns = {WT: np.zeros(features)}
    patterns.update({group: effect for group in TT7})
    patterns.update({group: rng.normal(0, 0.15, features) for group in ALTERNATIVES})
    patterns.update({group: patterns["tt4_11_SALK_020583"] for group in DOUBLES})
    values = {}
    genotypes = {}
    for group, pattern in patterns.items():
        n = 4 if group == "fls1_3_EMS" else 5
        for replicate in range(n):
            sample = f"{group}__{replicate}"
            log2_value = 10.0 + pattern + rng.normal(0, 0.08, features)
            values[sample] = np.exp2(log2_value)
            genotypes[sample] = group
    return pd.DataFrame(values), genotypes


def test_prepare_log2_does_not_double_log_or_drop_negative_centered_values() -> None:
    matrix = pd.DataFrame({"a": [0.5, 1.0, 2.0], "b": [0.25, 1.0, 4.0]})
    observed = prepare_log2(matrix, median_center=True)
    assert np.isfinite(observed.to_numpy()).all()
    assert np.allclose(observed.median(axis=0).to_numpy(), 0.0)


def test_exact_cluster_assignment_recovers_only_true_tt7_set() -> None:
    matrix, genotypes = synthetic_table()
    result = exact_t2(prepare_log2(matrix, median_center=True), genotypes)
    assert result["pass"] is True
    assert result["exact_cluster_test"]["observed_rank"] == 1
    assert result["exact_cluster_test"]["right_tail_p"] == 0.05
    assert result["exact_cluster_test"]["observed_statistic"] > 0.95
