from __future__ import annotations

import numpy as np
import pandas as pd

from calibrate_contrast_atlas_pattern_null import bh_adjust, calibrate_pair


def pattern_frame(study: str, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "study_id": study,
        "contrast_id": "c",
        "pattern_id": [f"p{i}" for i in range(len(values))],
        "pattern_beta": values,
        "pattern_se": np.full(len(values), 0.2),
    })


def test_pair_null_separates_aligned_and_opposite_patterns() -> None:
    values = np.asarray([-3.1, 0.4, 2.2, -1.5, 4.0, 1.1, -2.4, 3.3, 0.8, -0.7])
    query = pattern_frame("q", values)
    similar = calibrate_pair(query, pattern_frame("s", values), draws=999, seed=11, min_common=8)
    opposite = calibrate_pair(query, pattern_frame("o", -values), draws=999, seed=12, min_common=8)
    assert similar["p_similar"] <= 0.01
    assert opposite["p_opposite"] <= 0.01
    assert similar["observed_score"] > 0.99
    assert opposite["observed_score"] < -0.99


def test_pair_null_abstains_and_bh_is_monotone() -> None:
    short = pattern_frame("q", np.asarray([1.0, -1.0, 2.0]))
    result = calibrate_pair(short, pattern_frame("t", np.asarray([1.0, -1.0, 2.0])), 99, 1, 8)
    assert result["calibration_status"] == "abstain_insufficient_patterns_for_null"
    adjusted = bh_adjust(pd.Series([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted.to_numpy(), [0.03, 0.04, 0.04])
