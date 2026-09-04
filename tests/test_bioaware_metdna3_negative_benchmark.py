from __future__ import annotations

import numpy as np
import pandas as pd

from tasks.build_bioaware_metdna3_negative_dreams_benchmark import (
    candidate_window,
    strict_truth_top1,
)


def test_candidate_window_is_symmetric_and_identity_unique() -> None:
    mz = np.asarray([99.9995, 100.0000, 100.0005, 100.01])
    ik = np.asarray(["A", "A", "B", "C"])
    lower, upper, identities = candidate_window(mz, ik, 100.0, 10.0)
    assert (lower, upper) == (0, 3)
    assert identities.tolist() == ["A", "B"]


def test_ties_count_against_truth() -> None:
    tied = pd.DataFrame({
        "candidate_id": ["TRUTH", "WRONG"],
        "truth_candidate_id": ["TRUTH", "TRUTH"],
        "spectral_score": [0.8, 0.8],
    })
    unique = tied.copy()
    unique.loc[1, "spectral_score"] = 0.79
    assert not strict_truth_top1(tied)
    assert strict_truth_top1(unique)
