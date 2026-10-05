#!/usr/bin/env python
"""Dependency-free unit checks for BioAware B1 ranking semantics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from develop_bioaware_b1_multisource_action import (  # noqa: E402
    add_derived_features,
    paired_transition,
    source_from_unit,
    strict_spectral_metadata,
)


def candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q2"],
            "candidate_id": ["A" * 14, "B" * 14, "C" * 14, "D" * 14],
            "truth_candidate_id": ["A" * 14, "A" * 14, "C" * 14, "C" * 14],
            "truth_formula": ["C1H2"] * 2 + ["C2H4"] * 2,
            "spectral_score": [0.8, 0.7, 0.5, 0.5],
            "reference_spectra": [2, 1, 1, 1],
            "known_mass_candidate_fraction": [1, 1, 1, 1],
            "known_path_fraction": [1, 0, 1, 0],
            "known_inverse_depth_mean": [1, 0, 0.5, 0],
            "known_log_seed_support_mean": [1, 0, 1, 0],
            "known_log_degree": [1, 0, 1, 0],
            "edge0_complete_fraction": [1, 0, 1, 0],
            "edge0_bottleneck_mean": [0.8, 0, 0.7, 0],
            "unit_id": ["BV2cell__hilic"] * 4,
            "source": ["BV2cell"] * 4,
            "polarity": ["positive"] * 4,
        }
    )


def main() -> None:
    assert source_from_unit("Mouse_brain__rplc") == "Mouse_brain"
    metadata = strict_spectral_metadata(candidates())
    q1 = metadata.set_index("query_id").loc["q1"]
    q2 = metadata.set_index("query_id").loc["q2"]
    assert bool(q1.baseline_correct)
    assert not bool(q2.baseline_correct), "ties must count against truth"
    enriched = add_derived_features(candidates())
    assert np.isfinite(enriched.select_dtypes(include=[np.number]).to_numpy()).all()
    left = pd.DataFrame({"query_id": ["q1", "q2"], "gated_correct": [True, True]})
    right = pd.DataFrame({"query_id": ["q1", "q2"], "gated_correct": [False, True]})
    transition = paired_transition(left, right)
    assert transition == {
        "corrected_vs_right": 1,
        "introduced_vs_right": 0,
        "net": 1,
        "risk_net_lambda2": 1,
    }
    print("[test_bioaware_b1_multisource_action] PASS")


if __name__ == "__main__":
    main()
