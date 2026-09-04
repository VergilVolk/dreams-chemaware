from __future__ import annotations

import pandas as pd

from tasks.develop_bioaware_metdna3_negative_loso_ranker import (
    aggregate_known,
    spectral_top_gap,
)


def test_known_rotation_aggregation_is_candidate_specific(tmp_path) -> None:
    frame = pd.DataFrame({
        "query_id": ["q", "q", "q", "q"],
        "candidate_id": ["a", "a", "b", "b"],
        "mass_candidate": [1, 1, 1, 0],
        "path_available": [1, 0, 1, 0],
        "minimum_depth": [2, None, 1, None],
        "shortest_seed_count": [1, 0, 3, 0],
        "candidate_degree": [9, 9, 1, 1],
    })
    path = tmp_path / "paths.csv.gz"
    frame.to_csv(path, index=False, compression="gzip")
    result = aggregate_known(path).set_index("candidate_id")
    assert result.loc["a", "known_path_fraction"] == 0.5
    assert result.loc["a", "known_inverse_depth_mean"] == 0.25
    assert result.loc["b", "known_mass_candidate_fraction"] == 0.5
    assert result.loc["b", "known_inverse_depth_mean"] == 0.5


def test_gate_margin_is_top1_minus_top2_and_truth_independent() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["wrong", "truth", "other"],
        "spectral_score": [0.90, 0.88, 0.10],
    })
    assert abs(spectral_top_gap(frame) - 0.02) < 1e-12
