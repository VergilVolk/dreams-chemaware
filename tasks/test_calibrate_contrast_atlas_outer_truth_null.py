from __future__ import annotations

import pandas as pd

from calibrate_contrast_atlas_outer_truth_null import outer_truth_null


def strong_retrieval(n_queries: int = 10, n_targets: int = 8):
    registry_rows, truth_rows, candidate_rows = [], [], []
    for query in range(n_queries):
        query_id = f"q{query}"
        registry_rows.append({
            "query_id": query_id, "query_study_id": f"Q{query}", "query_contrast_id": "c",
            "cluster_id": query_id, "source_holdout": f"s{query % 2}", "instrument_holdout": "i",
        })
        for target in range(n_targets):
            truth_rows.append({
                "query_id": query_id, "target_study_id": f"T{target}", "target_contrast_id": "c",
                "expected_association": "similar_response" if target == 0 else "irrelevant",
                "truth_basis": "synthetic_smoke", "truth_data_role": "synthetic",
                "truth_frozen_before_scoring": True,
                "selection_blind_to_atlas_score": True,
            })
            candidate_rows.append({
                "world": "all", "query_study_id": f"Q{query}", "query_contrast_id": "c",
                "target_study_id": f"T{target}", "target_contrast_id": "c",
                "calibration_status": "calibrated_within_target_pattern_permutation",
                "calibrated_association": "similar_response",
                "q_similar": 0.001 if target == 0 else 0.05,
                "q_opposite": 0.9, "observed_score": 0.99 if target == 0 else 0.5,
                "target_source_family": "matched_source", "target_instrument": "matched_instrument",
            })
    return pd.DataFrame(candidate_rows), pd.DataFrame(registry_rows), pd.DataFrame(truth_rows)


def test_matched_outer_null_detects_strong_retrieval() -> None:
    candidates, registry, truth = strong_retrieval()
    report, null = outer_truth_null(
        candidates, registry, truth,
        ["target_source_family", "target_instrument"], draws=199, seed=9,
    )
    assert report["status"] == "CONTRAST_ATLAS_MATCHED_OUTER_TRUTH_NULL_COMPLETE"
    assert report["observed_endpoint"] == 1.0
    assert report["right_tail_p"] < 0.05
    assert report["pass_outer_truth_null"]
    assert len(null) == 199


def test_outer_null_abstains_without_movable_matched_strata() -> None:
    candidates, registry, truth = strong_retrieval(n_queries=2, n_targets=1)
    report, null = outer_truth_null(candidates, registry, truth, ["target_source_family"], draws=99)
    assert report["status"] == "ABSTAIN_INSUFFICIENT_MATCHED_OUTER_NULL"
    assert not report["pass_outer_truth_null"]
    assert null.empty
