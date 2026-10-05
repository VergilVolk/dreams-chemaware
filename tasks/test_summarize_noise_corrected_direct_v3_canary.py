"""Regression tests for the bounded direct-v3 canary summary."""
from __future__ import annotations

import numpy as np
import pandas as pd

from summarize_noise_corrected_direct_v3_canary import (
    _complete_metric_delta,
    _paired_arm_delta,
    _top1_outcomes,
)


def _table(rank: list[int], offset: float = 0.0) -> pd.DataFrame:
    rank_array = np.asarray(rank, dtype=np.int64)
    return pd.DataFrame({
        "candidate_rank": rank_array,
        "candidate_reciprocal_rank": 1.0 / rank_array,
        "candidate_macro_query_auc": np.asarray([0.4, 0.8]) + offset,
        "candidate_macro_query_auprc": 1.0 / rank_array,
        "candidate_positive_vs_best_negative_margin": np.asarray([-0.1, 0.1]) + offset,
        "candidate_signed_top1_top2_gap": np.asarray([-0.1, 0.1]) + offset,
    })


def test_paired_arm_delta_uses_right_minus_left_direction() -> None:
    result = _paired_arm_delta(
        _table([2, 1]), _table([1, 1], 0.1), np.asarray(["A", "B"]),
        repeats=100, seed=3,
    )
    assert result["recall@1_delta_pp"] == 50.0
    assert result["recall@2_delta_pp"] == 0.0
    assert result["mrr_delta_pp"] == 25.0
    assert np.isclose(result["mean_margin_delta"], 0.1)
    assert result["formula_cluster_recall1_delta"]["delta_pp"] == 50.0


def _metrics(offset: float = 0.0, mean_rank: float = 2.0) -> dict[str, object]:
    retrieval = {
        "queries": 2, "mrr": 0.6 + offset, "mean_rank": mean_rank,
        "median_rank": mean_rank, "macro_query_auroc": 0.7 + offset,
        "macro_query_auprc": 0.6 + offset,
        "mean_positive_vs_best_negative_margin": 0.1 + offset,
        "mean_top1_top2_gap": 0.2 + offset,
        "mean_signed_top1_top2_gap": 0.1 + offset,
        **{f"recall@{cutoff}": 0.5 + offset for cutoff in (1, 2, 3, 5, 10, 20)},
    }
    return {
        "retrieval": retrieval,
        "near_subset": dict(retrieval),
        "micro_candidate": {"molecules": 20, "auroc": 0.7 + offset, "auprc": 0.6 + offset},
        "massspecgym_10ppm_pooled_pairwise": {
            "spectrum_pairs": 40, "positive_pairs": 10, "negative_pairs": 30,
            "auroc": 0.7 + offset, "auprc": 0.6 + offset,
        },
        "massspecgym_mh_10ppm_pooled_pairwise": {
            "spectrum_pairs": 20, "positive_pairs": 5, "negative_pairs": 15,
            "auroc": 0.7 + offset, "auprc": 0.6 + offset,
        },
    }


def test_complete_metric_delta_covers_micro_near_and_10ppm_with_rank_direction() -> None:
    result = _complete_metric_delta(
        _metrics(), _metrics(offset=0.01, mean_rank=1.5),
    )
    assert np.isclose(result["retrieval.recall@20_delta_pp"], 1.0)
    assert np.isclose(result["near_subset.mrr_delta_pp"], 1.0)
    assert result["retrieval.mean_rank_reduction"] == 0.5
    assert np.isclose(result["micro_candidate.auroc_delta_pp"], 1.0)
    assert np.isclose(
        result["massspecgym_10ppm_pooled_pairwise.auprc_delta_pp"], 1.0,
    )
    assert np.isclose(
        result["massspecgym_mh_10ppm_pooled_pairwise.auroc_delta_pp"], 1.0,
    )


def test_top1_outcomes_report_near_corrected_introduced_and_risk_net() -> None:
    left = pd.DataFrame({"candidate_rank": [2, 1, 2, 1]})
    right = pd.DataFrame({"candidate_rank": [1, 2, 1, 1]})
    result = _top1_outcomes(
        left, right, np.asarray([True, True, False, False]),
    )
    assert result == {
        "corrected": 2, "introduced": 1, "risk_net_lambda2": 0,
        "near": {"corrected": 1, "introduced": 1, "risk_net_lambda2": -1},
    }


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(
        f"[test_summarize_noise_corrected_direct_v3_canary] PASS tests={len(tests)}",
        flush=True,
    )
