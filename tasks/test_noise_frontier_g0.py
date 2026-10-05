"""Fast invariant tests for the frozen frontier G0 experiment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_frontier_g0_core import (  # noqa: E402
    clustered_bootstrap,
    formula_folds,
    grouped_max,
    landmark_concordance,
    landmark_rank_profiles,
    landmark_response_concordance,
    pair_query_index,
    ranks_from_pair_scores,
    retrieval_summary,
    within_query_zscore,
)


def main() -> None:
    query_ptr = np.asarray([0, 2, 4], dtype=np.int64)
    molecule_ptr = np.asarray([0, 2, 3, 4, 6], dtype=np.int64)
    labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
    pair = np.asarray([0.8, 0.7, 0.75, 0.6, 0.5, 0.55])
    assert np.allclose(grouped_max(pair, molecule_ptr), [0.8, 0.75, 0.6, 0.55])
    assert np.array_equal(pair_query_index(query_ptr, molecule_ptr), [0, 0, 0, 1, 1, 1])
    rank = ranks_from_pair_scores(pair, query_ptr, molecule_ptr, labels)
    assert np.array_equal(rank, [1, 1])

    # Ties count against the positive.
    tied = pair.copy(); tied[2] = 0.8
    assert ranks_from_pair_scores(tied, query_ptr, molecule_ptr, labels)[0] == 2

    query_pair_ptr = molecule_ptr[query_ptr]
    standardized = within_query_zscore(pair, query_pair_ptr)
    assert abs(float(np.mean(standardized[:3]))) < 1e-7
    assert abs(float(np.std(standardized[:3])) - 1.0) < 1e-6

    # Pair-first fusion cannot borrow channel maxima from different references.
    channel_a = np.asarray([0.9, 0.0, 0.6, 0.5, 0.5, 0.5])
    channel_b = np.asarray([0.0, 0.9, 0.6, 0.5, 0.5, 0.5])
    correct_pair_fusion = channel_a + channel_b
    correct = ranks_from_pair_scores(correct_pair_fusion, query_ptr, molecule_ptr, labels)
    impossible_positive = grouped_max(channel_a, molecule_ptr) + grouped_max(channel_b, molecule_ptr)
    assert impossible_positive[0] > grouped_max(correct_pair_fusion, molecule_ptr)[0]
    assert correct[0] == 2

    responses = np.asarray([
        [[0.1], [0.2], [0.3], [0.4]],
        [[0.1], [0.2], [0.3], [0.4]],
        [[0.4], [0.3], [0.2], [0.1]],
    ], dtype=np.float32)
    profiles = landmark_rank_profiles(responses)
    valid = np.ones(4, dtype=bool)
    assert landmark_concordance(profiles[0], profiles[1], valid) == 1.0
    assert landmark_concordance(profiles[0], profiles[2], valid) < 0.5
    assert landmark_response_concordance(responses[0], responses[1], valid) == 1.0
    assert landmark_response_concordance(responses[0], responses[2], valid) < 0.5

    summary = retrieval_summary(np.asarray([1, 2]), np.asarray([2, 1]))
    assert summary["corrected"] == 1 and summary["introduced"] == 1
    assert summary["risk_utility_lambda2"] == -1
    bootstrap = clustered_bootstrap(
        np.asarray([1.0, 1.0, 0.0, 0.0]),
        np.asarray(["A", "A", "B", "B"]),
        500,
        7,
    )
    assert bootstrap["clusters"] == 2 and bootstrap["mean_delta"] == 0.5
    assert np.array_equal(
        formula_folds(np.asarray(["C6H12O6", "C6H12O6"]), 5, 19),
        formula_folds(np.asarray(["C6H12O6", "C6H12O6"]), 5, 19),
    )
    print("[test_noise_frontier_g0] PASS")


if __name__ == "__main__":
    main()
