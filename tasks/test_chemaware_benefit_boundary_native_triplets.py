"""CPU contracts for benefit-proven ChemAware native-triplet construction."""
from __future__ import annotations

import numpy as np

from build_chemaware_benefit_boundary_native_triplets import (
    NEGATIVE_DIRECTION_METRICS,
    POSITIVE_DIRECTION_METRICS,
    active_pairs,
    balanced_pair_candidates,
    benefit_consensus,
    minimum_balanced_cap,
)


def test_consensus_uses_worst_semantic_null() -> None:
    names = (*POSITIVE_DIRECTION_METRICS, *NEGATIVE_DIRECTION_METRICS)
    metric = np.zeros((4, 1, 1, len(names)), dtype=np.float32)
    # Correct beats all three nulls on two positive-direction metrics.
    metric[0, 0, 0, :2] = 1.0
    # It beats only two nulls on a third metric; worst-null consensus must stay 2.
    metric[0, 0, 0, 2] = 1.0
    metric[3, 0, 0, 2] = 2.0
    evidence = {
        "metric_names": np.asarray(names),
        "arm_metric": metric,
    }
    observed = benefit_consensus(evidence)
    assert observed.shape == (1, 1)
    assert int(observed[0, 0]) == 2


def test_active_pairs_are_unique_active_and_boundary_first() -> None:
    geometry = {
        "positive_rows": np.asarray([10, 11]),
        "negative_rows": np.asarray([20, 21]),
        "positive_scores": np.asarray([0.70, 0.60]),
        "negative_scores": np.asarray([0.65, 0.45]),
    }
    pairs = active_pairs(geometry, margin=0.1)
    signatures = [(p, n) for p, n, *_ in pairs]
    assert len(signatures) == len(set(signatures))
    assert all(hinge > 0.0 for _, _, hinge, _, _ in pairs)
    assert signatures[0] == (10, 20)
    assert (10, 21) not in signatures


def test_balanced_pair_candidates_preserve_query_breadth() -> None:
    def record(query: int, current_error: bool, count: int):
        return {
            "query": query,
            "current_error": current_error,
            "query_consensus": 2,
            "pairs": [
                (100 + query, 200 + query + depth, 0.5 - depth * 0.01, 0, depth)
                for depth in range(count)
            ],
        }

    records = [record(2, False, 3), record(1, True, 3), record(3, False, 1)]
    observed = list(balanced_pair_candidates(records, 2))
    queries = [int(item[0]["query"]) for item in observed]
    assert queries[:3] == [1, 2, 3]
    assert queries[3:] == [1, 2]
    assert queries.count(1) == 2 and queries.count(2) == 2
    assert queries.count(3) == 1

    cap, capacity = minimum_balanced_cap(records, target_events=5, maximum_cap=4)
    assert cap == 2
    assert capacity == 5
    cap, capacity = minimum_balanced_cap(records, target_events=7, maximum_cap=2)
    assert cap == 0
    assert capacity == 5


def main() -> None:
    test_consensus_uses_worst_semantic_null()
    test_active_pairs_are_unique_active_and_boundary_first()
    test_balanced_pair_candidates_preserve_query_breadth()
    print("PASS: ChemAware benefit-boundary native-triplet contracts", flush=True)


if __name__ == "__main__":
    main()
