#!/usr/bin/env python
"""Contracts for recovering the B development ranking candidate."""
from __future__ import annotations

import numpy as np

from finalize_wse_unified_ranking_candidate import adjudicate


def test_primary_ranking_can_pass_while_lambda2_utility_is_reported_failed() -> None:
    queries = 1000
    ptr = np.arange(0, 2 * queries + 1, 2, dtype=np.int64)
    labels = np.tile(np.asarray([1, 0], dtype=np.int8), queries)
    baseline = np.tile(np.asarray([1.0, 0.0], dtype=np.float32), queries)
    # The baseline is wrong on the final 200 queries.
    for query in range(800, 1000):
        baseline[ptr[query]:ptr[query + 1]] = (0.0, 1.0)
    candidate = baseline.copy()
    # B corrects all 200 baseline errors and introduces 101: positive accuracy,
    # but 200 - 2*101 = -2 under the separately reported lambda-2 utility.
    for query in range(101):
        candidate[ptr[query]:ptr[query + 1]] = (0.0, 1.0)
    for query in range(800, 1000):
        candidate[ptr[query]:ptr[query + 1]] = (1.0, 0.0)

    panel = {
        "base": baseline,
        "query_ptr": ptr,
        "labels": labels,
        "formulas": np.asarray([f"F{query}" for query in range(queries)]),
        "near": np.ones(queries, dtype=bool),
    }
    bundle = {
        "module_names": np.asarray(["official_dreams"]),
        "scores": baseline[None, :],
        "availability": np.ones((1, len(baseline)), dtype=np.float32),
    }
    scores = {
        "wse": baseline,
        "old_four_module": baseline,
        "A": baseline,
        "B": candidate,
    }
    result, selected = adjudicate(
        {"identity": panel, "formula": panel},
        {"identity": bundle, "formula": bundle},
        {"identity": scores, "formula": scores},
        seed=17, resamples=200,
    )
    assert selected, result
    assert all(result["primary_ranking_pass"].values())
    assert not any(result["legacy_lambda2_utility_pass"].values())


if __name__ == "__main__":
    test_primary_ranking_can_pass_while_lambda2_utility_is_reported_failed()
    print("PASS: WSE unified ranking candidate finalization contracts")
