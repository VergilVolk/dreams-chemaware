"""Regression test: evaluation stores may contain only requested query rows."""
from types import SimpleNamespace

import numpy as np

from chemaware_shared_v2_core import paired_evaluation, ranks_and_margins_for_queries


def main() -> None:
    graph = SimpleNamespace(
        n_queries=2,
        query_row=np.asarray([10, 20], dtype=np.int64),
        query_ptr=np.asarray([0, 2, 4], dtype=np.int64),
        molecule_ptr=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        pair_candidate_row=np.asarray([11, 12, 21, 22], dtype=np.int64),
        query_has_near=np.asarray([True, False]),
    )
    # Rows 10-12 from unevaluated query 0 are intentionally absent.
    store = SimpleNamespace(position={20: 0, 21: 1, 22: 2})
    encoded = np.asarray([
        [1.0, 0.0],
        [0.9, 0.0],
        [0.1, 0.9],
    ], dtype=np.float32)
    rank, margin = ranks_and_margins_for_queries(
        encoded, store, graph, np.asarray([1], dtype=np.int64),
    )
    assert rank.tolist() == [1]
    assert np.allclose(margin, [0.8])
    evaluation = paired_evaluation(
        encoded, encoded.copy(), store, graph, np.asarray([1], dtype=np.int64),
    )
    assert evaluation["summary"]["near_n"] == 0
    assert evaluation["summary"]["delta_near_recall1"] is None
    print("PASS: subset-only paired evaluation does not require whole-graph rows")


if __name__ == "__main__":
    main()
