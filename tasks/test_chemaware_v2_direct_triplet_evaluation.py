"""CPU contracts for direct-triplet candidate retrieval metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_v2_triplet_eval_core import (  # noqa: E402
    evaluate_graph, numerical_rank_replay_audit, paired_summary, summarize,
)


def main() -> None:
    # Query 0 truth wins; query 1 false candidate wins.  Each molecule has one
    # reference, which makes the expected strict ranks transparent.
    manifest = {
        "query_row": np.asarray([0, 3]),
        "query_ptr": np.asarray([0, 2, 4]),
        "molecule_ptr": np.asarray([0, 1, 2, 3, 4]),
        "pair_candidate_row": np.asarray([1, 2, 4, 5]),
    }
    rows = np.arange(6)
    encoded = np.asarray([
        [1.0, 0.0], [0.9, 0.1], [0.0, 1.0],
        [1.0, 0.0], [0.6, 0.8], [0.9, 0.1],
    ], dtype=np.float32)
    encoded /= np.linalg.norm(encoded, axis=1, keepdims=True)
    ranks, positive, negative, auc = evaluate_graph(
        encoded, rows, manifest, np.asarray([0, 1]),
    )
    assert np.array_equal(ranks, [1, 2])
    metrics = summarize(ranks, positive, negative, auc)
    assert metrics["recall1"] == 0.5
    paired = paired_summary(
        np.asarray([2, 1]), ranks, np.asarray(["A", "B"]), draws=100, seed=3,
    )
    assert paired["corrected_at_1"] == 1
    assert paired["introduced_at_1"] == 1
    assert paired["delta_recall1"] == 0.0

    # A frozen rank on an exact positive/negative boundary is isolated rather
    # than turning a cross-BLAS tie into either a gain or a regression.
    tied = encoded.copy()
    tied[2] = tied[1]
    tied_rank, *_ = evaluate_graph(tied, rows, manifest, np.asarray([0, 1]))
    stable, audit = numerical_rank_replay_audit(
        tied, rows, manifest, np.asarray([0, 1]),
        np.asarray([1, tied_rank[1]]), tied_rank,
    )
    assert np.array_equal(stable, [False, True])
    assert len(audit) == 1 and audit[0]["tie_explained"]

    # A real rank drift with a finite score gap must still fail closed.
    try:
        numerical_rank_replay_audit(
            encoded, rows, manifest, np.asarray([0, 1]),
            np.asarray([2, 2]), ranks,
        )
    except RuntimeError as error:
        assert "unexplained=1" in str(error)
    else:
        raise AssertionError("non-tie replay drift was accepted")
    print("PASS: ChemAware V2 direct-triplet evaluation contracts")


if __name__ == "__main__":
    main()
