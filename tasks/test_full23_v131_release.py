#!/usr/bin/env python3
"""Executable contracts for the Full23 v1.3.1 composition wrapper."""
from pathlib import Path

import numpy as np

from full23_tensor_set_readout_core import ENCODER_NAMES
from full23_v131_release import FrozenFull23V131
from unified_additive_release import REQUIRED_SCORE_NAMES, validate_axis


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "deploy" / "unified_additive_full23_v131"


def main() -> None:
    axis = validate_axis(
        np.asarray([0, 2, 4], dtype=np.int64),
        np.asarray(["q0", "q1"]),
        np.asarray(["q0", "q0", "q1", "q1"]),
        np.asarray(["c0", "c1", "c2", "c3"]),
        np.asarray(["q0:c0", "q0:c1", "q1:c2", "q1:c3"]),
    )
    evidence = {}
    for index, name in enumerate(REQUIRED_SCORE_NAMES):
        evidence[name] = np.asarray(
            [0.8, 0.2, 0.7, 0.3], dtype=np.float32
        ) + 0.01 * index

    rng = np.random.default_rng(20261102)
    banks = {
        name: rng.normal(size=(12, 1024)).astype(np.float32)
        for name in ENCODER_NAMES
    }
    query_positions = np.asarray(
        [[0, 0, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]],
        dtype=np.int64,
    )
    reference_positions = np.empty((4, len(ENCODER_NAMES), 3), dtype=np.int64)
    for row in range(4):
        reference_positions[row] = np.asarray(
            [[1 + row, 5 + row, 8 + row]] * len(ENCODER_NAMES), dtype=np.int64
        )
    reference_mask = np.ones_like(reference_positions, dtype=bool)

    ranker = FrozenFull23V131(
        PACKAGE / "assets" / "full23_v1_parent.pt",
        PACKAGE / "assets" / "full23_v131_query_centered_raw_penalty_readout.pt",
        device="cpu",
        batch_queries=1,
    )
    parent, _ = ranker.parent.score(axis, evidence)
    scores, report = ranker.score(
        axis,
        evidence,
        axis.candidate_pair_ids,
        query_positions,
        reference_positions,
        reference_mask,
        banks,
        {name: f"{index + 1:064x}" for index, name in enumerate(ENCODER_NAMES)},
    )
    assert np.all(np.isfinite(scores))
    assert report["residual_scored_queries"] == 1
    assert report["residual_scored_candidates"] == 2
    assert report["parent_fallback_nonempty_queries"] == 1
    assert np.array_equal(scores[2:], parent[2:])
    assert not np.array_equal(scores[:2], parent[:2])
    print("PASS: Full23 v1.3.1 executable release contracts")


if __name__ == "__main__":
    main()
