"""CPU contracts for ChemAware V2 direct-triplet conversion."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from build_chemaware_v2_direct_triplet_pools import (  # noqa: E402
    build_pool, correction_rows, match_controls,
)


def fixture():
    # Four queries, each with truth first and one false candidate.  Queries 0
    # and 1 are ChemAware corrections; 2 and 3 are eligible matched controls.
    manifest = {
        "query_row": np.asarray([0, 3, 6, 9]),
        "query_ptr": np.asarray([0, 2, 4, 6, 8]),
        "molecule_ptr": np.asarray([0, 2, 3, 5, 6, 8, 9, 11, 12]),
        "pair_candidate_row": np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    }
    policy = {
        "query": np.arange(4),
        "formula": np.asarray(["A", "B", "A", "B"]),
        "baseline_rank": np.asarray([2, 2, 2, 2]),
        "correct_rank": np.asarray([1, 1, 2, 2]),
        "baseline_candidate": np.asarray([1, 1, 1, 1]),
        "correct_selected_candidate_slot": np.asarray([0, 0, -1, -1]),
        "valid_candidate": np.asarray([[1], [1], [1], [1]], dtype=bool),
        "proposed_candidate": np.asarray([[0], [0], [0], [0]], dtype=np.int16),
    }
    return manifest, policy


def main() -> None:
    manifest, policy = fixture()
    corrected = correction_rows(policy)
    assert np.array_equal(corrected, [0, 1])
    controls = match_controls(policy, manifest, corrected, seed=7)
    assert set(controls.tolist()) == {2, 3}
    pool, audit = build_pool(policy, manifest, corrected)
    assert np.array_equal(pool["anchor_idx"], [0, 3])
    assert np.array_equal(pool["positive_idx"], [1, 4])
    assert np.array_equal(pool["negative_idx"], [2, 5])
    assert audit["eligible_anchors"] == 2
    assert audit["skipped_self_only_positive"] == 0
    print("PASS: ChemAware V2 direct triplet-pool contracts")


if __name__ == "__main__":
    main()
