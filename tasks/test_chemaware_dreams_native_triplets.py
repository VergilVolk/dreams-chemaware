"""Pure-CPU contracts for directional ChemAware-to-DreaMS triplets."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from build_chemaware_dreams_native_triplets import (  # noqa: E402
    CORRECTION, PROTECTION, build_pool, directional_events,
)


def main() -> None:
    manifest = {
        "query_row": np.asarray([0, 3, 6]),
        "query_ptr": np.asarray([0, 2, 4, 6]),
        "molecule_ptr": np.asarray([0, 2, 3, 5, 6, 8, 9]),
        "pair_candidate_row": np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    }
    policy = {
        "query": np.arange(3), "formula": np.asarray(["A", "B", "C"]),
        "baseline_rank": np.asarray([2, 1, 2]),
        "correct_rank": np.asarray([1, 2, 2]),
        "baseline_candidate": np.asarray([1, 0, 1]),
        "correct_selected_candidate_slot": np.asarray([0, 0, 0]),
        "valid_candidate": np.ones((3, 1), dtype=bool),
        "proposed_candidate": np.asarray([[0], [1], [1]], dtype=np.int16),
    }
    events = directional_events(policy)
    assert np.array_equal(events["event_type"], [CORRECTION, PROTECTION])
    assert int(events["excluded_wrong_to_wrong"]) == 1
    pool, audit = build_pool(policy, manifest)
    assert np.array_equal(pool["anchor_idx"], [0, 3])
    assert np.array_equal(pool["positive_idx"], [1, 4])
    assert np.array_equal(pool["negative_idx"], [2, 5])
    assert audit["correction_triplets"] == 1
    assert audit["protection_triplets"] == 1
    print("PASS: native DreaMS directional triplet contracts")


if __name__ == "__main__":
    main()
