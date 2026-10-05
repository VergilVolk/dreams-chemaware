from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_development_disagreement_core import (  # noqa: E402
    action_outcome,
    arbitrate_two_policies,
    disagreement_partition,
    formula_cluster_ci,
    intervention_labels,
    pair_audit,
)


def main() -> None:
    baseline = np.asarray([2, 1, 2, 1, 2])
    left_rank = np.asarray([1, 2, 2, 1, 1])
    right_rank = np.asarray([2, 1, 1, 2, 1])
    left_slot = np.asarray([0, 1, -1, -1, 2])
    right_slot = np.asarray([-1, -1, 0, 1, 2])
    outcome = action_outcome(baseline, left_rank)
    assert outcome["corrected_at_1"] == 2
    assert outcome["introduced_at_1"] == 1
    assert intervention_labels(baseline, left_rank).tolist() == [1, -1, 0, 0, 1]
    partitions = disagreement_partition(left_slot, right_slot)
    assert {name: int(np.sum(mask)) for name, mask in partitions.items()} == {
        "both_abstain": 0,
        "left_only": 2,
        "right_only": 2,
        "both_same_action": 1,
        "both_different_action": 0,
    }
    audit = pair_audit(
        "left", "right", baseline, left_rank, right_rank, left_slot, right_slot,
    )
    assert audit["partitions"]["left_only"]["queries"] == 2
    union_rank, union_slot, source = arbitrate_two_policies(
        baseline, left_rank, left_slot, right_rank, right_slot,
    )
    assert union_rank.tolist() == [1, 2, 1, 2, 1]
    assert union_slot.tolist() == [0, 1, 0, 1, 2]
    assert source.tolist() == [1, 1, 2, 2, 1]
    ci = formula_cluster_ci(
        np.asarray(["A", "A", "B", "B", "C"]),
        intervention_labels(baseline, left_rank), draws=200, seed=9,
    )
    assert len(ci) == 2 and ci[0] <= ci[1]
    print("PASS: ChemAware development disagreement core contracts")


if __name__ == "__main__":
    main()
