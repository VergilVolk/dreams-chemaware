"""CPU contracts for parent-margin soft residual fusion."""
from __future__ import annotations

import numpy as np

from audit_chemaware_soft_residual_fusion_policy import soft_fusion_rank


def main() -> None:
    table = {
        "valid": np.asarray([[True, True], [True, True]]),
        "proposed_candidate": np.asarray([[1, 2], [1, 2]]),
        "baseline_candidate": np.asarray([0, 0]),
        "baseline_rank": np.asarray([2, 1]),
        "rank": np.asarray([[1, 2], [2, 2]]),
        "molecule_labels": np.asarray([
            np.asarray([False, True, False]), np.asarray([True, False, False]),
        ], dtype=object),
    }
    score = [np.asarray([0.70, 0.68, 0.30]), np.asarray([0.70, 0.68, 0.30])]
    utility = np.asarray([[0.0, -1.0], [0.0, -1.0]])
    parent, active = soft_fusion_rank(table, score, utility, np.inf, 0.0)
    assert np.array_equal(parent, np.asarray([2, 1])) and not active.any()
    promoted, active = soft_fusion_rank(table, score, utility, -0.5, 0.10)
    assert np.array_equal(promoted, np.asarray([1, 2])) and active.all()
    no_excess, active = soft_fusion_rank(table, score, utility, 0.5, 0.10)
    assert np.array_equal(no_excess, parent) and not active.any()
    print("PASS: 7 ChemAware soft-residual-fusion contracts")


if __name__ == "__main__":
    main()
