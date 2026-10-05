from __future__ import annotations

import numpy as np

from chemaware_crossview_consensus_core import (
    aggregate,
    crossview_dominance_policy,
    crossview_policy,
)


def main() -> None:
    assert aggregate(np.asarray([1.0, 3.0]), "mean") == 2.0
    assert aggregate(np.asarray([1.0, 9.0, 3.0]), "median") == 3.0
    baseline = np.asarray([2, 2, 1])
    candidate = np.asarray([["t", "w"], ["t", "w"], ["t", "w"]])
    valid = np.ones((3, 2), dtype=bool)
    proposal = np.asarray([[1, 3], [1, 3], [1, 2]])
    utility = np.asarray([[4.0, 0.0], [3.0, 1.0], [5.0, -1.0]])
    identity = np.asarray(["x", "x", "x"])
    rank, selected, score, support = crossview_policy(
        baseline, candidate, valid, proposal, utility, identity,
        threshold=2.0, aggregation="median", min_context=2,
    )
    assert np.array_equal(rank, np.asarray([1, 1, 1]))
    assert np.array_equal(selected, np.asarray([0, 0, 0]))
    assert np.all(score >= 3.0)
    assert np.array_equal(support, np.asarray([2, 2, 2]))
    arms = np.stack((utility, utility - np.asarray([[2.0, -2.0]] * 3)))
    rank, selected, primary, dominance, support = crossview_dominance_policy(
        baseline, candidate, valid, proposal, arms, identity,
        primary_arm=0, absolute_threshold=2.0, dominance_threshold=1.0,
        aggregation="mean", min_context=2,
    )
    assert np.array_equal(rank, np.asarray([1, 1, 1]))
    assert np.array_equal(selected, np.asarray([0, 0, 0]))
    assert np.all(primary >= 3.5)
    assert np.all(dominance >= 2.0)
    assert np.array_equal(support, np.asarray([2, 2, 2]))
    print("PASS: ChemAware cross-view consensus contracts")


if __name__ == "__main__":
    main()
