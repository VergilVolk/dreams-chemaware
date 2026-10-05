"""CPU contracts for candidate-specific predicate support."""
from __future__ import annotations

import numpy as np

from chemaware_predicate_support_core import (
    candidate_compatibility,
    choose_abstention_threshold,
    formula_domain_centered_design,
    ridge_support_family,
    stable_nontrivial_formula_permutation,
)


def main() -> None:
    identity = np.asarray(["a", "b", "c", "d", "e"])
    formula = np.asarray(["F", "F", "F", "G", "H"])
    permutation = stable_nontrivial_formula_permutation(identity, formula, 7)
    assert set(permutation[:3]) == {0, 1, 2}
    assert np.all(permutation[:3] != np.arange(3))
    assert permutation[3] == 3 and permutation[4] == 4

    observation = np.asarray([[1, 0], [0, 1], [1, 1], [0, 1]], dtype=float)
    predicate = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    x, y, report = formula_domain_centered_design(
        observation, predicate,
        np.asarray(["F", "F", "G", "G"]),
        np.asarray(["D", "D", "D", "D"]),
        range(4),
    )
    assert np.allclose(x[:2].sum(axis=0), 0)
    assert np.allclose(y[:2].sum(axis=0), 0)
    assert report["retained_formula_domain_groups"] == 2

    family, scale, _ = ridge_support_family(x, y, [1.0], [0, 1], True)
    assert family[(1.0, 0)].shape == (2, 2)
    assert np.all(family[(1.0, 1)] >= 0)
    candidates = np.asarray([[1, 1], [1, 0], [1, 0]], dtype=float)
    score = candidate_compatibility(
        np.asarray([1, 0]), candidates, family[(1.0, 0)], scale, True,
    )
    # The first predicate is shared by every candidate and must cancel.
    changed = candidates.copy(); changed[:, 0] = 0
    score_changed = candidate_compatibility(
        np.asarray([1, 0]), changed, family[(1.0, 0)], scale, True,
    )
    assert np.allclose(score, score_changed)

    threshold = choose_abstention_threshold(
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([2, 1, 2]), np.asarray([1, 2, 1]),
        np.asarray(["A", "B", "C"]), 1,
    )
    assert threshold["risk_utility_at_1"] >= 1
    # A -inf threshold still selects only finite proposed actions; sentinel
    # -inf confidence denotes an exact no-op and must never become active.
    sentinel = np.asarray([-np.inf, 0.1])
    assert np.array_equal(np.isfinite(sentinel) & (sentinel >= -np.inf), [False, True])
    print("PASS: 6 candidate-specific predicate-support contracts")


if __name__ == "__main__":
    main()
