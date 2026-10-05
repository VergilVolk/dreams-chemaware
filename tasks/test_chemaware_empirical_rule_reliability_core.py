"""CPU contracts for empirical rule-reliability weighting."""

from __future__ import annotations

import numpy as np

from chemaware_empirical_rule_reliability_core import (
    aggregate_formula_contrasts,
    prevalence_matched_permutation,
    reliability_weights,
)


def test_formula_aggregation_prevents_replicate_mass_from_dominating() -> None:
    values = np.asarray([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    result, formula = aggregate_formula_contrasts(values, np.asarray(["A", "A", "B"]))
    assert np.array_equal(formula, ["A", "B"])
    assert np.allclose(result, [[2.0, 0.0], [0.0, 2.0]])


def test_reliability_is_nonnegative_and_rejects_negative_channels() -> None:
    values = np.asarray([
        [1.0, -1.0], [2.0, -2.0], [1.5, -1.5],
        [1.1, -1.1], [1.8, -1.8], [1.4, -1.4],
    ])
    weights, report = reliability_weights(
        values, np.asarray(list("ABCDEF")), np.asarray([2, 2]), 10, seed=7,
    )
    assert weights[0] > 0 and weights[1] == 0
    assert report["positive_weight_channels"] == 1


def test_permutation_preserves_weight_multiset_and_categories() -> None:
    weights = np.arange(10, dtype=np.float32)
    df = np.asarray([1, 1, 2, 2, 3, 1, 1, 2, 2, 3])
    output, report = prevalence_matched_permutation(weights, df, 5, seed=9, bins=2)
    assert np.array_equal(np.sort(output), np.sort(weights))
    assert set(output[:5]).issubset(set(weights[:5]))
    assert set(output[5:]).issubset(set(weights[5:]))
    assert report["weight_multiset_preserved"]


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} ChemAware empirical rule-reliability contracts")


if __name__ == "__main__":
    main()
