from __future__ import annotations

import numpy as np

from chemaware_orthogonal_rule_policy_core import (
    base_and_chemical_feature_indices,
    combine_residual_probability,
    combine_residual_score,
    permute_candidate_contrast_within_strata,
    signed_rule_contrast,
    symmetric_center_contrast,
    validate_matched_tables,
)


def toy_table(feature: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.asarray([[True, True], [True, False], [True, True]], dtype=bool)
    return {
        "feature": np.asarray(feature, dtype=np.float32),
        "valid": valid,
        "proposed_candidate": np.asarray([[1, 2], [0, -1], [1, 2]], dtype=np.int16),
        "baseline_candidate": np.asarray([0, 1, 0], dtype=np.int16),
        "baseline_rank": np.asarray([2, 1, 2], dtype=np.int16),
    }


def main() -> None:
    names = ["baseline_margin", "candidate_mass_max", "candidate_rule_max", "action_margin"]
    base, chemical = base_and_chemical_feature_indices(names)
    assert base.tolist() == [0, 1]
    assert chemical.tolist() == [2, 3]

    first = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
    second = first.copy()
    second[..., 2:] -= 2.0
    correct = toy_table(first)
    control = toy_table(second)
    validate_matched_tables(correct, control)
    delta = signed_rule_contrast(correct, control, chemical)
    assert np.all(delta[correct["valid"]] == 2.0)
    assert np.all(delta[~correct["valid"]] == 0.0)

    third = first.copy()
    third[..., 2:] += 1.0
    summary = symmetric_center_contrast(
        [correct, control, toy_table(third)], 0, chemical,
    )
    assert summary.shape == (3, 2, 12)
    # Against values two below and one above, the per-feature mean is +0.5,
    # lower/upper envelopes are -1/+2, and sign fractions are one half.
    assert np.allclose(summary[correct["valid"], 0:2], 0.5)
    assert np.allclose(summary[correct["valid"], 2:4], -1.0)
    assert np.allclose(summary[correct["valid"], 4:6], 2.0)
    assert np.allclose(summary[correct["valid"], 8:12], 0.5)

    permuted, source = permute_candidate_contrast_within_strata(
        delta, correct["valid"], correct["baseline_rank"], 17,
    )
    assert np.all(permuted[~correct["valid"]] == 0.0)
    assert np.all(source[~correct["valid"]] == -1)
    for count in (1, 2):
        for baseline_correct in (False, True):
            query = np.flatnonzero(
                (correct["valid"].sum(axis=1) == count)
                & ((correct["baseline_rank"] == 1) == baseline_correct)
            )
            before = np.sort(delta[query][correct["valid"][query]].reshape(-1))
            after = np.sort(permuted[query][correct["valid"][query]].reshape(-1))
            assert np.array_equal(before, after)

    combined = combine_residual_probability(
        np.asarray([0.2, 0.9]), np.asarray([-0.5, 0.5]), 0.5,
    )
    assert np.allclose(combined, [0.0, 1.0])
    score = combine_residual_score(
        np.asarray([0.2, 0.9]), np.asarray([-0.5, 0.5]), 0.5,
    )
    assert np.allclose(score, [-0.05, 1.15])
    print("PASS: 7 ChemAware orthogonal rule-policy core contracts")


if __name__ == "__main__":
    main()
