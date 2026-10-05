"""Numerical contracts for direct ChemAware rule supervision."""

from __future__ import annotations

import numpy as np

from chemaware_direct_chemical_prior_core import (
    candidate_center,
    centered_student_residual,
    compile_structure_fragment_prior,
    direct_prior_target,
)


def main() -> None:
    ptr = np.asarray([0, 3, 5], dtype=np.int64)
    evidence = np.asarray([[0.9, 0.1], [0.0, 0.0]], dtype=np.float32)
    presence = np.asarray(
        [
            [1.0, 0.0],  # query 0 positive candidate supports the strong rule
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    prior = compile_structure_fragment_prior(
        evidence, presence, ptr, rule_confidence=np.asarray([1.0, 0.5]),
    )
    assert prior.active_query.tolist() == [True, False]
    assert prior.raw_score[0] > prior.raw_score[1] > prior.raw_score[2]
    assert np.allclose(prior.centered_residual[:3].sum(), 0.0, atol=1e-7)
    assert np.array_equal(prior.centered_residual[3:], np.zeros(2, dtype=np.float32))

    # Candidate-wise additive offsets are ranking irrelevant and disappear.
    shifted = candidate_center(
        np.asarray([5.0, 4.0, 3.0, -7.0, -8.0]), ptr,
    )
    unshifted = candidate_center(
        np.asarray([2.0, 1.0, 0.0, 1.0, 0.0]), ptr,
    )
    assert np.allclose(shifted, unshifted)

    official = np.asarray([0.4, 0.3, 0.2, -0.1, -0.2], dtype=np.float32)
    assert np.array_equal(
        centered_student_residual(official, official, ptr),
        np.zeros_like(official),
    )
    assert np.array_equal(
        direct_prior_target(prior, 0.0), np.zeros_like(prior.centered_residual),
    )

    # Swapping candidate structures changes the chemical target while preserving
    # its candidate multiset and zero-sum capacity.
    swapped_presence = presence.copy()
    swapped_presence[[0, 1]] = swapped_presence[[1, 0]]
    swapped = compile_structure_fragment_prior(
        evidence, swapped_presence, ptr, rule_confidence=np.asarray([1.0, 0.5]),
    )
    assert prior.centered_residual[0] > prior.centered_residual[1]
    assert swapped.centered_residual[0] < swapped.centered_residual[1]
    assert np.allclose(
        np.sort(prior.centered_residual[:3]),
        np.sort(swapped.centered_residual[:3]),
    )

    print("PASS: direct structure-fragment candidate-prior contracts")


if __name__ == "__main__":
    main()
