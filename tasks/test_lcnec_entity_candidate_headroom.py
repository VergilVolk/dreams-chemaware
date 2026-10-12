"""Small deterministic contracts for the phenotype-blind candidate audit."""

import numpy as np

from audit_lcnec_entity_candidate_headroom import sqrt_cosine


def main() -> None:
    spectrum = (np.array([50.0, 75.0, 100.0]), np.array([4.0, 9.0, 16.0]))
    score, matched = sqrt_cosine(spectrum, spectrum, 0.02)
    assert abs(score - 1.0) < 1e-12
    assert matched == 3

    shifted = (np.array([50.01, 75.01, 120.0]), np.array([4.0, 9.0, 16.0]))
    score, matched = sqrt_cosine(spectrum, shifted, 0.02)
    assert 0.0 < score < 1.0
    assert matched == 2

    no_overlap = (np.array([10.0, 20.0]), np.array([1.0, 1.0]))
    score, matched = sqrt_cosine(spectrum, no_overlap, 0.02)
    assert score == 0.0
    assert matched == 0
    print("PASS: LCNEC candidate-headroom contracts")


if __name__ == "__main__":
    main()
