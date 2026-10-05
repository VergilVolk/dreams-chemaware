#!/usr/bin/env python
"""Dependency-free unit checks for BioAware B2 reaction transforms."""
from __future__ import annotations

import numpy as np

from audit_bioaware_b2_reaction_transform_action import (
    adduct_charge,
    formula_mass,
    greedy_shifted_cosine,
    transform_pair_features,
)


def tensor(precursor: float, peaks: list[tuple[float, float]]) -> np.ndarray:
    output = np.zeros((101, 2), dtype=np.float32)
    output[0] = (precursor, 1.1)
    for index, peak in enumerate(peaks, start=1):
        output[index] = peak
    return output


def main() -> None:
    assert abs(formula_mass("C6H12O6") - 180.06338810418) < 1e-8
    assert formula_mass("C6H12O6?") is None
    assert formula_mass("Xe2") is None
    assert adduct_charge("[M+H]+") == 1
    assert adduct_charge("[M-2H]2-") == 2
    assert adduct_charge("unknown") is None

    left_mz = np.array([50.0, 70.0])
    right_mz = np.array([40.0, 60.0])
    intensity = np.array([2 ** -0.5, 2 ** -0.5])
    direct, _ = greedy_shifted_cosine(left_mz, intensity, right_mz, intensity, (0.0,))
    shifted, fraction = greedy_shifted_cosine(left_mz, intensity, right_mz, intensity, (10.0,))
    assert direct == 0.0
    assert abs(shifted - 1.0) < 1e-12
    assert fraction == 1.0

    # CH2 differs by 14.015650064 Da.  Precursor and fragments carry exactly
    # that transformation, so the theoretical channel must be perfect while
    # the wrong-sign control is not.
    delta = formula_mass("CH2")
    seed = tensor(100.0, [(40.0, 1.0), (60.0, 1.0)])
    query = tensor(100.0 + delta, [(40.0 + delta, 1.0), (60.0 + delta, 1.0)])
    features = transform_pair_features(query, seed, "C2H4", "CH2", 1)
    assert features is not None
    assert features["mass_residual"] < 1e-5
    assert features["theoretical_shift"] > 0.999
    assert features["theoretical_hybrid"] > 0.999
    assert features["wrong_sign_hybrid"] < 1e-8
    assert features["transform_concordance"] > 0.999
    print("[test_bioaware_b2_reaction_transform_action] PASS", flush=True)


if __name__ == "__main__":
    main()
