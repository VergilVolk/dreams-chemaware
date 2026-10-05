"""CPU contracts for the monotone reliability-gated chemical kernel."""
from __future__ import annotations

import numpy as np

from chemaware_reliability_gated_kernel_core import (
    centered_reliability,
    empirical_percentile,
    gated_pair_score,
    monotone_gate,
    reliability_gated_embedding,
)


def main() -> None:
    calibration = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    percentile = empirical_percentile(np.asarray([0.0, 2.0, 5.0]), calibration)
    assert np.allclose(percentile, [0.0, 0.5, 1.0])
    gate = monotone_gate(calibration, calibration, floor=0.25, power=2.0)
    assert np.all(np.diff(gate) >= 0.0) and gate.min() >= 0.25 and gate.max() <= 1.0
    center = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bg_a = np.asarray([[0.9, 0.0], [0.0, 0.1]], dtype=np.float32)
    bg_b = np.asarray([[0.8, 0.0], [0.0, 0.2]], dtype=np.float32)
    reliability = centered_reliability(center, (bg_a, bg_b))
    assert reliability[1] > reliability[0]

    rng = np.random.default_rng(7)
    official = rng.normal(size=(9, 5)).astype(np.float32)
    mass = rng.normal(size=(9, 4)).astype(np.float32)
    chemical = rng.normal(size=(9, 3)).astype(np.float32)
    spectrum_gate = monotone_gate(np.arange(9), np.arange(9), floor=0.1, power=1.5)
    phi = reliability_gated_embedding(
        official, mass, chemical, spectrum_gate, mass_beta=0.2, chemical_beta=0.7,
    )
    left = np.asarray([0, 1, 2, 4, 7])
    right = np.asarray([8, 6, 5, 3, 1])
    direct = gated_pair_score(
        np.sum(official[left] * official[right], axis=1),
        np.sum(mass[left] * mass[right], axis=1),
        np.sum(chemical[left] * chemical[right], axis=1),
        spectrum_gate[left], spectrum_gate[right],
        mass_beta=0.2, chemical_beta=0.7,
    )
    assert np.max(np.abs(direct - np.sum(phi[left] * phi[right], axis=1))) < 1e-5
    assert np.linalg.eigvalsh(phi @ phi.T).min() > -1e-4
    print("PASS: 5 ChemAware reliability-gated kernel core contracts")


if __name__ == "__main__":
    main()
