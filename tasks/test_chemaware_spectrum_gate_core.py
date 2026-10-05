"""CPU contracts for the learned spectrum-only ChemAware gate."""
from __future__ import annotations

import numpy as np

from chemaware_spectrum_gate_core import (
    fit_cross_split_consensus_gate,
    fit_logistic_gate,
    gate_amplitude,
    gate_probability,
)


def main() -> None:
    rng = np.random.default_rng(73)
    feature = rng.normal(size=(400, 9))
    label = (feature[:, 0] - 0.5 * feature[:, 1] + 0.2 * rng.normal(size=400) > 0).astype(np.int8)
    model, report = fit_logistic_gate(feature, label, seed=19)
    probability = gate_probability(feature, model)
    assert probability.shape == (400,) and np.all((probability > 0) & (probability < 1))
    assert np.corrcoef(probability, label)[0, 1] > 0.7
    amplitude = gate_amplitude(probability, 0.25, 2.0)
    assert np.all(amplitude >= 0.5) and np.all(amplitude <= 1.0)
    query = rng.normal(size=(30, 7)).astype(np.float32)
    chemical = rng.normal(size=(30, 5)).astype(np.float32)
    embedding = np.concatenate((query, amplitude[:30, None] * chemical), axis=1)
    gram = embedding @ embedding.T
    assert np.linalg.eigvalsh(gram).min() > -1e-4
    assert report["rows"] == 400 and report["features"] == 9
    group = np.repeat(np.asarray([f"F{i}" for i in range(100)]), 4)
    consensus_model, consensus_report = fit_cross_split_consensus_gate(
        feature, label, group, seed=29,
    )
    consensus_probability = gate_probability(feature, consensus_model)
    assert np.corrcoef(consensus_probability, label)[0, 1] > 0.5
    assert consensus_report["formula_disjoint_halves"]
    assert 1 <= consensus_report["reproduced_features"] <= 9
    try:
        gate_amplitude(probability, -0.1, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative floor was accepted")
    print("PASS: 14 ChemAware spectrum-gate contracts")


if __name__ == "__main__":
    main()
