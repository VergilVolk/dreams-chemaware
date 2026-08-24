import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_directional_noise_v2_pool",
    ROOT / "tasks/audit_directional_noise_v2_pool.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def spectrum(peaks):
    body = np.zeros((2, 8), dtype=float)
    for index, (mz, intensity) in enumerate(peaks):
        body[:, index] = (mz, intensity)
    return body


def test_consensus_separates_core_and_low_prevalence_weak_peak():
    spectra = {
        1: spectrum([(100.0, 1.0), (150.0, 0.1)]),
        2: spectrum([(100.01, 0.8), (200.0, 0.4)]),
        3: spectrum([(100.0, 0.7), (210.0, 0.4)]),
    }
    labels = {1: "Orbitrap|ce-20", 2: "Orbitrap|ce-30", 3: "QTOF|ce-20"}
    rows = MODULE.classify_group_peaks(spectra, [1, 2, 3], labels, 0.02, 0.60, 0.40, 0.20)
    first = next(row for row in rows if row["row"] == 1)
    assert first["n_core_peaks"] == 1
    assert first["n_conditional_candidates"] == 1
    assert first["conditional_mz"] == [150.0]


def test_condition_diverse_selection_round_robins_conditions():
    rows = list(range(10))
    labels = {row: "A" if row < 8 else "B" for row in rows}
    selected = MODULE.choose_condition_diverse_rows(rows, labels, maximum=4, seed=3)
    assert len(selected) == 4
    assert any(labels[row] == "B" for row in selected)
