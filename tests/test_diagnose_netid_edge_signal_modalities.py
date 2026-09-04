from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "tasks" / "diagnose_netid_edge_signal_modalities.py"
SPEC = importlib.util.spec_from_file_location("diagnose_netid_edge_signal_modalities", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_peak_cosine_identical_spectrum_is_one() -> None:
    peaks = np.array([[50.0, 1.0], [75.0, 4.0], [90.0, 9.0]])
    score = MODULE.peak_cosine(peaks, 100.0, peaks, 100.0, 0.01, "direct")
    assert np.isclose(score, 1.0)


def test_modified_and_neutral_loss_capture_precursor_shift() -> None:
    left = np.array([[50.0, 1.0], [75.0, 4.0], [90.0, 9.0]])
    right = np.array([[60.0, 1.0], [85.0, 4.0], [100.0, 9.0]])
    assert np.isclose(MODULE.peak_cosine(left, 110.0, right, 120.0, 0.01, "modified"), 1.0)
    assert np.isclose(MODULE.peak_cosine(left, 110.0, right, 120.0, 0.01, "neutral_loss"), 1.0)
    assert np.isclose(MODULE.peak_cosine(left, 110.0, right, 120.0, 0.01, "direct"), 0.0)


def test_one_to_one_matching_prevents_peak_reuse() -> None:
    left = np.array([[50.000, 1.0], [50.005, 1.0]])
    right = np.array([[50.002, 1.0]])
    score = MODULE.peak_cosine(left, 100.0, right, 100.0, 0.01, "direct")
    assert np.isclose(score, 1.0 / np.sqrt(2.0))
