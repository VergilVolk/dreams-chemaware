from __future__ import annotations

import numpy as np

from tasks.metdna3_similarity import (
    metdna3_forward_dot,
    metdna3_reverse_dot,
    spectrum_from_dreams_tensor,
)


def tensor(precursor: float, peaks: list[tuple[float, float]]) -> np.ndarray:
    result = np.zeros((1 + max(4, len(peaks)), 2), dtype=np.float32)
    result[0] = (precursor, 1.1)
    result[1 : 1 + len(peaks)] = peaks
    return result


def test_identical_spectrum_has_unit_score() -> None:
    value = tensor(200.0, [(50.0, 1.0), (100.0, 0.5)])
    assert np.isclose(metdna3_forward_dot(value, value), 1.0)


def test_unmatched_peak_penalizes_forward_dot() -> None:
    left = tensor(200.0, [(50.0, 1.0), (100.0, 1.0)])
    right = tensor(200.0, [(50.0, 1.0)])
    assert np.isclose(metdna3_forward_dot(left, right), 1 / np.sqrt(2))


def test_metdna3_precursor_ceiling_removes_high_fragment() -> None:
    left = tensor(200.0, [(50.0, 1.0), (180.0, 10.0)])
    right = tensor(100.0, [(50.0, 1.0)])
    assert np.isclose(metdna3_forward_dot(left, right), 1.0)


def test_metdna3_reverse_ignores_extra_peaks_in_larger_precursor() -> None:
    larger = tensor(200.0, [(50.0, 1.0), (100.0, 1.0), (180.0, 10.0)])
    smaller = tensor(120.0, [(50.0, 1.0), (100.0, 1.0)])
    assert np.isclose(metdna3_reverse_dot(larger, smaller), 1.0)


def test_metdna3_reverse_penalizes_missing_reference_peak() -> None:
    larger = tensor(200.0, [(50.0, 1.0)])
    smaller = tensor(120.0, [(50.0, 1.0), (100.0, 1.0)])
    assert np.isclose(metdna3_reverse_dot(larger, smaller), 1 / np.sqrt(2))


def test_tensor_parser_excludes_precursor_and_padding() -> None:
    precursor, mz, intensity = spectrum_from_dreams_tensor(tensor(200.0, [(90.0, 0.5), (50.0, 1.0)]))
    assert precursor == 200.0
    assert mz.tolist() == [50.0, 90.0]
    assert intensity.tolist() == [1.0, 0.5]
