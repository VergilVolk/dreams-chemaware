from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("GLM_p0_spectral_coordinate_mapping.py")
SPEC = importlib.util.spec_from_file_location("p0_coordinate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def synthetic_panel() -> dict[str, np.ndarray]:
    # Two queries, each with one positive and two negative molecule coordinates.
    return {
        "query_row": np.array([0, 1]),
        "query_ik14": np.array(["A", "B"]),
        "query_formula": np.array(["F1", "F2"]),
        "near_query": np.array([True, False]),
        "independent_positive": np.array([True, True]),
        "query_ptr": np.array([0, 3, 6]),
        "molecule_ptr": np.array([0, 2, 3, 4, 5, 6, 7]),
        "molecule_label": np.array([1, 0, 0, 1, 0, 0]),
        "molecule_formula": np.array(["F1", "F1", "X", "F2", "Y", "Z"]),
        "candidate_row": np.array([2, 3, 4, 5, 6, 7, 8]),
    }


def test_rank_ties_count_against_positive_and_formula_error_is_visible() -> None:
    panel = synthetic_panel()
    # q0 positive max=0.8; same-formula negative ties at 0.8 -> rank 2/error.
    # q1 positive=0.9 and is uniquely best -> rank 1.
    scores = np.array([0.7, 0.8, 0.8, 0.1, 0.9, 0.4, 0.2])
    out = MODULE.evaluate_method(panel, scores)
    assert out["rank"].tolist() == [2, 1]
    assert out["correct"].tolist() == [False, True]
    assert out["same_formula_error"].tolist() == [True, False]
    assert out["cross_formula_error"].tolist() == [False, False]


def test_instrument_family_and_strict_cross_instrument_masks() -> None:
    panel = synthetic_panel()
    instruments = np.array(
        [
            "orbitrap_ft", "qtof", "qtof", "qtof", "orbitrap_ft",
            "qtof", "qtof", "orbitrap_ft", "qtof",
        ]
    )
    available, strict = MODULE.cross_instrument_masks(panel, instruments)
    assert available.tolist() == [True, False]
    assert strict.tolist() == [True, False]
    assert MODULE.instrument_family("LC-ESI-Q-Exactive Plus") == "orbitrap_ft"
    assert MODULE.instrument_family("LC-ESI-Maxis II HD Q-TOF Bruker") == "qtof"


def test_selective_risk_uses_high_gap_first() -> None:
    correct = np.array([True, False, True, False])
    confidence = np.array([0.9, 0.1, 0.8, 0.2])
    result = MODULE.selective_metrics(correct, confidence)
    assert result["risk_at_20pct"] == 0.0
    assert result["risk_at_40pct"] == 0.0
    assert result["risk_at_100pct"] == 0.5
