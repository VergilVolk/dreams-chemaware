import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_g8r_real_error_atlas", ROOT / "tasks/analyze_g8r_real_error_atlas.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_candidate_bins_are_explicit():
    values = MODULE.candidate_bin(pd.Series([2, 3, 4, 5, 8, 9, 20]))
    assert values.tolist() == ["2", "3-4", "3-4", "5-8", "5-8", "9+", "9+"]


def test_bh_qvalues_are_monotone_in_pvalue_order():
    p = np.asarray([0.04, 0.001, 0.02, 0.8])
    q = MODULE.bh_qvalues(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)
    assert np.all((q >= 0) & (q <= 1))


def test_score_arms_are_not_forced_mutually_exclusive():
    frame = pd.DataFrame({
        "dreams_correct": [False, False, False, False, True],
        "positive_score_reference_z": [-2.0, -2.0, 0.0, 0.0, -2.0],
        "negative_score_reference_z": [2.0, 0.0, 2.0, 0.0, 2.0],
        "hard_negative_top10_match_fraction": [1, 0, 1, 0, 0],
        "hard_negative_intensity_coverage_min": [1, 0, 1, 0, 0],
        "hard_negative_neutral_loss_sqrt_cosine": [1, 0, 1, 0, 0],
        "positive_cross_instrument": [False] * 5,
        "positive_collision_energy_delta": [0.0] * 5,
        "raw_features_favoring_positive": [0] * 5,
    })
    result, _ = MODULE.assign_screening_hypotheses(frame)
    assert result["score_error_family"].tolist() == [
        "positive_deficit_and_negative_excess",
        "positive_deficit_only",
        "negative_excess_only",
        "comparative_boundary_error",
        "official_correct",
    ]


def test_candidate_roles_use_positive_and_official_hard_negative():
    rows = []
    for label, identity, score in ((1, "POS", 0.4), (0, "N1", 0.7), (0, "N2", 0.6)):
        row = {
            "query_index": 0,
            "label": label,
            "candidate_ik14": identity,
            "candidate_formula": "C1",
            "candidate_smiles": "C",
            "candidate_scaffold": "",
            "mces_grade": "near" if not label else "identity",
            "dreams_score": score,
            "p2b_score": score,
            "dreams_winning_pair_row": 10 + len(rows),
        }
        for name in MODULE.RAW_FEATURES:
            row[f"dreams_pair_{name}"] = score
        rows.append(row)
    result = MODULE.get_candidate_roles(pd.DataFrame(rows)).iloc[0]
    assert result["positive_ik14"] == "POS"
    assert result["hard_negative_ik14"] == "N1"
    assert result["raw_features_favoring_positive"] == 0
    assert "positive_dreams_pair_row" not in result.index
