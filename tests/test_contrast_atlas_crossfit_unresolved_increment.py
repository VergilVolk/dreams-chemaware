from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from run_contrast_atlas_crossfit_unresolved_increment import (  # noqa: E402
    DEFAULT_SPLIT_SALT,
    double_center,
    endpoint_by_dyad,
    equal_norm_additive,
    fold_known_scores,
    fold_swaps,
    inchikey_fold,
    pairwise_ledger,
    sign_flip_delta,
)


def test_inchikey_split_is_deterministic_and_binary() -> None:
    keys = [
        "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
        "CCCCCCCCCCCCCC-DDDDDDDDDD-E",
        "EEEEEEEEEEEEEE-FFFFFFFFFF-G",
    ]
    first = [inchikey_fold(key, DEFAULT_SPLIT_SALT) for key in keys]
    second = [inchikey_fold(key, DEFAULT_SPLIT_SALT) for key in keys]
    assert first == second
    assert set(first).issubset({0, 1})


def test_fold_swaps_never_reuses_truth_as_train() -> None:
    frame = pd.DataFrame(
        {
            "known_fold_0_support": [3],
            "known_fold_1_support": [4],
            "known_fold_0_score": [0.2],
            "known_fold_1_score": [0.8],
        }
    )
    result = fold_swaps(frame).sort_values("truth_fold")
    assert result["train_fold"].tolist() == [1, 0]
    assert result["heldout_known_truth"].tolist() == [0.2, 0.8]
    assert result["train_known_score"].tolist() == [0.8, 0.2]


def test_known_support_counts_only_finite_response_pairs() -> None:
    keys = [
        "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
        "CCCCCCCCCCCCCC-DDDDDDDDDD-E",
    ]
    selected = pd.DataFrame(
        {
            "study_id": ["L", "L", "R", "R"],
            "entity_id": ["L1", "L2", "R1", "R2"],
            "library_inchikey": [keys[0], keys[1], keys[0], keys[1]],
        }
    )
    effects = pd.DataFrame(
        {
            "study_id": ["L", "L", "R", "R"],
            "entity_id": ["L1", "L2", "R1", "R2"],
            "contrast_id": ["c", "c", "c", "c"],
            "response_weight": [1.0, np.nan, 1.0, 1.0],
        }
    )
    result = fold_known_scores(effects, selected, DEFAULT_SPLIT_SALT)
    finite_fold = inchikey_fold(keys[0], DEFAULT_SPLIT_SALT)
    missing_fold = inchikey_fold(keys[1], DEFAULT_SPLIT_SALT)
    if finite_fold == missing_fold:
        assert result.loc[0, f"known_fold_{finite_fold}_support"] == 1
    else:
        assert result.loc[0, f"known_fold_{finite_fold}_support"] == 1
        assert result.loc[0, f"known_fold_{missing_fold}_support"] == 0


def test_double_center_removes_additive_axis_main_effects() -> None:
    rows = []
    for left_index, left in enumerate(("q1", "q2")):
        for right_index, right in enumerate(("c1", "c2", "c3")):
            rows.append(
                {
                    "truth_fold": 0,
                    "left_study_id": "L",
                    "right_study_id": "R",
                    "left_contrast_id": left,
                    "right_contrast_id": right,
                    "score": 10.0 * left_index + right_index,
                }
            )
    frame = pd.DataFrame(rows)
    centered = double_center(frame, "score")
    assert np.allclose(centered, 0.0)


def test_fixed_combination_is_truth_blind_and_equal_norm() -> None:
    frame = pd.DataFrame(
        {
            "truth_fold": [0, 0],
            "left_study_id": ["L", "L"],
            "right_study_id": ["R", "R"],
            "train_known_interaction": [3.0, 4.0],
            "unresolved_interaction": [0.0, 5.0],
            "heldout_known_truth_interaction": [1000.0, -1000.0],
        }
    )
    observed = equal_norm_additive(frame)
    changed = frame.copy()
    changed["heldout_known_truth_interaction"] *= -7
    assert np.allclose(observed, equal_norm_additive(changed))
    known_rms = np.sqrt(np.mean(np.asarray([3.0, 4.0]) ** 2))
    unresolved_rms = np.sqrt(np.mean(np.asarray([0.0, 5.0]) ** 2))
    expected = (
        np.asarray([3.0, 4.0]) / known_rms
        + np.asarray([0.0, 5.0]) / unresolved_rms
    ) / np.sqrt(2.0)
    assert np.allclose(observed, expected)


def test_pairwise_ledger_and_dyad_delta() -> None:
    frame = pd.DataFrame(
        {
            "truth_fold": [0, 0, 0],
            "left_study_id": ["L"] * 3,
            "left_contrast_id": ["q"] * 3,
            "right_study_id": ["R"] * 3,
            "right_contrast_id": ["a", "b", "c"],
            "heldout_known_truth_interaction": [3.0, 2.0, 1.0],
            "train_known_interaction": [1.0, 3.0, 2.0],
            "unresolved_interaction": [3.0, 2.0, 1.0],
            "combined_fixed_interaction": [3.0, 2.0, 1.0],
        }
    )
    pairs = pairwise_ledger(frame)
    assert len(pairs) == 3
    assert int(pairs["train_known_concordant"].sum()) == 1
    assert int(pairs["unresolved_only_concordant"].sum()) == 3
    dyads = endpoint_by_dyad(pairs)
    assert len(dyads) == 1
    assert dyads.loc[0, "train_known_concordance"] == 1 / 3
    assert dyads.loc[0, "train_known_plus_unresolved_fixed_concordance"] == 1.0
    assert np.isclose(dyads.loc[0, "fixed_combination_minus_train_known"], 2 / 3)


def test_small_dyad_sign_flip_is_exact() -> None:
    dyads = pd.DataFrame(
        {"fixed_combination_minus_train_known": [1.0, 2.0, -0.5]}
    )
    result = sign_flip_delta(dyads, replicates=3, seed=1)
    assert result["method"] == "exact_all_sign_configurations"
    assert result["null_configurations_or_replicates"] == 8
    assert result["one_sided_p"] == 0.25
