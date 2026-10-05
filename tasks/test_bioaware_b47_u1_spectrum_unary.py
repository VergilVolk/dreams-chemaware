#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_b47_u1_core import (  # noqa: E402
    UnaryRecipe, choose_recipe, same_top_unique_veto, score_recipe,
    strict_ranks, unary_features,
)


def test_features_handle_singletons_and_tied_maxima() -> None:
    score = np.asarray([0.7, 0.8, 0.8, 0.2, 0.6, 0.4], dtype=np.float32)
    ptr = np.asarray([0, 1, 4, 6], dtype=np.int64)
    feature = unary_features(score, ptr)
    assert np.allclose(feature.maximum, [0.7, 0.8, 0.6])
    assert np.allclose(feature.top2_mean, [0.7, 0.8, 0.5])
    assert feature.reference_count.tolist() == [1, 3, 2]


def test_recipe_is_exactly_max_at_zero_dose() -> None:
    score = np.asarray([0.1, 0.9, 0.6], dtype=np.float32)
    ptr = np.asarray([0, 2, 3], dtype=np.int64)
    feature = unary_features(score, ptr)
    recipe = UnaryRecipe("zero", "mean", 0.0, 0.0)
    assert np.array_equal(score_recipe(feature, recipe), feature.maximum)


def test_strict_rank_counts_ties_against_positive() -> None:
    score = np.asarray([0.8, 0.8, 0.7, 0.4, 0.5], dtype=np.float32)
    ptr = np.asarray([0, 3, 5], dtype=np.int64)
    assert strict_ranks(score, ptr).tolist() == [2, 2]


def test_recipe_selection_falls_back_without_safety() -> None:
    safe = {
        "delta_recall1": 0.01,
        "delta_mrr": 0.01,
        "near_delta_recall1": 0.01,
        "corrected_gt_2x_introduced": True,
        "formula_equal_delta_recall1": 0.01,
        "formula_equal_risk_utility": 0.01,
        "introduced": 1,
    }
    unsafe = dict(safe, corrected_gt_2x_introduced=False)
    baseline = dict(unsafe, delta_recall1=0.0)
    assert choose_recipe({"max": baseline, "good": safe, "bad": unsafe}) == "good"
    assert choose_recipe({"max": baseline, "bad": unsafe}) == "max"


def test_same_top_unique_veto_cannot_replace_winner_or_introduce_top1_error() -> None:
    baseline = np.asarray([0.8, 0.8, 0.4, 0.9, 0.7, 0.2], dtype=np.float32)
    calibrated = np.asarray([0.81, 0.79, 0.4, 0.7, 0.91, 0.2], dtype=np.float32)
    ptr = np.asarray([0, 3, 6], dtype=np.int64)
    output, gate = same_top_unique_veto(baseline, calibrated, ptr)
    # Query 0 is vetoed because evaluation ordering must not resolve its tie.
    # Query 1 is vetoed because calibration would switch the unique winner.
    assert gate.tolist() == [False, False]
    assert np.array_equal(output, baseline)
    old_rank, new_rank = strict_ranks(baseline, ptr), strict_ranks(output, ptr)
    assert old_rank.tolist() == [2, 1]
    assert new_rank.tolist() == [2, 1]


def test_same_top_unique_veto_can_only_preserve_a_unique_winner() -> None:
    baseline = np.asarray([0.9, 0.7, 0.4], dtype=np.float32)
    calibrated = np.asarray([0.91, 0.65, 0.5], dtype=np.float32)
    ptr = np.asarray([0, 3], dtype=np.int64)
    output, gate = same_top_unique_veto(baseline, calibrated, ptr)
    assert gate.tolist() == [True]
    assert np.argmax(output) == 0
    assert strict_ranks(baseline, ptr).tolist() == strict_ranks(output, ptr).tolist() == [1]


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print("[test_bioaware_b47_u1_spectrum_unary] PASS", flush=True)
