"""Fast contract tests for S3A direct clean-boundary routing."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from audit_noise_s3a_direct_boundary_transfer import restore_a4_query_rows, route_weights


def test_introduced_controls_are_targeted_without_upweighting_all_correct_queries() -> None:
    query = pd.DataFrame({
        "baseline_rank": [2, 2, 1, 1],
        "action_correctable": [True, False, False, False],
        "action_introduced": [False, False, True, False],
    })
    args = SimpleNamespace(
        route="action_correctable", correctable_weight=6.0,
        other_error_weight=1.0, correct_control_weight=1.0,
        introduced_control_weight=4.0,
    )
    assert np.array_equal(route_weights(query, args), np.asarray([6, 1, 4, 1], np.float32))


def test_none_preserves_historical_control_weight() -> None:
    query = pd.DataFrame({
        "baseline_rank": [1, 1],
        "action_correctable": [False, False],
        "action_introduced": [True, False],
    })
    args = SimpleNamespace(
        route="action_correctable", correctable_weight=6.0,
        other_error_weight=1.0, correct_control_weight=0.75,
        introduced_control_weight=None,
    )
    assert np.array_equal(route_weights(query, args), np.asarray([0.75, 0.75], np.float32))


def test_recurrence_changes_confidence_not_action_multiplicity() -> None:
    query = pd.DataFrame({
        "baseline_rank": [2, 2, 2],
        "action_correctable": [True, True, True],
        "action_introduced": [False, False, False],
        "positive_action_count": [1, 2, 8],
    })
    args = SimpleNamespace(
        route="action_recurrence", correctable_weight=6.0,
        other_error_weight=1.0, correct_control_weight=1.0,
        introduced_control_weight=None, recurrence_saturation=4.0,
    )
    # Recurrence is a capped per-query confidence. It never duplicates the
    # query or lets eight actions deliver eight times the maximum dose.
    assert np.allclose(route_weights(query, args), np.asarray([2.25, 3.5, 6.0]))


def test_mature_n_route_excludes_outcome_success_from_forbidden_family() -> None:
    query = pd.DataFrame({
        "baseline_rank": [2, 2],
        "action_correctable": [True, True],
        "mature_n_correctable": [True, False],
        "action_introduced": [False, False],
    })
    args = SimpleNamespace(
        route="mature_n", correctable_weight=6.0,
        other_error_weight=1.0, correct_control_weight=1.0,
        introduced_control_weight=None,
    )
    assert np.array_equal(route_weights(query, args), np.asarray([6.0, 1.0], np.float32))


def test_positive_rank_margin_keeps_pressure_after_bare_top1_is_reached() -> None:
    margin = torch.tensor([0.02], requires_grad=True)
    bare = F.softplus(-margin / 0.05)
    guarded = F.softplus((0.05 - margin) / 0.05)
    assert float(guarded) > float(bare)
    guarded.backward()
    assert margin.grad is not None and float(margin.grad) < 0.0


def test_a4_missing_query_row_is_restored_by_exact_query_index() -> None:
    ledger = pd.DataFrame({
        "stage": ["S3A", "A4"], "query_index": [1, 7],
        "query_row": [101.0, np.nan], "query_ik14": ["X", "Y"],
        "query_formula": ["F1", "F2"],
    })
    scan = pd.DataFrame({
        "query_index": [7], "query_row": [707], "query_ik14": ["Y"],
        "query_formula": ["F2"],
    })
    restored = restore_a4_query_rows(ledger, scan)
    assert restored.loc[restored.stage.eq("A4"), "query_row"].item() == 707


def main() -> None:
    test_introduced_controls_are_targeted_without_upweighting_all_correct_queries()
    test_none_preserves_historical_control_weight()
    test_recurrence_changes_confidence_not_action_multiplicity()
    test_mature_n_route_excludes_outcome_success_from_forbidden_family()
    test_positive_rank_margin_keeps_pressure_after_bare_top1_is_reached()
    test_a4_missing_query_row_is_restored_by_exact_query_index()
    print("[test_noise_s3a_direct_boundary_transfer] PASS tests=6")


if __name__ == "__main__":
    main()
