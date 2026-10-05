"""Fail-closed unit contracts for the formal best-action v6 repair."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from train_noise_corrected_routed_direct import (
    BoundaryExample,
    ProtectExample,
    REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_EXACT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_FLOAT_CONFIGURATION,
    REGISTERED_V3_EXACT_CONFIGURATION,
    REGISTERED_V3_FLOAT_CONFIGURATION,
    RoutedAction,
    _audit_corrective_example_materialization,
    _calibration_action_bank,
    _forward_memory_contract,
    _materialize_admitted_action_bank,
    _ndarray_sha256,
    _registered_report_configuration,
    _select_corrective_admission,
    _validate_registered_formal_v3_configuration,
)
from noise_corrected_shuffled_control_v3 import source_family_shuffled_action_bank


def _action(
    action_id: str,
    query: int,
    *,
    kind: str = "corrective",
    clean_rank: int = 2,
    action_rank: int = 1,
    action_margin: float = 0.1,
) -> dict[str, object]:
    return {
        "action_id": action_id,
        "query_index": query,
        "query_row": 100 + query,
        "source": "N_mature",
        "family": "candidate_gradient",
        "recipe_id": f"recipe-{action_id}",
        "supervision_kind": kind,
        "clean_rank": clean_rank,
        "action_rank": action_rank,
        "action_margin": action_margin,
        "control_semantic": "query_matched_no_op",
        "action_positive_row": 200 + query,
        "action_hard_negative_molecule_index": 1,
        "action_hard_negative_row": 300 + query,
        "control_positive_row": 200 + query,
        "control_hard_negative_molecule_index": 1,
        "control_hard_negative_row": 300 + query,
    }


def test_strict_admission_keeps_every_strong_view_without_one_best_compression() -> None:
    actions = pd.DataFrame([
        _action("q0-a", 0, action_margin=0.12),
        _action("q0-b", 0, action_margin=0.08),
        _action("floor", 1, action_margin=1e-7),
        _action("not-top1", 2, action_rank=2),
        _action("already-clean", 3, clean_rank=1),
        _action("harm", 4, kind="harmful", action_rank=3),
    ])
    selected, report = _select_corrective_admission(
        actions, mode="strict_top1", margin_floor=5e-6,
    )
    assert selected.action_id.tolist() == ["q0-a", "q0-b"]
    assert selected.query_index.nunique() == 1
    assert report["strict_top1_rows_before_margin_floor"] == 3
    assert report["selected_rows"] == 2
    assert report["one_best_query_compression_used"] is False


def test_strict_admission_rejects_nonfinite_rank_instead_of_treating_it_as_wrong() -> None:
    actions = pd.DataFrame([_action("bad-rank", 0, clean_rank=np.nan)])
    try:
        _select_corrective_admission(
            actions, mode="strict_top1", margin_floor=5e-6,
        )
    except RuntimeError as error:
        assert "finite positive integers" in str(error)
    else:
        raise AssertionError("strict admission accepted a non-finite clean rank")


def test_registered_calibration_uses_each_actual_training_bank() -> None:
    targeted = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    shuffled = targeted[::-1].copy()
    before = shuffled.copy()
    calibration, report = _calibration_action_bank(
        targeted, shuffled, arm_invariant_targeted=False,
    )
    assert calibration is shuffled
    assert np.array_equal(shuffled, before)
    assert report["arm_invariant"] is False
    assert report["policy"] == "arm_specific_training_action_bank"
    assert _ndarray_sha256(calibration) == _ndarray_sha256(shuffled)
    assert _ndarray_sha256(shuffled) != _ndarray_sha256(targeted)


def test_explicit_shared_calibration_helper_remains_nonformal() -> None:
    targeted = np.zeros((2, 3, 2), dtype=np.float32)
    shuffled = np.ones_like(targeted)
    calibration, report = _calibration_action_bank(
        targeted, shuffled, arm_invariant_targeted=True,
    )
    assert calibration is targeted
    assert report["arm_invariant"] is True


def test_admission_precedes_shuffle_and_excludes_nonstrict_donor() -> None:
    rows = [
        _action("strict-q0", 0),
        _action("strict-q1", 1),
        _action("nonstrict-sentinel", 2, action_rank=2),
        _action("harm", 3, kind="harmful", action_rank=3),
        _action("robust", 4, kind="robust", clean_rank=1),
    ]
    for index, row in enumerate(rows):
        row["recipe_id"] = "shared-recipe"
        row["action_tensor_index"] = index
    full = pd.DataFrame(rows)
    corrective, _ = _select_corrective_admission(
        full, mode="strict_top1", margin_floor=5e-6,
    )
    harmful = full.loc[full.supervision_kind.eq("harmful")].copy()
    robust = full.loc[full.supervision_kind.eq("robust")].copy()
    action_spectra = np.stack([
        np.full((3, 2), value, dtype=np.float32)
        for value in (10, 20, 999, 30, 40)
    ])
    control_spectra = -action_spectra
    (
        admitted, corrected, admitted_harmful, admitted_robust,
        targeted, controls, report,
    ) = _materialize_admitted_action_bank(
        corrective, harmful, robust, action_spectra, control_spectra,
    )
    assert admitted.action_id.tolist() == ["strict-q0", "strict-q1", "harm", "robust"]
    assert admitted.action_tensor_index.tolist() == list(range(4))
    assert corrected.action_id.tolist() == ["strict-q0", "strict-q1"]
    assert admitted_harmful.action_id.tolist() == ["harm"]
    assert admitted_robust.action_id.tolist() == ["robust"]
    assert report["rejected_corrective_rows_can_be_shuffle_donors"] is False
    shuffled, shuffle_report = source_family_shuffled_action_bank(
        admitted, targeted, controls, seed=7,
    )
    assert not np.any(shuffled == np.float32(999))
    assert set(np.unique(shuffled[:2, 0, 0]).tolist()) == {10.0, 20.0}
    assert shuffle_report["cross_query_rows"] == 2
    assert shuffle_report["all_nonfallback_donors_inside_input_panel"] is True
    assert shuffle_report["donor_tensor_index_sha256"]


def test_query_materialization_preserves_every_action_in_one_example_per_query() -> None:
    actions = pd.DataFrame([
        _action("q0-a", 0), _action("q0-b", 0), _action("q1-a", 1),
    ])
    examples = [
        BoundaryExample(
            query_index=0, query_row=10, identity="id0", formula="F0",
            actions=(
                RoutedAction("q0-a", 0, "N::N_mature|candidate_gradient"),
                RoutedAction("q0-b", 1, "N::N_mature|candidate_gradient"),
            ),
            positive_rows=(11,), negative_rows=((12,),),
        ),
        BoundaryExample(
            query_index=1, query_row=20, identity="id1", formula="F1",
            actions=(
                RoutedAction("q1-a", 2, "N::N_mature|candidate_gradient"),
            ),
            positive_rows=(21,), negative_rows=((22,),),
        ),
    ]
    report = _audit_corrective_example_materialization(actions, examples)
    assert report["exact_action_id_set_preserved"] is True
    assert report["one_boundary_example_per_query"] is True
    assert report["selected_action_rows"] == 3


def test_query_complete_forward_memory_bound_is_exact_and_fails_closed() -> None:
    actions = (
        RoutedAction("q0-a", 0, "N::N_mature|candidate_gradient"),
        RoutedAction("q0-b", 1, "N::N_mature|candidate_gradient"),
    )
    boundary = BoundaryExample(
        query_index=0, query_row=10, identity="id0", formula="F0",
        actions=actions, positive_rows=(11,), negative_rows=((12,),),
    )
    protective = ProtectExample(
        query_index=1, query_row=20, identity="id1", formula="F1",
        positive_rows=(21,), negative_rows=((22,),),
    )
    args = SimpleNamespace(
        batch_queries=4,
        protective_batch_queries=8,
        maximum_spectra_per_action_forward=7,
    )
    report = _forward_memory_contract(
        [boundary], [boundary], [boundary], [protective], args,
    )
    assert report["planned_worst_case_spectra_per_forward"] == 7
    assert report["gate_passed"] is True
    args.maximum_spectra_per_action_forward = 6
    try:
        _forward_memory_contract(
            [boundary], [boundary], [boundary], [protective], args,
        )
    except RuntimeError as error:
        assert "memory contract failed" in str(error)
    else:
        raise AssertionError("forward memory contract accepted an unsafe batch")


def test_formal_v6_configuration_is_exact_and_fails_closed() -> None:
    values = {
        **REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION,
        **REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION,
        "development": False,
        "corrective_objective_mode": "v3_direct",
    }
    _validate_registered_formal_v3_configuration(SimpleNamespace(**values))
    drifted = dict(values)
    drifted["arm_invariant_targeted_calibration"] = True
    try:
        _validate_registered_formal_v3_configuration(SimpleNamespace(**drifted))
    except RuntimeError as error:
        assert "arm_invariant_targeted_calibration" in str(error)
    else:
        raise AssertionError("formal v6 accepted non-V3 shared targeted calibration")
    wrong_allocator = dict(values)
    wrong_allocator["transfer_target_allocation"] = "mass_neutral_monotone"
    try:
        _validate_registered_formal_v3_configuration(SimpleNamespace(**wrong_allocator))
    except RuntimeError as error:
        assert "transfer_target_allocation" in str(error)
    else:
        raise AssertionError("formal v6 accepted the unheld allocator")
    wrong_objective = dict(values)
    wrong_objective["corrective_objective_mode"] = "full"
    try:
        _validate_registered_formal_v3_configuration(
            SimpleNamespace(**wrong_objective)
        )
    except RuntimeError as error:
        assert "v3_direct" in str(error)
    else:
        raise AssertionError("formal v6 accepted the old direct objective")


def test_v6_is_only_the_strict_action_admission_spliced_into_registered_v3() -> None:
    exact_diff = {
        key
        for key in (
            set(REGISTERED_V3_EXACT_CONFIGURATION)
            | set(REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION)
        )
        if REGISTERED_V3_EXACT_CONFIGURATION.get(key)
        != REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION.get(key)
    }
    float_diff = {
        key
        for key in (
            set(REGISTERED_V3_FLOAT_CONFIGURATION)
            | set(REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION)
        )
        if REGISTERED_V3_FLOAT_CONFIGURATION.get(key)
        != REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION.get(key)
    }
    assert exact_diff == {"direct_contract", "corrective_admission", "seed"}
    assert float_diff == {"corrective_margin_floor"}
    assert REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION[
        "corrective_admission"
    ] == "strict_top1"
    assert REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION[
        "corrective_margin_floor"
    ] == 5e-6
    assert REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION["seed"] == 20260908


def test_v6_restored_changes_only_the_registered_optimizer_boundary() -> None:
    exact_diff = {
        key
        for key in (
            set(REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION)
            | set(REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION)
        )
        if REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION.get(key)
        != REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION.get(key)
    }
    float_diff = {
        key
        for key in (
            set(REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION)
            | set(REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION)
        )
        if REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION.get(key)
        != REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION.get(key)
    }
    assert exact_diff == {
        "direct_contract",
        "materialize_optimizer_update_restoration",
        "continue_after_signal_gate_failure",
        "seed",
    }
    assert float_diff == set()
    values = {
        **REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION,
        **REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION,
        "development": False,
        "corrective_objective_mode": "v3_direct",
    }
    _validate_registered_formal_v3_configuration(SimpleNamespace(**values))
    assert values["materialize_optimizer_update_restoration"] is True
    assert values["optimizer_restoration_minimum_risk_component_retention"] == 0.90
    assert values["continue_after_signal_gate_failure"] is True
    assert values["seed"] == 20260911


def test_v7_restores_only_corrective_and_uses_the_low_loss_allocator() -> None:
    exact_diff = {
        key
        for key in (
            set(REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION)
            | set(REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_EXACT_CONFIGURATION)
        )
        if REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION.get(key)
        != REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_EXACT_CONFIGURATION.get(key)
    }
    float_diff = {
        key
        for key in (
            set(REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION)
            | set(REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_FLOAT_CONFIGURATION)
        )
        if REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION.get(key)
        != REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_FLOAT_CONFIGURATION.get(key)
    }
    assert exact_diff == {
        "direct_contract",
        "optimizer_restoration_scope",
        "reconcile_restored_adamw_first_moment",
        "transfer_target_allocation",
    }
    assert float_diff == {
        "minimum_optimizer_action_attributable_fraction_p10",
        "minimum_optimizer_restoration_target_reached_fraction",
    }
    values = {
        **REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_EXACT_CONFIGURATION,
        **REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_FLOAT_CONFIGURATION,
        "development": False,
        "corrective_objective_mode": "v3_direct",
    }
    _validate_registered_formal_v3_configuration(SimpleNamespace(**values))
    assert values["optimizer_restoration_scope"] == "corrective_only"
    assert values["reconcile_restored_adamw_first_moment"] is True
    assert values["transfer_target_allocation"] == "mass_neutral_monotone"
    assert values["minimum_optimizer_action_attributable_fraction_p10"] == 0.25
    assert values["minimum_optimizer_restoration_target_reached_fraction"] == 0.90
    assert values["seed"] == 20260911
    reported = _registered_report_configuration(SimpleNamespace(**values))
    assert set(reported) == (
        set(REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_EXACT_CONFIGURATION)
        | set(REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_FLOAT_CONFIGURATION)
    )
    assert reported["optimizer_restoration_scope"] == "corrective_only"
    assert reported["reconcile_restored_adamw_first_moment"] is True


def main() -> None:
    tests = [
        test_strict_admission_keeps_every_strong_view_without_one_best_compression,
        test_strict_admission_rejects_nonfinite_rank_instead_of_treating_it_as_wrong,
        test_registered_calibration_uses_each_actual_training_bank,
        test_explicit_shared_calibration_helper_remains_nonformal,
        test_admission_precedes_shuffle_and_excludes_nonstrict_donor,
        test_query_materialization_preserves_every_action_in_one_example_per_query,
        test_query_complete_forward_memory_bound_is_exact_and_fails_closed,
        test_formal_v6_configuration_is_exact_and_fails_closed,
        test_v6_is_only_the_strict_action_admission_spliced_into_registered_v3,
        test_v6_restored_changes_only_the_registered_optimizer_boundary,
        test_v7_restores_only_corrective_and_uses_the_low_loss_allocator,
    ]
    for test in tests:
        test()
    print(f"[test_noise_corrected_best_action_v6] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
