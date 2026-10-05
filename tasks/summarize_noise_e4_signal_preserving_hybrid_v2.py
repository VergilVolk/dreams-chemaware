"""Causal summary for corrected-E4 plus seven-source Hybrid V2/V3."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_noise_e4_best_actions_injector_v1_final import (
    _registered_metric_strict_or_boundary_failures,
    _registered_metric_violations,
)
from noise_final_core import sha256_file
from train_noise_final_r2_shared_encoder import formula_bootstrap_delta


EXPECTED_INITIAL_E8_SHA256 = (
    "8047b3f58c6808c86b320ac94b9e610610384040fa3a03e8d70550eb438a24af"
)
EXPECTED_BEST_ACTION_REPORT_SHA256 = (
    "246e7e871e6669fec9f2c62330a69bf9eb563a1b97b72b89732cecd682844349"
)
EXPECTED_BEST_ACTIONS_SHA256 = (
    "93f0785a69b5e323490a0b543059fa213fabb6ff848697831efba5b3fed667aa"
)
EXPECTED_BEST_ACTION_SPECTRA_SHA256 = (
    "6d57615aebbfb7bd6327bb7edb143c6837d2586221c61aa2ff45e833f5e1512a"
)
EXPECTED_E4_BASE_REPORT_SHA256 = (
    "9db644dc2592bb6eff2677779b731f6ee1f3f946f7df5868886b31c375cf3dfa"
)
EXPECTED_E4_BASE_ACTIONS_SHA256 = (
    "1ce8c412f9ee0ef7a4ba4758554313c85f123280d31e73004cb62b76887c7be5"
)
FORMULA_CI_FAMILYWISE_ALPHA = 0.05
FORMULA_CI_PRIMARY_COMPARISONS = 3


def _json_default(value: object) -> object:
    """Prevent a completed long run from failing on a NumPy scalar receipt."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-dir", type=Path, required=True)
    parser.add_argument("--shuffled-dir", type=Path, required=True)
    parser.add_argument("--best-action-dir", type=Path, required=True)
    parser.add_argument("--e4-base-action-dir", type=Path, required=True)
    parser.add_argument("--initial-e8-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--hybrid-version", choices=("v2", "v3"), default="v2",
    )
    return parser.parse_args()


def _nested_delta(candidate: object, reference: object) -> object:
    if isinstance(candidate, dict) and isinstance(reference, dict):
        return {
            key: _nested_delta(candidate[key], reference[key])
            for key in candidate.keys() & reference.keys()
        }
    if (
        isinstance(candidate, (int, float))
        and not isinstance(candidate, bool)
        and isinstance(reference, (int, float))
        and not isinstance(reference, bool)
    ):
        return float(candidate) - float(reference)
    return None


def _paired_counts(reference_rank: np.ndarray, candidate_rank: np.ndarray) -> dict[str, int]:
    corrected = (reference_rank != 1) & (candidate_rank == 1)
    introduced = (reference_rank == 1) & (candidate_rank != 1)
    return {
        "corrected": int(np.sum(corrected)),
        "introduced": int(np.sum(introduced)),
        "risk_net_lambda2": int(np.sum(corrected) - 2 * np.sum(introduced)),
    }


def _validate_optimizer_receipt(injection: dict, arm: str, version: str) -> None:
    """Recompute every registered V2 optimizer gate from persisted scalars."""
    if version == "v3":
        if (
            injection.get("gate_passed") is not True
            or int(injection.get("steps", -1)) != 30496
            or int(injection.get("action_active_steps", -1)) <= 0
            or float(injection.get("target_action_fraction", -1)) != 0.25
            or float(injection.get(
                "minimum_e4_projection_retention", -1,
            )) != 0.90
            or float(injection.get("maximum_update_norm_ratio", -1)) != 1.50
            or injection.get("exact_action_fraction_reached") is not True
            or injection.get("action_transmission_nonzero") is not True
            or injection.get("zero_action_exact_e4") is not True
        ):
            raise RuntimeError(f"{arm} V3 optimizer-boundary headline gate failed")
        fractions = injection.get("group_final_action_fraction_p50", {})
        projection = injection.get("group_minimum_e4_projection_retention", {})
        norm_ratio = injection.get("group_maximum_final_to_e4_norm_ratio", {})
        expected_groups = {"head", "backbone"}
        if any(set(values) != expected_groups for values in (
            fractions, projection, norm_ratio,
        )):
            raise RuntimeError(f"{arm} V3 optimizer parameter groups drifted")
        if (
            any(abs(float(value) - 0.25) > 1e-6
                for value in fractions.values())
            or min(map(float, projection.values())) + 1e-6 < 0.90
            or max(map(float, norm_ratio.values())) > 1.50 + 1e-6
        ):
            raise RuntimeError(f"{arm} V3 persisted optimizer scalars failed")
        return
    if (
        injection.get("gate_passed") is not True
        or int(injection.get("steps", -1)) != 30496
        or int(injection.get("semantic_active_steps", -1)) != 24576
        or int(injection.get("zero_semantic_steps", -1)) != 5920
        or float(injection.get("target_optimizer_action_fraction", -1)) != 0.25
        or float(injection.get("historical_e4_absolute_update_floor", -1)) != 0.90
        or float(injection.get("historical_e4_absolute_update_cap", -1)) != 1.50
        or injection.get("all_zero_semantic_shadow_updates_materialized") is not True
    ):
        raise RuntimeError(f"{arm} V2 optimizer-boundary headline gate failed")
    fraction = injection.get("optimizer_action_fraction_p10_by_group", {})
    projection = injection.get(
        "minimum_historical_e4_component_retention_by_group", {},
    )
    norm_min = injection.get(
        "minimum_final_to_shadow_e4_update_norm_ratio_by_group", {},
    )
    norm_max = injection.get(
        "maximum_final_to_shadow_e4_update_norm_ratio_by_group", {},
    )
    expected_groups = {"head", "backbone"}
    if any(set(values) != expected_groups for values in (
        fraction, projection, norm_min, norm_max,
    )):
        raise RuntimeError(f"{arm} V2 optimizer parameter groups drifted")
    if (
        any(abs(float(value) - 0.25) > 2e-6 for value in fraction.values())
        or float(injection.get("optimizer_action_fraction_max_abs_error", math.inf))
        > 2e-6
        or min(map(float, projection.values())) + 1e-6 < 0.90
        or min(map(float, norm_min.values())) + 1e-6 < 0.90
        or max(map(float, norm_max.values())) > 1.50 + 1e-6
        or float(injection.get(
            "minimum_final_to_historical_e4_update_norm_ratio", -math.inf,
        )) + 1e-6 < 0.90
        or float(injection.get(
            "maximum_final_to_historical_e4_update_norm_ratio", math.inf,
        )) > 1.50 + 1e-6
        or float(injection.get("maximum_virtual_adamw_relative_error", math.inf))
        > 1e-3
        or float(injection.get(
            "maximum_first_moment_reconstruction_relative_error", math.inf,
        )) > 1e-6
    ):
        raise RuntimeError(f"{arm} V2 persisted optimizer scalars fail recomputation")


def _validate_e4_base_exposure(schedule: dict, arm: str) -> None:
    """Keep historical E4 dose while reporting its row-level coverage honestly."""
    total = int(schedule.get("e4_base_action_rows", -1))
    physical = int(schedule.get("e4_base_physical_action_exposures", -1))
    unique = int(schedule.get("e4_base_unique_action_rows_exposed", -1))
    not_exposed = int(schedule.get("e4_base_action_rows_not_exposed", -1))
    fraction = float(schedule.get("e4_base_unique_action_row_coverage_fraction", -1.0))
    if (
        total != 190324
        or physical != 121984
        or not 0 < unique <= physical < total
        or not_exposed != total - unique
        or not math.isclose(fraction, unique / total, rel_tol=0.0, abs_tol=1e-12)
        or schedule.get(
            "e4_base_bank_is_validated_supplier_not_full_row_coverage"
        ) is not True
        or schedule.get(
            "e4_base_historical_identity_balanced_sampler_preserved"
        ) is not True
    ):
        raise RuntimeError(f"{arm} E4 base exposure accounting drifted")


def _validate_action_control(decision: dict, arm: str) -> None:
    control = decision.get("materialized_action_control", {})
    common = (
        int(control.get("rows", -1)) == 32114
        and int(control.get("selected_best_action_union_rows", -1)) == 32114
        and int(control.get("selected_strict_top1_corrective_queries", -1)) == 3482
        and control.get("selected_action_rows_preserved") is True
        and control.get("clean_and_action_active_candidate_row_union_preserved") is True
        and control.get("causal_control_donors_restricted_to_selected_union") is True
        and set(control.get("sources", [])) == {
            "N_mature", "P_guided_original", "E10B", "E11", "E12B",
            "A4_exact", "V4_gradient_path",
        }
    )
    if not common:
        raise RuntimeError(f"{arm} V2 action-panel receipt drifted")
    if arm == "targeted":
        if (
            control.get("strategy") != "true_query_matched_action_spectrum"
            or control.get("exact_actions_preserved") is not True
        ):
            raise RuntimeError("targeted V2 action tensor was not preserved")
        return
    if (
        control.get("strategy")
        != "supervision_source_family_exact_recipe_matched_cross_query_cyclic_shuffle"
        or control.get("matching_columns")
        != ["supervision_kind", "source", "family", "recipe_id"]
        or control.get("family_semantics_preserved") is not True
        or control.get("dose_and_action_semantics_preserved") is not True
        or control.get("all_nonfallback_donors_inside_input_panel") is not True
        or control.get("true_action_row_reused_for_same_query") is not False
        or control.get("exact_actions_preserved") is not False
        or int(control.get("cross_query_rows", -1))
        + int(control.get("paired_control_fallback_rows", -1)) != 32114
    ):
        raise RuntimeError("matched shuffled V2 action-view semantics drifted")


def _load_arm(path: Path, arm: str, args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    decision_path = path / "decision.json"
    ledger_path = path / "held_per_query.csv.gz"
    checkpoint_path = path / "final_shared_encoder.pt"
    for required in (decision_path, ledger_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "noise_final_e4a_direct_augmentation_complete":
        raise RuntimeError(f"{arm} training decision is incomplete")
    configuration = decision.get("configuration", {})
    is_v3 = args.hybrid_version == "v3"
    exact = {
        "action_selection": "materialized_routed",
        "materialized_action_arm": arm,
        "materialized_injection_mode": (
            "e4_live_shared_v3" if is_v3 else "e4_base_semantic_v2"
        ),
        "optimizer_boundary_mode": (
            "separated_e4_action_v3" if is_v3 else "signal_preserving_v2"
        ),
        "policy": "curriculum",
        "action_scope": "all",
        "outer_fold": 0,
        "formula_fold_seed": 20260825,
        "seed": args.seed,
        "epochs": 4,
        "batch_actions": 4,
        "views_per_identity": 4,
        "error_views_per_identity": 0,
        "positive_spectra": 4,
        "negative_molecules": 8,
        "unfreeze_blocks": 1,
        "direct_transfer_mode": "symmetric",
        "rank_reference_mode": "shared",
        "guided_noise_policy": "none",
        "pmt_arm": "none",
        "candidate_boundary_loss": False,
        "refresh_hard_negatives": False,
        "causal_arm": "legacy",
        "amp": False,
        "smoke": False,
    }
    drift = {
        key: {"expected": value, "observed": configuration.get(key)}
        for key, value in exact.items() if configuration.get(key) != value
    }
    floats = {
        "backbone_lr": 2e-6,
        "head_lr": 1e-5,
        "weight_decay": 1e-4,
        "rank_margin": 0.05,
        "temperature": 0.10,
        "lambda_clean_rank": 1.0,
        "lambda_aug_rank": 1.0,
        "lambda_consistency": 0.25,
        "lambda_margin_floor": 2.0,
        "lambda_preserve": 5.0,
        "margin_floor_slack": 0.005,
        "safety_ratio": 1.0,
        "safety_stream_weight": 1.0,
        "positive_stream_weight": 0.0,
        "grad_clip": 1.0,
        "injector_target_attributable_fraction": 0.25,
        "injector_minimum_protective_retention": 0.90,
        "injector_maximum_update_norm_ratio": 1.50,
    }
    for key, expected in floats.items():
        observed = configuration.get(key)
        if observed is None or not math.isclose(
            float(observed), expected, rel_tol=0.0, abs_tol=1e-15,
        ):
            drift[key] = {"expected": expected, "observed": observed}
    if drift:
        raise RuntimeError(f"{arm} {args.hybrid_version} configuration drifted: {drift}")

    contracts = decision.get("contracts", {})
    required_true = (
        (
            "clean_and_augmented_raw_spectra_train_same_encoder",
            "historical_e4_symmetric_shared_direct_loss_unmodified",
            "complete_seven_source_action_panel_used_without_one_best_compression",
            "materialized_all_strict_actions_preserved",
            "live_shared_v3_used",
            "live_shared_v3_clean_action_positive_negative_all_live",
            "live_shared_v3_every_audited_query_all_four_roles_live",
            "live_shared_v3_all_four_roles_live_through_ranking_paths",
            "live_shared_v3_targeted_shuffled_compared_before_injector",
            "live_shared_v3_action_rank_not_gated",
            "live_shared_v3_complete_e4_terms_preserved",
            "live_shared_v3_query_equal_dose",
            "live_shared_v3_same_query_actions_not_mixed",
            "live_shared_v3_independent_optimizer_states",
            "live_shared_v3_exact_action_fraction_reached",
            "live_shared_v3_action_transmission_nonzero",
            "inference_clean_spectrum_only",
        ) if is_v3 else (
            "clean_and_augmented_raw_spectra_train_same_encoder",
            "historical_e4_symmetric_shared_direct_loss_unmodified",
            "complete_seven_source_action_panel_used_without_one_best_compression",
            "materialized_all_strict_actions_preserved",
            "v2_starts_from_frozen_e8_not_official_restart",
            "v2_e8_initialization_is_floor_and_preservation_target",
            "v2_corrected_graph_e4_base_is_outcome_free",
            "v2_e4_base_and_later_actions_share_frozen_e8_geometry",
            "v2_later_actions_are_query_local_semantic_residual_only",
            "v2_every_later_identity_has_four_distinct_optimizer_steps_per_epoch",
            "v2_all_later_actions_exposed_with_family_local_coverage_first",
            "v2_later_source_family_equal_effective_dose",
            "v2_every_optimizer_step_audited",
            "v2_signal_and_e4_retention_gate_passed",
            "inference_clean_spectrum_only",
        )
    )
    bad = [key for key in required_true if contracts.get(key) is not True]
    if bad:
        raise RuntimeError(f"{arm} {args.hybrid_version} lost required contracts: {bad}")
    if (
        contracts.get("teacher")
        != "outer_train_action_routing_only_no_teacher_target"
        or contracts.get("P2b") != "forbidden"
        or contracts.get("P3_consumed") is not False
        or contracts.get("materialized_action_embedding_target_used") is not False
        or contracts.get("materialized_action_margin_target_used") is not False
    ):
        raise RuntimeError(f"{arm} V2 enabled a teacher or forbidden branch")

    schedule = decision.get(
        "live_shared_v3_schedule" if is_v3 else "signal_preserving_v2_schedule",
        {},
    )
    if is_v3:
        schedule_exact = {
            "action_count": 32114,
            "query_count": 3482,
            "views_per_query_per_epoch": 4,
            "optimizer_steps_per_epoch": 7624,
            "active_steps_per_epoch": 3484,
            "physical_action_exposures": 55712,
            "unique_actions_exposed": 32114,
            "all_unique_actions_exposed": True,
            "maximum_actions_per_query": 16,
            "e4_base_action_rows": 190324,
            "e4_base_identities": 7624,
            "e4_base_views_per_identity_per_epoch": 4,
            "e4_base_optimizer_steps_per_epoch": 7624,
            "same_query_actions_never_share_an_optimizer_step": True,
            "query_dose_equal_within_every_epoch": True,
            "action_rows_are_coverage_first_within_query": True,
            "source_families_are_interleaved_not_averaged": True,
            "complete_live_shared_e4_loss": True,
            "clean_action_positive_negative_all_trainable": True,
            "action_rank_hard_gate": False,
            "same_query_actions_mixed_before_optimizer": False,
            "teacher_embedding_or_margin_target": False,
        }
    else:
        schedule_exact = {
            "action_count": 32114,
            "identity_count": 1536,
            "epochs": 4,
            "optimizer_steps_per_epoch": [7624, 7624, 7624, 7624],
            "identity_bags_per_epoch": [6144, 6144, 6144, 6144],
            "bags_per_identity_per_epoch": 4,
            "minimum_identity_effective_weight_per_epoch": 4.0,
            "maximum_identity_effective_weight_per_epoch": 4.0,
            "maximum_step_effective_weight": 1.0,
            "active_optimizer_steps_per_epoch": 6144,
            "zero_semantic_optimizer_steps_per_epoch": 1480,
            "e4_base_action_rows": 190324,
            "e4_base_identities": 7624,
            "e4_base_views_per_identity_per_epoch": 4,
            "e4_base_optimizer_steps_per_epoch": 7624,
            "maximum_spectra_per_semantic_forward": 64,
            "all_unique_actions_exposed": True,
            "recycling_only_after_per_identity_source_family_complete_coverage": True,
            "identity_semantic_dose_equal": True,
            "source_family_semantic_dose_equal_within_identity": True,
            "every_source_family_present_on_every_identity_optimizer_step": True,
            "later_actions_share_encoder_and_are_not_teacher_targets": True,
            "each_later_identity_has_four_distinct_optimizer_opportunities_per_epoch": True,
            "later_source_family_equal_effective_dose_within_identity": True,
        }
    schedule_drift = {
        key: {"expected": value, "observed": schedule.get(key)}
        for key, value in schedule_exact.items() if schedule.get(key) != value
    }
    if schedule_drift:
        raise RuntimeError(f"{arm} {args.hybrid_version} schedule drifted: {schedule_drift}")
    _validate_optimizer_receipt(
        decision.get("optimizer_boundary_injection", {}), arm, args.hybrid_version,
    )
    _validate_action_control(decision, arm)

    provenance = decision.get("provenance", {})
    expected_provenance = {
        "initial_student_checkpoint_sha256": EXPECTED_INITIAL_E8_SHA256,
        "materialized_action_report_sha256": EXPECTED_BEST_ACTION_REPORT_SHA256,
        "materialized_training_actions_sha256": EXPECTED_BEST_ACTIONS_SHA256,
        "materialized_action_spectra_sha256": EXPECTED_BEST_ACTION_SPECTRA_SHA256,
        "corrected_e4_base_report_sha256": EXPECTED_E4_BASE_REPORT_SHA256,
        "corrected_e4_base_actions_sha256": EXPECTED_E4_BASE_ACTIONS_SHA256,
        "final_shared_encoder_sha256": sha256_file(checkpoint_path),
    }
    provenance_drift = {
        key: {"expected": value, "observed": provenance.get(key)}
        for key, value in expected_provenance.items()
        if provenance.get(key) != value
    }
    if provenance_drift:
        raise RuntimeError(f"{arm} V2 provenance drifted: {provenance_drift}")
    ledger = pd.read_csv(ledger_path, low_memory=False).sort_values(
        "query_index", kind="stable",
    ).reset_index(drop=True)
    required_ledger_columns = {
        "query_index", "query_row", "query_ik14", "query_formula", "has_near",
        "baseline_rank", "initialization_rank", "final_rank",
        "baseline_top_molecule_local", "initialization_top_molecule_local",
        "baseline_full_margin", "initialization_full_margin",
    }
    if missing := required_ledger_columns - set(ledger.columns):
        raise RuntimeError(f"{arm} held ledger lacks {sorted(missing)}")
    if ledger["query_index"].duplicated().any():
        raise RuntimeError(f"{arm} held ledger duplicated a query")
    return decision, ledger


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    artifact_hashes = {
        "initial_e8": sha256_file(args.initial_e8_checkpoint),
        "best_action_report": sha256_file(args.best_action_dir / "report.json"),
        "best_actions": sha256_file(args.best_action_dir / "training_actions.csv.gz"),
        "best_action_spectra": sha256_file(args.best_action_dir / "action_spectra.npz"),
        "e4_base_report": sha256_file(args.e4_base_action_dir / "report.json"),
        "e4_base_actions": sha256_file(
            args.e4_base_action_dir / "training_actions.csv.gz"
        ),
    }
    expected_hashes = {
        "initial_e8": EXPECTED_INITIAL_E8_SHA256,
        "best_action_report": EXPECTED_BEST_ACTION_REPORT_SHA256,
        "best_actions": EXPECTED_BEST_ACTIONS_SHA256,
        "best_action_spectra": EXPECTED_BEST_ACTION_SPECTRA_SHA256,
        "e4_base_report": EXPECTED_E4_BASE_REPORT_SHA256,
        "e4_base_actions": EXPECTED_E4_BASE_ACTIONS_SHA256,
    }
    if artifact_hashes != expected_hashes:
        raise RuntimeError(
            f"V2 frozen artifact hashes drifted: observed={artifact_hashes} "
            f"expected={expected_hashes}"
        )

    targeted, targeted_ledger = _load_arm(args.targeted_dir, "targeted", args)
    shuffled, shuffled_ledger = _load_arm(args.shuffled_dir, "shuffled", args)
    invariant_columns = (
        "query_index", "query_row", "query_ik14", "query_formula", "has_near",
        "baseline_rank", "initialization_rank", "baseline_top_molecule_local",
        "initialization_top_molecule_local", "baseline_full_margin",
        "initialization_full_margin",
    )
    for column in invariant_columns:
        if not np.array_equal(
            targeted_ledger[column].to_numpy(), shuffled_ledger[column].to_numpy(),
        ):
            raise RuntimeError(f"causal arms changed held ledger column {column}")
    if len(targeted.get("history", [])) != 4 or len(shuffled.get("history", [])) != 4:
        raise RuntimeError("V2 causal arms do not both contain four training epochs")
    for left, right in zip(targeted["history"], shuffled["history"]):
        history_keys = (
            (
                "e4_base_sampling_schedule_sha256",
                "safety_sampling_schedule_sha256",
                "steps",
                "later_action_active_steps",
                "later_action_zero_steps",
            ) if args.hybrid_version == "v3" else (
                "e4_base_sampling_schedule_sha256",
                "e4_base_action_exposure_sha256",
                "safety_sampling_schedule_sha256",
                "steps",
                "later_semantic_active_steps",
                "later_semantic_zero_steps",
            )
        )
        for key in history_keys:
            if left.get(key) != right.get(key):
                raise RuntimeError(f"causal arm training schedule drifted at {key}")
    schedule_key = (
        "live_shared_v3_schedule"
        if args.hybrid_version == "v3" else "signal_preserving_v2_schedule"
    )
    targeted_schedule = targeted.get(schedule_key, {})
    shuffled_schedule = shuffled.get(schedule_key, {})
    _validate_e4_base_exposure(targeted_schedule, "targeted")
    _validate_e4_base_exposure(shuffled_schedule, "shuffled")
    for key in (
        "e4_base_physical_action_exposures",
        "e4_base_unique_action_rows_exposed",
        "e4_base_action_rows_not_exposed",
        "e4_base_unique_queries_exposed",
    ):
        if targeted_schedule.get(key) != shuffled_schedule.get(key):
            raise RuntimeError(f"causal arms changed E4 base exposure at {key}")
    if (
        targeted_schedule.get("later_action_schedule_sha256")
        != shuffled_schedule.get("later_action_schedule_sha256")
    ):
        raise RuntimeError("targeted/shuffled later action schedule drifted")

    target_panel = targeted["held_clean"]["corrected_graph_registered_metrics"]
    shuffle_panel = shuffled["held_clean"]["corrected_graph_registered_metrics"]
    if target_panel["official"] != shuffle_panel["official"]:
        raise RuntimeError("causal arms disagree on official registered metrics")
    if target_panel["initialization"] != shuffle_panel["initialization"]:
        raise RuntimeError("causal arms disagree on E8 initialization metrics")
    official = target_panel["official"]
    initial = target_panel["initialization"]
    target = target_panel["student"]
    shuffled_metrics = shuffle_panel["student"]

    target_rank = targeted_ledger["final_rank"].to_numpy(np.int64)
    shuffled_rank = shuffled_ledger["final_rank"].to_numpy(np.int64)
    initial_rank = targeted_ledger["initialization_rank"].to_numpy(np.int64)
    official_rank = targeted_ledger["baseline_rank"].to_numpy(np.int64)
    formulas = targeted_ledger["query_formula"].astype(str).to_numpy()
    target_vs_shuffle_ci = formula_bootstrap_delta(
        shuffled_rank, target_rank, formulas, args.bootstrap_resamples, args.seed + 91,
        alpha=FORMULA_CI_FAMILYWISE_ALPHA / FORMULA_CI_PRIMARY_COMPARISONS,
    )
    target_vs_initial_ci = formula_bootstrap_delta(
        initial_rank, target_rank, formulas, args.bootstrap_resamples, args.seed + 92,
        alpha=FORMULA_CI_FAMILYWISE_ALPHA / FORMULA_CI_PRIMARY_COMPARISONS,
    )
    target_vs_official_ci = formula_bootstrap_delta(
        official_rank, target_rank, formulas, args.bootstrap_resamples, args.seed + 93,
        alpha=FORMULA_CI_FAMILYWISE_ALPHA / FORMULA_CI_PRIMARY_COMPARISONS,
    )
    paired_vs_shuffle = _paired_counts(shuffled_rank, target_rank)
    paired_vs_initial = _paired_counts(initial_rank, target_rank)
    paired_vs_official = _paired_counts(official_rank, target_rank)
    near = targeted_ledger["has_near"].to_numpy(bool)
    near_paired_vs_shuffle = _paired_counts(
        shuffled_rank[near], target_rank[near]
    )
    near_paired_vs_initial = _paired_counts(initial_rank[near], target_rank[near])
    near_paired_vs_official = _paired_counts(official_rank[near], target_rank[near])

    violations_official = _registered_metric_violations(target, official)
    violations_initial = _registered_metric_violations(target, initial)
    violations_shuffle = _registered_metric_violations(target, shuffled_metrics)
    strict_fail_official = _registered_metric_strict_or_boundary_failures(
        target, official,
    )
    strict_fail_initial = _registered_metric_strict_or_boundary_failures(
        target, initial,
    )
    strict_fail_shuffle = _registered_metric_strict_or_boundary_failures(
        target, shuffled_metrics,
    )
    official_r1 = float(official["retrieval"]["recall@1"])
    initial_r1 = float(initial["retrieval"]["recall@1"])
    target_r1 = float(target["retrieval"]["recall@1"])
    shuffled_r1 = float(shuffled_metrics["retrieval"]["recall@1"])
    gates = {
        "both_optimizer_boundary_contracts_passed": True,
        "targeted_formula_ci_vs_official_positive": bool(
            target_vs_official_ci["ci_low"] > 0
        ),
        "targeted_formula_ci_vs_initial_e8_positive": bool(
            target_vs_initial_ci["ci_low"] > 0
        ),
        "targeted_beats_shuffled_formula_ci": bool(
            target_vs_shuffle_ci["ci_low"] > 0
        ),
        "risk_net_lambda2_vs_official_positive": bool(
            paired_vs_official["risk_net_lambda2"] > 0
        ),
        "risk_net_lambda2_vs_initial_e8_positive": bool(
            paired_vs_initial["risk_net_lambda2"] > 0
        ),
        "risk_net_lambda2_vs_shuffled_positive": bool(
            paired_vs_shuffle["risk_net_lambda2"] > 0
        ),
        "near_risk_net_lambda2_vs_official_positive": bool(
            near_paired_vs_official["risk_net_lambda2"] > 0
        ),
        "near_risk_net_lambda2_vs_initial_e8_positive": bool(
            near_paired_vs_initial["risk_net_lambda2"] > 0
        ),
        "near_risk_net_lambda2_vs_shuffled_positive": bool(
            near_paired_vs_shuffle["risk_net_lambda2"] > 0
        ),
        "all_registered_metrics_nonnegative_vs_official": not violations_official,
        "all_registered_metrics_nonnegative_vs_initial_e8": not violations_initial,
        "all_registered_metrics_nonnegative_vs_shuffled": not violations_shuffle,
        "all_registered_metrics_strict_or_boundary_vs_official": not strict_fail_official,
        "all_registered_metrics_strict_or_boundary_vs_initial_e8": not strict_fail_initial,
        "all_registered_metrics_strict_or_boundary_vs_shuffled": not strict_fail_shuffle,
        "targeted_preservation_vs_initial_e8_ge_0_995": bool(
            float(targeted["held_clean"]["preservation_vs_initialization_mean"])
            >= 0.995
        ),
        "shuffled_preservation_vs_initial_e8_ge_0_995": bool(
            float(shuffled["held_clean"]["preservation_vs_initialization_mean"])
            >= 0.995
        ),
        "strict_four_pp_recall1_gain_vs_official": bool(
            target_r1 - official_r1 >= 0.04
        ),
    }
    report = {
        "status": (
            "noise_e4_live_shared_hybrid_v3_summary_complete"
            if args.hybrid_version == "v3"
            else "noise_e4_signal_preserving_hybrid_v2_summary_complete"
        ),
        "formal": False,
        "evaluation_role": "held-formula corrected development graph",
        "official": official,
        "initial_e8": initial,
        "targeted": target,
        "matched_shuffled": shuffled_metrics,
        "targeted_minus_official": _nested_delta(target, official),
        "targeted_minus_initial_e8": _nested_delta(target, initial),
        "targeted_minus_matched_shuffled": _nested_delta(target, shuffled_metrics),
        "recall1": {
            "official": official_r1,
            "initial_e8": initial_r1,
            "targeted": target_r1,
            "matched_shuffled": shuffled_r1,
            "targeted_minus_official_pp": 100.0 * (target_r1 - official_r1),
            "targeted_minus_initial_e8_pp": 100.0 * (target_r1 - initial_r1),
            "targeted_minus_matched_shuffled_pp": 100.0 * (target_r1 - shuffled_r1),
        },
        "paired_top1_vs_official": paired_vs_official,
        "paired_top1_vs_initial_e8": paired_vs_initial,
        "paired_top1_vs_matched_shuffled": paired_vs_shuffle,
        "near_paired_top1_vs_official": near_paired_vs_official,
        "near_paired_top1_vs_initial_e8": near_paired_vs_initial,
        "near_paired_top1_vs_matched_shuffled": near_paired_vs_shuffle,
        "formula_cluster_ci_familywise_control": {
            "method": "Bonferroni",
            "familywise_alpha": FORMULA_CI_FAMILYWISE_ALPHA,
            "primary_comparisons": FORMULA_CI_PRIMARY_COMPARISONS,
            "per_comparison_alpha": (
                FORMULA_CI_FAMILYWISE_ALPHA / FORMULA_CI_PRIMARY_COMPARISONS
            ),
            "simultaneous_confidence": 1.0 - FORMULA_CI_FAMILYWISE_ALPHA,
            "comparators": ["official", "initial_e8", "matched_shuffled"],
        },
        "formula_cluster_ci_vs_official": target_vs_official_ci,
        "formula_cluster_ci_vs_initial_e8": target_vs_initial_ci,
        "formula_cluster_ci_vs_matched_shuffled": target_vs_shuffle_ci,
        "registered_metric_violations_vs_official": violations_official,
        "registered_metric_violations_vs_initial_e8": violations_initial,
        "registered_metric_violations_vs_matched_shuffled": violations_shuffle,
        "registered_metric_strict_failures_vs_official": strict_fail_official,
        "registered_metric_strict_failures_vs_initial_e8": strict_fail_initial,
        "registered_metric_strict_failures_vs_matched_shuffled": strict_fail_shuffle,
        "per_arm_self_report_promotion_flags": {
            "causal_decision_role": "ignored",
            "reason": (
                "one arm cannot establish action-specific transfer; this summary "
                "recomputes every promotion gate against official, initial E8, "
                "and the matched shuffled arm"
            ),
            "targeted_pass_to_multifold": bool(
                targeted.get("pass_to_multifold", False)
            ),
            "shuffled_pass_to_multifold": bool(
                shuffled.get("pass_to_multifold", False)
            ),
        },
        "later_stream_attribution_limit": {
            "different_query_actions_are_mean_reduced_before_optimizer": bool(
                is_v3
            ),
            "exact_fraction_applies_to_full_later_e4_stream": bool(is_v3),
            "exact_fraction_is_not_targeted_minus_shuffled_fraction": bool(is_v3),
            "targeted_specific_transfer_requires_matched_shuffled_comparison": True,
            "action_active_step_fraction": (
                float(targeted_schedule["active_steps_per_epoch"])
                / float(targeted_schedule["optimizer_steps_per_epoch"])
                if is_v3 else None
            ),
            "nominal_full_training_action_fraction": (
                float(targeted_schedule["active_steps_per_epoch"])
                / float(targeted_schedule["optimizer_steps_per_epoch"])
                * float(
                    targeted["optimizer_boundary_injection"][
                        "target_action_fraction"
                    ]
                )
                if is_v3 else None
            ),
        },
        "gates": gates,
        "strict_full_panel_four_pp_achieved": bool(all(gates.values())),
        "e4_base_exposure": {
            key: targeted_schedule[key]
            for key in (
                "e4_base_action_rows",
                "e4_base_identities",
                "e4_base_physical_action_exposures",
                "e4_base_unique_action_rows_exposed",
                "e4_base_unique_action_row_coverage_fraction",
                "e4_base_action_rows_not_exposed",
                "e4_base_unique_queries_exposed",
                "e4_base_bank_is_validated_supplier_not_full_row_coverage",
                "e4_base_historical_identity_balanced_sampler_preserved",
            )
        },
        "provenance": artifact_hashes | {
            "targeted_decision_sha256": sha256_file(args.targeted_dir / "decision.json"),
            "targeted_checkpoint_sha256": sha256_file(
                args.targeted_dir / "final_shared_encoder.pt"
            ),
            "shuffled_decision_sha256": sha256_file(args.shuffled_dir / "decision.json"),
            "shuffled_checkpoint_sha256": sha256_file(
                args.shuffled_dir / "final_shared_encoder.pt"
            ),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Development result only. Four pp is an observed-result gate, not an "
            "action-oracle promise; MassSpecGym 10-ppm pooled AUROC is not the "
            "paper NIST20 0.85 replication."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".noise_e4_{args.hybrid_version}_summary_",
        dir=args.output_dir.parent,
    ))
    try:
        targeted_ledger.assign(
            matched_shuffled_rank=shuffled_rank,
            corrected_vs_shuffled=(shuffled_rank != 1) & (target_rank == 1),
            introduced_vs_shuffled=(shuffled_rank == 1) & (target_rank != 1),
        ).to_csv(
            staging / "targeted_vs_matched_shuffled_per_query.csv.gz",
            index=False,
            compression="gzip",
        )
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, default=_json_default), encoding="utf-8",
        )
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
