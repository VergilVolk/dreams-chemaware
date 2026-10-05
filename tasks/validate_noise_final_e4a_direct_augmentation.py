"""Fail-closed validator for an E4-A direct shared-embedding run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decision_path = args.output_dir / "decision.json"
    checkpoint_path = args.output_dir / "final_shared_encoder.pt"
    if not decision_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("E4-A output lacks decision or shared encoder checkpoint")
    report = json.loads(decision_path.read_text(encoding="utf-8"))
    if report.get("status") != "noise_final_e4a_direct_augmentation_complete":
        raise RuntimeError("unexpected E4-A status")
    contracts = report.get("contracts", {})
    expected = {
        "shared_query_reference_encoder": True,
        "model_weights_changed": True,
        "clean_and_augmented_raw_spectra_train_same_encoder": True,
        "inference_clean_spectrum_only": True,
        "P2b": "forbidden",
        "P3_consumed": False,
    }
    for key, value in expected.items():
        if contracts.get(key) != value:
            raise RuntimeError(f"E4-A contract failed: {key}={contracts.get(key)!r}")
    configuration = report.get("configuration", {})
    causal_arm = configuration.get("causal_arm", "legacy")
    if causal_arm != "legacy":
        if causal_arm not in {"clean_duplicate", "matched_random", "targeted"}:
            raise RuntimeError(f"unknown E4-A causal arm: {causal_arm}")
        causal_audit = report.get("causal_action_audit", {})
        if causal_audit.get("arm") != causal_arm or int(causal_audit.get("rows", 0)) < 1000:
            raise RuntimeError("E4-A causal action audit is missing or too small")
        causal_expected = {
            "causal_attribution_arm": causal_arm,
            "causal_arm_changes_only_action_view": True,
            "matched_control_selection_uses_outcome": False,
            "causal_sampler_keys_arm_invariant": True,
            "causal_candidate_references_arm_invariant": True,
        }
        for key, value in causal_expected.items():
            if contracts.get(key) != value:
                raise RuntimeError(f"E4-A causal contract failed: {key}")
        if causal_arm == "matched_random" and set(
            causal_audit.get("matched_control_index_counts", {})
        ) != {"0", "1"}:
            raise RuntimeError("E4-A matched-random arm did not use both frozen controls")
    guided_policy = configuration.get("guided_noise_policy", "none")
    action_selection = configuration.get("action_selection", "fixed")
    pmt_arm = configuration.get("pmt_arm", "none")
    candidate_boundary = bool(configuration.get("candidate_boundary_loss", False))
    boundary_version = configuration.get("candidate_boundary_version", "v1_edges")
    expected_outcome_loss = (
        guided_policy == "selected" or pmt_arm == "paired_target"
    )
    if contracts.get("action_outcomes_used_in_loss_or_sample_weight") is not expected_outcome_loss:
        raise RuntimeError(
            "E4-A outcome-loss contract disagrees with selected guided teacher mode"
        )
    transfer_mode = configuration.get("direct_transfer_mode", "symmetric")
    reference_mode = configuration.get("rank_reference_mode", "shared")
    if transfer_mode == "official_action":
        if contracts.get("teacher") != "frozen_official_raw_action_embedding":
            raise RuntimeError("official-action transfer did not declare its frozen raw-action teacher")
        if contracts.get("official_action_targets_frozen_before_optimizer") is not True:
            raise RuntimeError("official-action targets were not frozen before optimization")
        if contracts.get("action_selection") != "fixed":
            raise RuntimeError("E8 may not combine official-action transfer with outcome-mined actions")
    elif guided_policy == "selected":
        if contracts.get("teacher") != "outer_fold_isolated_privileged_action_margin":
            raise RuntimeError("selected E14 mode did not declare its outer-fold teacher")
        if report.get("guided_teacher_replay", {}).get(
            "action_margin_max_abs_error", 1.0
        ) > 2e-4:
            raise RuntimeError("selected E14 action teacher did not replay exactly")
        if not configuration.get("initial_student_checkpoint"):
            raise RuntimeError("selected E14 mode did not declare mature initialization")
    elif action_selection == "materialized_routed":
        injection_mode = configuration.get("materialized_injection_mode", "one_best_e4")
        optimizer_boundary_mode = configuration.get(
            "optimizer_boundary_mode", "ordinary_adamw",
        )
        v3_schedule = report.get("live_shared_v3_schedule", {})
        all_v3_actions_exposed = bool(
            v3_schedule.get("unique_actions_exposed")
            == v3_schedule.get("action_count")
        )
        materialized_expected = {
            "materialized_corrective_actions_only": True,
            "materialized_strict_clean_wrong_action_top1_only": (
                injection_mode != "complete_panel_historical_e4"
            ),
            "materialized_all_actions_exposed_before_recycling": (
                all_v3_actions_exposed
                if injection_mode == "e4_live_shared_v3" else True
            ),
            "materialized_action_control_changes_only_action_view": True,
            "materialized_action_embedding_target_used": False,
            "materialized_action_margin_target_used": False,
        }
        for key, value in materialized_expected.items():
            if contracts.get(key) != value:
                raise RuntimeError(f"materialized E4 contract failed: {key}")
        if injection_mode in {
            "complete_panel_historical_e4", "e4_base_semantic_v2",
            "e4_live_shared_v3",
        } and contracts.get(
            "materialized_selected_as_strict_corrective_in_frozen_e8_geometry"
        ) is not True:
            raise RuntimeError("complete action panel lost its frozen E8 semantics")
        if injection_mode == "one_best_e4":
            if contracts.get("materialized_action_loss_is_historical_e4") is not True:
                raise RuntimeError("materialized one-best E4 kernel contract failed")
        elif injection_mode == "multi_action_balanced":
            multi_expected = {
                "materialized_action_loss_is_historical_e4": False,
                "materialized_action_loss_is_balanced_multi_action_direct": True,
                "materialized_all_strict_actions_preserved": True,
                "materialized_satisfied_action_rank_gradient_gated": False,
                "materialized_identity_family_equal_effective_dose": True,
                "corrected_metrics_use_exact_query_matvec": True,
            }
            for key, value in multi_expected.items():
                if contracts.get(key) != value:
                    raise RuntimeError(f"materialized multi-action contract failed: {key}")
        elif injection_mode == "complete_panel_historical_e4":
            complete_expected = {
                "materialized_action_loss_is_historical_e4": True,
                "materialized_action_loss_is_balanced_multi_action_direct": False,
                "materialized_all_strict_actions_preserved": True,
                "materialized_satisfied_action_rank_gradient_gated": True,
                "materialized_identity_family_equal_effective_dose": False,
                "historical_e4_official_initialization_used": True,
                "historical_e4_official_cache_is_floor_and_preservation_target": True,
                "historical_e4_symmetric_shared_direct_loss_unmodified": True,
                "hybrid_changes_only_action_supplier_and_optimizer_boundary": True,
                "complete_seven_source_action_panel_used_without_one_best_compression": True,
                "complete_panel_unit_action_weights_without_identity_or_source_reweighting": True,
                "action_specific_to_clean_corrective_gradient_gate_passed": True,
                "pure_e4_control_retrained": False,
            }
            for key, value in complete_expected.items():
                if contracts.get(key) != value:
                    raise RuntimeError(f"complete-panel historical E4 contract failed: {key}")
            zero_change = report.get("zero_change_gate", {})
            if (
                zero_change.get("gate_passed") is not True
                or zero_change.get("verification")
                != "official_checkpoint_edge_scores_plus_candidate_comparison_tie_aware_ranks"
                or not math.isclose(
                    float(zero_change.get("score_tolerance", -1.0)), 5e-4,
                    rel_tol=0.0, abs_tol=1e-15,
                )
                or float(zero_change.get(
                    "maximum_pair_score_abs_error", 1.0,
                )) > 5e-4
                or int(zero_change.get("nonboundary_rank_mismatches", -1)) != 0
                or int(zero_change.get("rank_mismatches", -1))
                != int(zero_change.get("boundary_rank_mismatches", -2))
                or float(zero_change.get("preservation_mean", 0.0)) < 0.9999
            ):
                raise RuntimeError("complete-panel official zero-change gate failed")
            materialized = report.get("materialized_action_control", {})
            if (
                materialized.get("selected_actions_equal_unit_weight") is not True
                or materialized.get("identity_family_effective_dose_normalized")
                is not False
                or materialized.get("identity_effective_dose_equalized") is not False
                or materialized.get(
                    "historical_minimum_identity_exposure_budget_preserved"
                ) is not True
                or int(materialized.get("identity_exposure_minimum", 0)) < 16
            ):
                raise RuntimeError("complete-panel action exposure semantics drifted")
            if configuration.get("initial_student_checkpoint") not in {None, "None"}:
                raise RuntimeError("complete-panel hybrid did not start from official DreaMS")
            semantic = report.get("branch_gradient_audit", {})
            if (
                semantic.get("gate_passed") is not True
                or int(semantic.get("formula_microbatches", 0)) != 16
                or semantic.get("action_specific_definition")
                != "aug_rank_plus_0.25_symmetric_consistency"
                or semantic.get("clean_corrective_definition")
                != "clean_rank_plus_2.0_margin_floor"
                or semantic.get("shared_loss_terms_between_compared_objectives")
                is not False
                or semantic.get(
                    "preservation_excluded_as_separate_protective_component"
                ) is not True
                or semantic.get("positive_alignment_required")
                is not (configuration.get("materialized_action_arm") == "targeted")
                or set(semantic.get("parameter_groups", {})) != {"head", "backbone"}
            ):
                raise RuntimeError(
                    "complete-panel action-specific/clean-corrective gradient audit failed"
                )
            for group, group_report in semantic["parameter_groups"].items():
                if (
                    group_report.get("gate_passed") is not True
                    or not math.isclose(
                        float(group_report.get("action_gradient_nonzero_fraction", -1)),
                        1.0, rel_tol=0, abs_tol=0,
                    )
                    or not math.isclose(
                        float(group_report.get(
                            "clean_corrective_gradient_nonzero_fraction", -1,
                        )),
                        1.0, rel_tol=0, abs_tol=0,
                    )
                    or (
                        configuration.get("materialized_action_arm") == "targeted"
                        and float(group_report.get("alignment_median", 0.0)) <= 0
                    )
                ):
                    raise RuntimeError(
                        f"complete-panel semantic group gate failed: {group}"
                    )
        elif injection_mode == "e4_base_semantic_v2":
            v2_expected = {
                "materialized_action_loss_is_historical_e4": False,
                "materialized_action_loss_is_balanced_multi_action_direct": False,
                "materialized_all_strict_actions_preserved": True,
                "materialized_satisfied_action_rank_gradient_gated": True,
                "materialized_identity_family_equal_effective_dose": True,
                "historical_e4_official_initialization_used": False,
                "historical_e4_symmetric_shared_direct_loss_unmodified": True,
                "complete_seven_source_action_panel_used_without_one_best_compression": True,
                "v2_starts_from_frozen_e8_not_official_restart": True,
                "v2_e8_initialization_is_floor_and_preservation_target": True,
                "v2_corrected_graph_e4_base_is_outcome_free": True,
                "v2_e4_base_and_later_actions_share_frozen_e8_geometry": True,
                "v2_later_actions_are_query_local_semantic_residual_only": True,
                "v2_later_semantic_clean_boundary_rank_active": True,
                "v2_later_satisfied_action_rank_gradient_gated": True,
                "v2_later_symmetric_clean_action_consistency_preserved": True,
                "v2_every_later_identity_has_four_distinct_optimizer_steps_per_epoch": True,
                "v2_all_later_actions_exposed_with_family_local_coverage_first": True,
                "v2_later_source_family_equal_effective_dose": True,
                "v2_every_optimizer_step_audited": True,
                "v2_signal_and_e4_retention_gate_passed": True,
                "pure_e4_control_retrained": False,
            }
            for key, value in v2_expected.items():
                if contracts.get(key) != value:
                    raise RuntimeError(f"signal-preserving V2 contract failed: {key}")
            if not configuration.get("initial_student_checkpoint"):
                raise RuntimeError("signal-preserving V2 did not start from frozen E8")
            schedule = report.get("signal_preserving_v2_schedule", {})
            schedule_exact = {
                "action_count": 32114,
                "identity_count": 1536,
                "optimizer_steps_per_epoch": [7624] * 4,
                "identity_bags_per_epoch": [6144] * 4,
                "bags_per_identity_per_epoch": 4,
                "active_optimizer_steps_per_epoch": 6144,
                "zero_semantic_optimizer_steps_per_epoch": 1480,
                "e4_base_action_rows": 190324,
                "e4_base_identities": 7624,
                "e4_base_optimizer_steps_per_epoch": 7624,
                "maximum_spectra_per_semantic_forward": 64,
                "all_unique_actions_exposed": True,
                "recycling_only_after_per_identity_source_family_complete_coverage": True,
                "identity_semantic_dose_equal": True,
                "source_family_semantic_dose_equal_within_identity": True,
                "every_source_family_present_on_every_identity_optimizer_step": True,
                "later_source_family_equal_effective_dose_within_identity": True,
                "each_later_identity_has_four_distinct_optimizer_opportunities_per_epoch": True,
            }
            drift = {
                key: {"expected": value, "observed": schedule.get(key)}
                for key, value in schedule_exact.items()
                if schedule.get(key) != value
            }
            if drift:
                raise RuntimeError(f"signal-preserving V2 schedule failed: {drift}")
        elif injection_mode == "e4_live_shared_v3":
            v3_expected = {
                "materialized_action_loss_is_historical_e4": True,
                "materialized_action_loss_is_balanced_multi_action_direct": False,
                "materialized_all_strict_actions_preserved": True,
                "materialized_satisfied_action_rank_gradient_gated": False,
                "materialized_identity_family_equal_effective_dose": False,
                "historical_e4_official_initialization_used": False,
                "historical_e4_symmetric_shared_direct_loss_unmodified": True,
                "complete_seven_source_action_panel_used_without_one_best_compression": True,
                "live_shared_v3_used": True,
                "live_shared_v3_clean_action_positive_negative_all_live": True,
                "live_shared_v3_every_audited_query_all_four_roles_live": True,
                "live_shared_v3_all_four_roles_live_through_ranking_paths": True,
                "live_shared_v3_targeted_shuffled_compared_before_injector": True,
                "live_shared_v3_action_rank_not_gated": True,
                "live_shared_v3_complete_e4_terms_preserved": True,
                "live_shared_v3_query_equal_dose": True,
                "live_shared_v3_same_query_actions_not_mixed": True,
                "live_shared_v3_independent_optimizer_states": True,
                "live_shared_v3_exact_action_fraction_reached": True,
                "live_shared_v3_action_transmission_nonzero": True,
                "pure_e4_control_retrained": False,
            }
            for key, value in v3_expected.items():
                if contracts.get(key) != value:
                    raise RuntimeError(f"live-shared V3 contract failed: {key}")
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
                "e4_base_optimizer_steps_per_epoch": 7624,
                "same_query_actions_never_share_an_optimizer_step": True,
                "query_dose_equal_within_every_epoch": True,
                "source_families_are_interleaved_not_averaged": True,
                "complete_live_shared_e4_loss": True,
                "clean_action_positive_negative_all_trainable": True,
                "action_rank_hard_gate": False,
                "same_query_actions_mixed_before_optimizer": False,
                "teacher_embedding_or_margin_target": False,
            }
            drift = {
                key: {"expected": value, "observed": v3_schedule.get(key)}
                for key, value in schedule_exact.items()
                if v3_schedule.get(key) != value
            }
            if drift:
                raise RuntimeError(f"live-shared V3 schedule failed: {drift}")
            preinjection = report.get(
                "live_shared_v3_preinjection_gradient_audit", {},
            )
            if (
                preinjection.get("gate_passed") is not True
                or preinjection.get(
                    "clean_action_positive_negative_all_live"
                ) is not True
                or preinjection.get(
                    "clean_action_positive_negative_live_for_every_audited_query"
                ) is not True
                or preinjection.get(
                    "clean_action_positive_negative_ranking_paths_all_live"
                ) is not True
                or preinjection.get(
                    "ranking_paths_live_for_every_non_degenerate_audited_query"
                ) is not True
                or preinjection.get(
                    "targeted_and_shuffled_distinct_before_injector"
                ) is not True
                or int(preinjection.get("optimizer_steps", -1)) != 0
            ):
                raise RuntimeError("live-shared V3 pre-injection audit failed")
        else:
            raise RuntimeError(f"unknown materialized injection mode: {injection_mode}")
        if optimizer_boundary_mode == "action_injector_v1":
            injector_expected = {
                "action_injector_v1_used": True,
                "action_injector_v1_receives_complete_historical_e4_gradient": True,
                "action_injector_v1_action_selection_or_tensor_mutation": False,
                "action_injector_v1_historical_e4_loss_mutation": False,
                "action_injector_v1_every_optimizer_step_audited": True,
                "action_injector_v1_signal_and_safety_gate_passed": True,
            }
            for key, value in injector_expected.items():
                if contracts.get(key) != value:
                    raise RuntimeError(f"E4 ActionInjectorV1 contract failed: {key}")
            if injection_mode not in {
                "multi_action_balanced", "complete_panel_historical_e4",
            }:
                raise RuntimeError("E4 ActionInjectorV1 accepted one-best compression")
            injection = report.get("optimizer_boundary_injection", {})
            if (
                injection.get("enabled") is not True
                or injection.get("attribution_sampling") != "every_optimizer_step"
                or int(injection.get("steps", 0))
                != sum(int(record.get("steps", 0)) for record in report.get("history", []))
                or float(injection.get(
                    "optimizer_action_fraction_max_abs_error", 1.0,
                )) > 2e-6
                or min(injection.get(
                    "minimum_protective_component_retention_by_group", {"missing": 0.0},
                ).values()) + 1e-6 < 0.90
                or float(injection.get("maximum_update_norm_ratio", 2.0)) > 1.5000001
                or float(injection.get(
                    "maximum_first_moment_reconstruction_relative_error", 1.0,
                )) > 1e-6
                or injection.get("gate_passed") is not True
            ):
                raise RuntimeError("E4 ActionInjectorV1 signal report failed")
            if not report.get("provenance", {}).get("action_injector_v1_sha256"):
                raise RuntimeError("E4 ActionInjectorV1 provenance is missing")
            if not report.get("provenance", {}).get(
                "e4_action_injector_v1_bridge_sha256"
            ):
                raise RuntimeError("E4 ActionInjectorV1 bridge provenance is missing")
        elif optimizer_boundary_mode == "signal_preserving_v2":
            if injection_mode != "e4_base_semantic_v2":
                raise RuntimeError("signal-preserving optimizer accepted the wrong action mode")
            injection = report.get("optimizer_boundary_injection", {})
            action_fraction = injection.get(
                "optimizer_action_fraction_p10_by_group", {},
            )
            e4_projection = injection.get(
                "minimum_historical_e4_component_retention_by_group", {},
            )
            update_ratio_min = injection.get(
                "minimum_final_to_shadow_e4_update_norm_ratio_by_group", {},
            )
            update_ratio_max = injection.get(
                "maximum_final_to_shadow_e4_update_norm_ratio_by_group", {},
            )
            expected_groups = {"head", "backbone"}
            if (
                contracts.get("signal_preserving_v2_used") is not True
                or injection.get("enabled") is not True
                or injection.get("attribution_sampling") != "every_optimizer_step"
                or int(injection.get("steps", -1)) != 30496
                or int(injection.get("semantic_active_steps", -1)) != 24576
                or int(injection.get("zero_semantic_steps", -1)) != 5920
                or float(injection.get("target_optimizer_action_fraction", -1)) != 0.25
                or float(injection.get("historical_e4_absolute_update_floor", -1)) != 0.90
                or float(injection.get("historical_e4_absolute_update_cap", -1)) != 1.50
                or any(set(values) != expected_groups for values in (
                    action_fraction, e4_projection, update_ratio_min, update_ratio_max,
                ))
                or any(abs(float(value) - 0.25) > 2e-6
                       for value in action_fraction.values())
                or float(injection.get(
                    "optimizer_action_fraction_max_abs_error", 1.0,
                )) > 2e-6
                or min(map(float, e4_projection.values())) + 1e-6 < 0.90
                or min(map(float, update_ratio_min.values())) + 1e-6 < 0.90
                or max(map(float, update_ratio_max.values())) > 1.500001
                or float(injection.get(
                    "minimum_final_to_historical_e4_update_norm_ratio", -1.0,
                )) + 1e-6 < 0.90
                or float(injection.get(
                    "maximum_final_to_historical_e4_update_norm_ratio", 2.0,
                )) > 1.500001
                or float(injection.get(
                    "maximum_virtual_adamw_relative_error", 1.0,
                )) > 1e-3
                or float(injection.get(
                    "maximum_first_moment_reconstruction_relative_error", 1.0,
                )) > 1e-6
                or injection.get("all_zero_semantic_shadow_updates_materialized")
                is not True
                or injection.get("gate_passed") is not True
            ):
                raise RuntimeError("signal-preserving V2 injection report failed")
            provenance = report.get("provenance", {})
            if (
                not provenance.get("corrected_e4_base_report_sha256")
                or not provenance.get("corrected_e4_base_actions_sha256")
                or not provenance.get("e4_signal_preserving_hybrid_v2_sha256")
            ):
                raise RuntimeError("signal-preserving V2 provenance is incomplete")
        elif optimizer_boundary_mode == "separated_e4_action_v3":
            if injection_mode != "e4_live_shared_v3":
                raise RuntimeError("separated V3 optimizer accepted the wrong action mode")
            injection = report.get("optimizer_boundary_injection", {})
            fractions = injection.get("group_final_action_fraction_p50", {})
            projections = injection.get(
                "group_minimum_e4_projection_retention", {},
            )
            ratios = injection.get(
                "group_maximum_final_to_e4_norm_ratio", {},
            )
            expected_groups = {"head", "backbone"}
            if (
                contracts.get("live_shared_v3_used") is not True
                or injection.get("enabled") is not True
                or int(injection.get("steps", -1)) != 30496
                or int(injection.get("action_active_steps", -1)) != 13936
                or int(injection.get("zero_action_steps", -1)) != 16560
                or float(injection.get("target_action_fraction", -1)) != 0.25
                or float(injection.get(
                    "minimum_e4_projection_retention", -1,
                )) != 0.90
                or float(injection.get("maximum_update_norm_ratio", -1)) != 1.50
                or injection.get("exact_action_fraction_reached") is not True
                or injection.get("action_transmission_nonzero") is not True
                or injection.get("zero_action_exact_e4") is not True
                or any(set(values) != expected_groups for values in (
                    fractions, projections, ratios,
                ))
                or any(abs(float(value) - 0.25) > 1e-6
                       for value in fractions.values())
                or min(map(float, projections.values())) + 1e-6 < 0.90
                or max(map(float, ratios.values())) > 1.500001
                or injection.get("gate_passed") is not True
            ):
                raise RuntimeError("live-shared V3 injection report failed")
            provenance = report.get("provenance", {})
            if (
                not provenance.get("corrected_e4_base_report_sha256")
                or not provenance.get("corrected_e4_base_actions_sha256")
                or not provenance.get("e4_live_shared_hybrid_v3_sha256")
            ):
                raise RuntimeError("live-shared V3 provenance is incomplete")
        elif optimizer_boundary_mode != "ordinary_adamw":
            raise RuntimeError(
                f"unknown materialized optimizer boundary: {optimizer_boundary_mode}"
            )
        if contracts.get("teacher") != "outer_train_action_routing_only_no_teacher_target":
            raise RuntimeError("materialized E4 incorrectly declares a teacher target")
        if transfer_mode != "symmetric" or reference_mode != "shared":
            raise RuntimeError("materialized E4 did not preserve symmetric/shared transfer")
        control = report.get("materialized_action_control", {})
        if control.get("selected_action_rows_preserved") is not True:
            raise RuntimeError("materialized E4 selected action rows were not preserved")
        if configuration.get("materialized_action_arm") == "targeted":
            if control.get("exact_actions_preserved") is not True:
                raise RuntimeError("materialized E4 targeted action tensors were not preserved")
        elif (
            control.get("exact_actions_preserved") is not False
            or control.get("strategy")
            != "supervision_source_family_exact_recipe_matched_cross_query_cyclic_shuffle"
            or control.get("matching_columns")
            != ["supervision_kind", "source", "family", "recipe_id"]
            or control.get("family_semantics_preserved") is not True
            or control.get("dose_and_action_semantics_preserved") is not True
            or control.get("all_nonfallback_donors_inside_input_panel") is not True
            or control.get("true_action_row_reused_for_same_query") is not False
            or int(control.get("cross_query_rows", -1))
            + int(control.get("paired_control_fallback_rows", -1))
            != int(control.get("rows", -2))
        ):
            raise RuntimeError("materialized E4 causal control semantics drifted")
        if injection_mode == "one_best_e4":
            if control.get("one_maximum_margin_action_per_query") is not True:
                raise RuntimeError("materialized E4 did not restore the best-action union")
        elif injection_mode == "multi_action_balanced":
            if (
                control.get("multi_action_panel_all_strict") is not True
                or control.get("identity_family_effective_dose_normalized") is not True
                or control.get("all_strict_actions_exposed_before_recycling") is not True
                or control.get("identity_effective_dose_restored") is not True
            ):
                raise RuntimeError("materialized multi-action coverage/dose contract failed")
        elif injection_mode == "complete_panel_historical_e4":
            if (
                control.get("multi_action_panel_all_strict") is not True
                or control.get("selected_actions_equal_unit_weight") is not True
                or control.get("all_strict_actions_exposed_before_recycling") is not True
                or control.get("selected_in_e8_geometry_but_trained_from_official") is not True
                or int(control.get("selected_best_action_union_rows", -1)) != 32114
                or int(control.get("selected_strict_top1_corrective_queries", -1)) != 3482
            ):
                raise RuntimeError("complete-panel historical E4 coverage contract failed")
        elif injection_mode == "e4_base_semantic_v2":
            v2_schedule = report.get("signal_preserving_v2_schedule", {})
            total_base_rows = int(v2_schedule.get("e4_base_action_rows", -1))
            exposed_base_rows = int(
                v2_schedule.get("e4_base_unique_action_rows_exposed", -1)
            )
            if (
                control.get("multi_action_panel_all_strict") is not True
                or control.get("selected_in_e8_geometry_but_trained_from_official") is not False
                or int(control.get("selected_best_action_union_rows", -1)) != 32114
                or int(control.get("selected_strict_top1_corrective_queries", -1)) != 3482
                or v2_schedule.get("all_unique_actions_exposed") is not True
                or total_base_rows != 190324
                or int(v2_schedule.get("e4_base_physical_action_exposures", -1))
                != 121984
                or not 0 < exposed_base_rows < total_base_rows
                or int(v2_schedule.get("e4_base_action_rows_not_exposed", -1))
                != total_base_rows - exposed_base_rows
                or v2_schedule.get(
                    "e4_base_bank_is_validated_supplier_not_full_row_coverage"
                ) is not True
                or v2_schedule.get(
                    "e4_base_historical_identity_balanced_sampler_preserved"
                ) is not True
            ):
                raise RuntimeError("signal-preserving V2 action coverage contract failed")
        elif injection_mode == "e4_live_shared_v3":
            total_base_rows = int(v3_schedule.get("e4_base_action_rows", -1))
            exposed_base_rows = int(
                v3_schedule.get("e4_base_unique_action_rows_exposed", -1)
            )
            if (
                control.get("multi_action_panel_all_strict") is not True
                or control.get(
                    "selected_in_e8_geometry_but_trained_from_official"
                ) is not False
                or int(control.get("selected_best_action_union_rows", -1)) != 32114
                or int(control.get("selected_strict_top1_corrective_queries", -1))
                != 3482
                or total_base_rows != 190324
                or int(v3_schedule.get("e4_base_physical_action_exposures", -1))
                != 121984
                or not 0 < exposed_base_rows < total_base_rows
                or int(v3_schedule.get("e4_base_action_rows_not_exposed", -1))
                != total_base_rows - exposed_base_rows
                or v3_schedule.get(
                    "e4_base_bank_is_validated_supplier_not_full_row_coverage"
                ) is not True
                or v3_schedule.get(
                    "e4_base_historical_identity_balanced_sampler_preserved"
                ) is not True
            ):
                raise RuntimeError("live-shared V3 action coverage contract failed")
        if control.get(
            "clean_and_action_active_candidate_row_union_preserved"
        ) is not True:
            raise RuntimeError("materialized E4 did not preserve both active boundaries")
        selected_before_floor = int(control.get(
            "selected_best_action_union_rows_before_numerical_floor", -1,
        ))
        selected_after_floor = int(control.get("selected_best_action_union_rows", -1))
        numerical_excluded = int(control.get("numerical_boundary_actions_excluded", -1))
        if (
            selected_before_floor - selected_after_floor != numerical_excluded
            or numerical_excluded < 0
            or abs(float(control.get("strict_replay_action_margin_floor", -1.0)) - 5e-6)
            > 1e-12
        ):
            raise RuntimeError("materialized E4 numerical-boundary filter drifted")
        expected_sources = {
            "N_mature", "P_guided_original", "E10B", "E11", "E12B",
            "A4_exact", "V4_gradient_path",
        }
        if not expected_sources.issubset(set(control.get("qualifying_sources", []))):
            raise RuntimeError("materialized E4 qualifying action families are incomplete")
        if control.get("causal_control_donors_restricted_to_selected_union") is not True:
            raise RuntimeError("materialized E4 control donor pool differs from its action union")
        if control.get("routing_scores_not_used_as_loss_targets") is not True:
            raise RuntimeError("materialized E4 leaked routing scores into its loss")
        if injection_mode == "one_best_e4" and control.get(
            "selected_actions_equal_unit_weight"
        ) is not True:
            raise RuntimeError("materialized one-best actions lost unit weighting")
        if injection_mode == "multi_action_balanced" and (
            control.get("selected_actions_equal_unit_weight") is not False
            or not math.isclose(
                float(control.get("static_preclip_loss_scale", -1.0)), 0.16,
                rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise RuntimeError("materialized multi-action dose/preclip scale drifted")
        if injection_mode == "complete_panel_historical_e4" and (
            control.get("selected_actions_equal_unit_weight") is not True
            or not math.isclose(
                float(control.get("static_preclip_loss_scale", -1.0)), 1.0,
                rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise RuntimeError("complete-panel hybrid retained a failed dose/loss scale")
        if (
            injection_mode not in {"e4_base_semantic_v2", "e4_live_shared_v3"}
            and control.get("historical_identity_exposure_budget_restored") is not True
        ):
            raise RuntimeError("materialized E4 did not restore identity-balanced dose")
        clean_replay = report.get("materialized_clean_replay", {})
        if injection_mode == "complete_panel_historical_e4":
            if (
                clean_replay.get("exact_router_scoring_used") is not True
                or clean_replay.get("selection_geometry") != "frozen_E8"
                or clean_replay.get("training_initialization_geometry") != "official_DreaMS"
            ):
                raise RuntimeError("complete-panel hybrid confused selection/training geometry")
        elif injection_mode == "e4_base_semantic_v2":
            if (
                clean_replay.get("exact_router_scoring_used") is not True
                or clean_replay.get("selection_geometry") != "frozen_E8"
                or clean_replay.get("training_initialization_geometry") != "frozen_E8"
                or int(clean_replay.get("unexplained_rank_mismatches", -1)) != 0
                or float(clean_replay.get("margin_max_abs_error", 1.0)) > 5e-4
            ):
                raise RuntimeError("signal-preserving V2 confused E8 selection/training geometry")
        elif (
            clean_replay.get("exact_router_scoring_used") is not True
            or int(clean_replay.get("unexplained_rank_mismatches", -1)) != 0
            or float(clean_replay.get("margin_max_abs_error", 1.0)) > 5e-4
        ):
            raise RuntimeError("materialized E4 failed exact clean-geometry replay")
        replay = report.get("materialized_action_replay", {})
        if configuration.get("materialized_action_arm") == "targeted":
            if injection_mode == "complete_panel_historical_e4":
                if (
                    replay.get("evaluated") is not True
                    or replay.get("all_actions_encoded_exactly_once") is not True
                    or replay.get("stored_selection_geometry_enforced") is not False
                    or replay.get("candidate_reference_geometry") != "official_DreaMS"
                ):
                    raise RuntimeError("complete-panel action replay did not audit official init")
            elif (
                replay.get("evaluated") is not True
                or int(replay.get("requested_action_batch_size", -1))
                != int(configuration.get("eval_batch_size", -2))
                or int(replay.get("effective_action_batch_size", -1))
                > int(configuration.get("eval_batch_size", -2))
                or int(replay.get("observed_maximum_spectra_per_forward", -1))
                > int(configuration.get("eval_batch_size", -2))
                or replay.get("all_actions_encoded_exactly_once") is not True
                or replay.get("candidate_reference_geometry") != "current_E8"
                or replay.get("all_strict_top1_actions_reproduced") is not True
                or float(replay.get("margin_max_abs_error", 1.0)) > 5e-4
                or float(replay.get("clean_margin_max_abs_error", 1.0)) > 5e-4
            ):
                raise RuntimeError("materialized E4 failed exact best-action replay")
        elif replay.get("evaluated") is not False:
            raise RuntimeError("materialized E4 causal control unexpectedly replayed targets")
        if int(report.get("data", {}).get("materialized_unique_corrective_actions", 0)) < 1:
            raise RuntimeError("materialized E4 did not consume a corrective action")
        if report.get("provenance", {}).get(
            "final_shared_encoder_sha256"
        ) != sha256_file(checkpoint_path):
            raise RuntimeError("materialized E4 final checkpoint provenance drifted")
    elif pmt_arm == "paired_target":
        expected_pmt = {
            "pmt_target_control_same_batch": True,
            "pmt_harmful_target_weight_exact_zero": True,
            "pmt_all_corrective_actions_exposed_before_recycling": True,
            "pmt_control_has_identity_and_margin_floor": not candidate_boundary,
            "pmt_M2_predictions_used": False,
            "pmt_P_actions_used": False,
        }
        for key, value in expected_pmt.items():
            if contracts.get(key) != value:
                raise RuntimeError(f"E4-PMT contract failed: {key}")
        expected_teacher = (
            "forbidden" if candidate_boundary
            else "outer_train_current_geometry_target_control_advantage"
        )
        if contracts.get("teacher") != expected_teacher:
            raise RuntimeError("E4-PMT teacher provenance is not explicit")
        if candidate_boundary:
            expected_boundary = {
                "candidate_boundary_matrix_preserved": True,
                "candidate_boundary_scalar_teacher_used": False,
                "candidate_boundary_control_symmetrically_ranked": False,
                "candidate_boundary_control_gradient_stopped": True,
                "candidate_boundary_live_topk_refresh": True,
                "candidate_boundary_formula_stratified_gradient_calibration": True,
            }
            if boundary_version == "v1_edges":
                expected_boundary["candidate_boundary_action_not_weakened_by_calibration"] = True
            elif boundary_version == "v2_molecule_max":
                expected_boundary.update({
                    "candidate_boundary_uses_molecule_max": True,
                    "candidate_boundary_clean_primary_corrective_dose": True,
                    "candidate_boundary_v2_calibration_may_downscale": True,
                    "candidate_boundary_query_normalized_multi_action_dose": True,
                    "candidate_boundary_query_shared_candidate_references": True,
                    "candidate_boundary_noncorrective_action_safety_only": True,
                    "pmt_all_routed_actions_exposed_before_recycling": True,
                    "query_equal_action_weighting": True,
                })
                scale = float(report.get("effective_boundary_action_weight", -1.0))
                if scale < 0:
                    raise RuntimeError("candidate-boundary v2 reported an invalid action scale")
            else:
                raise RuntimeError(f"unknown candidate-boundary version: {boundary_version}")
            for key, value in expected_boundary.items():
                if contracts.get(key) != value:
                    raise RuntimeError(f"candidate-boundary contract failed: {key}")
    elif contracts.get("teacher") != "forbidden":
        raise RuntimeError(f"unexpected direct-transfer teacher: {contracts.get('teacher')!r}")
    if reference_mode == "official":
        if contracts.get("official_reference_anchors_training_only") is not True:
            raise RuntimeError("official reference mode lacks its training-only anchor contract")
    elif contracts.get("official_reference_anchors_training_only") is not False:
        raise RuntimeError("shared reference mode unexpectedly declares official anchors")
    held = report.get("held_clean", {})
    for key in ("baseline_recall1", "recall1", "delta_recall1", "corrected", "introduced"):
        if key not in held:
            raise RuntimeError(f"E4-A held-clean metric missing: {key}")
    if candidate_boundary or action_selection == "materialized_routed":
        metrics = held.get("complete_candidate_metrics", {})
        if set(metrics) != {
            "official", "initialization", "student", "student_minus_official",
            "student_minus_initialization",
        }:
            raise RuntimeError("candidate-boundary complete-candidate metrics are missing")
        required_metrics = {
            "recall1", "recall2", "recall3", "recall5", "recall10", "recall20",
            "mrr", "macro_query_auc", "micro_candidate_auc",
            "macro_query_auprc", "micro_candidate_auprc",
        }
        if required_metrics - set(metrics["student"]):
            raise RuntimeError("candidate-boundary metric panel is incomplete")
    if action_selection == "materialized_routed":
        registered = held.get("corrected_graph_registered_metrics", {})
        if set(registered) != {
            "official", "initialization", "student", "student_minus_official",
            "student_minus_initialization", "paired_top1_vs_official",
            "paired_top1_vs_initialization",
        }:
            raise RuntimeError("materialized E4 registered metric panel is incomplete")
        required_panels = {
            "retrieval", "near_subset", "micro_candidate",
            "massspecgym_10ppm_pooled_pairwise",
            "massspecgym_mh_10ppm_pooled_pairwise",
        }
        if required_panels - set(registered["student"]):
            raise RuntimeError("materialized E4 misses a registered metric family")
        required_retrieval = {
            "recall@1", "recall@2", "recall@3", "recall@5", "recall@10",
            "recall@20", "mrr", "mean_rank", "median_rank",
            "macro_query_auroc", "macro_query_auprc",
            "mean_positive_vs_best_negative_margin", "mean_top1_top2_gap",
            "mean_signed_top1_top2_gap",
        }
        if required_retrieval - set(registered["student"]["retrieval"]):
            raise RuntimeError("materialized E4 retrieval metric panel is incomplete")
    positive_weight = float(configuration.get("positive_stream_weight", 0.0))
    if positive_weight > 0:
        for key in (
            "real_cross_condition_positive_pairs_train_same_encoder",
            "positive_pair_selection_uses_model_outcome",
        ):
            if key not in contracts:
                raise RuntimeError(f"P/N/S contract missing: {key}")
        if contracts["real_cross_condition_positive_pairs_train_same_encoder"] is not True:
            raise RuntimeError("P-arm did not train the shared encoder")
        if contracts["positive_pair_selection_uses_model_outcome"] is not False:
            raise RuntimeError("P-arm pair selection used an outcome")
        for key in ("held_cross_condition_positive", "held_positive_clean"):
            if key not in held:
                raise RuntimeError(f"P/N/S held metric missing: {key}")
    print(
        "[validate_noise_final_e4a_direct_augmentation] PASS "
        f"policy={report['configuration']['policy']} "
        f"dR1={held['delta_recall1']:+.4f} C/I={held['corrected']}/{held['introduced']}"
    )


if __name__ == "__main__":
    main()
