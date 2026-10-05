"""Structural tests for the strict direct-v3 arm decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from summarize_noise_corrected_direct_v3_arms import (
    V6_HELD_METRIC_EVIDENCE_SCHEMA,
    _json_native, _metric_checks, sha256_file, state_sha256, summarize,
)
from train_noise_corrected_routed_direct import (
    REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_EXACT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_FLOAT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V6_INPUT_SHA256,
    REGISTERED_BEST_ACTION_V6_SCHEDULE_GEOMETRY,
)


def metrics(offset: float) -> dict[str, object]:
    retrieval = {
        "queries": 4,
        "mrr": 0.80 + offset,
        "mean_rank": 2.0 - offset,
        "median_rank": 1.0,
        "macro_query_auroc": 0.70 + offset,
        "macro_query_auprc": 0.60 + offset,
        "mean_positive_vs_best_negative_margin": 0.01 + offset,
        "mean_top1_top2_gap": 0.02 + offset,
        "mean_signed_top1_top2_gap": 0.01 + offset,
        **{f"recall@{cutoff}": 0.60 + offset for cutoff in (1, 2, 3, 5, 10, 20)},
    }
    return {
        "retrieval": retrieval,
        "near_subset": dict(retrieval),
        "micro_candidate": {"auroc": 0.75 + offset, "auprc": 0.65 + offset},
        "massspecgym_10ppm_pooled_pairwise": {
            "auroc": 0.76 + offset, "auprc": 0.66 + offset,
        },
        "massspecgym_mh_10ppm_pooled_pairwise": {
            "auroc": 0.77 + offset, "auprc": 0.67 + offset,
        },
    }


def table(candidate_rank: list[int]) -> pd.DataFrame:
    n_rows = len(candidate_rank)
    initial_rank = ([2, 2, 1, 1] * ((n_rows + 3) // 4))[:n_rows]
    query_index = np.arange(n_rows, dtype=np.int64)
    # Decouple formula clusters from the four-position rank pattern so every
    # small bootstrap cluster sees the same outcome mix, including in the
    # 18,333-row V6 fixture.
    query_formula = np.asarray(
        ["A", "B", "C", "D"], dtype=object,
    )[(query_index // 4) % 4]
    initial_rank_array = np.asarray(initial_rank, dtype=np.int64)
    candidate_rank_array = np.asarray(candidate_rank, dtype=np.int64)
    initial_correct = initial_rank_array == 1
    candidate_correct = candidate_rank_array == 1
    corrected = ~initial_correct & candidate_correct
    introduced = initial_correct & ~candidate_correct
    return pd.DataFrame({
        "query_index": query_index,
        "query_formula": query_formula,
        "near": np.ones(n_rows, dtype=bool),
        "initial_E8_rank": initial_rank_array,
        "candidate_rank": candidate_rank_array,
        "initial_E8_reciprocal_rank": 1.0 / initial_rank_array,
        "candidate_reciprocal_rank": 1.0 / candidate_rank_array,
        "initial_E8_macro_query_auc": np.where(initial_correct, 1.0, 0.5),
        "candidate_macro_query_auc": np.where(candidate_correct, 1.0, 0.5),
        "initial_E8_macro_query_auprc": 1.0 / initial_rank_array,
        "candidate_macro_query_auprc": 1.0 / candidate_rank_array,
        "initial_E8_positive_vs_best_negative_margin": np.where(
            initial_correct, 0.1, -0.1,
        ),
        "candidate_positive_vs_best_negative_margin": np.where(
            candidate_correct, 0.1, -0.1,
        ),
        "initial_E8_top1_top2_gap": np.full(n_rows, 0.1),
        "candidate_top1_top2_gap": np.full(n_rows, 0.1),
        "initial_E8_signed_top1_top2_gap": np.where(initial_correct, 0.1, -0.1),
        "candidate_signed_top1_top2_gap": np.where(candidate_correct, 0.1, -0.1),
        "corrected": corrected,
        "introduced": introduced,
        "risk_net": corrected.astype(np.int8) - introduced.astype(np.int8),
    })


def _evidence_arrays(n_rows: int) -> dict[str, np.ndarray]:
    query_index = np.arange(n_rows, dtype=np.int64)
    molecule_label = np.tile(np.asarray([1, 0, 0], dtype=np.uint8), n_rows)
    pair_label = np.tile(np.asarray([1, 0, 0, 0], dtype=np.uint8), n_rows)
    pair_is_mh = np.repeat(query_index % 2 == 0, 4).astype(np.bool_)
    initial_molecule = np.tile(
        np.asarray([0.60, 0.80, 0.20], dtype=np.float32), n_rows,
    )
    candidate_molecule = np.tile(
        np.asarray([0.90, 0.80, 0.20], dtype=np.float32), n_rows,
    )
    initial_pair = np.tile(
        np.asarray([0.60, 0.80, 0.40, 0.20], dtype=np.float32), n_rows,
    )
    candidate_pair = np.tile(
        np.asarray([0.90, 0.80, 0.40, 0.20], dtype=np.float32), n_rows,
    )
    return {
        "schema_version": np.asarray(V6_HELD_METRIC_EVIDENCE_SCHEMA),
        "query_index": query_index,
        "molecule_label": molecule_label,
        "pair_label": pair_label,
        "pair_is_mh": pair_is_mh,
        "official_molecule_score": initial_molecule,
        "official_pair_score": initial_pair,
        "initial_E8_molecule_score": initial_molecule.copy(),
        "initial_E8_pair_score": initial_pair.copy(),
        "candidate_molecule_score": candidate_molecule,
        "candidate_pair_score": candidate_pair,
    }


def _evidence_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    if mask is not None:
        labels = labels[mask]
        scores = scores[mask]
    return {
        "rows": int(len(labels)),
        "positive": int(np.sum(labels == 1)),
        "negative": int(np.sum(labels == 0)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def metric_panel(
    frame: pd.DataFrame,
    prefix: str,
    evidence: dict[str, np.ndarray],
    evidence_panel: str,
) -> dict[str, object]:
    rank = frame[f"{prefix}_rank"].to_numpy(dtype=np.int64)

    def summary(mask: np.ndarray) -> dict[str, float | int]:
        selected = frame.loc[mask]
        selected_rank = selected[f"{prefix}_rank"].to_numpy(dtype=np.int64)
        output: dict[str, float | int] = {
            "queries": int(len(selected)),
            "mrr": float(np.mean(1.0 / selected_rank)),
            "mean_rank": float(np.mean(selected_rank)),
            "median_rank": float(np.median(selected_rank)),
            "macro_query_auroc": float(selected[f"{prefix}_macro_query_auc"].mean()),
            "macro_query_auprc": float(selected[f"{prefix}_macro_query_auprc"].mean()),
            "mean_positive_vs_best_negative_margin": float(
                selected[f"{prefix}_positive_vs_best_negative_margin"].mean()
            ),
            "mean_top1_top2_gap": float(selected[f"{prefix}_top1_top2_gap"].mean()),
            "mean_signed_top1_top2_gap": float(
                selected[f"{prefix}_signed_top1_top2_gap"].mean()
            ),
        }
        output.update({
            f"recall@{cutoff}": float(np.mean(selected_rank <= cutoff))
            for cutoff in (1, 2, 3, 5, 10, 20)
        })
        return output

    all_rows = np.ones(len(frame), dtype=bool)
    near = frame["near"].to_numpy(dtype=bool)
    molecule = _evidence_metric(
        evidence["molecule_label"], evidence[f"{evidence_panel}_molecule_score"],
    )
    pair = _evidence_metric(
        evidence["pair_label"], evidence[f"{evidence_panel}_pair_score"],
    )
    mh_pair = _evidence_metric(
        evidence["pair_label"], evidence[f"{evidence_panel}_pair_score"],
        evidence["pair_is_mh"],
    )
    return {
        "retrieval": summary(all_rows),
        "near_subset": summary(near),
        "micro_candidate": {
            "molecules": molecule["rows"],
            "auroc": molecule["auroc"],
            "auprc": molecule["auprc"],
        },
        "massspecgym_10ppm_pooled_pairwise": {
            "spectrum_pairs": pair["rows"],
            "positive_pairs": pair["positive"],
            "negative_pairs": pair["negative"],
            "auroc": pair["auroc"],
            "auprc": pair["auprc"],
        },
        "massspecgym_mh_10ppm_pooled_pairwise": {
            "spectrum_pairs": mh_pair["rows"],
            "positive_pairs": mh_pair["positive"],
            "negative_pairs": mh_pair["negative"],
            "auroc": mh_pair["auroc"],
            "auprc": mh_pair["auprc"],
        },
    }


def test_numpy_scalar_summary_transport_is_strict_json() -> None:
    payload = {
        "schedule_and_signal_checks": {
            "finite": np.bool_(True),
            "count": np.int64(14_032),
            "fraction": np.float32(0.25),
        },
        "nested": [np.asarray([True, False], dtype=np.bool_)],
    }
    native = _json_native(payload)
    assert native["schedule_and_signal_checks"]["finite"] is True
    assert type(native["schedule_and_signal_checks"]["count"]) is int
    assert type(native["schedule_and_signal_checks"]["fraction"]) is float
    assert native["nested"] == [[True, False]]
    json.dumps(native)


def write_arm(
    root: Path,
    arm: str,
    ranks: list[int],
    *,
    direct_contract: str = "v3",
    materialize_registered_v6_rows: bool = True,
) -> None:
    root.mkdir()
    is_best_action = direct_contract in {
        "best_action_v6", "best_action_v6_restored",
        "best_action_v7_corrective_restored",
    }
    is_restored = direct_contract in {
        "best_action_v6_restored", "best_action_v7_corrective_restored",
    }
    is_corrective_restored = (
        direct_contract == "best_action_v7_corrective_restored"
    )
    training_seed = 20260911 if is_restored else (20260908 if is_best_action else 7)
    if is_best_action and materialize_registered_v6_rows:
        ranks = (ranks * ((18_333 + len(ranks) - 1) // len(ranks)))[:18_333]
    frame = table(ranks)
    evidence = _evidence_arrays(len(frame))
    checkpoint = root / "final_shared_encoder.pt"
    model_state = {"weight": torch.tensor([float(len(arm)), float(sum(ranks))])}
    torch.save({
        "status": "noise_corrected_routed_direct_shared_encoder",
        "model_state": model_state,
        "outer_fold": 0,
        "arm": arm,
        "inference_clean_only": True,
        "P2b_used": False,
        "corrective_objective_mode": "v3_direct",
        "direct_contract": direct_contract,
    }, checkpoint)
    initial = metric_panel(frame, "initial_E8", evidence, "initial_E8")
    candidate = metric_panel(frame, "candidate", evidence, "candidate")
    official = metric_panel(frame, "initial_E8", evidence, "official")
    initial_rank = frame["initial_E8_rank"].to_numpy(dtype=np.int64)
    candidate_rank = frame["candidate_rank"].to_numpy(dtype=np.int64)
    near = frame["near"].to_numpy(dtype=bool)
    corrected = int(np.sum((initial_rank > 1) & (candidate_rank == 1)))
    introduced = int(np.sum((initial_rank == 1) & (candidate_rank > 1)))
    near_corrected = int(np.sum(near & (initial_rank > 1) & (candidate_rank == 1)))
    near_introduced = int(np.sum(near & (initial_rank == 1) & (candidate_rank > 1)))
    held_rows = len(frame)
    graph_scope = (
        {
            "all_queries": 83619,
            "outer_train_queries": 65286,
            "outer_held_queries": 18333,
            "evaluated_outer_held_queries": 18333,
        }
        if is_best_action
        else {
            "all_queries": 2 * held_rows,
            "outer_train_queries": held_rows,
            "outer_held_queries": held_rows,
            "evaluated_outer_held_queries": held_rows,
        }
    )
    table_path = root / "held_per_query.csv.gz"
    frame.to_csv(table_path, index=False, compression="gzip")
    configuration: dict[str, object] = {
        "direct_contract": direct_contract,
        "seed": training_seed,
        "target_dense_corrective_to_risk_ratio": 1.0,
        "minimum_optimizer_action_alignment_p10": 0.05,
        "minimum_optimizer_action_attributable_fraction_p10": 0.10,
        "minimum_corrective_direction_retention_p10": 0.10,
    }
    if is_best_action:
        if is_corrective_restored:
            configuration = {
                **REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_EXACT_CONFIGURATION,
                **REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_FLOAT_CONFIGURATION,
            }
        elif is_restored:
            configuration = {
                **REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION,
                **REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION,
            }
        else:
            configuration = {
                **REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION,
                **REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION,
            }
        configuration["target_dense_corrective_to_risk_ratio"] = configuration[
            "target_corrective_to_risk_ratio"
        ]
    decision = {
        "status": "noise_corrected_routed_direct_complete",
        "formal": True,
        "arm": arm,
        "corrective_objective_mode": "v3_direct",
        "direct_contract": direct_contract,
        "outer_formula_fold": 0,
        "graph_scope": graph_scope,
        "configuration": configuration,
        "gradient_calibration": {
            "effective_branch_scale": {
                "transfer": 1.0,
                "payload": 0.25,
                "consistency": 0.25,
                "robust": 0.10,
                "harmful": 0.25,
            },
            "effective_action_to_risk_scale": 1.0,
            "effective_global_gradient_scale": 0.01,
            "effective_dense_corrective_to_risk_gradient_ratio": 1.0,
            "action_to_risk_scale_hit_cap": False,
            "branch_scale_hit_cap": {
                "payload": False,
                "consistency": False,
                "robust": False,
                "harmful": False,
            },
            "dense_corrective_to_risk_target_exactly_reached": True,
            "internal_action_combination_gate_passed": True,
            "corrective_direction_preserved_through_inner_semantic_projection": True,
            "corrective_robust_harmful_protective_gradients_separately_calibrated": True,
            "corrective_semantic_edge_fraction_median": {
                "active_transfer_fraction": (
                    0.0 if arm == "shuffled_action_control" else 0.10
                ),
            },
            "active_transfer_fraction_observed_gate_passed": (
                arm != "shuffled_action_control"
            ),
            "active_transfer_fraction_gate_applicable": (
                arm != "shuffled_action_control"
            ),
            "active_transfer_fraction_gate_passed": (
                None if arm == "shuffled_action_control" else True
            ),
            "corrective_mechanism_active_fraction": {
                "N": 0.10, "P": 0.10, "A4": 0.10,
            },
            "mechanism_active_transfer_gate_observed_passed": (
                arm != "shuffled_action_control"
            ),
            "mechanism_active_transfer_gate_applicable": (
                arm != "shuffled_action_control"
            ),
            "mechanism_active_transfer_gate_passed": (
                None if arm == "shuffled_action_control" else True
            ),
            "formula_stratified_without_replacement": True,
            "mechanism_prioritized_formula_diverse_prefix": True,
            "corrective_scale_uses_dense_corrective_branches_only": True,
            "sparse_auxiliary_branches_excluded_from_corrective_scale_denominator": True,
            "action_calibration_caps_not_silently_truncated": True,
            "calibration_mechanism_coverage": {
                panel: {
                    "available": ["N", "P", "A4"],
                    "selected": ["N", "P", "A4"],
                    "all_available_present": True,
                }
                for panel in ("corrective", "robust", "harmful")
            },
            "simultaneous_corrective_robust_harmful_risk_graph_retention": False,
        },
        "schedule": {
            "corrective_recycle_full_dose": True,
            "all_unique_action_panels_covered_before_recycling": True,
            "actual_scheduled_corrective_mechanism_epoch_mass_equalized": True,
            "robust_harmful_single_exposure_bounded_scale": True,
            "dense_auxiliary_epoch_mass_capped_at_one_batch_per_step": True,
            "maximum_auxiliary_microbatches_per_step": 4,
            "configured_maximum_auxiliary_microbatches_per_step": 4,
            "robust_harmful_evenly_interleaved_across_epoch": True,
            "robust_harmful_avoidable_step_overlap": False,
            "sparse_auxiliary_global_duty_compensation_applied": False,
            "partial_batch_query_mass_scaled_to_registered_size": True,
            "protective_microbatches_have_equal_epoch_weight": True,
            "clean_control_skips_no_gradient_action_forwards": True,
            "active_arm_epoch_transfer_edge_strata_recorded": True,
            "maximum_protective_microbatches_per_step": 4,
            "minimum_corrective_action_exposure_per_epoch": (
                4 if is_best_action else 2
            ),
            "maximum_corrective_action_exposure_per_epoch": (
                4 if is_best_action else 2
            ),
            "simultaneous_corrective_robust_harmful_graph_retention": False,
            "corrective_direction_preserved_through_inner_semantic_projection": True,
            "corrective_robust_harmful_protective_gradients_separately_recorded": True,
            "each_action_branch_vs_protective_cosine_recorded": True,
            **({
                "schedule_geometry": dict(
                    REGISTERED_BEST_ACTION_V6_SCHEDULE_GEOMETRY
                ),
                "schedule_geometry_by_epoch": [
                    dict(REGISTERED_BEST_ACTION_V6_SCHEDULE_GEOMETRY)
                    for _ in range(4)
                ],
                "schedule_geometry_matches_pre_model_preflight": True,
                "cap_safe_corrective_repartition_used": True,
                "cap_safe_corrective_repartition_preserved_query_order_and_coverage": True,
                "original_corrective_batches": 871,
                "cap_safe_corrective_batches": 877,
                "corrective_batches_added_by_cap_safe_repartition": 6,
                "required_optimizer_steps": 3508,
                "effective_corrective_recycle_factor": 4.0,
                "configured_maximum_corrective_recycle_factor": 4.0,
            } if is_best_action else {}),
        },
        "signal_transmission": {
            "passed": True,
            "clip_gate_passed": True,
            "optimizer_action_alignment_gate_passed": True,
            "optimizer_action_alignment_steps": (
                14_032 if is_restored and arm != "clean_control" else 64
            ),
            "optimizer_action_alignment_sampling": (
                "every_action_active_optimizer_step"
                if is_restored and arm != "clean_control"
                else "uniform_epoch_step_positions_including_endpoints"
            ),
            "optimizer_action_attributable_update_fraction_p10": 0.25,
            "optimizer_action_attributable_update_fraction_gate_passed": True,
            "optimizer_action_attributable_alignment_p10": 0.50,
            "optimizer_action_attributable_alignment_gate_passed": True,
            "corrective_direction_retention_after_risk_and_clip_p10": 0.50,
            "corrective_direction_retention_gate_passed": True,
            "optimizer_action_attributable_corrective_alignment_p10": 0.50,
            "optimizer_action_attributable_corrective_alignment_gate_passed": True,
            "legacy_90pct_end_to_end_loss_reproduced": False,
            "optimizer_counterfactual_virtual_step_max_relative_error": 1e-7,
            "optimizer_counterfactual_virtual_step_gate_passed": True,
            "optimizer_counterfactual_attribution_steps": (
                14_032 if is_restored and arm != "clean_control" else 64
            ),
            "optimizer_counterfactual_attribution_sampling": (
                "every_action_active_optimizer_step"
                if is_restored and arm != "clean_control"
                else "uniform_epoch_step_positions_including_endpoints"
            ),
            "parameter_group_action_signal": {
                name: {
                    "observations": (
                        14_032 if is_restored and arm != "clean_control" else 64
                    ),
                    "action_gradient_reaches_group_on_every_audit_step": True,
                    "pcgrad_clip_action_retention_p10": 0.50,
                    "optimizer_action_alignment_p10": 0.50,
                    "optimizer_action_attributable_update_fraction_p10": 0.25,
                    "optimizer_action_attributable_alignment_p10": 0.50,
                    "corrective_direction_retention_after_risk_and_clip_p10": 0.50,
                    "optimizer_action_attributable_corrective_alignment_p10": 0.50,
                    "protective_gradient_reaches_every_parameter_on_every_audit_step": True,
                    "gate_passed": True,
                }
                for name in ("head", "backbone")
            },
            "parameter_group_action_signal_gate_passed": True,
            "parameter_group_action_signal_sampling": (
                "every_action_active_optimizer_step"
                if is_restored and arm != "clean_control"
                else "uniform_epoch_step_positions_including_endpoints"
            ),
            "optimizer_update_restoration_materialized_every_active_step": bool(
                is_restored and arm != "clean_control"
            ),
            "optimizer_update_restoration_clean_control_noop": bool(
                is_restored and arm == "clean_control"
            ),
            "optimizer_update_restoration_minimum_observed_group_risk_retention": 0.95,
            "optimizer_update_restoration_all_group_targets_reached_fraction": 0.95,
            "optimizer_update_restoration_target_coverage_gate_passed": True,
            "optimizer_update_restoration_scope": (
                "corrective_only" if is_corrective_restored else "composite_action"
            ),
            "optimizer_update_restoration_corrective_only": is_corrective_restored,
            "optimizer_update_restoration_noncorrective_baseline": (
                "protective_plus_projected_robust_harmful"
                if is_corrective_restored else "protective_only"
            ),
            "protective_gradient_reaches_every_parameter_on_every_active_step": (
                is_corrective_restored and arm != "clean_control"
            ),
            "restored_adamw_first_moment_reconciled_every_active_step": (
                is_corrective_restored and arm != "clean_control"
            ),
            "restored_adamw_first_moment_max_reconstruction_relative_error": (
                1e-8 if is_corrective_restored else 0.0
            ),
            "restored_adamw_max_materialized_update_relative_error": (
                1e-4 if is_corrective_restored else 0.0
            ),
            "restored_adamw_max_fp32_parameter_replay_relative_error": (
                4e-4 if is_corrective_restored else 0.0
            ),
            "restored_adamw_reconstruction_target": (
                "realized_materialized_parameter_displacement"
                if is_corrective_restored else "not_applicable"
            ),
            "restored_adamw_verification_arithmetic": (
                "stable_decay_plus_adaptive_displacement"
                if is_corrective_restored else "not_applicable"
            ),
            "restored_adamw_second_moment_source": (
                "actual_combined_gradient" if is_corrective_restored else "unmodified"
            ),
            "signal_gate_failure": False,
        },
        "action_control": ({
            "cross_query_fraction": 1.0,
            "strategy": (
                "supervision_source_family_exact_recipe_matched_cross_query_cyclic_shuffle"
            ),
            "dose_and_action_semantics_preserved": True,
            "family_semantics_preserved": True,
            "donors_restricted_to_optimizer_admitted_panel": True,
            "all_nonfallback_donors_inside_input_panel": True,
            "donor_tensor_index_sha256": "donor-map",
        } if arm == "shuffled_action_control" else {}),
        "calibration_action_bank": {
            "policy": "arm_specific_training_action_bank",
            "arm_invariant": arm != "shuffled_action_control",
            "training_action_bank_mutated": False,
            "targeted_action_bank_sha256": "targeted-bank",
            "control_action_bank_sha256": "control-bank",
            "training_action_bank_sha256": (
                "shuffled-bank"
                if arm == "shuffled_action_control" else "targeted-bank"
            ),
            "calibration_action_bank_sha256": (
                "shuffled-bank"
                if arm == "shuffled_action_control" else "targeted-bank"
            ),
            "calibration_occurs_after_admission_and_shuffle": True,
        },
        "admitted_action_bank": {
            "strategy": "admission_then_exact_tensor_subset_then_reindex_before_shuffle",
            "admitted_rows": 297362,
            "admitted_rows_by_supervision": {
                "corrective": 32114,
                "harmful": 85959,
                "robust": 179289,
            },
            "admitted_action_ids_sha256": "admitted-ids",
            "admitted_semantic_boundary_sha256": "admitted-boundary",
            "shuffle_donor_pool_equals_optimizer_admitted_panel": True,
            "rejected_corrective_rows_can_be_shuffle_donors": False,
        },
        "action_forward_memory_contract": {
            "gate_passed": True,
            "queries_split_across_forwards": False,
            "maximum_spectra_per_action_forward": 512,
            "planned_worst_case_spectra_per_forward": 350,
        },
        "corrective_admission": {
            "mode": "strict_top1",
            "margin_floor": 5e-6,
            "candidate_corrective_rows": 12,
            "strict_top1_rows_before_margin_floor": 10,
            "selected_rows": 8,
            "selected_queries": 4,
            "all_selected_actions_preserved": True,
            "one_best_query_compression_used": False,
        },
        "corrective_query_materialization": {
            "selected_action_rows": 8,
            "materialized_action_rows": 8,
            "selected_queries": 4,
            "materialized_query_complete_examples": 4,
            "exact_action_id_set_preserved": True,
            "one_boundary_example_per_query": True,
        },
        "action_conditioned_boundary": {
            "enabled": True,
            **{
                panel: {
                    "all_selected_action_control_boundaries_retained": True,
                    "routed_hard_rows_never_truncated_within_molecule": True,
                    "candidate_boundary_cap_truncation": False,
                }
                for panel in ("corrective", "robust", "harmful")
            },
        },
        "mechanism_balance": {
            "query_or_action_dropped_for_balance": False,
            "coefficient_cap_passed": True,
            "maximum_allowed_coefficient": 16.0,
            "strategy": {
                "corrective": "global_identity_weighted_inverse_mechanism_incidence",
                "robust": "query_local_mechanism_mean",
                "harmful": "query_local_mechanism_mean",
            },
            **{
                panel: {
                    "balanced": True,
                    "maximum_mechanism_coefficient": 1.0,
                    "mechanisms": ["N", "P", "A4"],
                    "effective_epoch_mass": {"N": 1.0, "P": 1.0, "A4": 1.0},
                }
                for panel in ("corrective", "robust", "harmful")
            },
        },
        "contracts": {
            "action_and_control_candidate_switch_molecules_retained": True,
            "action_control_hard_rows_survive_epoch_refresh": True,
            "clean_action_control_topk_edge_union": True,
            "formula_stratified_gradient_calibration": True,
            "E8_symmetric_live_action_consistency_restored": True,
            "mature_E8_gradient_clip_restored": True,
            "bounded_memory_sequential_action_branch_backprop": True,
            "registered_formal_v3_configuration_verified": (
                direct_contract == "v3"
            ),
            "registered_formal_best_action_v6_configuration_verified": (
                is_best_action
            ),
            "registered_formal_best_action_v6_restored_configuration_verified": (
                is_restored and not is_corrective_restored
            ),
            "registered_formal_best_action_v7_corrective_restored_configuration_verified": (
                is_corrective_restored
            ),
            "all_causal_arms_calibrated_on_same_targeted_action_bank": False,
            "each_arm_calibrated_on_actual_post_shuffle_training_bank": True,
            "action_admission_precedes_tensor_subset_reindex_and_shuffle": True,
            "causal_arm_training_action_bank_hash_contract": True,
            "strict_top1_corrective_admission": (
                is_best_action
            ),
            "selected_corrective_action_ids_exactly_preserved_in_query_blocks": True,
            "formal_query_truncation_disabled": True,
            "source_manifest_exact_graph_alignment": True,
            "route_and_training_formula_fold_seed_exactly_aligned": True,
            "shuffled_control_matches_supervision_source_family_and_exact_recipe": True,
            "control_semantics_explicit_and_separate": True,
            "corrective_N_P_A4_global_epoch_mass_equalized_across_queries": True,
            "P_repeated_generation_source_does_not_multiply_family_dose": True,
            "selector_balances_source_then_family_before_recipe_multiplicity": True,
            "robust_harmful_use_query_local_not_sparse_global_equalization": True,
            "sparse_auxiliary_batches_evenly_interleaved": True,
            "dense_auxiliary_panels_cannot_multiply_calibrated_step_mass": True,
            "auxiliary_projected_against_corrective_before_fullgraph_risk": True,
            "corrective_direction_preserved_through_inner_semantic_projection": True,
            "semantic_action_branches_have_separate_gradient_ledgers": True,
            "global_mechanism_coefficient_cap_passed": True,
            "action_calibration_caps_not_silently_truncated": True,
            "clean_control_skips_no_gradient_action_forwards": True,
            "corrective_scale_uses_dense_corrective_branches_only": True,
            "sparse_auxiliary_branches_excluded_from_corrective_scale_denominator": True,
            "training_epoch_transfer_edge_strata_are_pooled_not_batch_averaged": True,
            "optimizer_counterfactual_action_attribution_measured": (
                arm != "clean_control"
            ),
            "optimizer_parameter_clones_limited_to_registered_audit_steps": (
                not is_restored
            ),
            "optimizer_update_restoration_materialized_every_active_step": bool(
                is_restored
            ),
            "head_and_backbone_action_signal_measured_separately": True,
            "corrective_direction_traced_through_risk_clip_and_adamw": True,
            "corrective_only_optimizer_residual_restored": is_corrective_restored,
            "restored_adamw_first_moment_reconciled": is_corrective_restored,
            "schedule_geometry_validated_before_model_load": True,
            "cap_safe_corrective_repartition_preserves_query_action_panel": True,
            "best_action_v6_exact_877_batch_schedule_verified": (
                is_best_action
            ),
            "near_corrected_introduced_risk_reported": True,
        },
        "evaluation": {"formal_held_graph": {
            "official": official,
            "initial_E8": initial,
            "candidate": candidate,
            "candidate_vs_initial_E8": {
                "corrected": corrected,
                "introduced": introduced,
                "risk_net_lambda2": corrected - 2 * introduced,
                "near": {
                    "corrected": near_corrected,
                    "introduced": near_introduced,
                    "risk_net_lambda2": near_corrected - 2 * near_introduced,
                },
            },
        }},
        "provenance": {
            "spectrum_data_sha256": (
                "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f"
            ),
            "architecture_checkpoint_sha256": (
                "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2"
            ),
            "initial_student_decision_sha256": (
                "0a7de098701e9ef5f6c3fa2ee62fcc2e1b301dab44878286bd0b61bc31e3e9d9"
            ),
            "candidate_graph_sha256": (
                REGISTERED_BEST_ACTION_V6_INPUT_SHA256["candidate_graph_sha256"]
                if is_best_action else "graph"
            ),
            "graph_report_sha256": (
                REGISTERED_BEST_ACTION_V6_INPUT_SHA256["graph_report_sha256"]
                if is_best_action else "graph-report"
            ),
            "source_manifest_sha256": "source-manifest",
            "routed_ledger_report_sha256": (
                REGISTERED_BEST_ACTION_V6_INPUT_SHA256[
                    "routed_ledger_report_sha256"
                ] if is_best_action else "ledger-report"
            ),
            "training_actions_sha256": (
                REGISTERED_BEST_ACTION_V6_INPUT_SHA256["training_actions_sha256"]
                if is_best_action else "actions"
            ),
            "action_spectra_sha256": (
                REGISTERED_BEST_ACTION_V6_INPUT_SHA256["action_spectra_sha256"]
                if is_best_action else "spectra"
            ),
            "initial_student_checkpoint_sha256": (
                REGISTERED_BEST_ACTION_V6_INPUT_SHA256[
                    "initial_student_checkpoint_sha256"
                ] if is_best_action else "initial"
            ),
            "official_checkpoint_sha256": (
                REGISTERED_BEST_ACTION_V6_INPUT_SHA256["official_checkpoint_sha256"]
                if is_best_action else "official"
            ),
            "v3_core_sha256": "v3-core",
            "v3_transfer_objective_sha256": "v3-transfer-objective",
            "v3_action_expansion_sha256": "v3-action-expansion",
            "optimizer_update_arbitration_sha256": "optimizer-restoration",
            "noise_v3_core_sha256": "noise-v3-core",
            "v3_action_router_sha256": "v3-router",
            "v3_shuffled_control_sha256": "v3-shuffle",
            "best_action_selector_sha256": "best-selector",
            "v3_action_panel_sha256": "v3-panel",
            "fullgraph_evaluator_sha256": "evaluator",
            "script_sha256": "trainer",
            "final_shared_encoder_sha256": sha256_file(checkpoint),
            "final_model_state_sha256": state_sha256(model_state),
            "held_per_query_sha256": sha256_file(table_path),
            "held_per_query_rows": held_rows,
        },
    }
    if is_best_action:
        evidence_path = root / "held_metric_evidence.npz"
        np.savez(evidence_path, **evidence)
        decision["provenance"]["held_metric_evidence_sha256"] = sha256_file(
            evidence_path
        )
        decision["provenance"]["held_metric_evidence_rows"] = {
            "held_queries": int(len(evidence["query_index"])),
            "molecules": int(len(evidence["molecule_label"])),
            "spectrum_pairs": int(len(evidence["pair_label"])),
            "mh_spectrum_pairs": int(np.sum(evidence["pair_is_mh"])),
        }
    (root / "decision.json").write_text(json.dumps(decision), encoding="utf-8")


def test_three_arm_summary_is_aligned_and_checks_every_metric_family() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_arm(root / "routed", "routed_direct", [1, 1, 1, 1])
        write_arm(root / "shuffled", "shuffled_action_control", [2, 1, 1, 1])
        write_arm(root / "clean", "clean_control", [2, 2, 1, 1])
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=200,
            formula_ci_familywise_hypotheses=4,
            seed=7,
        )
        report, paired = summarize(args)
        assert report["status"] == "noise_corrected_direct_v3_arm_summary_complete"
        assert report["training_seed"] == 7
        assert len(paired) == 4
        checks = report["candidate_vs_initial"]["metric_checks"]
        for required in (
            "recall@1", "recall@20", "mrr", "mean_rank",
            "macro_query_auroc", "macro_query_auprc",
            "mean_signed_top1_top2_gap",
            "micro_candidate.auroc", "massspecgym_10ppm_pooled_pairwise.auroc",
            "massspecgym_mh_10ppm_pooled_pairwise.auroc",
        ):
            assert required in checks and checks[required]
        assert report["promotion_checks"]["recall1_gain_at_least_threshold"]
        assert report["candidate_vs_initial"]["risk_checks"][
            "near_risk_net_lambda2_positive"
        ]
        schedule = report["schedule_and_signal_checks"]
        assert schedule["routed_semantic_transfer_gate_pass"]
        assert schedule["routed_each_mechanism_transfer_gate_pass"]
        assert schedule["shuffled_semantic_transfer_is_measured_not_forced"]
        assert schedule[
            "shuffled_each_mechanism_transfer_is_measured_not_forced"
        ]
        assert schedule[
            "control_semantics_and_global_mechanism_mass_are_explicit"
        ]
        assert schedule["all_available_mechanisms_enter_calibration_prefix"]
        assert schedule["auxiliary_accumulation_at_most_registered_four"]
        assert schedule["inner_semantic_projection_preserves_corrective"]


def test_best_action_v6_summary_requires_post_shuffle_actual_bank_calibration() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_arm(
            root / "routed", "routed_direct", [1, 1, 1, 1],
            direct_contract="best_action_v6",
        )
        write_arm(
            root / "shuffled", "shuffled_action_control", [2, 1, 1, 1],
            direct_contract="best_action_v6",
        )
        write_arm(
            root / "clean", "clean_control", [2, 2, 1, 1],
            direct_contract="best_action_v6",
        )
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=10_000,
            formula_ci_familywise_hypotheses=4,
            seed=20260908,
        )
        report, _ = summarize(args)
        assert report["direct_contract"] == "best_action_v6"
        assert report["schedule_and_signal_checks"][
            "registered_formal_direct_configuration_verified"
        ]
        assert report["schedule_and_signal_checks"][
            "post_shuffle_actual_bank_calibration"
        ]
        assert report["schedule_and_signal_checks"][
            "strict_top1_multi_action_admission"
        ]
        assert report["schedule_and_signal_checks"][
            "best_action_v6_cap_safe_871_to_877_schedule_verified"
        ]
        assert report["single_seed_gate_pass"]
        assert report["promote"] is False
        assert report["promotion_status"] == "pending_multiseed"
        path = root / "routed" / "decision.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        decision["schedule"]["schedule_geometry"][
            "cap_safe_corrective_batches"
        ] = 876
        path.write_text(json.dumps(decision), encoding="utf-8")
        try:
            summarize(args)
        except RuntimeError as error:
            assert "cap_safe_corrective_batches" in str(error)
        else:
            raise AssertionError("v6 summary accepted stale 871-batch scheduling")


def test_best_action_v6_accepts_arm_specific_scale_but_rejects_budget_miss() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for directory, arm, ranks in (
            ("routed", "routed_direct", [1, 1, 1, 1]),
            ("shuffled", "shuffled_action_control", [2, 1, 1, 1]),
            ("clean", "clean_control", [2, 2, 1, 1]),
        ):
            write_arm(
                root / directory, arm, ranks,
                direct_contract="best_action_v6",
            )
        path = root / "shuffled" / "decision.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        decision["gradient_calibration"]["effective_global_gradient_scale"] = 0.02
        path.write_text(json.dumps(decision), encoding="utf-8")
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=10_000,
            formula_ci_familywise_hypotheses=4,
            seed=20260908,
        )
        report, _ = summarize(args)
        assert report["schedule_and_signal_checks"][
            "post_shuffle_actual_bank_calibration"
        ]
        decision = json.loads(path.read_text(encoding="utf-8"))
        decision["gradient_calibration"][
            "effective_dense_corrective_to_risk_gradient_ratio"
        ] = 0.5
        path.write_text(json.dumps(decision), encoding="utf-8")
        try:
            summarize(args)
        except RuntimeError as error:
            assert "registered realized gradient budget" in str(error)
        else:
            raise AssertionError("v6 summary accepted a post-shuffle budget miss")


def test_best_action_v6_rejects_unregistered_data_or_architecture() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for directory, arm, ranks in (
            ("routed", "routed_direct", [1, 1, 1, 1]),
            ("shuffled", "shuffled_action_control", [2, 1, 1, 1]),
            ("clean", "clean_control", [2, 2, 1, 1]),
        ):
            write_arm(
                root / directory, arm, ranks,
                direct_contract="best_action_v6",
            )
        path = root / "clean" / "decision.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        decision["provenance"]["architecture_checkpoint_sha256"] = "drifted"
        path.write_text(json.dumps(decision), encoding="utf-8")
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=10_000,
            formula_ci_familywise_hypotheses=4,
            seed=20260908,
        )
        try:
            summarize(args)
        except RuntimeError as error:
            assert "provenance" in str(error) or "contract failed" in str(error)
        else:
            raise AssertionError("v6 summary accepted an unregistered architecture")


def test_best_action_v6_summary_rejects_an_unregistered_seed() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for directory, arm, ranks in (
            ("routed", "routed_direct", [1, 1, 1, 1]),
            ("shuffled", "shuffled_action_control", [2, 1, 1, 1]),
            ("clean", "clean_control", [2, 2, 1, 1]),
        ):
            write_arm(
                root / directory, arm, ranks,
                direct_contract="best_action_v6",
            )
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=10_000,
            formula_ci_familywise_hypotheses=4,
            seed=7,
        )
        try:
            summarize(args)
        except RuntimeError as error:
            assert "registered seed=20260908" in str(error)
        else:
            raise AssertionError("v6 summary accepted an unregistered seed")


def test_ceiling_recall_requires_exact_nonregression_not_impossible_improvement() -> None:
    initial = metrics(0.0)
    candidate = metrics(0.05)
    initial["retrieval"]["recall@20"] = 1.0
    initial["near_subset"]["recall@20"] = 1.0
    candidate["retrieval"]["recall@20"] = 1.0
    candidate["near_subset"]["recall@20"] = 1.0
    assert _metric_checks(initial, candidate)["recall@20"]
    candidate["retrieval"]["recall@20"] = 0.999
    assert not _metric_checks(initial, candidate)["recall@20"]


def test_summary_rejects_per_query_and_decision_recall_mismatch() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_arm(root / "routed", "routed_direct", [1, 1, 1, 1])
        write_arm(root / "shuffled", "shuffled_action_control", [2, 1, 1, 1])
        write_arm(root / "clean", "clean_control", [2, 2, 1, 1])
        path = root / "routed" / "decision.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        decision["evaluation"]["formal_held_graph"]["candidate"]["retrieval"][
            "recall@1"
        ] = 0.5
        path.write_text(json.dumps(decision), encoding="utf-8")
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=200,
            formula_ci_familywise_hypotheses=4,
            seed=7,
        )
        try:
            summarize(args)
        except RuntimeError as error:
            assert "candidate.retrieval.recall@1" in str(error)
        else:
            raise AssertionError("summary accepted a split held-result ledger")


def test_summary_blocks_end_to_end_ninety_percent_action_loss() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_arm(root / "routed", "routed_direct", [1, 1, 1, 1])
        write_arm(root / "shuffled", "shuffled_action_control", [2, 1, 1, 1])
        write_arm(root / "clean", "clean_control", [2, 2, 1, 1])
        path = root / "routed" / "decision.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        decision["signal_transmission"][
            "optimizer_action_attributable_update_fraction_p10"
        ] = 0.05
        decision["signal_transmission"][
            "optimizer_action_attributable_update_fraction_gate_passed"
        ] = False
        decision["signal_transmission"][
            "legacy_90pct_end_to_end_loss_reproduced"
        ] = True
        path.write_text(json.dumps(decision), encoding="utf-8")
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=200,
            formula_ci_familywise_hypotheses=4,
            seed=7,
        )
        report, _ = summarize(args)
        assert not report["schedule_and_signal_checks"][
            "active_arm_optimizer_counterfactual_attribution_valid"
        ]
        assert report["promote"] is False


def test_summary_blocks_backbone_only_ninety_percent_action_loss() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_arm(root / "routed", "routed_direct", [1, 1, 1, 1])
        write_arm(root / "shuffled", "shuffled_action_control", [2, 1, 1, 1])
        write_arm(root / "clean", "clean_control", [2, 2, 1, 1])
        path = root / "routed" / "decision.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        backbone = decision["signal_transmission"][
            "parameter_group_action_signal"
        ]["backbone"]
        backbone["optimizer_action_attributable_update_fraction_p10"] = 0.05
        backbone["gate_passed"] = False
        decision["signal_transmission"][
            "parameter_group_action_signal_gate_passed"
        ] = False
        path.write_text(json.dumps(decision), encoding="utf-8")
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=200,
            formula_ci_familywise_hypotheses=4,
            seed=7,
        )
        report, _ = summarize(args)
        assert not report["schedule_and_signal_checks"][
            "active_arm_optimizer_counterfactual_attribution_valid"
        ]
        assert report["promote"] is False


def test_summary_blocks_pre_risk_auxiliary_cancellation() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_arm(root / "routed", "routed_direct", [1, 1, 1, 1])
        write_arm(root / "shuffled", "shuffled_action_control", [2, 1, 1, 1])
        write_arm(root / "clean", "clean_control", [2, 2, 1, 1])
        path = root / "routed" / "decision.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        decision["gradient_calibration"][
            "corrective_direction_preserved_through_inner_semantic_projection"
        ] = False
        path.write_text(json.dumps(decision), encoding="utf-8")
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=200,
            formula_ci_familywise_hypotheses=4,
            seed=7,
        )
        report, _ = summarize(args)
        assert not report["schedule_and_signal_checks"][
            "inner_semantic_projection_preserves_corrective"
        ]
        assert report["promote"] is False


def test_summary_blocks_corrective_direction_loss_hidden_by_total_action_norm() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_arm(root / "routed", "routed_direct", [1, 1, 1, 1])
        write_arm(root / "shuffled", "shuffled_action_control", [2, 1, 1, 1])
        write_arm(root / "clean", "clean_control", [2, 2, 1, 1])
        path = root / "routed" / "decision.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        decision["signal_transmission"][
            "corrective_direction_retention_after_risk_and_clip_p10"
        ] = 0.05
        decision["signal_transmission"][
            "corrective_direction_retention_gate_passed"
        ] = False
        decision["signal_transmission"][
            "legacy_90pct_end_to_end_loss_reproduced"
        ] = True
        path.write_text(json.dumps(decision), encoding="utf-8")
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=200,
            formula_ci_familywise_hypotheses=4,
            seed=7,
        )
        report, _ = summarize(args)
        assert not report["schedule_and_signal_checks"][
            "active_arm_optimizer_counterfactual_attribution_valid"
        ]
        assert report["promote"] is False


def test_summary_rejects_tampered_final_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_arm(root / "routed", "routed_direct", [1, 1, 1, 1])
        write_arm(root / "shuffled", "shuffled_action_control", [2, 1, 1, 1])
        write_arm(root / "clean", "clean_control", [2, 2, 1, 1])
        with (root / "routed" / "final_shared_encoder.pt").open("ab") as stream:
            stream.write(b"tampered")
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=200,
            formula_ci_familywise_hypotheses=4,
            seed=7,
        )
        try:
            summarize(args)
        except RuntimeError as error:
            assert "checkpoint file hash differs" in str(error)
        else:
            raise AssertionError("summary accepted a tampered final checkpoint")


def test_best_action_v6_rejects_four_row_csv_spoofing_full_graph_counts() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for directory, arm, ranks in (
            ("routed", "routed_direct", [1, 1, 1, 1]),
            ("shuffled", "shuffled_action_control", [2, 1, 1, 1]),
            ("clean", "clean_control", [2, 2, 1, 1]),
        ):
            write_arm(
                root / directory, arm, ranks,
                direct_contract="best_action_v6",
                materialize_registered_v6_rows=False,
            )
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=10_000,
            formula_ci_familywise_hypotheses=4,
            seed=20260908,
        )
        try:
            summarize(args)
        except RuntimeError as error:
            assert "requires exactly 18333 actual held CSV rows" in str(error)
        else:
            raise AssertionError("v6 summary accepted four rows posing as 18,333")


def test_summary_rejects_same_recall1_but_different_rank_ledger() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_arm(root / "routed", "routed_direct", [1, 1, 1, 1])
        write_arm(root / "shuffled", "shuffled_action_control", [2, 1, 1, 1])
        write_arm(root / "clean", "clean_control", [2, 2, 1, 1])
        arm_root = root / "shuffled"
        table_path = arm_root / "held_per_query.csv.gz"
        frame = pd.read_csv(table_path)
        before_recall1 = float((frame["candidate_rank"] == 1).mean())
        frame.loc[0, "candidate_rank"] = 3
        frame.loc[0, "candidate_reciprocal_rank"] = 1.0 / 3.0
        frame.loc[0, "candidate_macro_query_auprc"] = 1.0 / 3.0
        frame.to_csv(table_path, index=False, compression="gzip")
        assert float((frame["candidate_rank"] == 1).mean()) == before_recall1
        decision_path = arm_root / "decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        # Give the attacker a valid file binding; the independent metric
        # reconstruction must still reject the stale decision metrics.
        decision["provenance"]["held_per_query_sha256"] = sha256_file(table_path)
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=200,
            formula_ci_familywise_hypotheses=4,
            seed=7,
        )
        try:
            summarize(args)
        except RuntimeError as error:
            assert "CSV/decision metric differs" in str(error)
            assert "candidate.retrieval" in str(error)
        else:
            raise AssertionError("summary accepted a same-Recall@1 split ledger")


def test_best_action_v6_rejects_registered_parameter_tampering() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for directory, arm, ranks in (
            ("routed", "routed_direct", [1, 1, 1, 1]),
            ("shuffled", "shuffled_action_control", [2, 1, 1, 1]),
            ("clean", "clean_control", [2, 2, 1, 1]),
        ):
            write_arm(
                root / directory, arm, ranks, direct_contract="best_action_v6",
            )
        base = {
            "routed_dir": root / "routed",
            "shuffled_dir": root / "shuffled",
            "clean_dir": root / "clean",
            "minimum_delta_recall1_pp": 4.0,
            "bootstrap_resamples": 10_000,
            "formula_ci_familywise_hypotheses": 4,
            "seed": 20260908,
        }
        for name, tampered in (
            ("minimum_delta_recall1_pp", 3.99),
            ("bootstrap_resamples", 9_999),
            ("formula_ci_familywise_hypotheses", 3),
        ):
            values = dict(base)
            values[name] = tampered
            try:
                summarize(argparse.Namespace(**values))
            except RuntimeError as error:
                assert f"registered {name}=" in str(error)
            else:
                raise AssertionError(f"v6 summary accepted tampered {name}")


def test_best_action_v6_requires_held_table_binding() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for directory, arm, ranks in (
            ("routed", "routed_direct", [1, 1, 1, 1]),
            ("shuffled", "shuffled_action_control", [2, 1, 1, 1]),
            ("clean", "clean_control", [2, 2, 1, 1]),
        ):
            write_arm(
                root / directory, arm, ranks, direct_contract="best_action_v6",
            )
        decision_path = root / "routed" / "decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["provenance"].pop("held_per_query_sha256")
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=10_000,
            formula_ci_familywise_hypotheses=4,
            seed=20260908,
        )
        try:
            summarize(args)
        except RuntimeError as error:
            assert "misses held-per-query hash/row provenance" in str(error)
        else:
            raise AssertionError("v6 summary accepted an unbound held table")


def test_summary_rejects_nonclosing_near_panel_count() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        write_arm(root / "routed", "routed_direct", [1, 1, 1, 1])
        write_arm(root / "shuffled", "shuffled_action_control", [2, 1, 1, 1])
        write_arm(root / "clean", "clean_control", [2, 2, 1, 1])
        decision_path = root / "clean" / "decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["evaluation"]["formal_held_graph"]["candidate"][
            "near_subset"
        ]["queries"] = 3
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=200,
            formula_ci_familywise_hypotheses=4,
            seed=7,
        )
        try:
            summarize(args)
        except RuntimeError as error:
            assert "near_subset query count does not close" in str(error)
        else:
            raise AssertionError("summary accepted a nonclosing near panel")


def test_best_action_v6_rejects_config_input_and_metric_evidence_spoofing() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        specifications = (
            ("routed", "routed_direct", [1, 1, 1, 1]),
            ("shuffled", "shuffled_action_control", [2, 1, 1, 1]),
            ("clean", "clean_control", [2, 2, 1, 1]),
        )
        for directory, arm, ranks in specifications:
            write_arm(
                root / directory, arm, ranks, direct_contract="best_action_v6",
            )
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=10_000,
            formula_ci_familywise_hypotheses=4,
            seed=20260908,
        )
        decision_paths = {
            directory: root / directory / "decision.json"
            for directory, _, _ in specifications
        }
        original = {
            name: path.read_text(encoding="utf-8")
            for name, path in decision_paths.items()
        }

        for name, path in decision_paths.items():
            decision = json.loads(original[name])
            decision["configuration"]["head_lr"] = 999.0
            path.write_text(json.dumps(decision), encoding="utf-8")
        try:
            summarize(args)
        except RuntimeError as error:
            assert "reported training configuration drifted" in str(error)
            assert "head_lr" in str(error)
        else:
            raise AssertionError("v6 summary accepted a uniformly tampered training config")
        for name, path in decision_paths.items():
            path.write_text(original[name], encoding="utf-8")

        for name, path in decision_paths.items():
            decision = json.loads(original[name])
            decision["provenance"]["candidate_graph_sha256"] = "unregistered-graph"
            path.write_text(json.dumps(decision), encoding="utf-8")
        try:
            summarize(args)
        except RuntimeError as error:
            assert "best-action v6 arm contract failed" in str(error)
        else:
            raise AssertionError("v6 summary accepted a uniformly unregistered graph")
        for name, path in decision_paths.items():
            path.write_text(original[name], encoding="utf-8")

        routed_path = decision_paths["routed"]
        decision = json.loads(original["routed"])
        decision["evaluation"]["formal_held_graph"]["candidate"][
            "micro_candidate"
        ]["auroc"] = 0.0
        routed_path.write_text(json.dumps(decision), encoding="utf-8")
        try:
            summarize(args)
        except RuntimeError as error:
            assert "candidate.micro_candidate.auroc" in str(error)
        else:
            raise AssertionError("v6 summary accepted a metric contradicted by evidence")


def test_best_action_v6_restored_requires_exhaustive_materialized_updates() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for directory, arm, ranks in (
            ("routed", "routed_direct", [1, 1, 1, 1]),
            ("shuffled", "shuffled_action_control", [2, 1, 1, 1]),
            ("clean", "clean_control", [2, 2, 1, 1]),
        ):
            write_arm(
                root / directory,
                arm,
                ranks,
                direct_contract="best_action_v6_restored",
            )
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=10_000,
            formula_ci_familywise_hypotheses=4,
            seed=20260911,
        )
        report, _ = summarize(args)
        assert report["direct_contract"] == "best_action_v6_restored"
        checks = report["schedule_and_signal_checks"]
        assert checks["optimizer_update_restoration_contract_valid"] is True
        assert checks["active_arm_optimizer_counterfactual_attribution_valid"] is True
        path = root / "routed" / "decision.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        decision["signal_transmission"]["signal_gate_failure"] = True
        path.write_text(json.dumps(decision), encoding="utf-8")
        failed, _ = summarize(args)
        assert failed["schedule_and_signal_checks"][
            "optimizer_update_restoration_contract_valid"
        ] is False
        assert failed["single_seed_gate_pass"] is False


def test_best_action_v7_requires_corrective_only_reconciled_restoration() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for directory, arm, ranks in (
            ("routed", "routed_direct", [1, 1, 1, 1]),
            ("shuffled", "shuffled_action_control", [2, 1, 1, 1]),
            ("clean", "clean_control", [2, 2, 1, 1]),
        ):
            write_arm(
                root / directory,
                arm,
                ranks,
                direct_contract="best_action_v7_corrective_restored",
            )
        args = argparse.Namespace(
            routed_dir=root / "routed",
            shuffled_dir=root / "shuffled",
            clean_dir=root / "clean",
            minimum_delta_recall1_pp=4.0,
            bootstrap_resamples=10_000,
            formula_ci_familywise_hypotheses=4,
            seed=20260911,
        )
        report, _ = summarize(args)
        assert report["direct_contract"] == "best_action_v7_corrective_restored"
        checks = report["schedule_and_signal_checks"]
        assert checks["optimizer_update_restoration_contract_valid"] is True

        path = root / "routed" / "decision.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        decision["signal_transmission"][
            "optimizer_update_restoration_all_group_targets_reached_fraction"
        ] = 0.89
        decision["signal_transmission"][
            "optimizer_update_restoration_target_coverage_gate_passed"
        ] = False
        path.write_text(json.dumps(decision), encoding="utf-8")
        failed_coverage, _ = summarize(args)
        assert failed_coverage["schedule_and_signal_checks"][
            "optimizer_update_restoration_contract_valid"
        ] is False
        assert failed_coverage["single_seed_gate_pass"] is False

        decision["signal_transmission"][
            "optimizer_update_restoration_all_group_targets_reached_fraction"
        ] = 0.95
        decision["signal_transmission"][
            "optimizer_update_restoration_target_coverage_gate_passed"
        ] = True
        decision["signal_transmission"][
            "optimizer_update_restoration_noncorrective_baseline"
        ] = "protective_only"
        path.write_text(json.dumps(decision), encoding="utf-8")
        failed, _ = summarize(args)
        assert failed["schedule_and_signal_checks"][
            "optimizer_update_restoration_contract_valid"
        ] is False
        assert failed["single_seed_gate_pass"] is False


def main() -> None:
    test_numpy_scalar_summary_transport_is_strict_json()
    test_three_arm_summary_is_aligned_and_checks_every_metric_family()
    test_best_action_v6_summary_requires_post_shuffle_actual_bank_calibration()
    test_best_action_v6_accepts_arm_specific_scale_but_rejects_budget_miss()
    test_best_action_v6_rejects_unregistered_data_or_architecture()
    test_best_action_v6_summary_rejects_an_unregistered_seed()
    test_ceiling_recall_requires_exact_nonregression_not_impossible_improvement()
    test_summary_rejects_per_query_and_decision_recall_mismatch()
    test_summary_blocks_end_to_end_ninety_percent_action_loss()
    test_summary_blocks_backbone_only_ninety_percent_action_loss()
    test_summary_blocks_pre_risk_auxiliary_cancellation()
    test_summary_blocks_corrective_direction_loss_hidden_by_total_action_norm()
    test_summary_rejects_tampered_final_checkpoint()
    test_best_action_v6_rejects_four_row_csv_spoofing_full_graph_counts()
    test_summary_rejects_same_recall1_but_different_rank_ledger()
    test_best_action_v6_rejects_registered_parameter_tampering()
    test_best_action_v6_requires_held_table_binding()
    test_summary_rejects_nonclosing_near_panel_count()
    test_best_action_v6_rejects_config_input_and_metric_evidence_spoofing()
    test_best_action_v6_restored_requires_exhaustive_materialized_updates()
    test_best_action_v7_requires_corrective_only_reconciled_restoration()
    print("[test_summarize_noise_corrected_direct_v3_arms] PASS tests=21")


if __name__ == "__main__":
    main()
