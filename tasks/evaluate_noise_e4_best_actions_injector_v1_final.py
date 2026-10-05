"""Evaluate best actions plus E4 loss and Injector V1 against stored E4.

This process is evaluation-only.  It never retrains the pure E4 comparator and
never feeds corrected-graph outcomes back into the shared encoder.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_noise_e4_faithful_v1 import nested_delta, paired_counts
from noise_corrected_fullgraph_evaluation import (
    full_metrics,
    held_metric_evidence,
    official_scores,
    paired_outcome_table,
    score_embedding_query_subset,
)
from noise_final_core import CandidateGraph, load_embedding_cache, sha256_file, stable_fold
from train_e1_identity import load_base_model, torch_load_compat
from train_noise_final_r2_shared_encoder import (
    SpectrumStore,
    encode_rows,
    formula_bootstrap_delta,
)


OFFICIAL_SHA256 = "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245"
GRAPH_SHA256 = "8a57bb3a9cccdf69a738f2b093bfad8f6fc4393b23f1342c7a035ac54ce58fa1"
SOURCE_MANIFEST_SHA256 = "504ce0a570ac1bc461e76cad81acb3ca3560e58b008ec2d8ff14036e944599f5"
EMBEDDING_CACHE_SHA256 = "18d7632adc67dcda5d650a5c0bc344d8db3878f9e1ff3f8d51a6d4918b96bbda"
DATA_SHA256 = "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f"
ARCHITECTURE_SHA256 = "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2"
HISTORICAL_E4_SHA256 = "8ee5b5aa604bda7f1e3cc99396773cebd10749e8820d02c3de048f2865c2509e"
HISTORICAL_E4_DECISION_SHA256 = "1d9144a93ca4e4162854cc67491db33612a8ace0e3457e1777ffb45ae5a125bd"
BEST_ACTION_REPORT_SHA256 = "246e7e871e6669fec9f2c62330a69bf9eb563a1b97b72b89732cecd682844349"
BEST_ACTIONS_SHA256 = "93f0785a69b5e323490a0b543059fa213fabb6ff848697831efba5b3fed667aa"
BEST_ACTION_SPECTRA_SHA256 = "6d57615aebbfb7bd6327bb7edb143c6837d2586221c61aa2ff45e833f5e1512a"

HYBRID_FORBIDDEN_CONTRACT = {
    # Outer-train routing provenance is allowed.  A teacher embedding or
    # teacher margin target entering the optimization objective is not.
    "teacher": "outer_train_action_routing_only_no_teacher_target",
    "P2b": "forbidden",
    "P3_consumed": False,
    "pure_e4_control_retrained": False,
    "action_injector_v1_action_selection_or_tensor_mutation": False,
    "action_injector_v1_historical_e4_loss_mutation": False,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-checkpoint", type=Path, required=True)
    parser.add_argument("--hybrid-decision", type=Path, required=True)
    parser.add_argument("--shuffled-checkpoint", type=Path, required=True)
    parser.add_argument("--shuffled-decision", type=Path, required=True)
    parser.add_argument("--historical-e4-checkpoint", type=Path, required=True)
    parser.add_argument("--historical-e4-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _exact_configuration(
    configuration: dict, *, hybrid: bool, hybrid_arm: str = "targeted",
) -> None:
    exact = {
        "policy": "curriculum", "action_scope": "all", "outer_fold": 0,
        "formula_fold_seed": 20260825, "epochs": 4, "batch_actions": 4,
        "views_per_identity": 4, "positive_spectra": 4,
        "negative_molecules": 8, "unfreeze_blocks": 1, "amp": False,
        "smoke": False,
    }
    if hybrid:
        exact.update({
            "action_selection": "materialized_routed",
            "materialized_injection_mode": "complete_panel_historical_e4",
            "materialized_action_arm": hybrid_arm, "error_views_per_identity": 0,
            "direct_transfer_mode": "symmetric", "rank_reference_mode": "shared",
            "guided_noise_policy": "none", "pmt_arm": "none",
            "candidate_boundary_loss": False, "refresh_hard_negatives": False,
            "causal_arm": "legacy", "optimizer_boundary_mode": "action_injector_v1",
        })
    drift = {
        key: {"expected": value, "observed": configuration.get(key)}
        for key, value in exact.items() if configuration.get(key) != value
    }
    floats = {
        "backbone_lr": 2e-6, "head_lr": 1e-5, "weight_decay": 1e-4,
        "rank_margin": 0.05, "temperature": 0.10,
        "lambda_clean_rank": 1.0, "lambda_aug_rank": 1.0,
        "lambda_consistency": 0.25, "lambda_margin_floor": 2.0,
        "lambda_preserve": 5.0, "margin_floor_slack": 0.005,
        "safety_ratio": 1.0, "grad_clip": 1.0,
    }
    if hybrid:
        floats.update({
            "safety_stream_weight": 1.0, "positive_stream_weight": 0.0,
            "injector_target_attributable_fraction": 0.25,
            "injector_minimum_protective_retention": 0.90,
            "injector_maximum_update_norm_ratio": 1.50,
        })
    for key, value in floats.items():
        observed = configuration.get(key)
        if observed is None or not math.isclose(float(observed), value, rel_tol=0, abs_tol=1e-15):
            drift[key] = {"expected": value, "observed": observed}
    if drift:
        label = "hybrid" if hybrid else "stored historical E4"
        raise RuntimeError(f"{label} configuration drifted: {drift}")


def hybrid_forbidden_contract_drift(contracts: dict) -> dict[str, dict[str, object]]:
    """Report forbidden-branch drift without conflating routing provenance."""
    return {
        key: {"expected": expected_value, "observed": contracts.get(key)}
        for key, expected_value in HYBRID_FORBIDDEN_CONTRACT.items()
        if contracts.get(key) != expected_value
    }


def _validate_hybrid(decision: dict, arm: str) -> None:
    if decision.get("status") != "noise_final_e4a_direct_augmentation_complete":
        raise RuntimeError("hybrid training decision is incomplete")
    _exact_configuration(
        decision.get("configuration", {}), hybrid=True, hybrid_arm=arm,
    )
    contracts = decision.get("contracts", {})
    required_true = (
        "clean_and_augmented_raw_spectra_train_same_encoder",
        "historical_e4_official_initialization_used",
        "historical_e4_official_cache_is_floor_and_preservation_target",
        "historical_e4_symmetric_shared_direct_loss_unmodified",
        "hybrid_changes_only_action_supplier_and_optimizer_boundary",
        "complete_seven_source_action_panel_used_without_one_best_compression",
        "complete_panel_unit_action_weights_without_identity_or_source_reweighting",
        "action_specific_to_clean_corrective_gradient_gate_passed",
        "materialized_all_strict_actions_preserved",
        "action_injector_v1_receives_complete_historical_e4_gradient",
        "action_injector_v1_every_optimizer_step_audited",
        "action_injector_v1_signal_and_safety_gate_passed",
        "inference_clean_spectrum_only",
    )
    bad = [key for key in required_true if contracts.get(key) is not True]
    if bad:
        raise RuntimeError(f"hybrid lost required E4/Injector contracts: {bad}")
    forbidden_drift = hybrid_forbidden_contract_drift(contracts)
    if forbidden_drift:
        raise RuntimeError(
            "hybrid enabled a forbidden branch or mutated E4 signal: "
            f"{forbidden_drift}"
        )
    provenance = decision.get("provenance", {})
    expected = {
        "materialized_action_report_sha256": BEST_ACTION_REPORT_SHA256,
        "materialized_training_actions_sha256": BEST_ACTIONS_SHA256,
        "materialized_action_spectra_sha256": BEST_ACTION_SPECTRA_SHA256,
        "official_checkpoint_sha256": OFFICIAL_SHA256,
        "graph_sha256": GRAPH_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "embedding_cache_sha256": EMBEDDING_CACHE_SHA256,
        "data_sha256": DATA_SHA256,
        "architecture_checkpoint_sha256": ARCHITECTURE_SHA256,
    }
    drift = {key: provenance.get(key) for key, value in expected.items() if provenance.get(key) != value}
    if drift:
        raise RuntimeError(f"hybrid frozen input provenance drifted: {drift}")
    injection = decision.get("optimizer_boundary_injection", {})
    if injection.get("gate_passed") is not True or int(injection.get("steps", 0)) <= 0:
        raise RuntimeError("hybrid Injector V1 did not pass every-step signal gate")
    zero_change = decision.get("zero_change_gate", {})
    if (
        zero_change.get("gate_passed") is not True
        or zero_change.get("verification")
        != "official_checkpoint_edge_scores_plus_candidate_comparison_tie_aware_ranks"
        or float(zero_change.get("maximum_pair_score_abs_error", 1.0)) > 5e-4
        or int(zero_change.get("nonboundary_rank_mismatches", -1)) != 0
        or int(zero_change.get("rank_mismatches", -1))
        != int(zero_change.get("boundary_rank_mismatches", -2))
    ):
        raise RuntimeError("hybrid official zero-change replay is not equivalent")
    semantic = decision.get("branch_gradient_audit", {})
    if (
        semantic.get("action_specific_definition")
        != "aug_rank_plus_0.25_symmetric_consistency"
        or semantic.get("clean_corrective_definition")
        != "clean_rank_plus_2.0_margin_floor"
        or semantic.get("shared_loss_terms_between_compared_objectives") is not False
        or semantic.get("preservation_excluded_as_separate_protective_component")
        is not True
        or semantic.get("positive_alignment_required") is not (arm == "targeted")
        or semantic.get("gate_passed") is not True
    ):
        raise RuntimeError("hybrid semantic gradient audit is not independent")
    for group, group_report in semantic.get("parameter_groups", {}).items():
        if (
            group_report.get("gate_passed") is not True
            or float(group_report.get("action_gradient_nonzero_fraction", -1)) != 1.0
            or float(group_report.get(
                "clean_corrective_gradient_nonzero_fraction", -1,
            )) != 1.0
            or (
                arm == "targeted"
                and float(group_report.get("alignment_median", 0.0)) <= 0
            )
        ):
            raise RuntimeError(f"hybrid semantic gradient group failed: {group}")
    if set(semantic.get("parameter_groups", {})) != {"head", "backbone"}:
        raise RuntimeError("hybrid semantic gradient audit lost a parameter group")


def _validate_historical(decision: dict, seed: int) -> None:
    if decision.get("status") != "noise_final_e4a_direct_augmentation_complete":
        raise RuntimeError("stored historical E4 decision is incomplete")
    configuration = decision.get("configuration", {})
    _exact_configuration(configuration, hybrid=False)
    if int(configuration.get("seed", -1)) != seed:
        raise RuntimeError("stored historical E4 seed does not match hybrid seed")
    if decision.get("data", {}).get("train_action_rows") != 28509:
        raise RuntimeError("stored historical E4 R0 action membership drifted")


def _load_encoded(
    model, checkpoint: Path, store: SpectrumStore, reachable: np.ndarray,
    args: argparse.Namespace, label: str, *, hybrid: bool,
    expected_arm: str = "targeted",
) -> np.ndarray:
    package = torch_load_compat(checkpoint, map_location="cpu")
    if (
        package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
        or not package.get("inference_clean_only") or package.get("P2b_used")
    ):
        raise RuntimeError(f"{label} checkpoint violates clean shared-encoder contract")
    if hybrid and (
        package.get("action_selection") != "materialized_routed"
        or package.get("materialized_action_arm") != expected_arm
        or package.get("materialized_injection_mode")
        != "complete_panel_historical_e4"
        or package.get("optimizer_boundary_mode") != "action_injector_v1"
        or package.get("initial_student_checkpoint_sha256") is not None
        or package.get("teacher_used") is not False
    ):
        raise RuntimeError("hybrid checkpoint identity does not match E4-R0 Injector V1")
    model.load_state_dict(package["model_state"], strict=True)
    del package
    model.eval()
    return encode_rows(
        model, store, reachable, torch.device(args.device),
        args.eval_batch_size, args.amp, label,
    )


def _paired_panel(
    reference: pd.DataFrame, candidate: pd.DataFrame, formulas: np.ndarray,
    resamples: int, seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    paired = paired_outcome_table(reference, candidate)
    ci = formula_bootstrap_delta(
        reference["rank"].to_numpy(np.int64),
        candidate["rank"].to_numpy(np.int64),
        formulas, resamples, seed,
    )
    return paired, {**paired_counts(paired), "formula_cluster_delta_recall1": ci}


def _registered_metric_violations(candidate: dict, reference: dict) -> list[str]:
    higher = []
    retrieval_higher = (
        "recall@1", "recall@2", "recall@3", "recall@5", "recall@10",
        "recall@20", "mrr", "macro_query_auroc", "macro_query_auprc",
        "mean_positive_vs_best_negative_margin", "mean_top1_top2_gap",
        "mean_signed_top1_top2_gap",
    )
    for panel in ("retrieval", "near_subset"):
        for metric in retrieval_higher:
            higher.append((f"{panel}.{metric}", candidate[panel][metric], reference[panel][metric]))
    for panel in (
        "micro_candidate", "massspecgym_10ppm_pooled_pairwise",
        "massspecgym_mh_10ppm_pooled_pairwise",
    ):
        for metric in ("auroc", "auprc"):
            higher.append((f"{panel}.{metric}", candidate[panel][metric], reference[panel][metric]))
    violations = [
        name for name, observed, baseline in higher
        if float(observed) + 1e-12 < float(baseline)
    ]
    for panel in ("retrieval", "near_subset"):
        for metric in ("mean_rank", "median_rank"):
            if float(candidate[panel][metric]) > float(reference[panel][metric]) + 1e-12:
                violations.append(f"{panel}.{metric}")
    return violations


def _registered_metric_strict_or_boundary_failures(
    candidate: dict, reference: dict,
) -> list[str]:
    """Require strict gain unless the reference is already at a hard boundary."""
    failures = []
    tolerance = 1e-12
    higher = (
        "recall@1", "recall@2", "recall@3", "recall@5", "recall@10",
        "recall@20", "mrr", "macro_query_auroc", "macro_query_auprc",
        "mean_positive_vs_best_negative_margin", "mean_top1_top2_gap",
        "mean_signed_top1_top2_gap",
    )
    for panel in ("retrieval", "near_subset"):
        for metric in higher:
            observed = float(candidate[panel][metric])
            baseline = float(reference[panel][metric])
            bounded_at_one = metric.startswith("recall@") or metric in {
                "mrr", "macro_query_auroc", "macro_query_auprc",
            }
            if not (
                observed > baseline + tolerance
                or (
                    bounded_at_one and baseline >= 1.0 - tolerance
                    and observed >= baseline - tolerance
                )
            ):
                failures.append(f"{panel}.{metric}")
    for panel in (
        "micro_candidate", "massspecgym_10ppm_pooled_pairwise",
        "massspecgym_mh_10ppm_pooled_pairwise",
    ):
        for metric in ("auroc", "auprc"):
            observed = float(candidate[panel][metric])
            baseline = float(reference[panel][metric])
            if not (
                observed > baseline + tolerance
                or (
                    baseline >= 1.0 - tolerance
                    and observed >= baseline - tolerance
                )
            ):
                failures.append(f"{panel}.{metric}")
    for panel in ("retrieval", "near_subset"):
        for metric in ("mean_rank", "median_rank"):
            observed = float(candidate[panel][metric])
            baseline = float(reference[panel][metric])
            if not (
                observed < baseline - tolerance
                or (
                    baseline <= 1.0 + tolerance
                    and observed <= baseline + tolerance
                )
            ):
                failures.append(f"{panel}.{metric}")
    return failures


def _validate_causal_pair(targeted: dict, shuffled: dict) -> None:
    """Bind the negative control to exactly one changed action-view payload."""
    targeted_history = targeted.get("history", [])
    shuffled_history = shuffled.get("history", [])
    if len(targeted_history) != 4 or len(shuffled_history) != 4:
        raise RuntimeError("hybrid causal arms do not both contain four epochs")
    invariant = (
        "action_sampling_schedule_sha256",
        "action_exposure_schedule_sha256",
        "safety_sampling_schedule_sha256",
        "steps",
    )
    for epoch, (left, right) in enumerate(
        zip(targeted_history, shuffled_history), start=1,
    ):
        drift = {
            key: {"targeted": left.get(key), "shuffled": right.get(key)}
            for key in invariant if left.get(key) != right.get(key)
        }
        if drift:
            raise RuntimeError(f"causal schedule drifted at epoch {epoch}: {drift}")
    control = shuffled.get("materialized_action_control", {})
    expected_columns = ["supervision_kind", "source", "family", "recipe_id"]
    if (
        control.get("matching_columns") != expected_columns
        or control.get("family_semantics_preserved") is not True
        or control.get("dose_and_action_semantics_preserved") is not True
        or control.get("true_action_row_reused_for_same_query") is not False
        or control.get("all_nonfallback_donors_inside_input_panel") is not True
        or int(control.get("rows", 0)) != 32114
        or int(control.get("cross_query_rows", -1))
        + int(control.get("paired_control_fallback_rows", -1)) != 32114
    ):
        raise RuntimeError("matched shuffled control semantics drifted")


def main() -> None:
    args = arguments()
    started = time.time()
    required = (
        args.graph, args.source_manifest, args.embedding_cache, args.data,
        args.official_checkpoint, args.architecture_checkpoint,
        args.hybrid_checkpoint, args.hybrid_decision,
        args.shuffled_checkpoint, args.shuffled_decision,
        args.historical_e4_checkpoint, args.historical_e4_decision,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("E4-R0 Injector evaluation requires an allocated GPU")
    if sha256_file(args.official_checkpoint) != OFFICIAL_SHA256:
        raise RuntimeError("official DreaMS checkpoint drifted")

    hybrid_decision = json.loads(args.hybrid_decision.read_text(encoding="utf-8"))
    historical_decision = json.loads(args.historical_e4_decision.read_text(encoding="utf-8"))
    shuffled_decision = json.loads(args.shuffled_decision.read_text(encoding="utf-8"))
    _validate_hybrid(hybrid_decision, "targeted")
    _validate_hybrid(shuffled_decision, "shuffled")
    hybrid_checkpoint_sha256 = sha256_file(args.hybrid_checkpoint)
    shuffled_checkpoint_sha256 = sha256_file(args.shuffled_checkpoint)
    if (
        hybrid_decision.get("provenance", {}).get("final_shared_encoder_sha256")
        != hybrid_checkpoint_sha256
        or shuffled_decision.get("provenance", {}).get(
            "final_shared_encoder_sha256"
        ) != shuffled_checkpoint_sha256
    ):
        raise RuntimeError("hybrid decision/checkpoint artifact binding failed")
    seed = int(hybrid_decision["configuration"]["seed"])
    if int(shuffled_decision["configuration"].get("seed", -1)) != seed:
        raise RuntimeError("targeted and shuffled arms do not share one seed")
    _validate_historical(historical_decision, seed)
    if (
        sha256_file(args.historical_e4_checkpoint) != HISTORICAL_E4_SHA256
        or sha256_file(args.historical_e4_decision)
        != HISTORICAL_E4_DECISION_SHA256
    ):
        raise RuntimeError("stored historical E4 artifact binding failed")
    _validate_causal_pair(hybrid_decision, shuffled_decision)

    graph = CandidateGraph(args.graph)
    held = np.asarray([
        query for query, formula in enumerate(graph.query_formula)
        if stable_fold(str(formula), 5, args.formula_fold_seed) == args.outer_fold
    ], dtype=np.int64)
    with np.load(args.source_manifest, allow_pickle=False) as source:
        for key, expected in (
            ("query_row", graph.query_row), ("query_ik14", graph.query_ik14),
            ("query_formula", graph.query_formula),
        ):
            if key not in source.files or not np.array_equal(np.asarray(source[key]), expected):
                raise RuntimeError(f"corrected source manifest drifted at {key}")
        query_adduct = np.asarray(source["query_adduct"], dtype=str)
    held_candidates = []
    for query in held:
        molecule_left, molecule_right = map(int, graph.query_ptr[int(query):int(query) + 2])
        pair_left, pair_right = map(int, graph.molecule_ptr[[molecule_left, molecule_right]])
        held_candidates.append(graph.pair_candidate_row[pair_left:pair_right])
    reachable = np.unique(np.concatenate((graph.query_row[held], *held_candidates))).astype(np.int64)

    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        torch.device(args.device), args.n_highest_peaks,
    )
    hybrid_encoded = _load_encoded(
        model, args.hybrid_checkpoint, store, reachable, args,
        "E4-R0-InjectorV1-corrected", hybrid=True,
    )
    historical_encoded = _load_encoded(
        model, args.historical_e4_checkpoint, store, reachable, args,
        "stored-E4-corrected", hybrid=False,
    )
    shuffled_encoded = _load_encoded(
        model, args.shuffled_checkpoint, store, reachable, args,
        "matched-shuffled-corrected", hybrid=True, expected_arm="shuffled",
    )
    hybrid_scores = score_embedding_query_subset(graph, reachable, hybrid_encoded, held)
    historical_scores = score_embedding_query_subset(graph, reachable, historical_encoded, held)
    shuffled_scores = score_embedding_query_subset(graph, reachable, shuffled_encoded, held)
    baseline_scores = official_scores(graph)
    official_metrics, official_table = full_metrics(
        graph, baseline_scores, query_adduct=query_adduct, queries=held,
    )
    historical_metrics, historical_table = full_metrics(
        graph, historical_scores, query_adduct=query_adduct, queries=held,
    )
    hybrid_metrics, hybrid_table = full_metrics(
        graph, hybrid_scores, query_adduct=query_adduct, queries=held,
    )
    shuffled_metrics, shuffled_table = full_metrics(
        graph, shuffled_scores, query_adduct=query_adduct, queries=held,
    )
    formulas = graph.query_formula[held]
    vs_official, panel_vs_official = _paired_panel(
        official_table, hybrid_table, formulas, args.bootstrap_resamples, seed,
    )
    vs_historical, panel_vs_historical = _paired_panel(
        historical_table, hybrid_table, formulas, args.bootstrap_resamples, seed + 1,
    )
    _, historical_vs_official = _paired_panel(
        official_table, historical_table, formulas, args.bootstrap_resamples, seed + 2,
    )
    targeted_vs_shuffled, panel_vs_shuffled = _paired_panel(
        shuffled_table, hybrid_table, formulas, args.bootstrap_resamples, seed + 3,
    )
    evidence = held_metric_evidence(
        graph,
        {
            "official": baseline_scores, "historical_e4": historical_scores,
            "targeted": hybrid_scores, "shuffled": shuffled_scores,
        },
        query_adduct, held,
    )

    _, cache_embeddings, cache_index = load_embedding_cache(args.embedding_cache)
    try:
        official_reachable = np.asarray([
            cache_embeddings[cache_index[int(row)]] for row in reachable
        ], dtype=np.float32)
    except KeyError as error:
        raise RuntimeError("official corrected cache misses a held graph row") from error
    cache_scores = score_embedding_query_subset(graph, reachable, official_reachable, held)
    cache_edge_errors = []
    for query in held:
        molecule_left, molecule_right = map(int, graph.query_ptr[int(query):int(query) + 2])
        pair_left, pair_right = map(int, graph.molecule_ptr[[molecule_left, molecule_right]])
        cache_edge_errors.append(float(np.max(np.abs(
            cache_scores.pair[pair_left:pair_right]
            - baseline_scores.pair[pair_left:pair_right]
        ))))
    official_cache_graph_max_abs_error = float(max(cache_edge_errors))
    if official_cache_graph_max_abs_error > 2e-5:
        raise RuntimeError(
            "official corrected cache does not reproduce frozen graph scores: "
            f"max_abs_error={official_cache_graph_max_abs_error:.8g}"
        )

    official_r1 = float(official_metrics["retrieval"]["recall@1"])
    historical_r1 = float(historical_metrics["retrieval"]["recall@1"])
    hybrid_r1 = float(hybrid_metrics["retrieval"]["recall@1"])
    historical_gain = historical_r1 - official_r1
    hybrid_gain = hybrid_r1 - official_r1
    retention = hybrid_gain / historical_gain if historical_gain > 0 else float("nan")
    violations_vs_official = _registered_metric_violations(hybrid_metrics, official_metrics)
    violations_vs_historical = _registered_metric_violations(hybrid_metrics, historical_metrics)
    strict_or_boundary_failures_vs_official = (
        _registered_metric_strict_or_boundary_failures(
        hybrid_metrics, official_metrics,
        )
    )
    strict_or_boundary_failures_vs_historical = (
        _registered_metric_strict_or_boundary_failures(
        hybrid_metrics, historical_metrics,
        )
    )
    gates = {
        "injector_signal_and_safety_passed": True,
        "historical_e4_top1_signal_not_lost": hybrid_r1 >= historical_r1,
        "historical_e4_gain_retention_ge_1": bool(np.isfinite(retention) and retention >= 1.0),
        "hybrid_formula_ci_vs_official_positive": (
            panel_vs_official["formula_cluster_delta_recall1"]["ci_low"] > 0
        ),
        "hybrid_formula_ci_vs_historical_nonnegative": (
            panel_vs_historical["formula_cluster_delta_recall1"]["ci_low"] >= 0
        ),
        "hybrid_formula_ci_vs_historical_positive": (
            panel_vs_historical["formula_cluster_delta_recall1"]["ci_low"] > 0
        ),
        "hybrid_risk_net_lambda2_vs_official_positive": (
            panel_vs_official["risk_net_lambda2"] > 0
        ),
        "hybrid_risk_net_lambda2_vs_historical_nonnegative": (
            panel_vs_historical["risk_net_lambda2"] >= 0
        ),
        "hybrid_risk_net_lambda2_vs_historical_positive": (
            panel_vs_historical["risk_net_lambda2"] > 0
        ),
        "targeted_beats_matched_shuffled_formula_ci": (
            panel_vs_shuffled["formula_cluster_delta_recall1"]["ci_low"] > 0
        ),
        "all_registered_metrics_nonnegative_vs_official": not violations_vs_official,
        "all_registered_metrics_nonnegative_vs_historical_e4": not violations_vs_historical,
        "all_registered_metrics_strictly_better_or_boundary_saturated_vs_official": (
            not strict_or_boundary_failures_vs_official
        ),
        "all_registered_metrics_strictly_better_or_boundary_saturated_vs_historical_e4": (
            not strict_or_boundary_failures_vs_historical
        ),
        "strict_four_pp_recall1_gain": hybrid_gain >= 0.04,
    }
    report = {
        "status": "noise_e4_best_actions_injector_v1_final_evaluation_complete",
        "formal": False,
        "evaluation_role": "registered train-side corrected development graph",
        "held_queries": int(len(held)),
        "official": official_metrics,
        "stored_historical_e4": historical_metrics,
        "best_actions_e4_injector_v1": hybrid_metrics,
        "matched_shuffled_control": shuffled_metrics,
        "historical_e4_minus_official": nested_delta(historical_metrics, official_metrics),
        "hybrid_minus_official": nested_delta(hybrid_metrics, official_metrics),
        "hybrid_minus_historical_e4": nested_delta(hybrid_metrics, historical_metrics),
        "targeted_minus_matched_shuffled": nested_delta(hybrid_metrics, shuffled_metrics),
        "stored_historical_e4_vs_official": historical_vs_official,
        "hybrid_vs_official": panel_vs_official,
        "hybrid_vs_historical_e4": panel_vs_historical,
        "targeted_vs_matched_shuffled": panel_vs_shuffled,
        "historical_e4_recall1_gain": historical_gain,
        "hybrid_recall1_gain": hybrid_gain,
        "historical_e4_gain_retention_fraction": retention,
        "registered_metric_violations_vs_official": violations_vs_official,
        "registered_metric_violations_vs_historical_e4": violations_vs_historical,
        "registered_metric_strict_or_boundary_failures_vs_official": (
            strict_or_boundary_failures_vs_official
        ),
        "registered_metric_strict_or_boundary_failures_vs_historical_e4": (
            strict_or_boundary_failures_vs_historical
        ),
        "gates": gates,
        "pass_without_losing_e4_signal": bool(all(
            value for key, value in gates.items() if key != "strict_four_pp_recall1_gain"
        )),
        "strict_four_pp_result_achieved": bool(gates["strict_four_pp_recall1_gain"]),
        "strict_four_pp_full_panel_result_achieved": bool(all(gates.values())),
        "contracts": {
            "pure_e4_comparator_reused_not_retrained": True,
            "evaluation_cannot_modify_training": True,
            "same_shared_encoder_for_queries_and_candidates": True,
            "complete_corrected_candidate_blocks": True,
            "massspecgym_pairwise_not_nist20_replication": True,
            "teacher_embedding_or_margin_target_used": False,
            "checkpoint_selection_from_held_metrics": False,
        },
        "runtime_seconds": time.time() - started,
        "model_initialization_loader": initialization,
        "official_cache_graph_max_abs_error": official_cache_graph_max_abs_error,
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "source_manifest_sha256": sha256_file(args.source_manifest),
            "embedding_cache_sha256": sha256_file(args.embedding_cache),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "hybrid_checkpoint_sha256": sha256_file(args.hybrid_checkpoint),
            "hybrid_decision_sha256": sha256_file(args.hybrid_decision),
            "shuffled_checkpoint_sha256": sha256_file(args.shuffled_checkpoint),
            "shuffled_decision_sha256": sha256_file(args.shuffled_decision),
            "historical_e4_checkpoint_sha256": sha256_file(args.historical_e4_checkpoint),
            "historical_e4_decision_sha256": sha256_file(args.historical_e4_decision),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Development evaluation, not a P3 test claim. The 4 pp gate is an observed-result "
            "requirement, never an action-headroom promise. MassSpecGym 10-ppm pooled AUROC "
            "is not the paper NIST20 0.85 replication."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_e4_best_injector_eval_", dir=args.output_dir.parent))
    try:
        vs_official.to_csv(staging / "hybrid_vs_official_per_query.csv.gz", index=False, compression="gzip")
        vs_historical.to_csv(staging / "hybrid_vs_historical_e4_per_query.csv.gz", index=False, compression="gzip")
        targeted_vs_shuffled.to_csv(
            staging / "targeted_vs_matched_shuffled_per_query.csv.gz",
            index=False, compression="gzip",
        )
        np.savez_compressed(staging / "held_metric_evidence.npz", **evidence)
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8",
        )
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
