#!/usr/bin/env python
"""Frozen structure-hidden MS1 positive-control for MSV000096214.

This script never opens the annotation sidecar.  It tests whether the genetic
counterfactual is recoverable before any spectral relation is introduced.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


WT = "WT_Col0"
TRUE_TARGETS = (
    "tt7_115_EMS",
    "tt7_8_SALK_039417",
    "tt7_9_SALK_seq_064306",
)
ALTERNATIVES = (
    "tt3_SALK_099848C",
    "tt4_11_SALK_020583",
    "fls1_3_EMS",
)
DOUBLES = {
    "tt7_115_EMS": "tt7_115__tt4_11",
    "tt7_8_SALK_039417": "tt7_8__tt4_11",
    "tt7_9_SALK_seq_064306": "tt7_9__tt4_11",
}
SEED = 20261012


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    values = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(labels, values))


def auprc(pos: np.ndarray, neg: np.ndarray) -> float:
    values = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(average_precision_score(labels, values))


def signature(
    matrix: np.ndarray,
    groups: np.ndarray,
    train_targets: tuple[str, str],
    excluded_wt_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    wt_indices = np.flatnonzero(groups == WT)
    if excluded_wt_index is not None:
        wt_indices = wt_indices[wt_indices != excluded_wt_index]
    wt_center = np.median(matrix[:, wt_indices], axis=1)
    deltas = []
    for genotype in train_targets:
        deltas.append(np.median(matrix[:, groups == genotype], axis=1) - wt_center)
    first, second = deltas
    concordant = first * second > 0
    weights = np.zeros(matrix.shape[0], dtype=float)
    weights[concordant] = np.sign(first[concordant]) * np.minimum(
        np.abs(first[concordant]), np.abs(second[concordant])
    )
    return wt_center, weights


def cosine_score(vector: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.linalg.norm(vector) * np.linalg.norm(weights))
    return 0.0 if denominator == 0 else float(np.dot(vector, weights) / denominator)


def score_samples(
    matrix: np.ndarray,
    modes: np.ndarray,
    groups: np.ndarray,
    train_targets: tuple[str, str],
) -> tuple[np.ndarray, dict[str, int]]:
    scores = np.zeros(matrix.shape[1], dtype=float)
    feature_counts: dict[str, int] = {}
    full_center, full_weights = signature(matrix, groups, train_targets)
    for mode in sorted(set(modes)):
        mask = modes == mode
        feature_counts[str(mode)] = int(np.count_nonzero(full_weights[mask]))
    for index in range(matrix.shape[1]):
        if groups[index] == WT:
            center, weights = signature(matrix, groups, train_targets, index)
        else:
            center, weights = full_center, full_weights
        per_mode = []
        for mode in sorted(set(modes)):
            mask = modes == mode
            per_mode.append(cosine_score(matrix[mask, index] - center[mask], weights[mask]))
        scores[index] = float(np.mean(per_mode))
    return scores, feature_counts


def evaluate_assignment(
    matrix: np.ndarray,
    modes: np.ndarray,
    groups: np.ndarray,
    targets: tuple[str, str, str],
    all_single_mutants: tuple[str, ...],
) -> dict[str, object]:
    complement = tuple(group for group in all_single_mutants if group not in targets)
    held_results = {}
    all_pairwise_aucs = []
    for held in targets:
        train = tuple(group for group in targets if group != held)
        scores, feature_counts = score_samples(matrix, modes, groups, train)  # type: ignore[arg-type]
        held_values = scores[groups == held]
        comparisons = {}
        for comparator in (WT, *complement):
            comparator_values = scores[groups == comparator]
            pair_auc = auc(held_values, comparator_values)
            all_pairwise_aucs.append(pair_auc)
            comparisons[comparator] = {
                "n_held": int(len(held_values)),
                "n_comparator": int(len(comparator_values)),
                "auroc": pair_auc,
                "auprc": auprc(held_values, comparator_values),
                "mean_margin": float(np.mean(held_values) - np.mean(comparator_values)),
            }
        held_results[held] = {
            "trained_on": list(train),
            "concordant_signature_features_by_mode": feature_counts,
            "comparisons": comparisons,
            "score_by_genotype": {
                genotype: float(np.mean(scores[groups == genotype]))
                for genotype in (WT, *targets, *complement)
            },
        }
    return {
        "targets": list(targets),
        "complement": list(complement),
        "held_results": held_results,
        "joint_min_pairwise_auroc": float(min(all_pairwise_aucs)),
        "joint_mean_pairwise_auroc": float(np.mean(all_pairwise_aucs)),
    }


def evaluate_g3(
    matrix: np.ndarray, modes: np.ndarray, groups: np.ndarray
) -> dict[str, object]:
    result = {}
    tt4 = "tt4_11_SALK_020583"
    for held in TRUE_TARGETS:
        train = tuple(group for group in TRUE_TARGETS if group != held)
        scores, _ = score_samples(matrix, modes, groups, train)  # type: ignore[arg-type]
        single = float(np.mean(scores[groups == held]))
        double = float(np.mean(scores[groups == DOUBLES[held]]))
        tt4_center = float(np.mean(scores[groups == tt4]))
        result[held] = {
            "single_mean_score": single,
            "double_mean_score": double,
            "tt4_mean_score": tt4_center,
            "single_minus_double_margin": single - double,
            "double_closer_to_tt4_than_single": abs(double - tt4_center) < abs(single - tt4_center),
        }
    return {
        "lineages": result,
        "all_three_single_to_double_margins_positive": all(
            value["single_minus_double_margin"] > 0 for value in result.values()
        ),
        "all_three_doubles_closer_to_tt4": all(
            value["double_closer_to_tt4_than_single"] for value in result.values()
        ),
    }


def preprocess(values: np.ndarray, modes: np.ndarray, median_center: bool) -> np.ndarray:
    matrix = np.log1p(values.astype(float))
    if median_center:
        for mode in sorted(set(modes)):
            mask = modes == mode
            matrix[mask, :] -= np.median(matrix[mask, :], axis=0, keepdims=True)
    return matrix


def run_world(values: np.ndarray, modes: np.ndarray, groups: np.ndarray, median_center: bool) -> dict:
    matrix = preprocess(values, modes, median_center)
    singles = tuple(sorted((*TRUE_TARGETS, *ALTERNATIVES)))
    assignments = []
    observed = None
    true_set = set(TRUE_TARGETS)
    for choice in itertools.combinations(singles, 3):
        result = evaluate_assignment(matrix, modes, groups, choice, singles)
        assignments.append({
            "targets": list(choice),
            "joint_min_pairwise_auroc": result["joint_min_pairwise_auroc"],
            "joint_mean_pairwise_auroc": result["joint_mean_pairwise_auroc"],
        })
        if set(choice) == true_set:
            observed = result
    if observed is None:
        raise AssertionError("true assignment absent")
    observed_stat = float(observed["joint_min_pairwise_auroc"])
    exact_p = sum(
        float(item["joint_min_pairwise_auroc"]) >= observed_stat - 1e-12
        for item in assignments
    ) / len(assignments)
    assignments.sort(key=lambda item: (-float(item["joint_min_pairwise_auroc"]), item["targets"]))
    g3 = evaluate_g3(matrix, modes, groups)
    held = observed["held_results"]
    all_g1_above_chance = all(
        float(held[group]["comparisons"][WT]["auroc"]) > 0.5 for group in TRUE_TARGETS
    )
    all_g2_margins_positive = all(
        float(held[group]["comparisons"][alt]["mean_margin"]) > 0
        for group in TRUE_TARGETS for alt in ALTERNATIVES
    )
    return {
        "preprocessing": "log1p_then_within_sample_within_polarity_median_center"
        if median_center else "raw_log1p",
        "observed_true_assignment": observed,
        "all_20_target_assignments_ranked": assignments,
        "exact_cluster_assignment_test": {
            "assignments": len(assignments),
            "observed_joint_min_pairwise_auroc": observed_stat,
            "right_tail_p": exact_p,
            "minimum_attainable_p": 1 / len(assignments),
            "observed_rank": next(
                index + 1 for index, item in enumerate(assignments)
                if set(item["targets"]) == true_set
            ),
        },
        "gates": {
            "g1_all_three_held_vs_wt_auroc_above_half": all_g1_above_chance,
            "g1_g2_true_assignment_uniquely_best_exact_p_0_05": exact_p == 0.05,
            "g2_all_nine_held_vs_alternative_mean_margins_positive": all_g2_margins_positive,
            "g3_all_three_single_to_double_margins_positive": g3["all_three_single_to_double_margins_positive"],
            "g3_all_three_doubles_closer_to_tt4": g3["all_three_doubles_closer_to_tt4"],
        },
        "g3": g3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    abundance_path = args.input / "structure_hidden_abundance.csv.gz"
    manifest_path = args.input / "sample_manifest.csv.gz"
    abundance = pd.read_csv(abundance_path)
    manifest = pd.read_csv(manifest_path)
    sample_ids = manifest["sample_id"].tolist()
    values = abundance[sample_ids].to_numpy(dtype=float)
    modes = abundance["ion_mode"].astype(str).to_numpy()
    groups = manifest["genotype"].astype(str).to_numpy()
    expected = {WT, *TRUE_TARGETS, *ALTERNATIVES, *DOUBLES.values()}
    if set(groups) != expected:
        raise ValueError(f"unexpected genotype set: {sorted(set(groups))}")

    primary = run_world(values, modes, groups, median_center=True)
    sensitivity = run_world(values, modes, groups, median_center=False)
    primary_pass = all(primary["gates"].values())
    sensitivity_same = primary["gates"] == sensitivity["gates"]
    report = {
        "status": "MSV000096214_STRUCTURE_HIDDEN_MS1_BASELINE_COMPLETE",
        "formal": True,
        "dataset": "MSV000096214",
        "data_role": "development/consumed",
        "sealed": False,
        "structure_or_annotation_used": False,
        "denominators": {
            "features": int(len(abundance)),
            "positive_mode_features": int((modes == "ESI+").sum()),
            "negative_mode_features": int((modes == "ESI-").sum()),
            "biological_samples": int(len(manifest)),
            "independent_tt7_alleles": 3,
            "alternative_single_mutants": 3,
            "double_mutant_lineages": 3,
            "exact_target_assignments": 20,
        },
        "primary": primary,
        "raw_log1p_sensitivity": sensitivity,
        "decision": {
            "all_primary_gates_pass": primary_pass,
            "sensitivity_gate_pattern_identical": sensitivity_same,
            "pass_to_spectral_increment": primary_pass and sensitivity_same,
        },
        "provenance": {
            "abundance_sha256": file_sha256(abundance_path),
            "sample_manifest_sha256": file_sha256(manifest_path),
            "seed_reserved_for_future_bootstrap": SEED,
        },
        "claim_limit": (
            "Structure-hidden MS1 genetic positive-control only. A pass would justify testing a frozen "
            "MS/MS relation layer; it would not establish unknown-entity, public-reference, mechanism, "
            "substrate, flux, or cross-study value."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "denominators": report["denominators"],
        "primary_exact_test": primary["exact_cluster_assignment_test"],
        "primary_gates": primary["gates"],
        "sensitivity_gates": sensitivity["gates"],
        "decision": report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
