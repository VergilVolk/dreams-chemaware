#!/usr/bin/env python
"""Adjudicate formula-OOF conditional-null candidate-energy arms.

This script separates three claims that must not be conflated: improvement
over the frozen spectral baseline, complementarity over either evidence-only
arm, and strict superadditivity beyond the sum of the two isolated gains.
All intervals resample molecular-formula clusters and all comparisons are
paired at query level.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from conditional_null_energy_core import strict_top1


ARM_NAMES = ("joint", "no_interaction", "spectral_only", "chem_only")
CONTRASTS = (
    "joint_minus_baseline",
    "joint_minus_spectral_only",
    "joint_minus_chem_only",
    "joint_minus_no_interaction",
    "strict_superadditivity",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARM_NAMES:
        parser.add_argument(f"--{arm.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261004)
    return parser.parse_args()


def load_arm(path: Path, expected_arm: str) -> tuple[dict[str, np.ndarray], dict]:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "CONDITIONAL_NULL_CANDIDATE_ENERGY_FROZEN":
        raise RuntimeError(f"unfrozen arm: {path}")
    if report.get("arm") != expected_arm:
        raise RuntimeError(f"arm-name mismatch: {path}")
    with np.load(path / "oof_scores.npz", allow_pickle=False) as body:
        arrays = {name: np.asarray(body[name]) for name in body.files}
    return arrays, report


def validate_alignment(arms: dict[str, dict[str, np.ndarray]]) -> None:
    reference = arms["joint"]
    required = (
        "baseline_scores", "labels", "query_ptr", "query_formula", "folds",
        "candidate_reference_count", "query_candidate_count", "query_has_near",
    )
    for name in required:
        if name not in reference:
            raise RuntimeError(f"joint ledger lacks {name}")
    for arm, body in arms.items():
        for name in required:
            if not np.array_equal(body[name], reference[name]):
                raise RuntimeError(f"{arm} ledger is misaligned at {name}")
        if body["candidate_scores"].shape != reference["baseline_scores"].shape:
            raise RuntimeError(f"{arm} candidate score shape drifted")
        if not np.all(np.isfinite(body["candidate_scores"])):
            raise RuntimeError(f"{arm} contains non-finite OOF scores")


def validate_reports(reports: dict[str, dict]) -> None:
    reference = reports["joint"]
    for arm, report in reports.items():
        if report.get("recipe") != reference.get("recipe"):
            raise RuntimeError(f"{arm} training recipe differs from the primary arm")
        if report.get("provenance") != reference.get("provenance"):
            raise RuntimeError(f"{arm} provenance differs from the primary arm")
    if reports["joint"].get("model_kind") != "anchored_evidence":
        raise RuntimeError("joint arm is not the preregistered anchored model")
    if reports["no_interaction"].get("model_kind") != "additive_two_head":
        raise RuntimeError("no-interaction arm is not the additive two-head comparator")


def formula_cluster_bootstrap(
    query_values: np.ndarray,
    formulas: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, list[float] | float | int]:
    values = np.asarray(query_values, dtype=np.float64)
    formulas = np.asarray(formulas, dtype=str)
    if values.ndim != 2 or values.shape[0] != len(formulas):
        raise ValueError("bootstrap values must be [query, contrast]")
    unique, inverse = np.unique(formulas, return_inverse=True)
    sums = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    np.add.at(sums, inverse, values)
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, values.shape[1]), dtype=np.float64)
    chunk = 256
    for start in range(0, resamples, chunk):
        stop = min(resamples, start + chunk)
        indices = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        draws[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
    return {
        "formula_clusters": int(len(unique)),
        "estimate": [float(value) for value in values.mean(axis=0)],
        "ci95": np.quantile(draws, [0.025, 0.975], axis=0).T.tolist(),
        # Five preregistered contrasts: Bonferroni family-wise alpha=0.05.
        "familywise95_ci": np.quantile(draws, [0.005, 0.995], axis=0).T.tolist(),
    }


def transition(reference: np.ndarray, candidate: np.ndarray) -> dict[str, int | float]:
    corrected = int(np.sum((reference == 0) & (candidate == 1)))
    introduced = int(np.sum((reference == 1) & (candidate == 0)))
    return {
        "delta_pp": 100.0 * float(np.mean(candidate - reference)),
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda_2": corrected - 2 * introduced,
    }


def subset_summary(
    selected: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    formulas: np.ndarray,
    resamples: int,
    seed: int,
    comparison: str,
) -> dict:
    count = int(np.sum(selected))
    if count < 2:
        return {"queries": count, "evaluable": False}
    delta = (candidate[selected] - reference[selected])[:, None]
    boot = formula_cluster_bootstrap(delta, formulas[selected], resamples, seed)
    return {
        "queries": count,
        "evaluable": True,
        "comparison": comparison,
        "baseline_top1": float(np.mean(reference[selected])),
        "joint_top1": float(np.mean(candidate[selected])),
        "delta_pp": 100.0 * float(np.mean(delta)),
        "formula_clusters": boot["formula_clusters"],
        "formula_cluster_ci95_pp": [100.0 * value for value in boot["ci95"][0]],
        **transition(reference[selected], candidate[selected]),
    }


def adjudicate(
    arms: dict[str, dict[str, np.ndarray]],
    bootstrap_resamples: int,
    bootstrap_seed: int,
    reports: dict[str, dict] | None = None,
) -> dict:
    validate_alignment(arms)
    if reports is not None:
        validate_reports(reports)
    joint = arms["joint"]
    query_ptr = np.asarray(joint["query_ptr"], dtype=np.int64)
    labels = np.asarray(joint["labels"], dtype=np.int8)
    formulas = np.asarray(joint["query_formula"], dtype=str)
    folds = np.asarray(joint["folds"], dtype=np.int8)
    reference_count = np.asarray(joint["candidate_reference_count"], dtype=np.float64)
    near = np.asarray(joint["query_has_near"], dtype=bool)
    correct = {
        "baseline": strict_top1(joint["baseline_scores"], labels, query_ptr).astype(np.float64),
    }
    for arm in ARM_NAMES:
        correct[arm] = strict_top1(
            arms[arm]["candidate_scores"], labels, query_ptr,
        ).astype(np.float64)
    correct["reference_count_only"] = strict_top1(
        reference_count, labels, query_ptr,
    ).astype(np.float64)
    contrast_matrix = np.column_stack((
        correct["joint"] - correct["baseline"],
        correct["joint"] - correct["spectral_only"],
        correct["joint"] - correct["chem_only"],
        correct["joint"] - correct["no_interaction"],
        correct["joint"] + correct["baseline"]
        - correct["spectral_only"] - correct["chem_only"],
    ))
    bootstrap = formula_cluster_bootstrap(
        contrast_matrix, formulas, bootstrap_resamples, bootstrap_seed,
    )
    estimates = dict(zip(CONTRASTS, bootstrap.pop("estimate")))
    ci95 = dict(zip(CONTRASTS, bootstrap.pop("ci95")))
    adjusted = dict(zip(CONTRASTS, bootstrap.pop("familywise95_ci")))
    transitions = {
        arm: transition(correct["baseline"], correct[arm]) for arm in ARM_NAMES
    }
    fold_deltas = {}
    for fold in sorted(np.unique(folds)):
        selected = folds == fold
        fold_deltas[str(int(fold))] = {
            name: 100.0 * float(np.mean(contrast_matrix[selected, index]))
            for index, name in enumerate(CONTRASTS)
        }
    joint_transition = transitions["joint"]
    multiplicity_matched = np.asarray([
        len(np.unique(reference_count[int(left):int(right)])) == 1
        for left, right in zip(query_ptr[:-1], query_ptr[1:])
    ], dtype=bool)
    strata = {
        "near_structure": subset_summary(
            near, correct["baseline"], correct["joint"], formulas,
            bootstrap_resamples, bootstrap_seed + 200, "joint_minus_baseline",
        ),
        "reference_multiplicity_matched": subset_summary(
            multiplicity_matched, correct["spectral_only"], correct["joint"], formulas,
            bootstrap_resamples, bootstrap_seed + 400, "joint_minus_spectral_only",
        ),
    }
    unique_synergy_corrected = int(np.sum(
        (correct["baseline"] == 0) & (correct["joint"] == 1)
        & (correct["spectral_only"] == 0) & (correct["chem_only"] == 0)
    ))
    unique_synergy_introduced = int(np.sum(
        (correct["baseline"] == 1) & (correct["joint"] == 0)
        & (correct["spectral_only"] == 1) & (correct["chem_only"] == 1)
    ))
    gates = {
        "joint_beats_baseline_adjusted_ci": adjusted["joint_minus_baseline"][0] > 0,
        "joint_beats_spectral_only_adjusted_ci": adjusted["joint_minus_spectral_only"][0] > 0,
        "joint_beats_chem_only_adjusted_ci": adjusted["joint_minus_chem_only"][0] > 0,
        "interaction_adds_value_adjusted_ci": adjusted["joint_minus_no_interaction"][0] > 0,
        "strict_superadditivity_adjusted_ci": adjusted["strict_superadditivity"][0] > 0,
        "risk_net_positive": joint_transition["risk_net_lambda_2"] > 0,
        "corrected_more_than_twice_introduced": (
            joint_transition["corrected"] > 2 * joint_transition["introduced"]
        ),
        "joint_baseline_delta_nonnegative_every_fold": all(
            row["joint_minus_baseline"] >= 0 for row in fold_deltas.values()
        ),
        "near_structure_not_harmed": (
            strata["near_structure"].get("evaluable", False)
            and strata["near_structure"]["formula_cluster_ci95_pp"][0] >= 0
        ),
        "multiplicity_matched_chem_increment_positive": (
            strata["reference_multiplicity_matched"].get("evaluable", False)
            and strata["reference_multiplicity_matched"]["formula_cluster_ci95_pp"][0] > 0
        ),
        "unique_synergy_corrected_exceeds_introduced": (
            unique_synergy_corrected > unique_synergy_introduced
        ),
    }
    base_pass = all((
        gates["joint_beats_baseline_adjusted_ci"],
        gates["risk_net_positive"],
        gates["corrected_more_than_twice_introduced"],
        gates["joint_baseline_delta_nonnegative_every_fold"],
        gates["near_structure_not_harmed"],
        gates["multiplicity_matched_chem_increment_positive"],
    ))
    complementarity_pass = base_pass and all((
        gates["joint_beats_spectral_only_adjusted_ci"],
        gates["joint_beats_chem_only_adjusted_ci"],
    ))
    superadditive_pass = complementarity_pass and all((
        gates["strict_superadditivity_adjusted_ci"],
        gates["interaction_adds_value_adjusted_ci"],
        gates["unique_synergy_corrected_exceeds_introduced"],
    ))
    if superadditive_pass:
        decision = "SUPERADDITIVE_DEVELOPMENT_SIGNAL_EXTERNAL_CONFIRMATION_REQUIRED"
    elif complementarity_pass:
        decision = "COMPLEMENTARY_JOINT_GAIN_NOT_SUPERADDITIVE"
    elif base_pass:
        decision = "JOINT_GAIN_WITHOUT_PROVEN_COMPONENT_COMPLEMENTARITY"
    else:
        decision = "STOP_NO_SAFE_JOINT_GAIN"
    return {
        "status": "CONDITIONAL_NULL_CANDIDATE_ENERGY_ADJUDICATED",
        "queries": int(len(formulas)),
        "formula_clusters": bootstrap["formula_clusters"],
        "primary_arm": "joint",
        "ablation_arms": ["no_interaction", "spectral_only", "chem_only"],
        "top1": {name: float(np.mean(value)) for name, value in correct.items()},
        "baseline_transitions": transitions,
        "paired_formula_cluster_contrasts": {
            name: {
                "estimate_pp": 100.0 * estimates[name],
                "ci95_pp": [100.0 * value for value in ci95[name]],
                "familywise95_ci_pp": [100.0 * value for value in adjusted[name]],
            }
            for name in CONTRASTS
        },
        "fold_deltas_pp": fold_deltas,
        "coverage_diagnostics": {
            "reference_count_only_top1": float(np.mean(correct["reference_count_only"])),
            "strata": strata,
        },
        "unique_synergy_transitions": {
            "corrected": unique_synergy_corrected,
            "introduced": unique_synergy_introduced,
        },
        "gates": gates,
        "decision": decision,
        "claim_limit": (
            "Formula-OOF development adjudication only. A passing result may justify one "
            "frozen external GNPS evaluation; it is not itself an external generalization claim."
        ),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_resamples < 1000:
        raise ValueError("formal adjudication requires at least 1000 bootstrap resamples")
    paths = {
        "joint": args.joint,
        "no_interaction": args.no_interaction,
        "spectral_only": args.spectral_only,
        "chem_only": args.chem_only,
    }
    loaded = {arm: load_arm(path, arm) for arm, path in paths.items()}
    arrays = {arm: pair[0] for arm, pair in loaded.items()}
    reports = {arm: pair[1] for arm, pair in loaded.items()}
    report = adjudicate(
        arrays, args.bootstrap_resamples, args.bootstrap_seed, reports=reports,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
