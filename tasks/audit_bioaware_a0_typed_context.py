#!/usr/bin/env python
"""A0 screen: does typed Rhea evidence beat an outcome-blind query permutation?

The input cohorts are already exposed development data.  This task does not fit
a model or choose a threshold.  It compares preregistered evidence columns with
a within-query candidate-assignment null that preserves candidate count and the
entire score multiset for every query while destroying candidate identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = (
    "raw_network_support",
    "complete_network_support",
    "dependency_corrected_network_support",
    "candidate_specific_network_support",
    "direction_supported_network_support",
    "fully_observed_hyperedge_path_count",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def query_arrays(frame: pd.DataFrame, feature: str) -> list[tuple[np.ndarray, int, bool]]:
    groups: list[tuple[np.ndarray, int, bool]] = []
    for query_id, group in frame.groupby("query_id", sort=False):
        if len(group) < 2:
            continue
        truths = group["truth_candidate_id"].astype(str).unique()
        if len(truths) != 1:
            raise RuntimeError(f"query {query_id} has {len(truths)} truths")
        truth_mask = group["candidate_id"].astype(str).to_numpy() == truths[0]
        if int(truth_mask.sum()) != 1:
            raise RuntimeError(f"query {query_id} truth is not unique in candidates")
        values = pd.to_numeric(group[feature], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"query {query_id} feature {feature} is non-finite")
        truth_index = int(np.flatnonzero(truth_mask)[0])
        spectral = pd.to_numeric(group["spectral_score"], errors="raise").to_numpy(float)
        spectral_top = int(np.sum(spectral >= spectral[truth_index])) == 1
        groups.append((values, truth_index, spectral_top))
    return groups


def metrics(groups: list[tuple[np.ndarray, int, bool]]) -> dict:
    margins: list[float] = []
    top1: list[bool] = []
    pairwise: list[float] = []
    baseline_wrong_top1: list[bool] = []
    baseline_correct_top1: list[bool] = []
    for values, truth_index, baseline_correct in groups:
        truth = float(values[truth_index])
        wrong = np.delete(values, truth_index)
        margins.append(truth - float(wrong.max()))
        correct = bool(np.sum(values >= truth) == 1)  # ties count against truth
        top1.append(correct)
        pairwise.extend(
            (
                (truth > wrong).astype(float)
                + 0.5 * (truth == wrong).astype(float)
            ).tolist()
        )
        (baseline_correct_top1 if baseline_correct else baseline_wrong_top1).append(correct)
    return {
        "queries": len(groups),
        "queries_with_any_nonzero_evidence": int(
            np.sum([np.any(values != 0) for values, _, _ in groups])
        ),
        "queries_with_feature_variation": int(
            np.sum([np.ptp(values) > 0 for values, _, _ in groups])
        ),
        "mean_truth_minus_best_wrong": float(np.mean(margins)),
        "median_truth_minus_best_wrong": float(np.median(margins)),
        "feature_top1": float(np.mean(top1)),
        "pairwise_accuracy_with_half_ties": float(np.mean(pairwise)),
        "baseline_wrong_queries": len(baseline_wrong_top1),
        "baseline_wrong_feature_top1": int(np.sum(baseline_wrong_top1)),
        "baseline_correct_queries": len(baseline_correct_top1),
        "baseline_correct_feature_top1": int(np.sum(baseline_correct_top1)),
    }


def permutation_null(
    groups: list[tuple[np.ndarray, int, bool]],
    *,
    repeats: int,
    rng: np.random.Generator,
) -> dict:
    observed = metrics(groups)
    margin_null = np.empty(repeats, dtype=float)
    top1_null = np.empty(repeats, dtype=float)
    pairwise_null = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        permuted = [
            (rng.permutation(values), truth_index, baseline_correct)
            for values, truth_index, baseline_correct in groups
        ]
        result = metrics(permuted)
        margin_null[repeat] = result["mean_truth_minus_best_wrong"]
        top1_null[repeat] = result["feature_top1"]
        pairwise_null[repeat] = result["pairwise_accuracy_with_half_ties"]

    def summary(values: np.ndarray, observed_value: float) -> dict:
        return {
            "mean": float(values.mean()),
            "p95": float(np.quantile(values, 0.95)),
            "empirical_one_sided_p": float(
                (1 + np.sum(values >= observed_value)) / (len(values) + 1)
            ),
        }

    return {
        "observed": observed,
        "within_query_candidate_permutation": {
            "repeats": repeats,
            "margin": summary(
                margin_null, observed["mean_truth_minus_best_wrong"]
            ),
            "top1": summary(top1_null, observed["feature_top1"]),
            "pairwise": summary(
                pairwise_null, observed["pairwise_accuracy_with_half_ties"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/validation/bioaware_reaction_context_directional_20260830"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/bioaware_a0_typed_context_20260830.json"),
    )
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"fail-closed: output exists: {args.output}")
    if args.permutations < 1000:
        raise ValueError("A0 requires at least 1000 permutations")

    inputs = sorted(args.input_dir.glob("*__candidate_context.csv.gz"))
    if not inputs:
        raise FileNotFoundError(f"no candidate context files in {args.input_dir}")
    rng = np.random.default_rng(args.seed)
    datasets: dict[str, dict] = {}
    for path in inputs:
        name = path.name.removesuffix("__candidate_context.csv.gz")
        frame = pd.read_csv(path)
        missing = set(FEATURES) - set(frame)
        if missing:
            raise RuntimeError(f"{path} missing A0 features: {sorted(missing)}")
        feature_reports = {
            feature: permutation_null(
                query_arrays(frame, feature),
                repeats=args.permutations,
                rng=rng,
            )
            for feature in FEATURES
        }
        datasets[name] = {
            "queries": int(frame["query_id"].nunique()),
            "evaluable_multi_candidate_queries": int(
                sum(len(group) >= 2 for _, group in frame.groupby("query_id", sort=False))
            ),
            "excluded_single_candidate_queries": int(
                sum(len(group) < 2 for _, group in frame.groupby("query_id", sort=False))
            ),
            "candidates": int(len(frame)),
            "baseline_recall1": float(
                frame.groupby("query_id", sort=False)["baseline_correct"].first().mean()
            ),
            "features": feature_reports,
            "input_sha256": sha256(path),
        }

    # Passing requires replication on the deployment-like automatic-seed cohort.
    # Published-seed headroom and MTBLS13729 are already exposed diagnostics.
    auto = datasets.get("mtbls1905_auto", {})
    replicated = []
    decision_audit: dict[str, dict] = {}
    multiplicity_alpha = 0.05 / len(FEATURES)
    for feature in FEATURES:
        report = auto.get("features", {}).get(feature)
        if report is None:
            continue
        observed = report["observed"]
        null = report["within_query_candidate_permutation"]
        feature_pass = (
            observed["queries"] >= 50
            and observed["baseline_wrong_queries"] >= 15
            and observed["queries_with_feature_variation"] >= 15
            and observed["baseline_wrong_feature_top1"] >= 3
            and
            observed["mean_truth_minus_best_wrong"] > 0
            and null["margin"]["empirical_one_sided_p"] <= multiplicity_alpha
            and null["pairwise"]["empirical_one_sided_p"] <= multiplicity_alpha
        )
        decision_audit[feature] = {
            "queries_ge_50": observed["queries"] >= 50,
            "baseline_errors_ge_15": observed["baseline_wrong_queries"] >= 15,
            "varying_queries_ge_15": (
                observed["queries_with_feature_variation"] >= 15
            ),
            "baseline_errors_rescued_ge_3": (
                observed["baseline_wrong_feature_top1"] >= 3
            ),
            "positive_mean_margin": observed["mean_truth_minus_best_wrong"] > 0,
            "bonferroni_margin_p_pass": (
                null["margin"]["empirical_one_sided_p"] <= multiplicity_alpha
            ),
            "bonferroni_pairwise_p_pass": (
                null["pairwise"]["empirical_one_sided_p"] <= multiplicity_alpha
            ),
            "pass": bool(feature_pass),
        }
        if feature_pass:
            replicated.append(feature)

    report = {
        "status": "bioaware_a0_typed_context_complete",
        "formal": False,
        "datasets_are_exposed_development": True,
        "null": (
            "within-query candidate assignment permutation; preserves each query's "
            "candidate count and feature-score multiset"
        ),
        "datasets": datasets,
        "replicated_features_on_automatic_seed_cohort": replicated,
        "automatic_seed_feature_decision_audit": decision_audit,
        "multiplicity_alpha": multiplicity_alpha,
        "pass_to_context_reranker_training": bool(replicated),
        "next_required_decoy": (
            "degree-, reaction-size-, and mass-difference-matched Rhea edge rewiring "
            "on a new cohort; the current permutation is the first falsification gate"
        ),
        "claim_limit": (
            "A0 is an exposed-data mechanism screen. Published-seed results are "
            "headroom, not deployable annotation performance."
        ),
        "parameters": {"permutations": args.permutations, "seed": args.seed},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
