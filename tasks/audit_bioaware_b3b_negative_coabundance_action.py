#!/usr/bin/env python
"""Decisive negative-ion-only repair of BioAware B3 co-abundance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b3_reaction_coabundance_action import (  # noqa: E402
    MATCHED_RANDOM_FEATURES,
    atomic_json,
)
from develop_bioaware_b1_multisource_action import (  # noqa: E402
    EXPECTED_SOURCES,
    evaluate_model,
    fit_ranker,
    paired_cluster_bootstrap,
    paired_transition,
    sha256,
    summarize,
)


# Frozen before B3b outcomes: the two models differ by exactly one coordinate.
COMPARATOR_FEATURES = [
    name for name in MATCHED_RANDOM_FEATURES
    if name not in {
        "coabundance_random_positive_top3_mean",
        "coabundance_random_negative_top3_mean",
        "coabundance_random_sign_stability_top3_mean",
    }
]
ACTION_FEATURES = COMPARATOR_FEATURES + ["coabundance_abs_excess_top3_mean"]


def negative_source_loso(
    candidates: pd.DataFrame, recipe: str, features: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    negative = candidates.loc[candidates.polarity.astype(str).eq("negative")].copy()
    outputs = []
    reports = []
    for held_source in EXPECTED_SOURCES:
        test = negative.loc[negative.source.eq(held_source)].copy()
        held_identities = set(test.truth_candidate_id.astype(str))
        held_formulas = set(test.truth_formula.astype(str))
        train = negative.loc[
            ~negative.source.eq(held_source)
            & ~negative.truth_candidate_id.astype(str).isin(held_identities)
            & ~negative.truth_formula.astype(str).isin(held_formulas)
        ].copy()
        query_rows = train[["query_id", "truth_candidate_id", "truth_formula"]].drop_duplicates()
        if query_rows.query_id.nunique() < 50:
            raise RuntimeError(f"{held_source}: fewer than 50 negative-ion training queries")
        if set(query_rows.truth_candidate_id.astype(str)) & held_identities:
            raise RuntimeError(f"{held_source}: truth-identity leakage")
        if set(query_rows.truth_formula.astype(str)) & held_formulas:
            raise RuntimeError(f"{held_source}: truth-formula leakage")
        scaler, model = fit_ranker(train, features)
        outputs.append(evaluate_model(test, scaler, model, features, held_source, recipe))
        reports.append({
            "held_source": held_source,
            "train_queries_after_purge": int(query_rows.query_id.nunique()),
            "train_identities_after_purge": int(query_rows.truth_candidate_id.nunique()),
            "train_formulas_after_purge": int(query_rows.truth_formula.nunique()),
            "test_queries": int(test.query_id.nunique()),
            "identity_overlap": 0,
            "formula_overlap": 0,
            "coefficients": {
                name: float(value)
                for name, value in zip(features, model.coef_[0], strict=True)
            },
        })
    combined = pd.concat(outputs, ignore_index=True)
    if len(combined) != negative.query_id.nunique():
        raise RuntimeError("negative source-LOSO changed query coverage")
    return combined, reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b3-candidate-features", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if not args.b3_candidate_features.is_file():
        raise FileNotFoundError(args.b3_candidate_features)
    candidates = pd.read_csv(args.b3_candidate_features)
    required = set(ACTION_FEATURES) | {
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "source", "polarity", "baseline_correct",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise RuntimeError(f"B3 feature cache missing columns: {sorted(missing)}")
    negative_queries = candidates.loc[candidates.polarity.eq("negative"), [
        "query_id", "truth_candidate_id", "truth_formula", "source"
    ]].drop_duplicates()
    if len(negative_queries) != 548:
        raise RuntimeError(f"negative protocol changed: {len(negative_queries)} != 548")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    reports = {}
    for recipe, features in (
        ("matched_random_minimal", COMPARATOR_FEATURES),
        ("reaction_coabundance_minimal", ACTION_FEATURES),
    ):
        result, folds = negative_source_loso(candidates, recipe, features)
        path = args.output_dir / f"{recipe}__transitions.csv.gz"
        result.to_csv(path, index=False, compression="gzip")
        results[recipe] = result
        reports[recipe] = {
            "features": features,
            "pooled": summarize(result),
            "by_source": {
                source: summarize(result.loc[result.source.eq(source)])
                for source in EXPECTED_SOURCES
            },
            "folds": folds,
            "transitions_sha256": sha256(path),
        }

    action = results["reaction_coabundance_minimal"]
    comparator = results["matched_random_minimal"]
    transition = paired_transition(action, comparator)
    identity_ci = paired_cluster_bootstrap(
        action, comparator, "truth_candidate_id", args.bootstrap_resamples, args.seed + 1
    )
    formula_ci = paired_cluster_bootstrap(
        action, comparator, "truth_formula", args.bootstrap_resamples, args.seed + 2
    )
    by_source = {}
    for source in EXPECTED_SOURCES:
        left = action.loc[action.source.eq(source)]
        right = comparator.loc[comparator.source.eq(source)]
        by_source[source] = {
            **paired_transition(left, right),
            "delta": float(left.gated_correct.mean() - right.gated_correct.mean()),
        }
    paired = action[["query_id", "gated_correct", "truth_candidate_id"]].merge(
        comparator[["query_id", "gated_correct"]], on="query_id",
        suffixes=("_action", "_comparator"), validate="one_to_one",
    )
    corrected_identities = int(paired.loc[
        paired.gated_correct_action & ~paired.gated_correct_comparator,
        "truth_candidate_id",
    ].nunique())
    gates = {
        "increment_ge_3pp": formula_ci["mean"] >= 0.03,
        "identity_ci_low_positive": identity_ci["ci_low"] > 0,
        "formula_ci_low_positive": formula_ci["ci_low"] > 0,
        "corrected_gt_2x_introduced": transition["corrected_vs_right"] > 2 * transition["introduced_vs_right"],
        "corrected_identities_ge_20": corrected_identities >= 20,
        "all_four_sources_nonnegative": all(item["delta"] >= 0 for item in by_source.values()),
    }
    report = {
        "status": "bioaware_b3b_negative_coabundance_action_complete",
        "formal": True,
        "protocol": "negative-ion-only source-LOSO; action and comparator differ by one excess-correlation coordinate",
        "candidate_protocol": {
            "queries": int(len(negative_queries)),
            "identities": int(negative_queries.truth_candidate_id.nunique()),
            "formulas": int(negative_queries.truth_formula.nunique()),
            "sources": int(negative_queries.source.nunique()),
            "baseline_recall1": float(negative_queries.merge(
                candidates[["query_id", "baseline_correct"]].drop_duplicates(), on="query_id"
            ).baseline_correct.mean()),
        },
        "recipes": reports,
        "primary_action_vs_matched_random": {
            "transition": transition,
            "corrected_identities": corrected_identities,
            "identity_cluster_bootstrap": identity_ci,
            "formula_cluster_bootstrap": formula_ci,
            "by_source": by_source,
        },
        "gates": gates,
        "pass_to_context_model": bool(all(gates.values())),
        "contracts": {
            "negative_training_only": True,
            "positive_interventions": 0,
            "single_incremental_reaction_coordinate": True,
            "held_source_truth_identity_and_formula_purged": True,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "b3_candidate_features": sha256(args.b3_candidate_features),
            "script": sha256(Path(__file__)),
        },
        "decision": "Failure terminates six-replicate co-abundance as a BioAware action; success permits frozen context-model validation only.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
