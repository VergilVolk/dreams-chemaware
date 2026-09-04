#!/usr/bin/env python
"""Candidate-label permutation null for the discovered negative BioAware recipe.

The full network feature block is permuted within each query.  This preserves
the candidate count, marginal feature distributions, and within-block feature
covariance, while destroying the candidate-specific network assignment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from audit_bioaware_metdna3_negative_loso_ablation import ABLATIONS
    from audit_bioaware_metdna3_negative_source_loso import biological_source
    from develop_bioaware_metdna3_negative_loso_ranker import evaluate_fold, summarize
except ModuleNotFoundError:  # pragma: no cover
    from tasks.audit_bioaware_metdna3_negative_loso_ablation import ABLATIONS
    from tasks.audit_bioaware_metdna3_negative_source_loso import biological_source
    from tasks.develop_bioaware_metdna3_negative_loso_ranker import evaluate_fold, summarize


def permute_candidate_feature_blocks(
    frame: pd.DataFrame, features: list[str], rng: np.random.Generator
) -> pd.DataFrame:
    output = frame.copy()
    source_values = frame[features].to_numpy(float)
    target_values = source_values.copy()
    for indices in frame.groupby("query_id", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        target_values[indices] = source_values[rng.permutation(indices)]
    output.loc[:, features] = target_values
    return output


def source_formula_purged_run(
    candidates: pd.DataFrame,
    features: list[str],
    require_raw_step0_edge: bool,
) -> pd.DataFrame:
    outputs = []
    for source in sorted(candidates["biological_source"].unique()):
        test = candidates[candidates["biological_source"].eq(source)].copy()
        train = candidates[~candidates["biological_source"].eq(source)].copy()
        test_ids = set(test["truth_candidate_id"].astype(str))
        test_formulas = set(test["truth_formula"].astype(str))
        train = train[
            (~train["truth_candidate_id"].astype(str).isin(test_ids))
            & (~train["truth_formula"].astype(str).isin(test_formulas))
        ]
        result, _ = evaluate_fold(
            train, test, source, features=features,
            require_raw_step0_edge=require_raw_step0_edge,
        )
        outputs.append(result)
    transitions = pd.concat(outputs, ignore_index=True)
    expected_queries = int(candidates["query_id"].nunique())
    if len(transitions) != expected_queries or transitions["query_id"].duplicated().any():
        raise RuntimeError("source/formula-purged coverage changed")
    return transitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-features", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_loso_ranker_v3_identity_purged/candidate_features.csv.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_candidate_permutation_v1"),
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--recipe-name", choices=sorted(ABLATIONS),
        default="network_only_same_edge_gate",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    candidates = pd.read_csv(args.candidate_features)
    candidates["biological_source"] = candidates["unit_id"].astype(str).map(biological_source)
    recipe = ABLATIONS[args.recipe_name]
    features = recipe["features"]
    # DreaMS cosine is a fixed baseline feature.  The null destroys only the
    # candidate-specific assignment of the network evidence block.
    permutation_features = [feature for feature in features if feature != "spectral_score"]
    if not permutation_features:
        raise RuntimeError("candidate-network permutation requires at least one network feature")
    observed_transitions = source_formula_purged_run(
        candidates, features, recipe["require_raw_step0_edge"]
    )
    observed = summarize(observed_transitions)
    rng = np.random.default_rng(args.seed)
    null_rows = []
    for repeat in range(args.repeats):
        permuted = permute_candidate_feature_blocks(candidates, permutation_features, rng)
        transitions = source_formula_purged_run(
            permuted, features, recipe["require_raw_step0_edge"]
        )
        summary = summarize(transitions)
        null_rows.append({"repeat": repeat, **summary})
        if (repeat + 1) % 10 == 0 or repeat + 1 == args.repeats:
            print(f"[candidate permutation] {repeat + 1}/{args.repeats}", flush=True)
    null = pd.DataFrame(null_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    null.to_csv(args.output_dir / "candidate_permutation_null.csv.gz", index=False, compression="gzip")
    metrics = {}
    for metric in ("delta_recall1", "risk_weighted_net_lambda2", "corrected", "introduced"):
        values = null[metric].to_numpy(float)
        observed_value = float(observed[metric])
        metrics[metric] = {
            "observed": observed_value,
            "null_mean": float(values.mean()),
            "null_p05": float(np.quantile(values, 0.05)),
            "null_p95": float(np.quantile(values, 0.95)),
            "empirical_one_sided_p_ge_observed": float(
                (1 + np.sum(values >= observed_value)) / (1 + len(values))
            ),
        }
    report = {
        "status": "bioaware_metdna3_negative_candidate_permutation_complete",
        "formal": True,
        "protocol": "source+identity+formula-purged LOSO; within-query joint network-feature permutation; frozen recipe and gate",
        "recipe_name": args.recipe_name,
        "model_features": features,
        "permuted_network_features": permutation_features,
        "require_raw_step0_edge": bool(recipe["require_raw_step0_edge"]),
        "repeats": args.repeats,
        "observed": observed,
        "null_metrics": metrics,
        "contracts": {
            "candidate_count_preserved": True,
            "within_query_feature_multiset_preserved": True,
            "network_feature_covariance_preserved_as_joint_block": True,
            "candidate_specific_network_assignment_destroyed": True,
            "threshold_or_recipe_retuned": False,
            "P2b": "forbidden", "phenotype": "forbidden",
        },
        "pass": metrics["delta_recall1"]["empirical_one_sided_p_ge_observed"] <= 0.05,
        "claim_limit": "Permutation falsification on opened sources; not independent external confirmation.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
