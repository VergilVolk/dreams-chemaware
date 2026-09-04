#!/usr/bin/env python
"""Leave-biological-source-out stress test for the frozen network-only recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from audit_bioaware_metdna3_negative_loso_ablation import ABLATIONS
    from develop_bioaware_metdna3_negative_loso_ranker import (
        evaluate_fold, formula_bootstrap, summarize,
    )
except ModuleNotFoundError:  # pragma: no cover
    from tasks.audit_bioaware_metdna3_negative_loso_ablation import ABLATIONS
    from tasks.develop_bioaware_metdna3_negative_loso_ranker import (
        evaluate_fold, formula_bootstrap, summarize,
    )


def biological_source(unit: str) -> str:
    if "__" not in unit:
        raise ValueError(f"unit lacks chromatography suffix: {unit}")
    return unit.rsplit("__", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-features", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_loso_ranker_v3_identity_purged/candidate_features.csv.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_source_loso_v1"),
    )
    parser.add_argument(
        "--recipe-name", choices=sorted(ABLATIONS),
        default="network_only_same_edge_gate",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    candidates = pd.read_csv(args.candidate_features)
    candidates["biological_source"] = candidates["unit_id"].astype(str).map(biological_source)
    recipe = ABLATIONS[args.recipe_name]
    reports = {}
    all_transitions = {}
    for purge_formula in (False, True):
        label = "source_identity_formula_purged" if purge_formula else "source_identity_purged"
        outputs = []
        folds = []
        for source in sorted(candidates["biological_source"].unique()):
            test = candidates[candidates["biological_source"].eq(source)].copy()
            train = candidates[~candidates["biological_source"].eq(source)].copy()
            test_ids = set(test["truth_candidate_id"].astype(str))
            train = train[~train["truth_candidate_id"].astype(str).isin(test_ids)]
            test_formulas = set(test["truth_formula"].astype(str))
            if purge_formula:
                train = train[~train["truth_formula"].astype(str).isin(test_formulas)]
            if set(train["truth_candidate_id"].astype(str)) & test_ids:
                raise RuntimeError(f"{label}/{source}: truth identity overlap")
            if purge_formula and set(train["truth_formula"].astype(str)) & test_formulas:
                raise RuntimeError(f"{label}/{source}: truth formula overlap")
            result, fold = evaluate_fold(
                train, test, source, features=recipe["features"],
                require_raw_step0_edge=recipe["require_raw_step0_edge"],
            )
            fold.update({
                "heldout_biological_source": source,
                "test_units": sorted(test["unit_id"].unique()),
                "train_sources": sorted(train["biological_source"].unique()),
                "train_identities": int(train["truth_candidate_id"].nunique()),
                "train_formulas": int(train["truth_formula"].nunique()),
                "truth_identity_overlap": 0,
                "truth_formula_overlap": 0 if purge_formula else None,
                "result": summarize(result),
            })
            outputs.append(result)
            folds.append(fold)
        transitions = pd.concat(outputs, ignore_index=True)
        expected_queries = int(candidates["query_id"].nunique())
        if len(transitions) != expected_queries or transitions["query_id"].duplicated().any():
            raise RuntimeError(f"{label}: expected {expected_queries} unique queries")
        reports[label] = {
            "pooled": summarize(transitions),
            "formula_cluster_bootstrap_vs_dreams": formula_bootstrap(
                transitions, args.bootstrap_resamples, 20260901
            ),
            "folds": folds,
        }
        all_transitions[label] = transitions
        print(f"[source LOSO] {label}: {reports[label]['pooled']}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, transitions in all_transitions.items():
        transitions.to_csv(args.output_dir / f"{label}__transitions.csv.gz", index=False, compression="gzip")
    report = {
        "status": "bioaware_metdna3_negative_source_loso_complete",
        "formal": True,
        "protocol": "four-fold leave-biological-source-out; test identities purged; optional test-formula purge; explicitly selected frozen ablation recipe and gate",
        "recipe_name": args.recipe_name,
        "features": recipe["features"],
        "require_raw_step0_edge": bool(recipe["require_raw_step0_edge"]),
        "results": reports,
        "contracts": {
            "same_source_other_chromatography_in_training": False,
            "threshold_or_recipe_retuned": False,
            "P2b": "forbidden", "phenotype": "forbidden",
            "shared_embedding_changed": False,
        },
        "claim_limit": "Opened four-source transfer stress test, not a new external confirmation.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
