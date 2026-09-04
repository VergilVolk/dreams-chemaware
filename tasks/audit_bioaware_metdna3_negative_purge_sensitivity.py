#!/usr/bin/env python
"""Formula- and reaction-component-purged sensitivity for negative BioAware.

This is a post-discovery stress test.  It does not choose thresholds or feature
families.  The network-only recipe and deployment gate are copied unchanged
from the identity-purged ablation.
"""
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


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def step0_components(edges_path: Path, identities: pd.Series) -> dict[str, str]:
    edges = pd.read_csv(edges_path)
    edges = edges[edges["minimum_step"].eq(0)]
    union = UnionFind()
    for row in edges.itertuples(index=False):
        union.union(str(row.ik14_a), str(row.ik14_b))
    result = {}
    for value in identities.astype(str).unique():
        result[value] = union.find(value) if value in union.parent else f"singleton:{value}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-features", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_loso_ranker_v3_identity_purged/candidate_features.csv.gz"),
    )
    parser.add_argument(
        "--edges", type=Path,
        default=Path("data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_purge_sensitivity_v1"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidate_features)
    component = step0_components(args.edges, candidates["truth_candidate_id"])
    candidates["truth_step0_component"] = candidates["truth_candidate_id"].astype(str).map(component)
    recipe = ABLATIONS["network_only_same_edge_gate"]
    all_reports = {}
    for purge in ("identity", "formula", "step0_component"):
        outputs = []
        folds = []
        for unit in sorted(candidates["unit_id"].unique()):
            test = candidates[candidates["unit_id"].eq(unit)].copy()
            train = candidates[~candidates["unit_id"].eq(unit)].copy()
            test_ids = set(test["truth_candidate_id"].astype(str))
            train = train[~train["truth_candidate_id"].astype(str).isin(test_ids)]
            if purge == "formula":
                heldout = set(test["truth_formula"].astype(str))
                train = train[~train["truth_formula"].astype(str).isin(heldout)]
                overlap = set(train["truth_formula"].astype(str)) & heldout
            elif purge == "step0_component":
                heldout = set(test["truth_step0_component"].astype(str))
                train = train[~train["truth_step0_component"].astype(str).isin(heldout)]
                overlap = set(train["truth_step0_component"].astype(str)) & heldout
            else:
                overlap = set(train["truth_candidate_id"].astype(str)) & test_ids
            if overlap:
                raise RuntimeError(f"{purge}/{unit}: purge overlap remains")
            if train["query_id"].nunique() < 50:
                raise RuntimeError(f"{purge}/{unit}: fewer than 50 training queries")
            result, fold = evaluate_fold(
                train, test, str(unit), features=recipe["features"],
                require_raw_step0_edge=recipe["require_raw_step0_edge"],
            )
            fold.update({
                "purge": purge,
                "train_identities": int(train["truth_candidate_id"].nunique()),
                "train_formulas": int(train["truth_formula"].nunique()),
                "train_step0_components": int(train["truth_step0_component"].nunique()),
                "purged_overlap": 0,
                "result": summarize(result),
            })
            outputs.append(result)
            folds.append(fold)
        transitions = pd.concat(outputs, ignore_index=True)
        if len(transitions) != 595 or transitions["query_id"].duplicated().any():
            raise RuntimeError(f"{purge}: query coverage changed")
        transitions.to_csv(
            args.output_dir / f"network_only__{purge}_purged_transitions.csv.gz",
            index=False, compression="gzip",
        )
        all_reports[purge] = {
            "pooled": summarize(transitions),
            "formula_cluster_bootstrap_vs_dreams": formula_bootstrap(
                transitions, args.bootstrap_resamples, 20260901
            ),
            "folds": folds,
        }
        print(f"[purge] {purge}: {all_reports[purge]['pooled']}", flush=True)
    report = {
        "status": "bioaware_metdna3_negative_purge_sensitivity_complete",
        "formal": True,
        "recipe": "post-discovery frozen network-only scoring plus raw step0 safety gate",
        "features": recipe["features"],
        "purges": all_reports,
        "contracts": {
            "threshold_or_recipe_retuned": False,
            "identity_purge_always_applied": True,
            "P2b": "forbidden", "phenotype": "forbidden",
            "shared_embedding_changed": False,
        },
        "claim_limit": "Opened-cohort sensitivity only; reaction-component purge is underpowered and not an external confirmation.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
