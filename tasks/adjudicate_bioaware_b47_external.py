#!/usr/bin/env python
"""Apply the preregistered integrated BioAware claim gates to B47 results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bioaware_unified_benchmark_core import validate_candidate_manifest


CONTROL_METHODS = (
    "catalogue_degree", "network", "degree_rewired",
    "seed_context_permuted", "matched_non_neighbor",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads((args.evaluation / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_unified_benchmark_complete":
        raise RuntimeError("B47 evaluation is incomplete")
    if report.get("benchmark_track") != "sample_context" or report.get("protocol_scope") != "sealed_external":
        raise RuntimeError("B47 result is not a sealed sample-context evaluation")
    key = "bioaware_exact_event_vs_frozen_unary"
    comparison = report.get("comparisons", {}).get(key)
    if not isinstance(comparison, dict):
        raise RuntimeError(f"missing primary comparison: {key}")
    per_query = pd.read_csv(args.evaluation / "per_query.csv.gz")
    per_query["query_id"] = per_query["query_id"].astype(str)
    pivot = per_query.pivot(index="query_id", columns="method_id", values="top1")
    required_methods = {"frozen_unary", "bioaware_exact_event", *CONTROL_METHODS}
    missing = required_methods - set(pivot.columns)
    if missing:
        raise RuntimeError(f"B47 evaluation misses preregistered controls: {sorted(missing)}")
    delta = pivot["bioaware_exact_event"].astype(float) - pivot["frozen_unary"].astype(float)
    query_meta = per_query.loc[
        per_query["method_id"].eq("frozen_unary"),
        [column for column in ("query_id", "source", "near_query") if column in per_query],
    ].set_index("query_id")
    source_delta = delta.groupby(query_meta["source"]).mean()
    near_delta = (
        float(delta.loc[query_meta["near_query"].astype(bool)].mean())
        if "near_query" in query_meta and query_meta["near_query"].astype(bool).any()
        else 0.0
    )
    manifest = validate_candidate_manifest(pd.read_csv(args.candidate_manifest))
    truth = manifest.loc[manifest["is_truth"], ["query_id", "candidate_id"]]
    corrected_queries = delta.index[(pivot["frozen_unary"] == 0) & (pivot["bioaware_exact_event"] == 1)]
    corrected_identities = int(
        truth.loc[truth["query_id"].isin(corrected_queries), "candidate_id"].nunique()
    )
    control_advantage = {
        method: float((pivot["bioaware_exact_event"].astype(float) - pivot[method].astype(float)).mean())
        for method in CONTROL_METHODS
    }
    formula_ci = comparison["formula_cluster_top1_ci"]
    source_ci = comparison["source_cluster_top1_ci"]
    gates = {
        "delta_recall_at_1_at_least_3pp": float(comparison["delta_recall_at_1"]) >= 0.03,
        "formula_ci_low_positive": float(formula_ci["ci_low"]) > 0,
        "source_ci_low_positive": float(source_ci["ci_low"]) > 0,
        "corrected_gt_2x_introduced": int(comparison["corrected"]) > 2 * int(comparison["introduced"]),
        "at_least_50_corrected_identities": corrected_identities >= 50,
        "every_source_nonnegative": bool((source_delta >= 0).all()),
        "near_stratum_nonnegative": near_delta >= 0,
        "beats_catalogue_network_and_nulls": all(value > 0 for value in control_advantage.values()),
    }
    passed = all(gates.values())
    output = {
        "status": "BIOAWARE_B47_EXTERNAL_ADJUDICATION_COMPLETE",
        "gates": gates,
        "promotion_authorized": passed,
        "verdict": "PROMOTE_SPARSE_CONTEXT_RESIDUAL" if passed else "STOP_NO_BIOAWARE_GAIN_CLAIM",
        "primary_comparison": comparison,
        "corrected_identities": corrected_identities,
        "per_source_delta_recall_at_1": {str(k): float(v) for k, v in source_delta.items()},
        "near_delta_recall_at_1": near_delta,
        "exact_event_minus_control_recall_at_1": control_advantage,
        "claim_limit": "Failure of any preregistered gate forbids a BioAware performance claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
