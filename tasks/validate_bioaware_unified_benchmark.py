#!/usr/bin/env python
"""Fail-closed validator for a BioAware unified benchmark output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    query_path = args.output_dir / "per_query.csv.gz"
    if not report_path.is_file() or not query_path.is_file():
        raise FileNotFoundError("benchmark report or per-query ledger is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_unified_benchmark_complete":
        raise RuntimeError("unexpected benchmark status")
    if report.get("formal") is not True:
        raise RuntimeError("benchmark is not formal")
    if report.get("benchmark_track") not in {
        "spectrum_reference", "spectrum_structure", "sample_context"
    }:
        raise RuntimeError("unknown benchmark track")
    if report.get("protocol_scope") not in {
        "opened_development", "public_test", "sealed_external"
    }:
        raise RuntimeError("unknown protocol scope")
    frame = pd.read_csv(query_path)
    required = {
        "method_id", "query_id", "formula_cluster", "source", "polarity",
        "strict_rank", "top1", "reciprocal_rank", "query_auroc",
        "truth_margin_vs_best_negative",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"per-query ledger missing columns: {sorted(missing)}")
    denominator = int(report["candidate_denominator"]["queries"])
    methods = set(report["methods"])
    if set(frame["method_id"].astype(str)) != methods:
        raise RuntimeError("report and per-query method sets differ")
    for method_id, group in frame.groupby("method_id", sort=False):
        if len(group) != denominator or group["query_id"].nunique() != denominator:
            raise RuntimeError(f"{method_id}: query denominator changed")
        if not group["strict_rank"].ge(1).all():
            raise RuntimeError(f"{method_id}: invalid rank")
        if not group["query_auroc"].between(0, 1).all():
            raise RuntimeError(f"{method_id}: invalid query AUROC")
        numeric = group[[
            "strict_rank", "reciprocal_rank", "query_auroc",
            "truth_margin_vs_best_negative",
        ]].to_numpy(float)
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"{method_id}: non-finite metric")
    contracts = report.get("contracts", {})
    if contracts.get("candidate_sets_identical_across_methods") is not True:
        raise RuntimeError("candidate-set identity contract is absent")
    if contracts.get("pairwise_dreams_auc_not_computed_by_this_candidate_ranker") is not True:
        raise RuntimeError("candidate and DreaMS pairwise AUROC were conflated")
    print(
        "[validate_bioaware_unified_benchmark] PASS",
        f"methods={len(methods)} queries={denominator:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()

