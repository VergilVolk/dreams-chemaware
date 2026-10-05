#!/usr/bin/env python
"""Evaluate arbitrary candidate scorers with one frozen BioAware metric engine.

Example
-------
python -u tasks/evaluate_bioaware_unified_benchmark.py \
  --candidate-manifest data/benchmark/candidates.csv.gz \
  --method official_dreams=data/benchmark/dreams.csv.gz \
  --method bioaware=data/benchmark/bioaware.csv.gz \
  --baseline-method official_dreams \
  --benchmark-track sample_context \
  --protocol-scope opened_development \
  --output-dir data/validation/bioaware_benchmark_run_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

from bioaware_unified_benchmark_core import (
    compare_methods,
    evaluate_method,
    summarize_method,
    validate_candidate_manifest,
)


METHOD_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="METHOD_ID=CSV",
        help="Repeat once per candidate scorer.",
    )
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument(
        "--benchmark-track",
        required=True,
        choices=("spectrum_reference", "spectrum_structure", "sample_context"),
    )
    parser.add_argument(
        "--protocol-scope",
        required=True,
        choices=("opened_development", "public_test", "sealed_external"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int)
    parser.add_argument("--expected-candidate-rows", type=int)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261003)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise RuntimeError(f"unsupported table format: {path}")


def parse_methods(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise RuntimeError(f"method must use METHOD_ID=CSV syntax: {value}")
        method_id, raw_path = value.split("=", 1)
        method_id = method_id.strip()
        if not METHOD_PATTERN.fullmatch(method_id):
            raise RuntimeError(f"invalid method ID: {method_id!r}")
        if method_id in output:
            raise RuntimeError(f"duplicate method ID: {method_id}")
        output[method_id] = Path(raw_path)
    return output


def subgroup_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    panels: dict[str, pd.DataFrame] = {"overall": frame}
    if "near_query" in frame.columns:
        panels["near"] = frame.loc[frame["near_query"].astype(bool)]
        panels["non_near"] = frame.loc[~frame["near_query"].astype(bool)]
    for polarity, subset in frame.groupby("polarity", sort=True):
        panels[f"polarity::{polarity}"] = subset
    for source, subset in frame.groupby("source", sort=True):
        panels[f"source::{source}"] = subset
    return {
        name: summarize_method(subset)
        for name, subset in panels.items()
        if len(subset) > 0
    }


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise RuntimeError("report contains a non-finite number")
        return number
    return value


def main() -> None:
    args = arguments()
    methods = parse_methods(args.method)
    if args.baseline_method not in methods:
        raise RuntimeError("baseline method is not present in --method inputs")
    if args.bootstrap_resamples < 100:
        raise RuntimeError("bootstrap-resamples must be at least 100")
    if args.expected_queries is not None and args.expected_queries < 1:
        raise RuntimeError("expected-queries must be positive")
    if args.expected_candidate_rows is not None and args.expected_candidate_rows < 2:
        raise RuntimeError("expected-candidate-rows must be at least two")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite benchmark output: {args.output_dir}")

    candidates = validate_candidate_manifest(read_table(args.candidate_manifest))
    n_queries = int(candidates["query_id"].nunique())
    if args.expected_queries is not None and n_queries != args.expected_queries:
        raise RuntimeError(
            f"query count mismatch: expected {args.expected_queries}, observed {n_queries}"
        )
    if (
        args.expected_candidate_rows is not None
        and len(candidates) != args.expected_candidate_rows
    ):
        raise RuntimeError(
            "candidate row count mismatch: expected "
            f"{args.expected_candidate_rows}, observed {len(candidates)}"
        )

    per_method: dict[str, pd.DataFrame] = {}
    method_reports: dict[str, object] = {}
    method_hashes: dict[str, str] = {}
    for method_id, path in methods.items():
        predictions = read_table(path)
        per_query = evaluate_method(candidates, predictions, method_id)
        per_method[method_id] = per_query
        method_reports[method_id] = {"panels": subgroup_metrics(per_query)}
        method_hashes[method_id] = sha256(path)
        print(
            f"[benchmark] {method_id}: n={len(per_query):,} "
            f"R1={float(per_query['top1'].mean()):.6f}",
            flush=True,
        )

    baseline = per_method[args.baseline_method]
    comparisons: dict[str, object] = {}
    paired_ledgers: list[pd.DataFrame] = []
    for offset, (method_id, per_query) in enumerate(per_method.items()):
        if method_id == args.baseline_method:
            continue
        comparison, paired = compare_methods(
            baseline,
            per_query,
            resamples=args.bootstrap_resamples,
            seed=args.seed + 10 * offset,
        )
        comparisons[f"{method_id}_vs_{args.baseline_method}"] = comparison
        paired.insert(0, "contender_method", method_id)
        paired.insert(1, "baseline_method", args.baseline_method)
        paired_ledgers.append(paired)

    report = json_ready({
        "status": "bioaware_unified_benchmark_complete",
        "formal": True,
        "benchmark_track": args.benchmark_track,
        "protocol_scope": args.protocol_scope,
        "baseline_method": args.baseline_method,
        "candidate_denominator": {
            "queries": n_queries,
            "candidate_rows": int(len(candidates)),
            "formula_clusters": int(candidates["formula_cluster"].nunique()),
            "sources": int(candidates["source"].nunique()),
            "near_queries": (
                int(candidates.groupby("query_id")["near_query"].first().sum())
                if "near_query" in candidates.columns else None
            ),
        },
        "methods": method_reports,
        "comparisons": comparisons,
        "tie_contract": {
            "candidate_rank": "rank = 1 + number of negative scores >= truth score",
            "query_auroc": "standard pair statistic with half credit for score ties",
        },
        "contracts": {
            "candidate_sets_identical_across_methods": True,
            "one_truth_per_query": True,
            "missing_queries_forbidden": True,
            "opened_and_sealed_scopes_not_pooled": True,
            "pairwise_dreams_auc_not_computed_by_this_candidate_ranker": True,
        },
        "parameters": {
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "seed": int(args.seed),
        },
        "provenance": {
            "candidate_manifest": str(args.candidate_manifest),
            "candidate_manifest_sha256": sha256(args.candidate_manifest),
            "prediction_sha256": method_hashes,
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Protocol-bound candidate-ranking comparison only. It does not by itself "
            "establish external generalization, reaction-specific causality, shared-"
            "embedding gain, or SOTA."
        ),
    })

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=args.output_dir.name + ".partial.", dir=args.output_dir.parent
    ))
    try:
        all_query = pd.concat(per_method.values(), ignore_index=True)
        all_query.to_csv(temporary / "per_query.csv.gz", index=False)
        if paired_ledgers:
            pd.concat(paired_ledgers, ignore_index=True).to_csv(
                temporary / "paired_comparisons.csv.gz", index=False
            )
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"[frozen] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

