#!/usr/bin/env python
"""Fold the frozen B35 per-query ledger into the unified paired comparison.

B35's artifact stores one row per rotation instance; physical queries repeat
under KGMN seed masks and must carry one vote each.  This script verifies
rank consistency within each physical query, folds to the physical
denominator, and feeds the paired official-versus-BioAware frames into the
unified engine's `compare_methods`.  No per-candidate score is invented; the
paired half of the unified ruler is the correct interface for this artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_unified_benchmark_core import compare_methods  # noqa: E402

FROZEN_DELTA_RECALL_AT_1 = 0.02882
FROZEN_CORRECTED = 50
FROZEN_INTRODUCED = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-query", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite B35 unified output: {output}")
    output.mkdir(parents=True)

    raw = pd.read_csv(args.per_query.resolve())
    required = {
        "physical_key", "source", "polarity", "evaluation_stratum",
        "truth_formula", "official_rank", "bioaware_rank",
        "official_reciprocal_rank", "bioaware_reciprocal_rank",
        "official_query_auroc", "bioaware_query_auroc", "intervene",
    }
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"B35 per-query artifact missing columns: {sorted(missing)}")

    inconsistent = (
        raw.groupby("physical_key")[["official_rank", "bioaware_rank", "source", "polarity"]]
        .nunique()
        .gt(1)
        .any(axis=1)
        .sum()
    )
    if inconsistent:
        raise RuntimeError(
            f"{inconsistent} physical queries have inconsistent repeated rows"
        )
    folded = raw.drop_duplicates("physical_key", keep="first").reset_index(drop=True)

    baseline = pd.DataFrame({
        "query_id": folded["physical_key"].astype(str),
        "formula_cluster": folded["truth_formula"].astype(str),
        "source": folded["source"].astype(str),
        "polarity": folded["polarity"].astype(str),
        "top1": folded["official_rank"].eq(1),
        "reciprocal_rank": pd.to_numeric(folded["official_reciprocal_rank"], errors="raise"),
        "query_auroc": pd.to_numeric(folded["official_query_auroc"], errors="raise"),
    })
    contender = pd.DataFrame({
        "query_id": folded["physical_key"].astype(str),
        "formula_cluster": folded["truth_formula"].astype(str),
        "source": folded["source"].astype(str),
        "polarity": folded["polarity"].astype(str),
        "top1": folded["bioaware_rank"].eq(1),
        "reciprocal_rank": pd.to_numeric(folded["bioaware_reciprocal_rank"], errors="raise"),
        "query_auroc": pd.to_numeric(folded["bioaware_query_auroc"], errors="raise"),
    })

    comparison, _joined = compare_methods(
        baseline, contender, resamples=args.bootstrap_resamples, seed=args.seed
    )
    gate_pass = (
        abs(comparison["delta_recall_at_1"] - FROZEN_DELTA_RECALL_AT_1) < 5e-4
        and comparison["corrected"] == FROZEN_CORRECTED
        and comparison["introduced"] == FROZEN_INTRODUCED
    )

    by_stratum = {}
    for stratum, group in folded.groupby("evaluation_stratum", sort=True):
        keys = set(group["physical_key"].astype(str))
        base = baseline[baseline["query_id"].isin(keys)]
        test = contender[contender["query_id"].isin(keys)]
        stats, _ = compare_methods(base, test, resamples=1000, seed=args.seed + 7)
        by_stratum[str(stratum)] = {
            "queries": int(len(base)),
            "delta_recall_at_1": stats["delta_recall_at_1"],
            "corrected": stats["corrected"],
            "introduced": stats["introduced"],
        }
    by_source = {}
    for source, group in folded.groupby("source", sort=True):
        keys = set(group["physical_key"].astype(str))
        base = baseline[baseline["query_id"].isin(keys)]
        test = contender[contender["query_id"].isin(keys)]
        stats, _ = compare_methods(base, test, resamples=1000, seed=args.seed + 13)
        by_source[str(source)] = {
            "queries": int(len(base)),
            "delta_recall_at_1": stats["delta_recall_at_1"],
        }

    payload = {
        "status": "bioaware_trackc_b35_unified_comparison",
        "scope": "opened_development_cross_fitted",
        "rotation_rows": int(len(raw)),
        "physical_queries": int(len(folded)),
        "comparison": comparison,
        "by_stratum": by_stratum,
        "by_source": by_source,
        "known_answer_gate": {
            "frozen_delta_recall_at_1": FROZEN_DELTA_RECALL_AT_1,
            "frozen_corrected": FROZEN_CORRECTED,
            "frozen_introduced": FROZEN_INTRODUCED,
            "pass": bool(gate_pass),
        },
        "claim_limit": (
            "B30/B35 opened cross-fitted development result; the B44 external "
            "reversal remains in the limitation table. Not a sealed-external row."
        ),
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        temporary = Path(handle.name)
    temporary.replace(output / "report.json")
    folded[[
        "physical_key", "source", "polarity", "evaluation_stratum",
        "truth_formula", "official_rank", "bioaware_rank", "intervene", "b30_veto",
    ]].to_csv(output / "physical_query_ledger.csv.gz", index=False, compression="gzip")
    print(json.dumps({
        "status": payload["status"],
        "physical_queries": payload["physical_queries"],
        "delta_recall_at_1": comparison["delta_recall_at_1"],
        "corrected": comparison["corrected"],
        "introduced": comparison["introduced"],
        "mcnemar_exact_p": comparison["mcnemar_exact_p"],
        "gate_pass": bool(gate_pass),
    }, indent=2), flush=True)
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
