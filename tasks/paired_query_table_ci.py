#!/usr/bin/env python
"""Formula-cluster paired CI between two methods from saved GNPS query tables.

The article benchmark report pairs every method against ``official_dreams``
only.  This script closes the remaining pairwise questions (for example
weighted entropy vs Noise V1, or frozen P2b vs its base embedding) from the
already-saved ``queries_<panel>_<method>.csv.gz`` tables, without touching
any score or rerunning anything.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_gnps_gold_silver_10ppm_embeddings import cluster_ci

PANELS = ("identity_disjoint", "formula_disjoint")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument("--candidate-method", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261003)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    report: dict[str, object] = {
        "status": "gnps_article_paired_table_ci_complete",
        "baseline_method": args.baseline_method,
        "candidate_method": args.candidate_method,
        "bootstrap_resamples": args.bootstrap_resamples,
        "panels": {},
    }
    for panel_index, panel in enumerate(PANELS):
        baseline = pd.read_csv(
            args.evaluation / f"queries_{panel}_{args.baseline_method}.csv.gz",
        ).sort_values("query_index").reset_index(drop=True)
        candidate = pd.read_csv(
            args.evaluation / f"queries_{panel}_{args.candidate_method}.csv.gz",
        ).sort_values("query_index").reset_index(drop=True)
        panel_block: dict[str, object] = {
            "queries": int(len(baseline)),
        }
        formulas = baseline["query_formula"].astype(str).to_numpy()
        near_mask = baseline["near"].to_numpy(bool)
        for label, column, near in (
            ("recall_at_1", "rank", False), ("near_recall_at_1", "rank", True),
            ("mrr", "reciprocal_rank", False),
        ):
            if column == "rank":
                left_values = (baseline["rank"].to_numpy(float) == 1).astype(float)
                right_values = (candidate["rank"].to_numpy(float) == 1).astype(float)
            else:
                left_values = baseline[column].to_numpy(float)
                right_values = candidate[column].to_numpy(float)
            values = right_values - left_values
            selected_formulas, selected_values = (
                (formulas[near_mask], values[near_mask]) if near
                else (formulas, values)
            )
            panel_block[label] = cluster_ci(
                selected_formulas, selected_values, args.bootstrap_resamples,
                args.bootstrap_seed + panel_index * 1000, hypotheses=24,
            )
            panel_block[f"{label}_queries"] = int(len(selected_values))
        report["panels"][panel] = panel_block
    report["claim_limit"] = (
        "Post-hoc pairwise comparisons over the same frozen tables; familywise "
        "correction assumes 24 hypotheses per metric."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
