#!/usr/bin/env python
"""Evaluate Contrast Atlas selective risk--coverage over frozen FDR thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adjudicate_contrast_atlas_unknown_increment import metrics
from build_contrast_atlas_query_outcomes import build_outcomes


DEFAULT_THRESHOLDS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20)


def selective_curve(
    candidates: pd.DataFrame,
    registry: pd.DataFrame,
    truth: pd.DataFrame,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    if not thresholds or any(value <= 0 or value > 1 for value in thresholds):
        raise ValueError("FDR thresholds must lie in (0,1]")
    if tuple(sorted(set(thresholds))) != thresholds:
        raise ValueError("FDR thresholds must be unique and increasing")
    rows = []
    for threshold in thresholds:
        outcomes, _ = build_outcomes(candidates, registry, truth, threshold)
        for world, group in outcomes.groupby("world", sort=True):
            summary = metrics(group)
            answered = int(group["answered"].sum())
            incorrect_answered = int((group["answered"] & ~group["correct"]).sum())
            rows.append({
                "world": world,
                "fdr_threshold": threshold,
                **summary,
                "answered_queries": answered,
                "incorrect_answered": incorrect_answered,
                "selective_risk": incorrect_answered / answered if answered else 0.0,
            })
    curve = pd.DataFrame(rows)
    for world, group in curve.groupby("world"):
        ordered = group.sort_values("fdr_threshold")
        if (np.diff(ordered["coverage"].to_numpy(float)) < -1e-12).any():
            raise RuntimeError(f"coverage is not monotone for {world}; candidate acceptance contract drift")
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrated-candidates", type=Path, required=True)
    parser.add_argument("--query-registry", type=Path, required=True)
    parser.add_argument("--truth-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_THRESHOLDS))
    parser.add_argument("--risk-ceiling", type=float, default=0.10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    thresholds = tuple(float(item) for item in args.thresholds.split(",") if item.strip())
    candidates = pd.read_csv(args.calibrated_candidates)
    registry = pd.read_csv(args.query_registry)
    truth = pd.read_csv(args.truth_ledger)
    curve = selective_curve(candidates, registry, truth, thresholds)
    eligible = curve.loc[curve["selective_risk"] <= args.risk_ceiling]
    operating_points = {}
    for world, group in eligible.groupby("world"):
        best = group.sort_values(["coverage", "fdr_threshold"], ascending=[False, True]).iloc[0]
        operating_points[world] = {
            "fdr_threshold": float(best["fdr_threshold"]),
            "coverage": float(best["coverage"]),
            "selective_risk": float(best["selective_risk"]),
            "balanced_endpoint": float(best["balanced_endpoint"]),
        }
    args.output_dir.mkdir(parents=True)
    curve.to_csv(args.output_dir / "selective_curve.csv", index=False)
    report = {
        "status": "CONTRAST_ATLAS_SELECTIVE_CURVE_COMPLETE",
        "formal": True,
        "thresholds": list(thresholds),
        "risk_ceiling": args.risk_ceiling,
        "queries": int(registry["query_id"].nunique()),
        "operating_points_development_only": operating_points,
        "claim_limit": "Operating-point selection is development-only and must be frozen before sealed evaluation.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
