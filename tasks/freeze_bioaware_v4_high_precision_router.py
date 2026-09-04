#!/usr/bin/env python
"""Freeze a high-precision BioAware V4 router after a three-cohort audit.

V4 is deliberately narrower than V3.  It keeps the already frozen V3 family
weights, disables the brittle path-only expert, and selects one rank-expert
advantage threshold using only explicitly consumed development cohorts.  The
remaining external panels are untouched and must be evaluated without refit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLDS = (0.025, 0.05, 0.075, 0.10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(frame: pd.DataFrame, threshold: float) -> dict:
    required = {
        "truth_candidate_id", "baseline_candidate_id", "proposed_candidate_id",
        "changes_top1", "proposed_unique", "spectral_margin", "fusion_advantage",
        "support_count",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"transition table is missing V4 columns: {sorted(missing)}")
    intervene = (
        frame["changes_top1"].astype(bool)
        & frame["proposed_unique"].astype(bool)
        & (frame["spectral_margin"].astype(float) <= 0.05 + 1e-12)
        & (frame["fusion_advantage"].astype(float) >= threshold - 1e-12)
        & (frame["support_count"].astype(int) >= 2)
    )
    final = frame["baseline_candidate_id"].astype(str).where(
        ~intervene, frame["proposed_candidate_id"].astype(str)
    )
    truth = frame["truth_candidate_id"].astype(str)
    baseline = frame["baseline_candidate_id"].astype(str).eq(truth)
    candidate = final.eq(truth)
    corrected = int((~baseline & candidate).sum())
    introduced = int((baseline & ~candidate).sum())
    return {
        "queries": int(len(frame)), "interventions": int(intervene.sum()),
        "baseline_recall1": float(baseline.mean()), "recall1": float(candidate.mean()),
        "delta_recall1": float(candidate.mean() - baseline.mean()),
        "corrected": corrected, "introduced": introduced,
        "risk_weighted_net_lambda2": corrected - 2 * introduced,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-artifact", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--internal-rplc", type=Path, required=True)
    parser.add_argument("--consumed-external", type=Path, required=True)
    parser.add_argument("--consumed-external-name", default="BV2cell__hilic")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.v3_artifact, args.development, args.internal_rplc, args.consumed_external]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    v3 = json.loads(args.v3_artifact.read_text(encoding="utf-8"))
    if v3.get("status") != "bioaware_v3_consensus_router_artifact_frozen":
        raise RuntimeError("unexpected V3 artifact")
    cohorts = {
        "consumed_development": pd.read_csv(args.development),
        "consumed_internal_rplc": pd.read_csv(args.internal_rplc),
        f"consumed_external_{args.consumed_external_name}": pd.read_csv(args.consumed_external),
    }
    if any(frame["query_id"].duplicated().any() for frame in cohorts.values()):
        raise RuntimeError("V4 selection cohorts must be one row per query")
    candidates = []
    for threshold in THRESHOLDS:
        results = {name: evaluate(frame, threshold) for name, frame in cohorts.items()}
        total_corrected = sum(row["corrected"] for row in results.values())
        total_introduced = sum(row["introduced"] for row in results.values())
        minimum_risk = min(row["risk_weighted_net_lambda2"] for row in results.values())
        all_zero_introduced = all(row["introduced"] == 0 for row in results.values())
        candidates.append({
            "minimum_fusion_advantage": threshold, "cohorts": results,
            "total_corrected": total_corrected, "total_introduced": total_introduced,
            "minimum_cohort_risk_net": minimum_risk,
            "all_cohorts_zero_introduced": all_zero_introduced,
        })
    eligible = [
        row for row in candidates
        if row["all_cohorts_zero_introduced"]
        and row["minimum_cohort_risk_net"] > 0
        and row["total_corrected"] > 0
    ]
    if not eligible:
        raise RuntimeError("no preregistered high-precision threshold passes every consumed cohort")
    # Maximise verified corrections under the zero-introduction constraint;
    # prefer the larger threshold on a tie.
    selected = max(eligible, key=lambda row: (
        row["total_corrected"], row["minimum_cohort_risk_net"],
        row["minimum_fusion_advantage"],
    ))
    artifact = {
        "status": "bioaware_v4_high_precision_router_artifact_frozen",
        "formal": True,
        "router": {
            "type": "rank_consensus_only_with_high_precision_abstention",
            "family_features": v3["router"]["rank_consensus_expert"]["family_features"],
            "raw_families": v3["router"]["rank_consensus_expert"]["raw_families"],
            "weights": v3["router"]["rank_consensus_expert"]["weights"],
            "gate": {
                "maximum_spectral_margin": 0.05,
                "minimum_fusion_advantage": selected["minimum_fusion_advantage"],
                "minimum_support_families": 2,
            },
            "depth3_expert_enabled": False,
            "fallback": "exact official DreaMS order",
        },
        "selection": {
            "objective": "maximum corrections subject to zero introduced errors and positive lambda=2 risk net in every consumed cohort",
            "threshold_grid": list(THRESHOLDS),
            "all_candidates": candidates,
            "selected": selected,
        },
        "consumed_sets": {
            "development": str(args.development),
            "internal_rplc": str(args.internal_rplc),
            "external_panel": args.consumed_external_name,
            "external_transition": str(args.consumed_external),
        },
        "confirmatory_external_panels": {
            "excluded": [args.consumed_external_name],
            "required": 7,
            "rule": "zero refit; pooled formula-cluster CI lower bound >0; corrected > introduced; lambda=2 net >0; no panel degradation",
        },
        "contracts": {
            "P2b": "forbidden", "phenotype": "forbidden",
            "weights_refit": False, "depth3_disabled": True,
            "consumed_external_not_confirmatory": True,
            "evaluation_must_load_this_artifact_without_refit": True,
        },
        "provenance": {
            "v3_artifact_sha256": sha256(args.v3_artifact),
            "development_sha256": sha256(args.development),
            "internal_rplc_sha256": sha256(args.internal_rplc),
            "consumed_external_sha256": sha256(args.consumed_external),
        },
        "claim_limit": "V4 is frozen after consuming one external panel. Only the seven named untouched panels plus real graph-decoy tests can support a high-precision external claim.",
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "artifact.json"
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
