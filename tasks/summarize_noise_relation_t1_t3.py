#!/usr/bin/env python
"""Compare two T1/T3 shared encoders directly with the Stage-1 warm start."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_gnps_gold_silver_10ppm_embeddings import paired_summary


METRIC_COLUMNS = (
    "rank", "reciprocal_rank", "macro_query_auc", "macro_query_auprc",
    "positive_vs_best_negative_margin", "top1_top2_gap", "signed_top1_top2_gap",
)


def candidate_view(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    output = source[["query_index", "query_row", "query_ik14", "query_formula", "near"]].copy()
    for name in METRIC_COLUMNS:
        output[name] = source[f"candidate_{name}"].to_numpy()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replicate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (
        args.stage1 / "held_per_query.csv.gz", args.stage1 / "report.json",
        args.primary / "held_per_query.csv.gz", args.primary / "report.json",
        args.replicate / "held_per_query.csv.gz", args.replicate / "report.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline = candidate_view(args.stage1 / "held_per_query.csv.gz")
    report = {
        "status": "noise_relation_t1_t3_summary_complete",
        "baseline": "Stage-1 targeted champion",
        "selection": "primary seed 3407 predeclared; replicate seed 3408 is confirmation, not model selection",
        "seeds": {},
    }
    for offset, (name, directory) in enumerate((
        ("primary_seed_3407", args.primary),
        ("replicate_seed_3408", args.replicate),
    )):
        candidate = candidate_view(directory / "held_per_query.csv.gz")
        paired, _ = paired_summary(
            baseline, candidate, args.bootstrap_resamples,
            20260930 + offset * 1000, hypotheses=24,
        )
        evaluation = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        report["seeds"][name] = {
            "candidate_full_metrics": evaluation["candidate"],
            "candidate_vs_stage1": paired,
        }
    primary = report["seeds"]["primary_seed_3407"]["candidate_vs_stage1"]
    replicate = report["seeds"]["replicate_seed_3408"]["candidate_vs_stage1"]
    report["algorithmic_result"] = {
        "primary_recall1_delta_pp": primary["formula_cluster_paired_ci"]["recall@1"]["delta_pp"],
        "primary_recall1_ci_low_pp": primary["formula_cluster_paired_ci"]["recall@1"]["ci_low_pp"],
        "primary_risk_net_lambda2": primary["risk_net_lambda2"],
        "replicate_recall1_delta_pp": replicate["formula_cluster_paired_ci"]["recall@1"]["delta_pp"],
        "replicate_recall1_ci_low_pp": replicate["formula_cluster_paired_ci"]["recall@1"]["ci_low_pp"],
        "replicate_risk_net_lambda2": replicate["risk_net_lambda2"],
    }
    report["claim_limit"] = (
        "Direct shared-encoder T1/T3 continuation result on the corrected MassSpecGym held fold. "
        "No reranker, teacher embedding, or candidate feature is used at inference."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
