#!/usr/bin/env python
"""Summarize the single full-MassSpecGym model against official and V1 on GNPS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact(report: dict) -> dict:
    output = {}
    for panel, block in report["panels"].items():
        paired = block["paired"]
        candidate = block["candidate"]
        output[panel] = {
            "baseline_recall_at_1": block["baseline"]["retrieval"]["recall@1"],
            "candidate_recall_at_1": candidate["retrieval"]["recall@1"],
            "candidate_mrr": candidate["retrieval"]["mrr"],
            "candidate_macro_query_auroc": candidate["retrieval"]["macro_query_auroc"],
            "candidate_macro_query_auprc": candidate["retrieval"]["macro_query_auprc"],
            "candidate_micro_candidate_auroc": candidate["micro_candidate"]["auroc"],
            "candidate_micro_candidate_auprc": candidate["micro_candidate"]["auprc"],
            "candidate_pooled_10ppm_auroc": candidate["gnps_10ppm_pooled_pairwise"]["auroc"],
            "candidate_pooled_10ppm_auprc": candidate["gnps_10ppm_pooled_pairwise"]["auprc"],
            "recall_at_1_delta_pp": paired["formula_cluster_paired_ci"]["recall@1"]["delta_pp"],
            "recall_at_1_ci_low_pp": paired["formula_cluster_paired_ci"]["recall@1"]["ci_low_pp"],
            "recall_at_1_ci_high_pp": paired["formula_cluster_paired_ci"]["recall@1"]["ci_high_pp"],
            "corrected": paired["corrected"],
            "introduced": paired["introduced"],
            "risk_net_lambda2": paired["risk_net_lambda2"],
            "near_recall_at_1": candidate["near_subset"]["recall@1"],
            "near_corrected": paired["near_corrected"],
            "near_introduced": paired["near_introduced"],
            "near_risk_net_lambda2": paired["near_risk_net_lambda2"],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-vs-new", type=Path, required=True)
    parser.add_argument("--v1-vs-new", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--triplet-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "status": "noise_massspecgym_full_native_gnps_summary_complete",
        "training": load(args.training_report),
        "triplets": load(args.triplet_report),
        "gnps_vs_official": compact(load(args.official_vs_new)),
        "gnps_vs_v1": compact(load(args.v1_vs_new)),
        "claim_limit": (
            "This is the frozen GNPS result of one full-MassSpecGym native triplet run. "
            "It is not NIST20 replication and no unreported model-selection arm exists."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
