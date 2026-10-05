#!/usr/bin/env python
"""Condense MassSpecGym OOF and three frozen GNPS fusion comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--wse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact(report: dict) -> dict:
    output = {}
    for panel, body in report["panels"].items():
        baseline = body["baseline"]
        candidate = body["candidate"]
        paired = body["paired"]
        ci = paired["formula_cluster_paired_ci"]["recall@1"]
        output[panel] = {
            "baseline_recall1": baseline["retrieval"]["recall@1"],
            "fusion_recall1": candidate["retrieval"]["recall@1"],
            "recall1_delta_pp": ci["delta_pp"],
            "recall1_formula_ci_pp": [ci["ci_low_pp"], ci["ci_high_pp"]],
            "baseline_mrr": baseline["retrieval"]["mrr"],
            "fusion_mrr": candidate["retrieval"]["mrr"],
            "baseline_macro_query_auroc": baseline["retrieval"]["macro_query_auroc"],
            "fusion_macro_query_auroc": candidate["retrieval"]["macro_query_auroc"],
            "baseline_pooled_pairwise_auroc": baseline["gnps_10ppm_pooled_pairwise"]["auroc"],
            "fusion_pooled_pairwise_auroc": candidate["gnps_10ppm_pooled_pairwise"]["auroc"],
            "corrected": paired["corrected"],
            "introduced": paired["introduced"],
            "risk_net_lambda2": paired["risk_net_lambda2"],
            "near_recall1": candidate["near_subset"]["recall@1"],
            "near_risk_net_lambda2": paired["near_risk_net_lambda2"],
        }
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = load(args.model_report)
    report = {
        "status": "NOISE_MSG_FUSION_SUMMARY_COMPLETE",
        "massspecgym_formula_oof": model["oof"],
        "gnps_frozen": {
            "official_vs_fusion": compact(load(args.official)),
            "noise_v1_vs_fusion": compact(load(args.v1)),
            "wse_vs_fusion": compact(load(args.wse)),
        },
        "claim_boundary": (
            "Learned candidate reranker over deployment-visible spectrum-pair scores; "
            "not a new shared encoder and not an embedding-gain claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
