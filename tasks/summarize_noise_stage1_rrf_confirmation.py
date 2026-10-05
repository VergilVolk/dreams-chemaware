#!/usr/bin/env python
"""Join internal and independent GNPS fixed-RRF confirmation reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


RANK_KEYS = ("recall@1", "recall@2", "recall@3", "recall@5", "recall@10", "recall@20", "mrr", "macro_query_auroc", "macro_query_auprc")
AUC_KEYS = ("auroc", "auprc")


def nonregressions(stage1: dict, rrf: dict) -> dict[str, bool]:
    checks = {
        f"retrieval.{key}": float(rrf["retrieval"][key]) >= float(stage1["retrieval"][key])
        for key in RANK_KEYS
    }
    checks.update({
        f"near_subset.{key}": float(rrf["near_subset"][key]) >= float(stage1["near_subset"][key])
        for key in RANK_KEYS
    })
    for family in ("micro_candidate", "gnps_10ppm_pooled_pairwise", "gnps_mh_10ppm_pooled_pairwise"):
        if family in stage1 and family in rrf:
            for key in AUC_KEYS:
                checks[f"{family}.{key}"] = float(rrf[family][key]) >= float(stage1[family][key])
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--gnps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    internal = json.loads(args.internal.read_text(encoding="utf-8"))
    gnps = json.loads(args.gnps.read_text(encoding="utf-8"))
    internal_paired = internal["fixed_rrf_vs_stage1"]
    internal_gate = {
        "stage1_reproduced": all(internal["stage1_reproduction"].values()),
        "recall1_delta_positive": internal_paired["formula_cluster_paired_ci"]["recall@1"]["delta_pp"] > 0,
        "recall1_formula_ci_low_positive": internal_paired["formula_cluster_paired_ci"]["recall@1"]["ci_low_pp"] > 0,
        "risk_net_lambda2_positive": internal_paired["risk_net_lambda2"] > 0,
    }
    panel_gates = {}
    for name, panel in gnps["panels"].items():
        paired = panel["fixed_rrf_vs_stage1"]
        checks = nonregressions(panel["stage1"], panel["fixed_rrf"])
        panel_gates[name] = {
            "all_registered_rank_auc_metrics_nonregressing": all(checks.values()),
            "metric_checks": checks,
            "recall1_formula_ci_low_positive": paired["formula_cluster_paired_ci"]["recall@1"]["ci_low_pp"] > 0,
            "risk_net_lambda2_positive": paired["risk_net_lambda2"] > 0,
        }
    independent_pass = all(
        gate["all_registered_rank_auc_metrics_nonregressing"]
        and gate["recall1_formula_ci_low_positive"]
        and gate["risk_net_lambda2_positive"]
        for gate in panel_gates.values()
    )
    report = {
        "status": "noise_stage1_rrf_confirmation_summary_complete",
        "internal_exploratory_gate": internal_gate,
        "independent_gnps_panel_gates": panel_gates,
        "independent_confirmation_pass": independent_pass,
        "reranker_promotion_authorized": independent_pass and all(internal_gate.values()),
        "embedding_gain_claim_authorized": False,
        "scale_dependent_margin_gate": (
            "not applied: RRF score gaps and cosine score gaps have different arbitrary scales; "
            "rank, AUC/AP, corrected/introduced and formula-cluster CI are the valid comparisons"
        ),
        "claim_limit": (
            "A pass promotes only the fixed rank-fusion reranker. It does not establish a better "
            "shared encoder and does not count as a Noise fine-tuning pp gain."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
