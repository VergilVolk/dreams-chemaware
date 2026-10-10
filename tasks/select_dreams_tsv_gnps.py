#!/usr/bin/env python
"""Select the better of the two fixed merge arms on consumed GNPS development."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PANELS = ("identity_disjoint", "formula_disjoint")
METRIC_PATHS = {
    "recall@1": ("retrieval", "recall@1"),
    "recall@3": ("retrieval", "recall@3"),
    "recall@5": ("retrieval", "recall@5"),
    "recall@10": ("retrieval", "recall@10"),
    "recall@20": ("retrieval", "recall@20"),
    "mrr": ("retrieval", "mrr"),
    "macro_query_auroc": ("retrieval", "macro_query_auroc"),
    "macro_query_auprc": ("retrieval", "macro_query_auprc"),
    "near_recall@1": ("near_subset", "recall@1"),
    "near_mrr": ("near_subset", "mrr"),
    "near_macro_query_auroc": ("near_subset", "macro_query_auroc"),
    "near_macro_query_auprc": ("near_subset", "macro_query_auprc"),
    "micro_candidate_auroc": ("micro_candidate", "auroc"),
    "micro_candidate_auprc": ("micro_candidate", "auprc"),
    "pooled_pairwise_auroc": ("gnps_10ppm_pooled_pairwise", "auroc"),
    "pooled_pairwise_auprc": ("gnps_10ppm_pooled_pairwise", "auprc"),
}


def recall1(report: dict, panel: str, side: str) -> float:
    return float(report["panels"][panel][side]["retrieval"]["recall@1"])


def nested_float(body: dict, path: tuple[str, str]) -> float:
    return float(body[path[0]][path[1]])


def panel_summary(panel_report: dict) -> dict:
    metrics = {}
    for name, path in METRIC_PATHS.items():
        baseline = nested_float(panel_report["baseline"], path)
        candidate = nested_float(panel_report["candidate"], path)
        metrics[name] = {
            "official": baseline,
            "candidate": candidate,
            "delta": candidate - baseline,
            "delta_pp": 100 * (candidate - baseline),
        }
    paired = panel_report["paired"]
    return {
        "metrics": metrics,
        "corrected_at_1": int(paired["corrected"]),
        "introduced_at_1": int(paired["introduced"]),
        "risk_net_lambda2": int(paired["risk_net_lambda2"]),
        "formula_cluster_paired_ci": paired["formula_cluster_paired_ci"],
        "near_formula_cluster_paired_ci": paired["near_formula_cluster_paired_ci"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-report", type=Path, required=True)
    parser.add_argument("--tsv-report", type=Path, required=True)
    parser.add_argument("--linear-checkpoint", type=Path, required=True)
    parser.add_argument("--tsv-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    linear = json.loads(args.linear_report.read_text(encoding="utf-8"))
    tsv = json.loads(args.tsv_report.read_text(encoding="utf-8"))
    scores = {}
    for name, report in (("linear", linear), ("tsv", tsv)):
        values = {}
        deltas = []
        for panel in PANELS:
            summary = panel_summary(report["panels"][panel])
            values[panel] = summary
            baseline = recall1(report, panel, "baseline")
            candidate = recall1(report, panel, "candidate")
            deltas.append(candidate - baseline)
        values["minimum_two_panel_delta_recall1"] = min(deltas)
        values["mean_two_panel_delta_recall1"] = sum(deltas) / len(deltas)
        values["stable_plus_3pp_target_attained"] = all(delta >= 0.03 for delta in deltas)
        scores[name] = values
    # "Stable" means the worst of the identity/formula gains is primary.
    # Mean gain breaks a tie; an exact tie conservatively retains the linear control.
    linear_key = (
        scores["linear"]["minimum_two_panel_delta_recall1"],
        scores["linear"]["mean_two_panel_delta_recall1"],
    )
    tsv_key = (
        scores["tsv"]["minimum_two_panel_delta_recall1"],
        scores["tsv"]["mean_two_panel_delta_recall1"],
    )
    winner = "tsv" if tsv_key > linear_key else "linear"
    checkpoint = args.tsv_checkpoint if winner == "tsv" else args.linear_checkpoint
    selected = args.output.parent / "selected_checkpoint.pt"
    if selected.exists():
        raise FileExistsError(selected)
    try:
        os.link(checkpoint, selected)
        storage = "hardlink"
    except OSError:
        import shutil
        shutil.copy2(checkpoint, selected)
        storage = "copy"
    result = {
        "status": "DREAMS_TSV_GNPS_DEVELOPMENT_SELECTION_COMPLETE",
        "dataset": "GNPS Gold/Silver 10 ppm",
        "dataset_status": "development_consumed",
        "panels": {"identity_disjoint_queries": 10995, "formula_disjoint_queries": 5261},
        "primary_endpoint": (
            "maximum worst-panel Recall@1 gain vs official across identity/formula; "
            "mean gain is the fixed tie-breaker"
        ),
        "scores": scores,
        "winner": winner,
        "selected_checkpoint": str(selected),
        "selected_checkpoint_storage": storage,
        "massspecgym_accessed": False,
        "enveda_accessed": False,
        "claim_limit": (
            "Development selection only. Enveda-180 remains unopened for the final external test."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
