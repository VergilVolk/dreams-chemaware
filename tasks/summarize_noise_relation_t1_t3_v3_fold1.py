#!/usr/bin/env python
"""Select the query-balanced targeted arm against its matched control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-report", type=Path, required=True)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    targeted = json.loads(args.targeted_report.read_text(encoding="utf-8"))
    control = json.loads(args.control_report.read_text(encoding="utf-8"))
    if (
        targeted.get("status") != "noise_relation_t1_t3_v3_training_complete"
        or control.get("status") != "noise_relation_t1_t3_v3_training_complete"
        or targeted.get("arm") != "targeted"
        or control.get("arm") != "control"
    ):
        raise RuntimeError("V3 fold-1 summary received the wrong arm reports")
    for key in ("seed", "rotation"):
        if targeted.get(key) != control.get(key):
            raise RuntimeError(f"V3 targeted/control {key} mismatch")
    if targeted["selection"] != control["selection"]:
        raise RuntimeError("V3 targeted/control selected different action events")
    for key in (
        "queries", "action_queries", "optimizer_steps",
        "one_selected_action_per_action_query",
    ):
        if targeted["schedule"].get(key) != control["schedule"].get(key):
            raise RuntimeError(f"V3 targeted/control schedule mismatch: {key}")

    t = targeted["fold1_diagnostics"]
    c = control["fold1_diagnostics"]
    if t["before"] != c["before"]:
        raise RuntimeError("V3 targeted/control warm-start metrics differ")
    deltas = {
        "targeted_vs_warm_recall1_pp": 100.0 * float(t["recall1_delta"]),
        "control_vs_warm_recall1_pp": 100.0 * float(c["recall1_delta"]),
        "targeted_minus_control_recall1_pp": 100.0 * (
            float(t["after"]["recall_at_1_strict"])
            - float(c["after"]["recall_at_1_strict"])
        ),
        "targeted_vs_warm_mrr_pp": 100.0 * float(t["mrr_delta"]),
        "control_vs_warm_mrr_pp": 100.0 * float(c["mrr_delta"]),
        "targeted_minus_control_mrr_pp": 100.0 * (
            float(t["after"]["mean_reciprocal_rank"])
            - float(c["after"]["mean_reciprocal_rank"])
        ),
        "targeted_minus_control_margin": (
            float(t["after"]["mean_margin"])
            - float(c["after"]["mean_margin"])
        ),
    }
    gates = {
        "targeted_improves_warm_recall1": deltas["targeted_vs_warm_recall1_pp"] > 0.0,
        "targeted_improves_warm_mrr": deltas["targeted_vs_warm_mrr_pp"] > 0.0,
        "targeted_beats_matched_control_recall1": (
            deltas["targeted_minus_control_recall1_pp"] > 0.0
        ),
        "targeted_beats_matched_control_mrr": (
            deltas["targeted_minus_control_mrr_pp"] > 0.0
        ),
    }
    proceed = all(gates.values())
    report = {
        "status": "noise_relation_t1_t3_v3_fold1_selection_complete",
        "protocol": (
            "one query once; one exact action per action query; 0.5 clean T1/T3 "
            "+ 0.5 native exact triplet inside action queries; targeted/control "
            "share every relation, batch and optimizer step"
        ),
        "targeted": t,
        "matched_control": c,
        "deltas": deltas,
        "gates": gates,
        "proceed_to_outer_held": proceed,
        "no_go_reason": None if proceed else (
            "query-balanced targeted arm did not improve both warm start and "
            "matched control on fold-1 Recall@1 and MRR"
        ),
        "claim_limit": (
            "Fold-1 is a train-side safety and action-specificity screen.  "
            "It is not an outer-held performance claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

