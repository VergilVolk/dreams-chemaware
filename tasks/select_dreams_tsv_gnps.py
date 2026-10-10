#!/usr/bin/env python
"""Select the better of the two fixed merge arms on consumed GNPS development."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PANELS = ("identity_disjoint", "formula_disjoint")


def recall1(report: dict, panel: str, side: str) -> float:
    return float(report["panels"][panel][side]["retrieval"]["recall@1"])


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
            baseline = recall1(report, panel, "baseline")
            candidate = recall1(report, panel, "candidate")
            values[panel] = {
                "official_recall1": baseline,
                "candidate_recall1": candidate,
                "delta_recall1": candidate - baseline,
                "delta_recall1_pp": 100 * (candidate - baseline),
            }
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
