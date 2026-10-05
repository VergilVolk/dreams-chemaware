#!/usr/bin/env python
"""Summarize the WSE-discordant native continuation against official and V1 on GNPS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PANELS = ("identity_disjoint", "formula_disjoint")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def panel_delta(report: dict, panel: str) -> dict:
    paired = report["panels"][panel]["paired"]
    recall = paired["formula_cluster_paired_ci"]["recall@1"]
    near = paired["near_formula_cluster_paired_ci"]["recall@1"]
    return {
        "recall_at_1_delta_pp": recall["delta_pp"],
        "recall_at_1_ci_low_pp": recall["ci_low_pp"],
        "recall_at_1_ci_high_pp": recall["ci_high_pp"],
        "near_recall_at_1_delta_pp": near["delta_pp"],
        "near_recall_at_1_ci_low_pp": near["ci_low_pp"],
        "corrected": paired["corrected"],
        "introduced": paired["introduced"],
        "risk_net_lambda2": paired["risk_net_lambda2"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplet-report", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, action="append", required=True)
    parser.add_argument(
        "--vs-official", type=Path, action="append", required=True,
        help="paired report against official DreaMS, one per seed, same order as training reports",
    )
    parser.add_argument(
        "--vs-v1", type=Path, action="append", required=True,
        help="paired report against the Noise V1 champion, same order",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (
        len(args.training_report) == len(args.vs_official) == len(args.vs_v1)
    ):
        raise RuntimeError("per-seed report lists must have equal length")

    seeds = []
    for training, vs_official, vs_v1 in zip(
        args.training_report, args.vs_official, args.vs_v1, strict=True,
    ):
        training_report = load(training)
        if training_report.get("status") != "NOISE_WSE_DISCORDANT_NATIVE_TRAINING_COMPLETE":
            raise RuntimeError("WSE-native training report is incomplete")
        seeds.append({
            "seed": training_report["seed"],
            "training": training_report["training"],
            "gnps_vs_official": {
                panel: panel_delta(load(vs_official), panel) for panel in PANELS
            },
            "gnps_vs_v1": {
                panel: panel_delta(load(vs_v1), panel) for panel in PANELS
            },
        })
    report = {
        "status": "noise_wse_discordant_gnps_summary_complete",
        "triplets": load(args.triplet_report),
        "seeds": seeds,
        "claim_limit": (
            "Direction summary of one native continuation per seed. GNPS never entered "
            "mining or training; no MassSpecGym held-fold claim exists for this lineage "
            "because every corrected query is training content."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
