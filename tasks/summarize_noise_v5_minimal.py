#!/usr/bin/env python
"""Pre-registered verdict for the V5 minimal residual round.

Gates (all three must pass for promotion; any failure is a NO-GO that keeps
the V1 champion untouched):

G1 train-side safety  - both arms' development recall@1 must not fall below
                        the atlas V1-current baseline (0.965015...), because a
                        damaged control arm makes the causal contrast void.
G2 GNPS benefit       - targeted vs V1 recall@1 delta >= 0 on BOTH GNPS
                        panels (point estimate; the conservative reading of
                        "small but definitely beneficial").
G3 causal direction   - targeted vs control recall@1 delta >= 0 on BOTH GNPS
                        panels, so any G2 gain is attributable to the action
                        content rather than the shared continuation.

No held fold-0 evaluation happens here and no performance number in the
output is claimable; promotion only certifies direction plus safety.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PANELS = ("identity_disjoint", "formula_disjoint")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_status(report: dict[str, object], status: str, label: str) -> None:
    if report.get("status") != status:
        raise RuntimeError(f"{label} report status is not {status}")


def panel_recall_delta(eval_report: dict[str, object], panel: str) -> tuple[float, int, int, int]:
    paired = eval_report["panels"][panel]["paired"]
    delta_pp = float(
        paired["formula_cluster_paired_ci"]["recall@1"]["delta_pp"],
    )
    return (
        delta_pp,
        int(paired["corrected"]),
        int(paired["introduced"]),
        int(paired["risk_net_lambda2"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-report", type=Path, required=True)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--atlas-report", type=Path, required=True)
    parser.add_argument("--dev-targeted", type=Path, required=True)
    parser.add_argument("--dev-control", type=Path, required=True)
    parser.add_argument("--gnps-targeted-vs-v1", type=Path, required=True)
    parser.add_argument("--gnps-control-vs-v1", type=Path, required=True)
    parser.add_argument("--gnps-targeted-vs-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    targeted = load_json(args.targeted_report)
    control = load_json(args.control_report)
    atlas = load_json(args.atlas_report)
    dev_targeted = load_json(args.dev_targeted)
    dev_control = load_json(args.dev_control)
    gnps_targeted_vs_v1 = load_json(args.gnps_targeted_vs_v1)
    gnps_control_vs_v1 = load_json(args.gnps_control_vs_v1)
    gnps_targeted_vs_control = load_json(args.gnps_targeted_vs_control)
    require_status(
        targeted, "noise_v5_minimal_training_complete", "targeted training",
    )
    require_status(
        control, "noise_v5_minimal_training_complete", "control training",
    )
    require_status(
        atlas, "noise_v1_current_residual_atlas_complete", "atlas",
    )
    require_status(dev_targeted, "noise_dev_graph_recall_scored", "dev targeted")
    require_status(dev_control, "noise_dev_graph_recall_scored", "dev control")
    for label, report in (
        ("targeted vs V1", gnps_targeted_vs_v1),
        ("control vs V1", gnps_control_vs_v1),
        ("targeted vs control", gnps_targeted_vs_control),
    ):
        if report.get("status") != "gnps_gold_silver_10ppm_embedding_evaluation_complete":
            raise RuntimeError(f"{label} GNPS evaluation is incomplete")
        for panel in PANELS:
            if panel not in report.get("panels", {}):
                raise RuntimeError(f"{label} GNPS evaluation misses panel {panel}")

    baseline_dev_recall = float(atlas["graph"]["development_recall_at_1"])
    dev_targeted_recall = float(dev_targeted["recall_at_1"])
    dev_control_recall = float(dev_control["recall_at_1"])
    g1 = (
        dev_targeted_recall >= baseline_dev_recall
        and dev_control_recall >= baseline_dev_recall
    )
    g2_detail = {}
    g2 = True
    for panel in PANELS:
        delta_pp, corrected, introduced, risk_net = panel_recall_delta(
            gnps_targeted_vs_v1, panel,
        )
        g2_detail[panel] = {
            "recall_at_1_delta_pp": delta_pp,
            "corrected": corrected,
            "introduced": introduced,
            "risk_net_lambda2": risk_net,
        }
        if delta_pp < 0.0:
            g2 = False
    g3_detail = {}
    g3 = True
    for panel in PANELS:
        delta_pp, corrected, introduced, risk_net = panel_recall_delta(
            gnps_targeted_vs_control, panel,
        )
        g3_detail[panel] = {
            "recall_at_1_delta_pp": delta_pp,
            "corrected": corrected,
            "introduced": introduced,
            "risk_net_lambda2": risk_net,
        }
        if delta_pp < 0.0:
            g3 = False
    promoted = bool(g1 and g2 and g3)
    report = {
        "status": (
            "noise_v5_minimal_promotion_granted"
            if promoted else "noise_v5_minimal_no_go"
        ),
        "promoted": promoted,
        "gates": {
            "G1_train_side_safety": {
                "passed": g1,
                "atlas_v1_dev_recall_at_1": baseline_dev_recall,
                "targeted_dev_recall_at_1": dev_targeted_recall,
                "control_dev_recall_at_1": dev_control_recall,
            },
            "G2_gnps_benefit_vs_v1": {"passed": g2, "panels": g2_detail},
            "G3_causal_targeted_vs_control": {"passed": g3, "panels": g3_detail},
        },
        "gradient_audit": {
            "note": (
                "same seed and unit order in both arms; first-unit gradients "
                "are directly comparable"
            ),
            "targeted_first_unit_grad_norm_l2": targeted["training"][
                "first_unit_grad_norm_l2"
            ],
            "control_first_unit_grad_norm_l2": control["training"][
                "first_unit_grad_norm_l2"
            ],
            "targeted_mean_unit_grad_norm_l2": targeted["training"][
                "mean_unit_grad_norm_l2"
            ],
            "control_mean_unit_grad_norm_l2": control["training"][
                "mean_unit_grad_norm_l2"
            ],
            "targeted_active_fraction": targeted["training"]["active_fraction"],
            "control_active_fraction": control["training"]["active_fraction"],
            "targeted_mean_hinge": targeted["training"]["mean_hinge"],
            "control_mean_hinge": control["training"]["mean_hinge"],
        },
        "control_vs_v1_reference": {
            panel: panel_recall_delta(gnps_control_vs_v1, panel)
            for panel in PANELS
        },
        "claim_limit": (
            "Development numbers are train-side damage detectors; GNPS deltas "
            "certify direction and safety only.  No MassSpecGym held fold-0 "
            "evaluation was performed and no performance is claimed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
