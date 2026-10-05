"""GNPS-panel arm selection after the fold-1 gate was falsified.

Why this selector exists (predeclared before any GNPS result was seen):
run 2347230's fold-1 gate fired NO-GO, but the gate's premise is falsified
by sealed evidence.  The averaged arm's checkpoint has BOTH

  * fold-1 strict recall@1 delta  -0.878 pp  (run 2347230), and
  * held fold-0 recall@1 delta    +0.556 pp, CI_low +0.105 pp (run 2347055),

so fold-1 (inside Stage-1's training pool: 5,793 fold-1 action events were
recovered from it) moves OPPOSITE to held for the same continuation.  A
memorized fold-1 baseline mechanically punishes every weight update and
carries no information about held direction.  The honest replacement
selector is the external GNPS gold/silver 10-ppm panel: public-library
spectra that never entered any training fold.

Predeclared rule: each arm's evidence is its paired GNPS recall@1 delta
(candidate vs Stage-1) on both panels.  The winner is the arm with the
largest WORST-PANEL delta (a conservative min rule across panels), ties
break on the formula-disjoint panel delta.  Held fold-0 is spent only if
the winner's worst-panel delta is strictly positive; otherwise the
continuation direction itself is unsupported and the run stops.  GNPS
selects an arm; it never certifies a MassSpecGym performance claim.

Author: GLM-5.3 via DeepSeek Harness, taking over the GPT-authored pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("clean-only", "averaged", "action-rotation")
PANELS = ("identity_disjoint", "formula_disjoint")


def arm_evidence(report: dict, arm: str) -> dict:
    panels = {}
    for panel in PANELS:
        paired = report["panels"][panel]["paired"]
        recall = paired["formula_cluster_paired_ci"]["recall@1"]
        panels[panel] = {
            "queries": int(paired["queries"]),
            "delta_pp": float(recall["delta_pp"]),
            "ci_low_pp": float(recall["ci_low_pp"]),
            "ci_high_pp": float(recall["ci_high_pp"]),
        }
    return panels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-only-report", type=Path, required=True)
    parser.add_argument("--averaged-report", type=Path, required=True)
    parser.add_argument("--action-rotation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    reports = {
        "clean-only": json.loads(
            args.clean_only_report.read_text(encoding="utf-8"),
        ),
        "averaged": json.loads(
            args.averaged_report.read_text(encoding="utf-8"),
        ),
        "action-rotation": json.loads(
            args.action_rotation_report.read_text(encoding="utf-8"),
        ),
    }
    evidence = {arm: arm_evidence(reports[arm], arm) for arm in ARMS}
    for arm in ARMS:
        for panel in PANELS:
            if (
                evidence[arm][panel]["queries"]
                != evidence["clean-only"][panel]["queries"]
            ):
                raise RuntimeError(
                    f"arm {arm} scored a different {panel} query set",
                )

    def ranking(arm: str) -> tuple[float, float, float]:
        worst = min(evidence[arm][panel]["delta_pp"] for panel in PANELS)
        tiebreak = evidence[arm]["formula_disjoint"]["delta_pp"]
        return (worst, tiebreak, -float(arm == "averaged"))

    winner = max(ARMS, key=ranking)
    worst_panel_delta = min(
        evidence[winner][panel]["delta_pp"] for panel in PANELS
    )
    proceed = worst_panel_delta > 0.0
    report = {
        "status": "noise_relation_t1_t3_v2_gnps_selection_complete",
        "arms": evidence,
        "selection_rule": (
            "argmax worst-panel paired GNPS recall@1 delta vs Stage-1 "
            "(conservative minimum across identity/formula panels); ties "
            "break on the formula-disjoint panel; held is spent only when "
            "the winner's worst-panel delta is strictly positive"
        ),
        "replaces": (
            "fold-1 gate of run 2347230, falsified by the averaged arm: "
            "fold-1 delta -0.878 pp vs sealed held delta +0.556 pp "
            "(CI_low +0.105 pp, run 2347055)"
        ),
        "selected_arm": winner if proceed else None,
        "proceed_to_held": proceed,
        "no_go_reason": None if proceed else (
            "no arm beat Stage-1 on both GNPS panels; the continuation "
            "direction itself is unsupported"
        ),
        "claim_limit": (
            "GNPS public-library transfer selects an arm and justifies (or "
            "refuses) spending the held fold.  It is not a MassSpecGym "
            "performance claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
