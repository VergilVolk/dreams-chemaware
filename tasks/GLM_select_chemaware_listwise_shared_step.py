"""Select one shared listwise step using Arm 1 only.

Arm 2 is never inspected for step selection.  The corresponding Arm-2
checkpoint at the Arm-1-selected step is carried forward, preventing separate
role-2 winner's-curse selection from masquerading as a chemical increment.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, action="append", required=True)
    return parser.parse_args()


def step_from_name(name: str, arm: str) -> int:
    match = re.fullmatch(rf"{arm}_step(\d+)", name)
    if match is None:
        raise ValueError(f"invalid {arm} checkpoint name: {name}")
    return int(match.group(1))


def main() -> None:
    args = arguments()
    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if int(report.get("formula_role", -1)) != 2:
        raise RuntimeError("shared-step selection requires formula role 2")
    if report.get("formula_role_contract_passed") is not True:
        raise RuntimeError("shared-step selection lacks the frozen role-2 contract")
    rows = {row["name"]: row for row in report["results"]}
    if "phaseA_2pp" not in rows:
        raise KeyError("phaseA_2pp is absent")
    expected = sorted(set(int(step) for step in args.expected_step))
    arm1_names = sorted(
        (name for name in rows if name.startswith("arm1_step")),
        key=lambda name: step_from_name(name, "arm1"),
    )
    observed = [step_from_name(name, "arm1") for name in arm1_names]
    if observed != expected:
        raise RuntimeError(
            f"Arm-1 checkpoint steps drifted: expected={expected} observed={observed}"
        )
    candidates = []
    for name in arm1_names:
        step = step_from_name(name, "arm1")
        arm2_name = f"arm2_step{step}"
        if arm2_name not in rows:
            raise KeyError(f"paired Arm-2 checkpoint is absent: {arm2_name}")
        paired = rows[name].get("paired_vs_phaseA_2pp")
        if paired is None:
            raise KeyError(f"{name} lacks paired_vs_phaseA_2pp")
        utility = int(paired["corrected_at_1"]) - 2 * int(paired["introduced_at_1"])
        candidates.append({
            "step": step,
            "arm1_name": name,
            "arm2_name": arm2_name,
            "arm1_checkpoint": rows[name]["checkpoint"],
            "arm2_checkpoint": rows[arm2_name]["checkpoint"],
            "arm1_risk_utility_vs_phasea": utility,
            "arm1_paired_vs_phasea": paired,
        })
    selected = max(candidates, key=lambda row: (
        int(row["arm1_risk_utility_vs_phasea"]),
        float(row["arm1_paired_vs_phasea"]["delta_recall1"]),
        float(row["arm1_paired_vs_phasea"]["delta_mrr"]),
        -int(row["step"]),
    ))
    output = {
        "status": "GLM_LISTWISE_SHARED_STEP_SELECTED_FROM_ARM1_ONLY",
        "selection_panel": "formula_role_2",
        "chemical_arm_used_for_selection": False,
        "selection_rule": (
            "maximize Arm-1 corrected-2*introduced vs Phase-A; then delta "
            "Recall@1, delta MRR, earlier step; carry Arm 2 at the same step"
        ),
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.output.with_suffix(".step.txt").write_text(
        f"{selected['step']}\n", encoding="utf-8",
    )
    args.output.with_suffix(".arm1_checkpoint.txt").write_text(
        f"{selected['arm1_checkpoint']}\n", encoding="utf-8",
    )
    args.output.with_suffix(".arm2_checkpoint.txt").write_text(
        f"{selected['arm2_checkpoint']}\n", encoding="utf-8",
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
