"""Issue the frozen release decision for a ChemAware surgical embedding run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


NULL_PREFIX = "null_"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-delta-recall1", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if int(report.get("formula_role", -1)) != 3:
        raise RuntimeError("surgical release decision requires formula role 3")
    rows = {row["name"]: row for row in report["results"]}
    required = {"official", "stage1_base", "surgical_correct"}
    if missing := sorted(required - set(rows)):
        raise RuntimeError(f"role-3 evaluation lacks required arms: {missing}")
    nulls = [row for name, row in rows.items() if name.startswith(NULL_PREFIX)]
    if len(nulls) != 3:
        raise RuntimeError(f"expected three frozen null arms, found {len(nulls)}")
    correct = rows["surgical_correct"]
    stage1 = rows["stage1_base"]
    paired_official = correct["paired_vs_official"]
    paired_stage1 = correct["paired_vs_stage1_base"]
    metrics = correct["metrics"]
    stage1_metrics = stage1["metrics"]
    risk = int(paired_stage1["corrected_at_1"]) - 2 * int(
        paired_stage1["introduced_at_1"]
    )
    maximum_null_recall1 = max(float(row["metrics"]["recall1"]) for row in nulls)
    gates = {
        "beats_stage1_recall1": float(paired_stage1["delta_recall1"]) > 0.0,
        "stage1_formula_ci_positive": float(
            paired_stage1["formula_cluster_bootstrap_delta_recall1_ci95"][0]
        ) > 0.0,
        "stage1_risk_utility_positive": risk > 0,
        "stage1_mrr_nonnegative": float(paired_stage1["delta_mrr"]) >= 0.0,
        "stage1_recall3_nonnegative": (
            float(metrics["recall3"]) >= float(stage1_metrics["recall3"])
        ),
        "stage1_micro_auc_nonnegative": (
            float(metrics["micro_auc"]) >= float(stage1_metrics["micro_auc"])
        ),
        "stage1_macro_auc_nonnegative": (
            float(metrics["macro_auc"]) >= float(stage1_metrics["macro_auc"])
        ),
        "correct_exceeds_every_null_recall1": (
            float(metrics["recall1"]) > maximum_null_recall1
        ),
        "five_pp_target_reached": (
            float(paired_official["delta_recall1"]) >= args.target_delta_recall1
        ),
    }
    method_valid = all(
        value for key, value in gates.items() if key != "five_pp_target_reached"
    )
    target_reached = method_valid and gates["five_pp_target_reached"]
    output = {
        "status": (
            "CHEMAWARE_SURGICAL_FIVE_PP_TARGET_REACHED"
            if target_reached
            else (
                "CHEMAWARE_SURGICAL_VALID_INCREMENT_BELOW_FIVE_PP"
                if method_valid
                else "CHEMAWARE_SURGICAL_RELEASE_FAIL"
            )
        ),
        "formula_role": 3,
        "method_valid": bool(method_valid),
        "five_pp_target_reached": bool(target_reached),
        "target_delta_recall1": float(args.target_delta_recall1),
        "delta_recall1_vs_official": float(paired_official["delta_recall1"]),
        "delta_recall1_vs_stage1": float(paired_stage1["delta_recall1"]),
        "formula_cluster_ci95_vs_stage1": paired_stage1[
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ],
        "corrected_vs_stage1": int(paired_stage1["corrected_at_1"]),
        "introduced_vs_stage1": int(paired_stage1["introduced_at_1"]),
        "risk_utility_vs_stage1": int(risk),
        "maximum_null_recall1": float(maximum_null_recall1),
        "gates": gates,
        "claim_boundary": (
            "The five-pp claim is permitted only when every method-validity gate "
            "and the explicit total delta threshold pass on frozen formula role 3."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
