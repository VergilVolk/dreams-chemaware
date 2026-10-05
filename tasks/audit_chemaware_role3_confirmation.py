"""Decide whether one role-2-selected checkpoint confirms on formula role 3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--base-name", default="phaseA_2pp")
    parser.add_argument(
        "--protected-baseline-name", default="phaseA_2pp",
        help="Protected checkpoint that the candidate may not regress on any release metric.",
    )
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def decision(
    report: dict, base_name: str, candidate_name: str,
    protected_baseline_name: str = "phaseA_2pp",
) -> dict:
    if int(report.get("formula_role", -1)) != 3:
        raise RuntimeError("confirmation decision must use formula role 3")
    if report.get("formula_role_contract_passed") is not True:
        raise RuntimeError("role-3 evaluation lacks an exact frozen panel contract")
    rows = {row["name"]: row for row in report["results"]}
    if (
        base_name not in rows
        or candidate_name not in rows
        or protected_baseline_name not in rows
    ):
        raise KeyError("role-3 evaluation lacks base, protected baseline, or candidate")
    base = rows[base_name]
    candidate = rows[candidate_name]
    paired_name = f"paired_vs_{base_name}"
    paired = candidate.get(paired_name)
    if paired is None:
        raise KeyError(f"candidate lacks {paired_name}")
    metrics = candidate["metrics"]
    base_metrics = base["metrics"]
    protected_metrics = rows[protected_baseline_name]["metrics"]
    corrected = int(paired["corrected_at_1"])
    introduced = int(paired["introduced_at_1"])
    gates = {
        "recall1_strictly_improves": float(paired["delta_recall1"]) > 0.0,
        "formula_cluster_ci_strictly_positive": float(
            paired["formula_cluster_bootstrap_delta_recall1_ci95"][0]
        ) > 0.0,
        "corrected_exceeds_twice_introduced": corrected > 2 * introduced,
        "mrr_nonnegative": float(paired["delta_mrr"]) >= 0.0,
        "recall3_nonnegative": float(metrics["recall3"]) >= float(base_metrics["recall3"]),
        "micro_auc_nonnegative": float(metrics["micro_auc"]) >= float(base_metrics["micro_auc"]),
        "macro_auc_nonnegative": float(metrics["macro_auc"]) >= float(base_metrics["macro_auc"]),
        "protected_recall1_nonnegative": (
            float(metrics["recall1"]) >= float(protected_metrics["recall1"])
        ),
        "protected_recall3_nonnegative": (
            float(metrics["recall3"]) >= float(protected_metrics["recall3"])
        ),
        "protected_mrr_nonnegative": (
            float(metrics["mrr"]) >= float(protected_metrics["mrr"])
        ),
        "protected_micro_auc_nonnegative": (
            float(metrics["micro_auc"]) >= float(protected_metrics["micro_auc"])
        ),
        "protected_macro_auc_nonnegative": (
            float(metrics["macro_auc"]) >= float(protected_metrics["macro_auc"])
        ),
    }
    return {
        "status": (
            "CHEMAWARE_ROLE3_CONFIRMATION_PASS"
            if all(gates.values()) else "CHEMAWARE_ROLE3_CONFIRMATION_FAIL"
        ),
        "formula_role": 3,
        "base_name": base_name,
        "candidate_name": candidate_name,
        "protected_baseline_name": protected_baseline_name,
        "confirmed": all(gates.values()),
        "paired_vs_base": paired,
        "candidate_metrics": metrics,
        "base_metrics": base_metrics,
        "protected_baseline_metrics": protected_metrics,
        "gates": gates,
    }


def main() -> None:
    args = arguments()
    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    output = decision(
        report, args.base_name, args.candidate_name, args.protected_baseline_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".confirmed.txt").write_text(
        ("1" if output["confirmed"] else "0") + "\n", encoding="utf-8",
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
