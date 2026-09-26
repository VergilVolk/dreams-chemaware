"""Select one short residual-continuation checkpoint on formula role 2."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--base-name", default="stage1_base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-positive-formula-ci", action="store_true",
        help="Require the formula-cluster Recall@1 CI lower bound to exceed zero.",
    )
    return parser.parse_args()


def step_from_checkpoint(path: str) -> int:
    match = re.search(r"step-(\d+)\.ckpt$", path.replace("\\", "/"))
    if match is None:
        raise ValueError(f"cannot recover fixed step from checkpoint path: {path}")
    return int(match.group(1))


def main() -> None:
    args = arguments()
    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if int(report.get("formula_role", -1)) != 2:
        raise RuntimeError("residual checkpoint selection must use formula role 2")
    rows = {row["name"]: row for row in report["results"]}
    if args.base_name not in rows:
        raise KeyError(f"base checkpoint is absent: {args.base_name}")
    base = rows[args.base_name]
    paired_field = f"paired_vs_{args.base_name}"
    candidates = []
    for name, row in rows.items():
        if name in {"official", args.base_name}:
            continue
        paired = row.get(paired_field)
        if paired is None:
            raise KeyError(f"{name} lacks {paired_field}")
        metrics = row["metrics"]
        base_metrics = base["metrics"]
        risk_utility = int(paired["corrected_at_1"]) - 2 * int(paired["introduced_at_1"])
        gates = {
            "recall1_strictly_improves": float(paired["delta_recall1"]) > 0.0,
            "risk_utility_positive": risk_utility > 0,
            "mrr_nonnegative": float(paired["delta_mrr"]) >= 0.0,
            "recall3_nonnegative": float(metrics["recall3"]) >= float(base_metrics["recall3"]),
            "micro_auc_nonnegative": float(metrics["micro_auc"]) >= float(base_metrics["micro_auc"]),
            "macro_auc_nonnegative": float(metrics["macro_auc"]) >= float(base_metrics["macro_auc"]),
        }
        if args.require_positive_formula_ci:
            gates["formula_cluster_ci_strictly_positive"] = float(
                paired["formula_cluster_bootstrap_delta_recall1_ci95"][0]
            ) > 0.0
        candidates.append({
            "name": name,
            "checkpoint": row["checkpoint"],
            "step": step_from_checkpoint(row["checkpoint"]),
            "risk_utility_vs_base": risk_utility,
            "paired_vs_base": paired,
            "metrics": metrics,
            "gates": gates,
            "admissible": all(gates.values()),
        })
    admissible = [row for row in candidates if row["admissible"]]
    if admissible:
        selected = max(admissible, key=lambda row: (
            int(row["risk_utility_vs_base"]),
            float(row["paired_vs_base"]["delta_recall1"]),
            float(row["paired_vs_base"]["delta_mrr"]),
            -int(row["step"]),
        ))
        advanced = True
    else:
        selected = {
            "name": args.base_name,
            "checkpoint": base["checkpoint"],
            "step": 0,
            "risk_utility_vs_base": 0,
            "paired_vs_base": None,
            "metrics": base["metrics"],
            "gates": {},
            "admissible": True,
        }
        advanced = False
    output = {
        "status": (
            "CHEMAWARE_RESIDUAL_CHECKPOINT_SELECTED"
            if advanced else "CHEMAWARE_RESIDUAL_CHECKPOINT_RETAIN_BASE"
        ),
        "formula_role": 2,
        "base_name": args.base_name,
        "advanced_beyond_base": advanced,
        # Backward-compatible field retained for older consumers.  New code
        # must use advanced_beyond_base because the comparator may be Phase A.
        "advanced_beyond_stage1": advanced,
        "selection_rule": (
            f"maximize corrected-2*introduced versus {args.base_name}; require positive "
            "Recall@1 and MRR plus nonnegative Recall@3, micro-AUC and macro-AUC"
            + (
                "; require strictly positive formula-cluster Recall@1 CI"
                if args.require_positive_formula_ci else ""
            )
        ),
        "selected": selected,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    args.output.with_suffix(".checkpoint.txt").write_text(
        str(selected["checkpoint"]) + "\n", encoding="utf-8",
    )
    args.output.with_suffix(".step.txt").write_text(
        str(selected["step"]) + "\n", encoding="utf-8",
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
