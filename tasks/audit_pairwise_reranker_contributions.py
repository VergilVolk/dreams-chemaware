"""Audit incremental contributions and failure transitions of pairwise rerankers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = [
    "dreams_only", "dreams_plus_raw", "dreams_plus_burden",
    "dreams_plus_token", "dreams_plus_raw_burden", "dreams_plus_raw_burden_token",
]


def formula_bootstrap(frame: pd.DataFrame, column: str, iterations: int, seed: int) -> list[float]:
    values = frame.groupby("formula")[column].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for i in range(iterations):
        draws[i] = rng.choice(values, len(values), replace=True).mean()
    return np.quantile(draws, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranker-dir", type=Path, default=Path("data/validation/pairwise_delta_reranker"))
    parser.add_argument("--audit", type=Path, default=Path("data/validation/large_observability_residual_audit/confirmation_query_audit.csv"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/pairwise_reranker_contribution_audit"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.audit)[[
        "query_index", "audit_quadrant", "robust_model_residual_candidate", "ring_class"
    ]]
    frames = {}
    for model in MODELS:
        frame = pd.read_csv(args.reranker_dir / f"{model}_confirmation_queries.csv")
        frames[model] = frame.merge(audit, on="query_index", how="left", validate="one_to_one")
    comparisons = [
        ("dreams_plus_raw", "dreams_only", "raw_increment"),
        ("dreams_plus_burden", "dreams_only", "panel_burden_increment"),
        ("dreams_plus_raw_burden", "dreams_plus_raw", "panel_burden_over_raw"),
        ("dreams_plus_raw_burden_token", "dreams_plus_raw", "panel_and_token_over_raw"),
        ("dreams_plus_raw_burden_token", "dreams_plus_raw_burden", "token_over_raw_panel"),
    ]
    rows, strata_rows = [], []
    for index, (model_name, reference_name, label) in enumerate(comparisons):
        model = frames[model_name]
        reference = frames[reference_name]
        merged = reference[["query_index", "formula", "top1", "mrr"]].merge(
            model[["query_index", "top1", "mrr"]], on="query_index", suffixes=("_reference", "_model"),
            validate="one_to_one",
        ).merge(audit, on="query_index", how="left", validate="one_to_one")
        merged["top1_difference"] = merged["top1_model"].astype(float) - merged["top1_reference"].astype(float)
        merged["mrr_difference"] = merged["mrr_model"] - merged["mrr_reference"]
        rows.append({
            "comparison": label, "model": model_name, "reference": reference_name,
            "top1_increment": float(merged["top1_difference"].mean()),
            "top1_formula_bootstrap_ci95": formula_bootstrap(merged, "top1_difference", args.bootstrap, 20260813 + index),
            "mrr_increment": float(merged["mrr_difference"].mean()),
            "mrr_formula_bootstrap_ci95": formula_bootstrap(merged, "mrr_difference", args.bootstrap, 20260913 + index),
            "wrong_to_correct": int(((~merged["top1_reference"]) & merged["top1_model"]).sum()),
            "correct_to_wrong": int((merged["top1_reference"] & (~merged["top1_model"])).sum()),
            "net_correct_queries": int(merged["top1_difference"].sum()),
        })
        for stratum, group in merged.groupby("audit_quadrant", dropna=False):
            strata_rows.append({
                "comparison": label, "audit_quadrant": str(stratum), "queries": len(group),
                "reference_top1": float(group["top1_reference"].mean()),
                "model_top1": float(group["top1_model"].mean()),
                "top1_increment": float(group["top1_difference"].mean()),
                "wrong_to_correct": int(((~group["top1_reference"]) & group["top1_model"]).sum()),
                "correct_to_wrong": int((group["top1_reference"] & (~group["top1_model"])).sum()),
            })
    summary = pd.DataFrame(rows)
    strata = pd.DataFrame(strata_rows)
    summary.to_csv(args.output_dir / "incremental_contributions.csv", index=False)
    strata.to_csv(args.output_dir / "failure_type_transitions.csv", index=False)
    report = {
        "status": "pairwise_reranker_contribution_audit",
        "comparisons": rows,
        "decision_rule": (
            "A component has independent confirmation evidence only if its paired formula-bootstrap interval "
            "is above zero; otherwise it remains mechanistically motivated but performance-unproven."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(strata.to_string(index=False))


if __name__ == "__main__":
    main()
