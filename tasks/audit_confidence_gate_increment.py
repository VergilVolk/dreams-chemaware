"""Audit the independent contribution of the confidence gate and rule panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def formula_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n: int, seed: int) -> list[float]:
    merged = a[["query_index", "formula", "top1"]].merge(
        b[["query_index", "formula", "top1"]], on=["query_index", "formula"],
        suffixes=("_a", "_b"), validate="one_to_one",
    )
    by_formula = {
        formula: group["top1_b"].to_numpy(float) - group["top1_a"].to_numpy(float)
        for formula, group in merged.groupby("formula")
    }
    formulas = np.array(list(by_formula), dtype=object)
    rng = np.random.default_rng(seed)
    values = np.empty(n)
    for index in range(n):
        sampled = rng.choice(formulas, size=len(formulas), replace=True)
        delta = np.concatenate([by_formula[formula] for formula in sampled])
        values[index] = delta.mean()
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def compare(a: pd.DataFrame, b: pd.DataFrame, n: int, seed: int) -> dict:
    merged = a.merge(
        b, on=["query_index", "ik14", "formula"], suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    result = {
        "top1_a": float(merged["top1_a"].mean()),
        "top1_b": float(merged["top1_b"].mean()),
        "delta_b_minus_a": float(merged["top1_b"].mean() - merged["top1_a"].mean()),
        "formula_bootstrap_ci95": formula_bootstrap(a, b, n, seed),
        "wrong_to_correct": int(((~merged["top1_a"]) & merged["top1_b"]).sum()),
        "correct_to_wrong": int((merged["top1_a"] & (~merged["top1_b"])).sum()),
    }
    if "chosen_ik14_a" in merged and "chosen_ik14_b" in merged:
        result["same_choice_fraction"] = float(
            (merged["chosen_ik14_a"] == merged["chosen_ik14_b"]).mean()
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gated-dir", type=Path, default=Path("data/validation/confidence_gated_reranker"))
    parser.add_argument("--ungated-dir", type=Path, default=Path("data/validation/pairwise_delta_reranker"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/confidence_gate_audit"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gated_raw = pd.read_csv(args.gated_dir / "raw_confirmation_gated_queries.csv")
    gated_panel = pd.read_csv(args.gated_dir / "raw_panel_confirmation_gated_queries.csv")
    ungated_raw = pd.read_csv(args.ungated_dir / "dreams_plus_raw_confirmation_queries.csv")
    ungated_panel = pd.read_csv(args.ungated_dir / "dreams_plus_raw_burden_confirmation_queries.csv")

    report = {
        "rule_panel_increment_with_same_gate": compare(gated_raw, gated_panel, args.bootstrap, 20260813),
        "gate_increment_for_raw": compare(ungated_raw, gated_raw, args.bootstrap, 20260814),
        "gate_increment_for_raw_panel": compare(ungated_panel, gated_panel, args.bootstrap, 20260815),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
