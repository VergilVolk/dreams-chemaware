#!/usr/bin/env python
"""Audit method complementarity from frozen GNPS per-query tables.

This is a diagnostic, not a deployable oracle.  It never changes scores and
never selects a method for a query.  It reports where two or more already
evaluated methods are correct on different queries, thereby measuring the
maximum information available to a future train-side selector or curriculum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PANELS = ("identity_disjoint", "formula_disjoint")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--official-method", default="official_dreams")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_table(root: Path, panel: str, method: str) -> pd.DataFrame:
    path = root / f"queries_{panel}_{method}.csv.gz"
    table = pd.read_csv(path).sort_values("query_index").reset_index(drop=True)
    required = {"query_index", "query_row", "query_ik14", "query_formula", "near", "rank"}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError(f"{path} misses columns: {sorted(missing)}")
    return table


def aligned(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    columns = ("query_index", "query_row", "query_ik14", "query_formula", "near")
    return all(left[column].equals(right[column]) for column in columns)


def overlap_block(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    both = a & b
    union = a | b
    return {
        "both": int(both.sum()),
        "a_only": int((a & ~b).sum()),
        "b_only": int((~a & b).sum()),
        "neither": int((~union).sum()),
        "union": int(union.sum()),
        "jaccard": float(both.sum() / union.sum()) if union.any() else 1.0,
    }


def main() -> None:
    args = arguments()
    methods = list(dict.fromkeys(args.method))
    if len(methods) < 2 or len(methods) != len(args.method):
        raise RuntimeError("provide at least two unique --method entries")
    if args.output.exists():
        raise FileExistsError(args.output)

    report: dict[str, object] = {
        "status": "noise_gnps_method_complementarity_complete",
        "methods": methods,
        "official_method": args.official_method,
        "panels": {},
    }
    for panel in PANELS:
        tables = {
            method: load_table(args.evaluation, panel, method)
            for method in [args.official_method, *methods]
        }
        reference = tables[args.official_method]
        for method, table in tables.items():
            if not aligned(reference, table):
                raise RuntimeError(f"unaligned query table: {panel}/{method}")
        correct = {
            method: table["rank"].to_numpy(np.int64) == 1
            for method, table in tables.items()
        }
        official = correct[args.official_method]
        method_correct = {method: correct[method] for method in methods}
        method_union = np.logical_or.reduce(list(method_correct.values()))
        best_recall = max(float(values.mean()) for values in method_correct.values())
        panel_report: dict[str, object] = {
            "queries": int(len(reference)),
            "absolute": {
                method: {
                    "correct": int(values.sum()),
                    "recall_at_1": float(values.mean()),
                }
                for method, values in method_correct.items()
            },
            "pairwise": {},
            "oracle_union": {
                "correct": int(method_union.sum()),
                "recall_at_1": float(method_union.mean()),
                "gain_over_official_pp": float(100.0 * (method_union.mean() - official.mean())),
                "gain_over_best_constituent_pp": float(100.0 * (method_union.mean() - best_recall)),
                "all_methods_wrong": int((~method_union).sum()),
            },
        }
        for index, first in enumerate(methods):
            for second in methods[index + 1:]:
                first_correct = method_correct[first]
                second_correct = method_correct[second]
                key = f"{first}__{second}"
                block = overlap_block(first_correct, second_correct)
                union = first_correct | second_correct
                block.update({
                    "oracle_union_recall_at_1": float(union.mean()),
                    "oracle_gain_over_best_pp": float(
                        100.0 * (union.mean() - max(first_correct.mean(), second_correct.mean()))
                    ),
                    "first_corrects_official": int((~official & first_correct).sum()),
                    "second_corrects_official": int((~official & second_correct).sum()),
                    "shared_corrections_of_official": int((~official & first_correct & second_correct).sum()),
                    "first_introduces_vs_official": int((official & ~first_correct).sum()),
                    "second_introduces_vs_official": int((official & ~second_correct).sum()),
                })
                panel_report["pairwise"][key] = block

        near = reference["near"].to_numpy(bool)
        near_union = method_union[near]
        near_official = official[near]
        panel_report["near_oracle_union"] = {
            "queries": int(near.sum()),
            "recall_at_1": float(near_union.mean()),
            "gain_over_official_pp": float(100.0 * (near_union.mean() - near_official.mean())),
            "all_methods_wrong": int((~near_union).sum()),
        }
        report["panels"][panel] = panel_report

    report["claim_limit"] = (
        "Label-aware OR is an information upper bound only. It is not a deployable score, "
        "not an article performance claim, and may only motivate train-side selection or mining."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
