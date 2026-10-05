#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.input_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b44_massbank_one_time_evaluation_complete":
        raise RuntimeError("B44 result status is invalid")
    if report.get("formal") is not True or report.get("primary_panel", "").split(";")[0] != "NEGATIVE":
        raise RuntimeError("B44 formal primary-panel contract failed")
    query = pd.read_csv(args.input_dir / "per_query.csv.gz")
    candidates = pd.read_csv(args.input_dir / "candidate_scores.csv.gz")
    if len(query) != report["panel"]["queries"] or query["query_id"].duplicated().any():
        raise RuntimeError("B44 per-query result coverage drift")
    if candidates.duplicated(["query_id", "candidate_ik14"]).any():
        raise RuntimeError("B44 candidate score rows are duplicated")
    if not set(report["gates"]).issuperset({
        "negative_primary_formula_ci_positive",
        "negative_primary_real_minus_null_formula_ci_positive",
    }):
        raise RuntimeError("B44 result is missing preregistered gates")
    print("[validate_bioaware_b44_massbank_once] PASS", {
        "queries": len(query),
        "delta": report["by_polarity"]["NEGATIVE"]["paired"]["delta_recall_at_1"],
        "pass": report["pass_external_confirmation"],
    }, flush=True)


if __name__ == "__main__":
    main()
