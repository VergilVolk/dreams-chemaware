#!/usr/bin/env python
"""Fail-closed validator for the locked BioAware v2-0 comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "mtbls13729_bioaware_v2_two_layer_comparison_complete":
        raise RuntimeError("unexpected BioAware v2 comparison status")
    if not report.get("formal") or report.get("queries") != 21:
        raise RuntimeError("locked v2-0 comparison must contain the original 21 queries")
    if report["expanded_seed_rows"] <= report["archived_seed_rows"]:
        raise RuntimeError("expanded seed pool did not exceed archived v1 seed pool")
    for name in ["archived_v1", "expanded_rhea_only", "two_layer"]:
        result = report["results"][name]
        expected = result["baseline_recall1"] + (
            result["corrected"] - result["introduced"]
        ) / result["n_queries"]
        if not np.isclose(expected, result["bioaware_recall1"], atol=1e-12):
            raise RuntimeError(f"{name}: transition counts do not reproduce Recall@1")
    archived = report["results"]["archived_v1"]
    if not (
        np.isclose(archived["baseline_recall1"], 20 / 21)
        and np.isclose(archived["bioaware_recall1"], 19 / 21)
        and archived["corrected"] == 0
        and archived["introduced"] == 1
    ):
        raise RuntimeError("archived BioAware v1 result was not exactly reproduced")
    scored = pd.read_csv(args.output_dir / "two_layer_candidate_scores.csv.gz")
    decisions = pd.read_csv(args.output_dir / "two_layer_query_decisions.csv")
    if scored["query_id"].nunique() != 21 or len(decisions) != 21:
        raise RuntimeError("two-layer output query count mismatch")
    if scored.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("two-layer output contains duplicate candidates")
    if "Level 2a-supported spectral pseudo-truth" not in report["reference_truth"]:
        raise RuntimeError("pseudo-truth limitation is missing")
    print(
        "[validate_bioaware_v2_two_layer] PASS "
        f"seeds={report['expanded_seed_rows']} pass={report['gates']['pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
