#!/usr/bin/env python
"""Fail-closed validator for the BioAware B35 retrieval benchmark."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    per_query_path = args.output_dir / "per_query.csv.gz"
    if not report_path.is_file() or not per_query_path.is_file():
        raise FileNotFoundError("B35 report or per-query ledger is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(per_query_path)
    if report.get("status") != "bioaware_b35_real_library_retrieval_complete":
        raise RuntimeError("unexpected B35 status")
    if len(frame) != 1738 or frame["query_id"].nunique() != 1738:
        raise RuntimeError("B35 row-level coverage changed")
    if frame["physical_key"].nunique() != 1631:
        raise RuntimeError("B35 physical-query coverage changed")
    counts = frame.groupby("polarity")["query_id"].nunique().to_dict()
    if counts != {"negative": 860, "positive": 878}:
        raise RuntimeError(f"B35 polarity coverage changed: {counts}")
    negative = frame.loc[frame["polarity"].eq("negative")]
    if not math.isclose(float(negative["official_rank"].eq(1).mean()), 0.6604651162790698, abs_tol=1e-15):
        raise RuntimeError("B35 official negative Recall@1 replay changed")
    if not math.isclose(float(negative["bioaware_rank"].eq(1).mean()), 0.7197674418604652, abs_tol=1e-15):
        raise RuntimeError("B35 B30 negative Recall@1 replay changed")
    positive = frame.loc[frame["polarity"].eq("positive")]
    if not positive["official_rank"].equals(positive["bioaware_rank"]):
        raise RuntimeError("B35 positive abstention changed a rank")
    if report["auc_definitions"].get("exactly_comparable_to_DreaMS_paper_0_85") is not False:
        raise RuntimeError("B35 improperly claims exact paper-AUROC comparability")
    if report["contracts"].get("opened_development_queries") is not True:
        raise RuntimeError("B35 opened-data disclosure changed")
    if report["contracts"].get("independent_blind_test") is not False:
        raise RuntimeError("B35 blind-test disclosure changed")
    if not all(report["gates"].values()) or not report.get("pass_opened_real_library_benchmark"):
        raise RuntimeError(f"B35 gates failed: {report['gates']}")
    primary = report["panels"]["primary_all_real_physical_queries"]
    if primary["n_queries"] != 1631:
        raise RuntimeError("B35 primary panel size changed")
    for model in ("official_dreams", "dreams_plus_bioaware_b30"):
        metrics = primary["metrics"][model]
        for key in (
            "recall_at_1", "recall_at_2", "recall_at_5", "recall_at_10",
            "recall_at_20", "mrr", "macro_query_auroc",
            "micro_within_query_auroc", "pooled_candidate_auroc",
        ):
            if key not in metrics or not 0 <= float(metrics[key]) <= 1:
                raise RuntimeError(f"B35 invalid {model}/{key}")
    print(
        "[validate_bioaware_b35_real_library_retrieval] PASS",
        json.dumps({
            "primary_n": primary["n_queries"],
            "official": primary["metrics"]["official_dreams"],
            "bioaware": primary["metrics"]["dreams_plus_bioaware_b30"],
            "delta": primary["metrics"]["delta_bioaware_minus_dreams"],
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
