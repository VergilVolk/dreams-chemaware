#!/usr/bin/env python
"""Adjudicate ChemProp2 M1 across prespecified library-score sensitivities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_report(path: Path) -> dict:
    report_path = path / "report.json"
    if not report_path.is_file():
        raise RuntimeError(f"missing report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "chemprop2_response_reproducibility_m1_complete":
        raise RuntimeError(f"invalid M1 report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score070", type=Path,
        default=Path("data/validation/chemprop2_response_m1_score070_20261011"),
    )
    parser.add_argument(
        "--score080", type=Path,
        default=Path("data/validation/chemprop2_response_m1_v2_20261011"),
    )
    parser.add_argument(
        "--score090", type=Path,
        default=Path("data/validation/chemprop2_response_m1_score090_20261011"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/validation/chemprop2_response_m1_sensitivity_20261011.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")
    reports = {
        "0.70": read_report(args.score070),
        "0.80": read_report(args.score080),
        "0.90": read_report(args.score090),
    }
    source_hashes = [report["dataset"]["source_hashes"] for report in reports.values()]
    if any(item != source_hashes[0] for item in source_hashes[1:]):
        raise RuntimeError("source hashes differ across score sensitivities")

    rows: dict[str, dict[str, object]] = {}
    for threshold, report in reports.items():
        prefix = "log2_feature_pseudocount_median_centered|trajectory_auc|"
        known = report["retrieval"][prefix + "library_supported"]
        unresolved = report["retrieval"][prefix + "unresolved"]
        delta = report["primary_deltas_cluster_bootstrap_by_drug"][
            "unresolved_minus_library_supported_mrr"
        ]
        controls = report["primary_negative_controls"]
        rows[threshold] = {
            "library_supported_features": report["feature_universes"]["library_supported"],
            "unresolved_features": report["feature_universes"]["unresolved"],
            "library_supported_mrr": known["mrr"],
            "unresolved_mrr": unresolved["mrr"],
            "unresolved_minus_library_supported_mrr": delta,
            "coordinate_permutation_empirical_p_mrr": controls[
                "independent_coordinate_permutation"
            ]["empirical_p_mrr_ge_observed"],
            "count_matched_unresolved_mrr_q025": controls[
                "count_matched_unresolved"
            ]["mrr"]["q025"],
            "unresolved_above_coordinate_permutation": controls[
                "independent_coordinate_permutation"
            ]["empirical_p_mrr_ge_observed"] <= 0.05,
            "unknown_increment_ci_positive": delta["ci95_low"] > 0,
            "count_matched_unknown_above_known": controls[
                "count_matched_unresolved"
            ]["mrr"]["q025"] > known["mrr"],
        }

    response_signal_robust = all(
        bool(row["unresolved_above_coordinate_permutation"]) for row in rows.values()
    )
    unknown_increment_robust = all(
        bool(row["unknown_increment_ci_positive"])
        and bool(row["count_matched_unknown_above_known"])
        for row in rows.values()
    )
    output = {
        "status": "chemprop2_response_m1_sensitivity_adjudicated",
        "formal": False,
        "evidence_role": "development_consumed",
        "dataset": "MSV000094899",
        "denominator": "36 leave-one-replicate queries from 12 drugs x 3 replicates",
        "source_hashes": source_hashes[0],
        "threshold_results": rows,
        "adjudication": {
            "unresolved_response_signal_robust_to_library_threshold": response_signal_robust,
            "pass_response_object_to_cross_design_m1b": response_signal_robust,
            "unknown_increment_robust_to_library_threshold": unknown_increment_robust,
            "pass_unknown_increment_claim": unknown_increment_robust,
            "reason": (
                "Unresolved profiles beat coordinate permutation at every threshold, but the "
                "unresolved-versus-library-supported MRR increment and count-matched control "
                "do not pass at MQScore 0.70. Library-score thresholding is not a valid final "
                "known/unresolved truth definition."
            ),
        },
        "next_decisive_work": [
            "repeat M1 across the independent two-timepoint design for shared drugs",
            "replace score-threshold known status with authentic-standard or frozen author identity state",
            "test spectral response transport against exact-feature, MS1-only, FBMN, and shuffled-relation controls",
            "do not claim mechanism retrieval or cross-source transfer from this result",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output["adjudication"], indent=2))


if __name__ == "__main__":
    main()
