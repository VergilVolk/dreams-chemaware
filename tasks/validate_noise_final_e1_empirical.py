#!/usr/bin/env python
"""Fail-closed validator for E1 empirical noise calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def monotone(values: list[float]) -> bool:
    return values[0] <= values[1] <= values[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    names = [
        "e1_report.json",
        "frozen_empirical_noise_recipe.json",
        "replicate_group_summary.csv",
        "consensus_peak_calibration.csv.gz",
        "pairwise_spectrum_variation.csv.gz",
        "matched_peak_jitter_strata.csv",
        "e1_peak_prevalence_landscape.png",
        "e1_empirical_jitter_by_condition.png",
        "e1_pairwise_dropout_reliability.png",
    ]
    missing = [name for name in names if not (directory / name).exists()]
    if missing:
        raise RuntimeError(f"incomplete E1 output: {missing}")
    report = json.loads((directory / "e1_report.json").read_text(encoding="utf-8"))
    recipe = json.loads((directory / "frozen_empirical_noise_recipe.json").read_text(encoding="utf-8"))
    if report.get("status") != "noise_final_e1_empirical_calibration_complete":
        raise RuntimeError("unexpected E1 status")
    if not report.get("formal") or not report.get("pass_to_e2") or not all(report["gates"].values()):
        raise RuntimeError("formal E1 gates did not pass")
    if report["contracts"].get("P3_identity_overlap") != 0 or report["contracts"].get("P2b") != "forbidden":
        raise RuntimeError("E1 scope contract failed")
    if sha256(directory / "frozen_empirical_noise_recipe.json") != report["provenance"]["recipe_sha256"]:
        raise RuntimeError("frozen recipe hash mismatch")
    dropout_relations = recipe["peak_dropout"]["reliable_identity_equal_by_condition_relation"]
    intensity = recipe["abs_log_intensity_jitter"]
    mz = recipe["abs_mz_jitter_ppm"]
    for relation, dropout in dropout_relations.items():
        if not monotone([dropout["q25"], dropout["q50"], dropout["q75"]]):
            raise RuntimeError(f"dropout quantiles are not monotone for {relation}")
    screening = recipe["peak_dropout"]["e2_screening_grid"]
    if sum(int(value["eligible_for_e2"]) for value in screening.values()) < 2:
        raise RuntimeError("fewer than two reliable dropout relations enter E2")
    for relation, value in screening.items():
        if any(dose <= 0 or dose > 0.30 for dose in value["doses"]):
            raise RuntimeError(f"unsafe E2 dropout dose for {relation}")
    if not monotone([intensity["q25"], intensity["q50"], intensity["q75"]]):
        raise RuntimeError("intensity quantiles are not monotone")
    if not monotone([mz["q25"], mz["q50"], mz["q75"]]):
        raise RuntimeError("m/z quantiles are not monotone")
    groups = pd.read_csv(directory / "replicate_group_summary.csv")
    strata = pd.read_csv(directory / "matched_peak_jitter_strata.csv")
    pairs = pd.read_csv(directory / "pairwise_spectrum_variation.csv.gz")
    if groups["ik14"].nunique() != report["identities"] or len(strata) < 8:
        raise RuntimeError("E1 support tables disagree with report")
    required_pair_columns = {
        "source_peak_clusters", "target_peak_clusters", "matched_clusters",
        "peak_jaccard", "reliable_for_dose",
    }
    if not required_pair_columns.issubset(pairs.columns):
        raise RuntimeError("E1 pairwise reliability columns are incomplete")
    if int(pairs["reliable_for_dose"].sum()) != report["reliable_pairwise_variants"]:
        raise RuntimeError("E1 reliable pair count disagrees with report")
    print(
        json.dumps(
            {
                "status": "noise_final_e1_validation_passed",
                "groups": report["eligible_identity_adduct_groups"],
                "identities": report["identities"],
                "formulas": report["formulas"],
                "cross_condition_groups": report["cross_condition_groups"],
                "consensus_peak_clusters": report["consensus_peak_clusters"],
                "matched_peak_pairs": report["matched_peak_pairs"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
