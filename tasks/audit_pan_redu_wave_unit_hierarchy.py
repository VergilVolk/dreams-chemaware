#!/usr/bin/env python3
"""Audit reported sample identifiers before any biological effect is fitted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest, sep="\t", dtype=str, keep_default_na=False)
    rows = []
    for contrast_id, block in frame.groupby("contrast_id", sort=True):
        by_id = block.groupby("reported_unique_subject_id")
        file_counts = by_id.size()
        level_counts = by_id["level"].nunique()
        rows.append({
            "contrast_id": contrast_id,
            "dataset_accession": block["dataset_accession"].iloc[0],
            "preliminary_source_family": block["preliminary_source_family"].iloc[0],
            "comparison_stratum": block["comparison_stratum"].iloc[0],
            "n_files": len(block),
            "n_reported_unit_identifiers": len(file_counts),
            "median_files_per_reported_id": float(file_counts.median()),
            "max_files_per_reported_id": int(file_counts.max()),
            "reported_ids_with_multiple_files": int((file_counts > 1).sum()),
            "reported_ids_assigned_multiple_contrast_levels": int((level_counts > 1).sum()),
            "one_file_per_reported_id": bool((file_counts == 1).all()),
            "level_assignment_consistent_within_reported_id": bool((level_counts == 1).all()),
            "design_asset_required": True,
            "biological_independence_status": "not_adjudicated",
        })
    audit = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "unit_hierarchy_audit.tsv", sep="\t", index=False)
    report = {
        "schema": "pan_redu_wave_unit_hierarchy_audit_v1",
        "denominators": {
            "contrast_panels": len(audit),
            "panels_one_file_per_reported_id": int(audit["one_file_per_reported_id"].sum()),
            "panels_with_repeated_files_per_reported_id": int((~audit["one_file_per_reported_id"]).sum()),
            "panels_with_level_consistency": int(audit["level_assignment_consistent_within_reported_id"].sum()),
            "panels_with_reported_id_crossing_levels": int((~audit["level_assignment_consistent_within_reported_id"]).sum()),
        },
        "decision": (
            "No panel is statistically ready from ReDU identifiers alone. "
            "Panels crossing contrast levels fail closed; all others require design-asset adjudication."
        ),
        "leakage_status": "metadata_only_no_feature_abundance_read",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
