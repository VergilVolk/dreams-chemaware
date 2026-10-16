#!/usr/bin/env python3
"""Select a repository-scale raw tranche after reusing deposited quant/MS2 pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition", required=True, type=Path)
    parser.add_argument("--wave", required=True, type=Path)
    parser.add_argument("--processed-pairs", required=True, type=Path)
    parser.add_argument("--source-count", type=int, default=20)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    acquisition = pd.read_csv(args.acquisition, sep="\t", dtype=str, keep_default_na=False)
    acquisition = acquisition.loc[acquisition["acquisition_keep"].str.lower().eq("true")].copy()
    acquisition["deposited_bytes"] = pd.to_numeric(acquisition["deposited_bytes"], errors="raise").astype("int64")
    wave = pd.read_csv(args.wave, sep="\t", dtype=str, keep_default_na=False)
    pairs = pd.read_csv(args.processed_pairs, sep="\t", dtype=str, keep_default_na=False)
    pairs = pairs.loc[pairs["schema_qualified"].str.lower().eq("true")].copy()

    accession_sources = wave[["dataset_accession", "preliminary_source_family"]].drop_duplicates()
    processed_sources = set(
        accession_sources.loc[
            accession_sources["dataset_accession"].isin(pairs["dataset_accession"]),
            "preliminary_source_family",
        ]
    )
    remaining = acquisition.loc[~acquisition["preliminary_source_family"].isin(processed_sources)].copy()
    panels = (
        wave.groupby("preliminary_source_family")["contrast_id"]
        .nunique().rename("candidate_panels")
    )
    sources = (
        remaining.groupby("preliminary_source_family")
        .agg(
            accessions=("dataset_accession", lambda values: ",".join(sorted(set(values)))),
            files=("filename", "size"),
            bytes=("deposited_bytes", "sum"),
        )
        .join(panels, how="left")
        .reset_index()
    )
    sources["candidate_panels"] = sources["candidate_panels"].fillna(0).astype(int)
    sources["gib"] = sources["bytes"] / (2 ** 30)
    sources["gib_per_candidate_panel"] = sources["gib"] / sources["candidate_panels"].clip(lower=1)
    selected = sources.sort_values(
        ["gib_per_candidate_panel", "gib", "preliminary_source_family"]
    ).head(args.source_count).copy()
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected_ids = set(selected["preliminary_source_family"])
    manifest = remaining.loc[remaining["preliminary_source_family"].isin(selected_ids)].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "selected_sources.tsv", sep="\t", index=False)
    manifest.to_csv(args.output_dir / "raw_acquisition_manifest.tsv", sep="\t", index=False)
    selected_wave = wave.loc[wave["preliminary_source_family"].isin(selected_ids)].copy()
    selected_wave.to_csv(args.output_dir / "selected_contrast_panels.tsv", sep="\t", index=False)
    report = {
        "schema": "pan_redu_high_value_raw_tranche_v1",
        "selection_rule": "lowest_GiB_per_candidate_panel_after_reusing_schema_qualified_processed_sources",
        "denominators": {
            "processed_accessions_reused": int(pairs["dataset_accession"].nunique()),
            "processed_source_families_reused_in_wave": len(processed_sources),
            "selected_new_source_families": int(selected["preliminary_source_family"].nunique()),
            "selected_accessions": int(manifest["dataset_accession"].nunique()),
            "selected_candidate_panels": int(selected_wave["contrast_id"].nunique()),
            "selected_raw_files": len(manifest),
            "selected_raw_bytes": int(manifest["deposited_bytes"].sum()),
            "selected_raw_gib": float(manifest["deposited_bytes"].sum() / (2 ** 30)),
        },
        "leakage_status": "metadata_file_size_and_processed_schema_availability_only",
        "claim_boundary": [
            "Cost-efficient acquisition does not establish biological truth or analysis readiness.",
            "Processed sources are reused only as inputs; they do not count as independent raw reprocessing validations.",
            "The full post-deduplication acquisition manifest remains frozen and can be added without changing the method.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
