#!/usr/bin/env python
"""Freeze an outcome-unopened within-study ST001154 HILIC extension panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from audit_st001154_bioaware_external_readiness import checksum  # noqa: E402
from freeze_st001154_hilic_confirmation_samples import (  # noqa: E402
    archive_inventory,
    evenly_spaced_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-workbook", type=Path,
        default=Path(
            "data/reference/ST001154_negative_pilot_20260901/"
            "KOMP_All_AssaysSampleDetails.xlsx"
        ),
    )
    parser.add_argument(
        "--files-json", type=Path,
        default=Path(
            "data/reference/bioaware_public_cohort_probe_20260901/"
            "ST001154__files__json"
        ),
    )
    parser.add_argument(
        "--opened-selection", type=Path, nargs="+",
        default=[Path(
            "data/validation/bioaware_st001154_hilic_confirmation_selection_v1/"
            "samples.csv"
        )],
    )
    parser.add_argument("--pilot-file", default="KOMP_HILIC_NEG_345321_152.raw")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_st001154_hilic_extension_selection_v1"),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    for path in (args.sample_workbook, args.files_json, *args.opened_selection):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    samples = pd.read_excel(args.sample_workbook, sheet_name="HILICNEG")
    required = {"KOMPLABEL", "FileName", "Order", "SAMPLETYPE"}
    if not required.issubset(samples.columns):
        raise RuntimeError("HILICNEG workbook lacks required acquisition columns")
    opened = pd.concat(
        [pd.read_csv(path) for path in args.opened_selection], ignore_index=True
    )
    if opened["FileName"].duplicated().any() or opened["KOMPLABEL"].duplicated().any():
        raise RuntimeError("opened selection inputs overlap each other")
    opened_files = set(opened["FileName"].astype(str))
    opened_labels = set(opened["KOMPLABEL"].astype(str))
    eligible = samples.loc[
        samples["SAMPLETYPE"].astype(str).eq("StudySample")
        & ~samples["FileName"].astype(str).eq(args.pilot_file)
        & ~samples["FileName"].astype(str).isin(opened_files)
        & ~samples["KOMPLABEL"].astype(str).isin(opened_labels)
    ].copy()
    eligible = eligible.sort_values(["Order", "FileName"], kind="stable")
    technical_reinjections_excluded = int(eligible["KOMPLABEL"].duplicated().sum())
    eligible = eligible.drop_duplicates("KOMPLABEL", keep="first")
    selected = evenly_spaced_samples(eligible, args.n_samples)
    if set(selected["FileName"].astype(str)) & opened_files:
        raise RuntimeError("extension panel overlaps the opened selection")
    inventory = archive_inventory(args.files_json)
    selected["archive_bytes"] = selected["FileName"].map(inventory)
    if selected["archive_bytes"].isna().any() or (selected["archive_bytes"] <= 0).any():
        raise RuntimeError("selected extension RAW is missing from archive inventory")
    selected = selected[["KOMPLABEL", "FileName", "Order", "SAMPLETYPE", "archive_bytes"]]
    forbidden = ("treatment", "genotype", "sex", "zygosity", "phenotype")
    if any(token in column.lower() for column in selected.columns for token in forbidden):
        raise RuntimeError("phenotype-like column entered extension selection")
    args.output_dir.mkdir(parents=True)
    sample_path = args.output_dir / "samples.csv"
    selected.to_csv(sample_path, index=False)
    report = {
        "status": "bioaware_st001154_hilic_confirmation_samples_frozen",
        "formal": True,
        "selection": (
            "eight acquisition-order-evenly-spaced StudySample RAW files after excluding "
            "the previously opened eight-sample panel"
        ),
        "samples": int(len(selected)),
        "opened_samples_excluded": int(len(opened_files)),
        "technical_reinjections_excluded_before_selection": technical_reinjections_excluded,
        "total_archive_bytes": int(selected["archive_bytes"].sum()),
        "minimum_order": int(selected["Order"].min()),
        "maximum_order": int(selected["Order"].max()),
        "phenotype_columns_used": False,
        "outcome_opened": False,
        "validation_role": "within-study prospective extension; not an independent cohort",
        "provenance": {
            "sample_workbook_sha256": checksum(args.sample_workbook),
            "files_json_sha256": checksum(args.files_json),
            "opened_selection_sha256": {
                str(path): checksum(path) for path in args.opened_selection
            },
            "samples_sha256": checksum(sample_path),
            "script_sha256": checksum(Path(__file__)),
        },
        "claim_limit": (
            "Sample selection only. The panel is spectrum-outcome-unopened but belongs to the "
            "same study as the diagnostic cohort, so it cannot establish independent SOTA."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(selected.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
