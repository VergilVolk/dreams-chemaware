#!/usr/bin/env python
"""Freeze phenotype-blind ST001154 HILIC-negative confirmation samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from audit_st001154_bioaware_external_readiness import checksum
except ModuleNotFoundError:  # imported as tasks.* during tests
    from tasks.audit_st001154_bioaware_external_readiness import checksum


def archive_inventory(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        str(row["name"]): int(row.get("size", 0) or 0)
        for rows in payload.get("compressed_file_content", {}).values()
        for row in rows
    }


def evenly_spaced_samples(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count < 1 or len(frame) < count:
        raise ValueError("sample count is invalid for the available frame")
    ordered = frame.sort_values(["Order", "FileName"], kind="stable").reset_index(drop=True)
    positions = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
    if len(np.unique(positions)) != count:
        raise RuntimeError("evenly spaced sample positions are not unique")
    return ordered.iloc[positions].copy().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-workbook",
        type=Path,
        default=Path(
            "data/reference/ST001154_negative_pilot_20260901/"
            "KOMP_All_AssaysSampleDetails.xlsx"
        ),
    )
    parser.add_argument(
        "--files-json",
        type=Path,
        default=Path(
            "data/reference/bioaware_public_cohort_probe_20260901/"
            "ST001154__files__json"
        ),
    )
    parser.add_argument("--pilot-file", default="KOMP_HILIC_NEG_345321_152.raw")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/validation/bioaware_st001154_hilic_confirmation_selection_v1"
        ),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    samples = pd.read_excel(args.sample_workbook, sheet_name="HILICNEG")
    required = {"KOMPLABEL", "FileName", "Order", "SAMPLETYPE"}
    if not required.issubset(samples.columns):
        raise RuntimeError(f"HILICNEG workbook lacks {sorted(required-set(samples.columns))}")
    eligible = samples.loc[
        samples["SAMPLETYPE"].astype(str).eq("StudySample")
        & ~samples["FileName"].astype(str).eq(args.pilot_file)
    ].copy()
    if eligible["FileName"].duplicated().any():
        raise RuntimeError("eligible confirmation RAW filenames are not unique")
    eligible = eligible.sort_values(["Order", "FileName"], kind="stable")
    technical_reinjections_excluded = int(eligible["KOMPLABEL"].duplicated().sum())
    eligible = eligible.drop_duplicates("KOMPLABEL", keep="first")
    if eligible["KOMPLABEL"].duplicated().any():
        raise RuntimeError("biological sample deduplication failed")
    selected = evenly_spaced_samples(eligible, args.n_samples)
    inventory = archive_inventory(args.files_json)
    selected["archive_bytes"] = selected["FileName"].map(inventory)
    if selected["archive_bytes"].isna().any() or (selected["archive_bytes"] <= 0).any():
        raise RuntimeError("selected confirmation RAW is missing from archive inventory")
    selected = selected[["KOMPLABEL", "FileName", "Order", "SAMPLETYPE", "archive_bytes"]]
    if any(
        token in column.lower()
        for column in selected.columns
        for token in ("treatment", "genotype", "sex", "zygosity", "phenotype")
    ):
        raise RuntimeError("phenotype-like columns entered the frozen sample selection")

    args.output_dir.mkdir(parents=True)
    sample_path = args.output_dir / "samples.csv"
    selected.to_csv(sample_path, index=False)
    report = {
        "status": "bioaware_st001154_hilic_confirmation_samples_frozen",
        "formal": True,
        "selection": "eight acquisition-order-evenly-spaced StudySample RAW files",
        "pilot_file_excluded": args.pilot_file,
        "samples": int(len(selected)),
        "technical_reinjections_excluded_before_selection": technical_reinjections_excluded,
        "total_archive_bytes": int(selected["archive_bytes"].sum()),
        "minimum_order": int(selected["Order"].min()),
        "maximum_order": int(selected["Order"].max()),
        "phenotype_columns_used": False,
        "outcome_opened": False,
        "provenance": {
            "sample_workbook_sha256": checksum(args.sample_workbook),
            "files_json_sha256": checksum(args.files_json),
            "samples_sha256": checksum(sample_path),
            "script_sha256": checksum(Path(__file__)),
        },
        "claim_limit": "Sample selection only; no spectrum, candidate, DreaMS, or BioAware outcome was inspected.",
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(selected.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
