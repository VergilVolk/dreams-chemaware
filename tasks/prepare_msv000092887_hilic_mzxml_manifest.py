#!/usr/bin/env python3
"""Freeze the MSV000092887 HILIC-positive mzXML download universe.

The file inventory is metadata only.  Biological roles are joined from the
already-frozen assay ledger; they are recorded for provenance but must not be
read by the later spectrum-to-feature linker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
from pathlib import Path

import pandas as pd


DATASET = "MSV000092887"
COLLECTION = "HILIC Metabolomics -MZXML"
PREFIX = f"peak/{COLLECTION}/"
BASE = "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(file_index: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    required_index = {"dataset", "collection", "filepath", "size"}
    required_ledger = {"assay_name", "biological_unit_id", "pair_id", "group"}
    if missing := required_index - set(file_index):
        raise ValueError(f"file index lacks {sorted(missing)}")
    if missing := required_ledger - set(ledger):
        raise ValueError(f"assay ledger lacks {sorted(missing)}")

    selected = file_index.loc[
        file_index.dataset.eq(DATASET)
        & file_index.collection.eq(COLLECTION)
        & file_index.filepath.str.startswith(PREFIX)
        & file_index.filepath.str.lower().str.endswith(".mzxml")
    ].copy()
    selected["assay_name"] = selected.filepath.map(lambda value: Path(value).stem)
    if selected.filepath.duplicated().any() or selected.assay_name.duplicated().any():
        raise ValueError("HILIC-positive paths and assay names must be unique")

    biological = ledger[["assay_name", "biological_unit_id", "pair_id", "group"]].copy()
    if biological.assay_name.duplicated().any():
        raise ValueError("biological assay names must be unique")
    selected = selected.merge(biological, on="assay_name", how="left", validate="one_to_one")
    selected["role"] = selected.biological_unit_id.notna().map(
        {True: "biological", False: "blank_or_qc"}
    )
    selected["expected_bytes"] = pd.to_numeric(selected["size"], errors="raise").astype("int64")
    selected["remote_descriptor"] = selected.filepath.map(lambda value: f"f.{DATASET}/{value}")
    selected["download_url"] = selected.remote_descriptor.map(
        lambda value: BASE + urllib.parse.urlencode({"file": value})
    )
    selected["spectrum_linkage_visibility"] = "phenotype_blind"
    selected = selected.sort_values(["role", "assay_name"]).reset_index(drop=True)

    if len(selected) != 129:
        raise ValueError(f"expected 129 HILIC-positive mzXML files, got {len(selected)}")
    if int(selected.expected_bytes.sum()) != 838_574_158:
        raise ValueError("HILIC-positive byte denominator changed")
    biological_rows = selected.role.eq("biological")
    if int(biological_rows.sum()) != 78:
        raise ValueError(f"expected 78 biological files, got {int(biological_rows.sum())}")
    if int(selected.loc[biological_rows, "expected_bytes"].sum()) != 509_571_140:
        raise ValueError("biological-file byte denominator changed")
    if selected.loc[biological_rows, ["biological_unit_id", "pair_id", "group"]].isna().any().any():
        raise ValueError("biological files lack design provenance")
    if len(set(biological.assay_name) - set(selected.assay_name)):
        raise ValueError("at least one biological assay is absent from the file index")

    return selected[[
        "dataset", "collection", "filepath", "assay_name", "role",
        "biological_unit_id", "pair_id", "group", "expected_bytes",
        "remote_descriptor", "download_url", "spectrum_linkage_visibility",
    ]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-index", required=True, type=Path)
    parser.add_argument("--assay-ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    file_index = pd.read_csv(args.file_index, low_memory=False)
    ledger = pd.read_csv(args.assay_ledger, sep="\t", dtype=str, keep_default_na=False)
    manifest = build_manifest(file_index, ledger)
    args.output_dir.mkdir(parents=True)
    phenotype_blind_manifest = manifest[[
        "dataset", "collection", "filepath", "assay_name", "expected_bytes",
        "remote_descriptor", "download_url", "spectrum_linkage_visibility",
    ]]
    phenotype_blind_manifest.to_csv(
        args.output_dir / "mzxml_manifest.tsv", sep="\t", index=False
    )

    biological = manifest.role.eq("biological")
    report = {
        "schema": "msv000092887_hilic_positive_mzxml_manifest_v2_phenotype_blind",
        "status": "DOWNLOAD_UNIVERSE_FROZEN",
        "evidence_role": {
            "dataset_status": "development_consumed",
            "leakage_state": "outcome_open_but_manifest_construction_metadata_only",
            "external_validation_claim": False,
        },
        "denominators": {
            "files": int(len(manifest)),
            "bytes": int(manifest.expected_bytes.sum()),
            "biological_files": int(biological.sum()),
            "biological_bytes": int(manifest.loc[biological, "expected_bytes"].sum()),
            "blank_or_qc_files": int((~biological).sum()),
            "blank_or_qc_bytes": int(manifest.loc[~biological, "expected_bytes"].sum()),
        },
        "provenance": {
            "file_index_sha256": sha256(args.file_index),
            "assay_ledger_sha256": sha256(args.assay_ledger),
        },
        "linkage_contract": {
            "download_universe_includes_all_129_files": True,
            "download_completed": False,
            "reason": "blank and QC files are retained for acquisition-domain and false-link calibration",
            "phenotype_columns_absent_from_linkage_manifest": [
                "biological_unit_id", "pair_id", "group"
            ],
            "published_alignment_tolerances": {
                "retention_time_minutes": 0.05,
                "ms1_da": 0.005
            },
            "published_tolerances_are_initial_audit_values_not_posthoc_tunable_success_parameters": True
        },
        "claim_boundary": "This freezes acquisition only; it does not establish feature-level MS2 membership or a biological response replication."
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
