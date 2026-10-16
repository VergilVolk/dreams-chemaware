#!/usr/bin/env python3
"""Qualify deposited feature-table/MGF pairs without using biological outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd


def normalized_id(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def sample_stem(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    text = re.sub(r"\s+(?:filtered\s+)?Peak area$", "", text, flags=re.I)
    return Path(text.rsplit("/", 1)[-1]).stem.casefold()


def read_quant(path: Path) -> tuple[set[str], list[str], int, int, int]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.read(65536)
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        delimiter = ","
    # Some deposited MZmine tables contain a trailing empty header while a
    # minority of data rows contain additional empty cells.  A dataframe
    # parser rejects those files even though the only fields needed here are
    # the header and feature identifier.  Stream the CSV and record every
    # width mismatch instead of silently repairing the quantitative matrix.
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader)
        normalized = {str(column).strip().casefold(): index for index, column in enumerate(header)}
        id_index = 0
        for name in ("row id", "feature_id", "feature id", "id", "row_id"):
            if name in normalized:
                id_index = normalized[name]
                break
        ids: set[str] = set()
        n_rows = 0
        width_mismatches = 0
        for row in reader:
            if not row or not any(str(value).strip() for value in row):
                continue
            n_rows += 1
            if len(row) != len(header):
                width_mismatches += 1
            if id_index < len(row) and str(row[id_index]).strip():
                ids.add(normalized_id(row[id_index]))
    sample_columns = [
        str(column) for column in header
        if re.search(r"\.mz(?:ml|xml)(?:\s+(?:filtered\s+)?Peak area)?$", str(column), flags=re.I)
        or str(column).strip().casefold().endswith(" peak area")
    ]
    return ids, sample_columns, n_rows, len(header), width_mismatches


def read_mgf_ids(path: Path) -> tuple[set[str], int]:
    ids: set[str] = set()
    blocks = 0
    current: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                blocks += 1
                current = {}
            elif line == "END IONS":
                value = current.get("FEATURE_ID") or current.get("SCANS") or current.get("TITLE")
                if value:
                    ids.add(normalized_id(value))
            elif "=" in line and not line[0].isdigit():
                key, value = line.split("=", 1)
                current[key.upper()] = value
    return ids, blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checksums", required=True, type=Path)
    parser.add_argument("--file-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    assets = pd.read_csv(args.checksums, sep="\t", dtype=str, keep_default_na=False)
    files = pd.read_csv(args.file_manifest, sep="\t", dtype=str, keep_default_na=False)
    selected_stems = {
        accession: {sample_stem(value) for value in block["filename"]}
        for accession, block in files.groupby("dataset_accession")
    }

    quant_records = []
    mgf_records = []
    for row in assets.to_dict("records"):
        path = Path(row["local_path"])
        if path.suffix.lower() == ".mgf":
            ids, blocks = read_mgf_ids(path)
            mgf_records.append({**row, "feature_ids": ids, "n_spectra": blocks})
        else:
            ids, sample_columns, n_rows, n_columns, width_mismatches = read_quant(path)
            stems = {sample_stem(value) for value in sample_columns}
            target = selected_stems.get(row["dataset_accession"], set())
            quant_records.append({
                **row, "feature_ids": ids, "sample_columns": sample_columns,
                "n_rows": n_rows, "n_columns": n_columns,
                "row_width_mismatches": width_mismatches,
                "n_sample_columns": len(sample_columns),
                "selected_file_columns_matched": len(stems & target),
                "selected_file_column_coverage": len(stems & target) / len(target) if target else 0.0,
            })

    pair_rows = []
    for quant in quant_records:
        for mgf in mgf_records:
            if quant["dataset_accession"] != mgf["dataset_accession"]:
                continue
            overlap = quant["feature_ids"] & mgf["feature_ids"]
            denominator = min(len(quant["feature_ids"]), len(mgf["feature_ids"]))
            pair_rows.append({
                "dataset_accession": quant["dataset_accession"],
                "quant_path": quant["filepath"],
                "quant_local_path": quant["local_path"],
                "mgf_path": mgf["filepath"],
                "mgf_local_path": mgf["local_path"],
                "quant_features": len(quant["feature_ids"]),
                "mgf_spectra": mgf["n_spectra"],
                "mgf_feature_ids": len(mgf["feature_ids"]),
                "shared_feature_ids": len(overlap),
                "feature_id_overlap_fraction": len(overlap) / denominator if denominator else 0.0,
                "quant_sample_columns": quant["n_sample_columns"],
                "quant_row_width_mismatches": quant["row_width_mismatches"],
                "selected_file_columns_matched": quant["selected_file_columns_matched"],
                "selected_file_column_coverage": quant["selected_file_column_coverage"],
            })
    pairs = pd.DataFrame(pair_rows)
    if len(pairs):
        pairs["qualification_score"] = pairs[[
            "feature_id_overlap_fraction", "selected_file_column_coverage"
        ]].min(axis=1)
        pairs["schema_qualified"] = (
            (pairs["feature_id_overlap_fraction"] >= 0.90)
            & (pairs["selected_file_column_coverage"] >= 0.50)
            & (pairs["shared_feature_ids"] >= 50)
            & (pairs["quant_sample_columns"] >= 10)
        )
        best = (
            pairs.sort_values(
                ["dataset_accession", "schema_qualified", "qualification_score", "shared_feature_ids"],
                ascending=[True, False, False, False],
            ).groupby("dataset_accession", as_index=False).head(1)
        )
    else:
        best = pairs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "all_pair_diagnostics.tsv", sep="\t", index=False)
    best.to_csv(args.output_dir / "best_pair_per_accession.tsv", sep="\t", index=False)
    report = {
        "schema": "pan_redu_wave_processed_pair_audit_v1",
        "denominators": {
            "quant_tables": len(quant_records),
            "mgf_files": len(mgf_records),
            "candidate_pairings": len(pairs),
            "accessions_evaluated": int(pairs["dataset_accession"].nunique()) if len(pairs) else 0,
            "accessions_with_schema_qualified_pair": int(best["schema_qualified"].sum()) if len(best) else 0,
        },
        "leakage_status": "schema_and_identifier_join_only_no_biological_effects_opened",
        "claim_boundary": (
            "A schema-qualified pair still requires sample-design, blank/QC, preprocessing, "
            "biological-unit, and spectrum-quality adjudication before response estimation."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
