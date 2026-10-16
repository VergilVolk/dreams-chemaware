#!/usr/bin/env python3
"""Find sample-file and biological-unit mappings in deposited design tables."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


SUBJECT_RE = re.compile(r"subject|participant|patient|individual|animal|mouse|rat|host|donor", re.I)
UNIT_ID_RE = re.compile(
    r"(?:subject|participant|patient|individual|animal|mouse|rat|host|donor)"
    r".*(?:id(?:entifier)?|number|_no)$|^Subject$",
    re.I,
)
DESIGN_RE = re.compile(r"group|condition|treatment|diet|disease|health|sex|gender|age|genotype|time|visit", re.I)
STANDARDIZED_SUBJECT_RE = re.compile(
    r"^(?:Study_)?(?:SubjectIdentifierAsRecorded|UniqueSubjectID)$|"
    r"^Study_UniqueSubjectID$|^ATTRIBUTE_Subject_|^Subject_(?:Age|Country|Health|Human)",
    re.I,
)


def normalize_cell(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    name = text.rsplit("/", 1)[-1]
    return Path(name).stem.casefold()


def read_tables(path: Path) -> dict[str, pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return {
            str(name): frame
            for name, frame in pd.read_excel(path, sheet_name=None, dtype=str).items()
        }
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {"json": pd.json_normalize(payload)}
    try:
        frame = pd.read_csv(path, sep=None, engine="python", dtype=str, on_bad_lines="skip")
    except UnicodeDecodeError:
        frame = pd.read_csv(
            path, sep=None, engine="python", dtype=str, on_bad_lines="skip", encoding="latin-1"
        )
    return {"table": frame}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checksums", required=True, type=Path)
    parser.add_argument("--file-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    assets = pd.read_csv(args.checksums, sep="\t", dtype=str, keep_default_na=False)
    files = pd.read_csv(args.file_manifest, sep="\t", dtype=str, keep_default_na=False)
    selected_by_accession = {
        accession: {normalize_cell(value) for value in block["filename"]}
        for accession, block in files.groupby("dataset_accession")
    }
    rows = []
    failures = []
    for asset in assets.to_dict("records"):
        path = Path(asset["local_path"])
        accession = asset["dataset_accession"]
        selected = selected_by_accession.get(accession, set())
        try:
            tables = read_tables(path)
        except Exception as error:
            failures.append({
                "dataset_accession": accession, "local_path": str(path), "error": repr(error)
            })
            continue
        for sheet, frame in tables.items():
            frame = frame.dropna(how="all").dropna(axis=1, how="all")
            if frame.empty:
                continue
            best_column = ""
            best_overlap = 0
            for column in frame.columns:
                values = {normalize_cell(value) for value in frame[column].dropna()}
                overlap = len(values & selected)
                if overlap > best_overlap:
                    best_column, best_overlap = str(column), overlap
            subject_columns = [str(column) for column in frame.columns if SUBJECT_RE.search(str(column))]
            native_subject_columns = [
                column for column in subject_columns if not STANDARDIZED_SUBJECT_RE.search(column)
            ]
            nonstandard_unit_id_columns = [
                str(column) for column in frame.columns
                if UNIT_ID_RE.search(str(column)) and not STANDARDIZED_SUBJECT_RE.search(str(column))
            ]
            design_columns = [str(column) for column in frame.columns if DESIGN_RE.search(str(column))]
            rows.append({
                "dataset_accession": accession,
                "asset_path": asset["filepath"],
                "local_path": str(path),
                "sheet": sheet,
                "n_rows": len(frame),
                "n_columns": len(frame.columns),
                "best_file_column": best_column,
                "selected_file_overlap": best_overlap,
                "selected_files_in_accession": len(selected),
                "selected_file_coverage": best_overlap / len(selected) if selected else 0.0,
                "subject_like_columns": ";".join(subject_columns),
                "nonstandard_subject_like_columns": ";".join(native_subject_columns),
                "nonstandard_unit_id_columns": ";".join(nonstandard_unit_id_columns),
                "design_like_columns": ";".join(design_columns),
                "has_subject_like_column": bool(subject_columns),
                "has_nonstandard_subject_like_column": bool(native_subject_columns),
                "has_nonstandard_unit_id_column": bool(nonstandard_unit_id_columns),
                "has_design_like_column": bool(design_columns),
            })
    diagnostics = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(args.output_dir / "design_table_diagnostics.tsv", sep="\t", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "parse_failures.tsv", sep="\t", index=False)
    best = (
        diagnostics.sort_values(
            ["dataset_accession", "selected_file_overlap", "has_nonstandard_unit_id_column", "has_design_like_column"],
            ascending=[True, False, False, False],
        ).groupby("dataset_accession", as_index=False).head(1)
        if len(diagnostics) else diagnostics
    )
    best.to_csv(args.output_dir / "best_design_table_per_accession.tsv", sep="\t", index=False)
    mapped = best.loc[best["selected_file_overlap"] > 0] if len(best) else best
    report = {
        "schema": "pan_redu_wave_design_table_audit_v1",
        "denominators": {
            "downloaded_assets": int(len(assets)),
            "parsed_tables_or_sheets": int(len(diagnostics)),
            "parse_failures": int(len(failures)),
            "accessions_with_any_selected_file_mapping": int(mapped["dataset_accession"].nunique()) if len(mapped) else 0,
            "accessions_with_mapped_table_and_subject_column": int(
                mapped.loc[mapped["has_subject_like_column"]].dataset_accession.nunique()
            ) if len(mapped) else 0,
            "accessions_with_mapped_nonstandard_unit_id_column": int(
                mapped.loc[mapped["has_nonstandard_unit_id_column"]].dataset_accession.nunique()
            ) if len(mapped) else 0,
        },
        "decision_boundary": (
            "Column-name and filename overlap is readiness evidence only. "
            "The selected table must still be manually adjudicated before defining biological units."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
