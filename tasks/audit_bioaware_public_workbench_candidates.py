#!/usr/bin/env python
"""Fail-closed readiness audit for public BioAware confirmation cohorts.

This script intentionally stops before raw-data download or DreaMS encoding.  A
study is eligible only when the deposited metadata supports all of the pieces
needed for a zero-tune, structure-resolved negative-MS/MS retrieval benchmark.
Study-level statements such as "identified with authentic standards" are not
silently promoted to per-row structure truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


STUDIES = ("ST001264", "ST003550", "ST000923")
NEGATIVE = "NEGATIVE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_concatenated_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        value, position = decoder.raw_decode(text, position)
        if not isinstance(value, dict):
            raise TypeError(f"expected object in {path}, got {type(value).__name__}")
        objects.append(value)
    return objects


def file_inventory(payload: dict[str, Any]) -> dict[str, Any]:
    archive_rows = payload.get("compressed_file_content", {})
    raw_files = 0
    raw_bytes = 0
    all_bytes = 0
    for rows in archive_rows.values():
        for row in rows:
            size = int(row.get("size", 0) or 0)
            all_bytes += size
            if str(row.get("name", "")).lower().endswith(".raw"):
                raw_files += 1
                raw_bytes += size
    return {
        "top_level_files": payload.get("files", []),
        "raw_files": raw_files,
        "raw_bytes": raw_bytes,
        "archive_uncompressed_bytes": all_bytes,
    }


def metabolite_rows(obj: dict[str, Any]) -> list[dict[str, Any]]:
    rows = obj.get("MS_METABOLITE_DATA", {}).get("Metabolites", [])
    return rows if isinstance(rows, list) else []


def field_count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(bool(str(row.get(field, "")).strip()) for row in rows)


def confidence_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields = sorted({str(key) for row in rows for key in row})
    pattern = re.compile(r"confidence|level|reliab|identification|annotation", re.I)
    return [field for field in fields if pattern.search(field)]


def distinct_hmdb(rows: list[dict[str, Any]]) -> set[str]:
    answer: set[str] = set()
    for row in rows:
        category = str(row.get("Category", "")).strip()
        if re.fullmatch(r"HMDB\d+", category):
            answer.add(category)
    return answer


def audit_study(study: str, source: Path) -> dict[str, Any]:
    paths = {
        label: source / f"{study}__{label}__json"
        for label in ("summary", "analysis", "mwtab", "files")
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        return {"study_id": study, "status": "missing_metadata", "missing": missing}

    summary = read_json(paths["summary"])
    analyses = read_json(paths["analysis"])
    objects = read_concatenated_json(paths["mwtab"])
    inventory = file_inventory(read_json(paths["files"]))
    negative_objects = [
        obj for obj in objects if str(obj.get("MS", {}).get("ION_MODE", "")).upper() == NEGATIVE
    ]
    negative_ids = [
        str(obj.get("METABOLOMICS WORKBENCH", {}).get("ANALYSIS_ID", ""))
        for obj in negative_objects
    ]
    negative_rows = [row for obj in negative_objects for row in metabolite_rows(obj)]
    unique_names = {str(row.get("Metabolite", "")).strip() for row in negative_rows if row.get("Metabolite")}
    hmdb = distinct_hmdb(negative_rows)
    structure_counts = {
        "inchi_key_rows": field_count(negative_rows, "inchi_key"),
        "pubchem_id_rows": field_count(negative_rows, "pubchem_id"),
        "kegg_id_rows": field_count(negative_rows, "kegg_id"),
        "unique_hmdb_ids": len(hmdb),
    }
    confidence = confidence_fields(negative_rows)
    metadata_text = " ".join(
        json.dumps(obj.get(section, {}), ensure_ascii=False)
        for obj in negative_objects
        for section in ("ANALYSIS", "MS", "SAMPLEPREP")
    )
    explicit_dda = bool(re.search(r"data[- ]?dependent|\bDDA\b|Top\s*\d+", metadata_text, re.I))
    explicit_ms2 = bool(re.search(r"MS/MS|ddMS2|fragment", metadata_text, re.I))

    result: dict[str, Any] = {
        "study_id": study,
        "title": summary.get("study_title"),
        "species": summary.get("species"),
        "negative_analysis_ids": negative_ids,
        "negative_metabolite_rows": len(negative_rows),
        "negative_unique_names": len(unique_names),
        "structure_counts": structure_counts,
        "row_confidence_fields": confidence,
        "explicit_negative_dda": explicit_dda,
        "explicit_negative_ms2": explicit_ms2,
        "files": inventory,
        "analysis_metadata": analyses,
        "provenance": {label: sha256(path) for label, path in paths.items()},
    }

    if study == "ST001264":
        results_path = source / "ST001264_AN002101_Results.txt"
        if not results_path.exists():
            result["status"] = "blocked_missing_negative_results"
            result["pass"] = False
            return result
        with results_path.open("r", encoding="utf-8-sig") as handle:
            first = handle.readline().rstrip("\r\n").split("\t")
            handle.readline()
            data_rows = sum(1 for line in handle if line.strip())
        result["negative_results"] = {
            "rows": data_rows,
            "header_columns": first,
            "structure_or_annotation_column_present": any(
                re.search(r"name|inchi|smiles|formula|annotation|confidence|level", column, re.I)
                for column in first
            ),
            "sha256": sha256(results_path),
        }
        result["blocking_reasons"] = [
            "DDA and MS/MS are explicit, but the deposited negative result table contains only m/z_RT and sample intensities.",
            "The metadata allows either m/z-RT or MS/MS matching, and no per-row Level-1 confidence field is deposited.",
            "A structure-resolved truth manifest cannot be reconstructed without the authors' annotation export.",
        ]
    elif study == "ST003550":
        polarity_counts: dict[str, int] = {}
        for row in negative_rows:
            polarity = str(row.get("Polarity", "missing"))
            polarity_counts[polarity] = polarity_counts.get(polarity, 0) + 1
        result["deposited_row_polarity_counts"] = polarity_counts
        result["blocking_reasons"] = [
            "The two mwTab analysis blocks repeat the same isotopologue panel rather than supplying independent structure identities.",
            f"Only {len(hmdb)} unique HMDB identifiers are present, below the preregistered external-cohort identity requirement.",
            "No per-row Level-1 confidence field is deposited; authentic-standard language is limited to selected targets.",
        ]
    elif study == "ST000923":
        result["blocking_reasons"] = [
            "The study-level statement reports authentic-standard confirmation, but negative rows contain no InChIKey, PubChem, KEGG, or per-row confidence field.",
            "The deposited acquisition metadata does not explicitly document DDA or ddMS2 for the negative analyses.",
            "The raw archive is too large to justify download before row-resolved truth and MS2 availability are established.",
        ]
    else:  # pragma: no cover
        raise AssertionError(study)

    gates = {
        "negative_analysis_present": bool(negative_objects),
        "explicit_negative_ms2": explicit_ms2,
        "row_resolved_structure_truth": (
            structure_counts["inchi_key_rows"] > 0
            or structure_counts["pubchem_id_rows"] > 0
            or structure_counts["unique_hmdb_ids"] >= 30
        ),
        "row_resolved_confidence": bool(confidence),
        "at_least_30_truth_identities": (
            structure_counts["inchi_key_rows"] >= 30
            or structure_counts["pubchem_id_rows"] >= 30
            or structure_counts["unique_hmdb_ids"] >= 30
        ),
        "raw_data_present": inventory["raw_files"] > 0,
    }
    if study == "ST001264":
        gates["annotation_export_present"] = result["negative_results"][
            "structure_or_annotation_column_present"
        ]
    result["gates"] = gates
    result["pass"] = all(gates.values())
    result["status"] = "eligible" if result["pass"] else "blocked_before_raw_download"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/reference/bioaware_public_cohort_probe_20260901"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/bioaware_public_workbench_candidate_audit_v1/report.json"),
    )
    args = parser.parse_args()

    reports = [audit_study(study, args.source) for study in STUDIES]
    eligible = [report["study_id"] for report in reports if report.get("pass")]
    payload = {
        "status": "bioaware_public_workbench_candidate_audit_complete",
        "formal": True,
        "contract": {
            "required": [
                "negative biological MS/MS",
                "per-row structure-resolved truth",
                "per-row confidence sufficient for Level-1 evaluation",
                "at least 30 truth identities before candidate-window filtering",
                "raw data present",
            ],
            "study_level_authentic_standard_statement_is_not_row_truth": True,
            "large_raw_archives_downloaded_only_after_metadata_pass": True,
        },
        "studies": reports,
        "eligible_studies": eligible,
        "decision": (
            "Proceed to raw download and frozen BioAware evaluation."
            if eligible
            else "No audited study is currently eligible; request the missing row-resolved annotation/MS2 export or find another cohort."
        ),
        "claim_limit": "Metadata readiness audit only; no DreaMS or BioAware performance is measured.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
