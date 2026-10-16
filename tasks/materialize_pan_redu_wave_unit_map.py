#!/usr/bin/env python3
"""Materialize conservative biological-unit maps from native design columns."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from audit_pan_redu_wave_design_tables import normalize_cell, read_tables


UNIT_PRIORITY = (
    re.compile(r"^subject$", re.I),
    re.compile(r"host_subject_id$", re.I),
    re.compile(r"animal_id$", re.I),
    re.compile(r"animal_number$", re.I),
    re.compile(r"mouse_no$", re.I),
    re.compile(r"subject_id$", re.I),
    re.compile(r"participant.*id$|patient.*id$|individual.*id$|donor.*id$", re.I),
)


def choose_unit_column(columns: list[str]) -> str:
    eligible = [column for column in columns if not column.casefold().endswith("taxid")]
    for pattern in UNIT_PRIORITY:
        matches = [column for column in eligible if pattern.search(column)]
        if matches:
            return sorted(matches, key=lambda value: (len(value), value.casefold()))[0]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-tables", required=True, type=Path)
    parser.add_argument("--contrast-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-units-per-level", type=int, default=5)
    args = parser.parse_args()
    best = pd.read_csv(args.best_tables, sep="\t", dtype=str, keep_default_na=False)
    contrast = pd.read_csv(args.contrast_manifest, sep="\t", dtype=str, keep_default_na=False)
    source_by_accession = (
        contrast[["dataset_accession", "preliminary_source_family"]]
        .drop_duplicates().set_index("dataset_accession")["preliminary_source_family"].to_dict()
    )

    mappings: dict[tuple[str, str], set[str]] = {}
    table_records = []
    for row in best.to_dict("records"):
        candidate_columns = [
            value for value in row.get("nonstandard_unit_id_columns", "").split(";") if value
        ]
        unit_column = choose_unit_column(candidate_columns)
        if not unit_column:
            continue
        tables = read_tables(Path(row["local_path"]))
        table = tables[row["sheet"]]
        file_column = row["best_file_column"]
        if file_column not in table or unit_column not in table:
            continue
        source_family = source_by_accession[row["dataset_accession"]]
        nonmissing = 0
        for file_value, unit_value in zip(table[file_column], table[unit_column]):
            file_key = normalize_cell(file_value)
            unit = str(unit_value or "").strip()
            if not file_key or not unit or unit.casefold() in {"nan", "none", "missing value"}:
                continue
            nonmissing += 1
            mappings.setdefault((source_family, file_key), set()).add(unit)
        table_records.append({
            "dataset_accession": row["dataset_accession"],
            "preliminary_source_family": source_family,
            "asset_path": row["asset_path"],
            "sheet": row["sheet"],
            "file_column": file_column,
            "unit_column": unit_column,
            "nonmissing_mapping_rows": nonmissing,
        })

    output_rows = []
    for row in contrast.to_dict("records"):
        file_key = normalize_cell(row["filename"])
        candidates = mappings.get((row["preliminary_source_family"], file_key), set())
        if len(candidates) == 1:
            raw_unit = next(iter(candidates))
            unit_id = f"{row['preliminary_source_family']}::{raw_unit}"
            status = "native_design_unit_mapped"
        elif len(candidates) > 1:
            raw_unit, unit_id, status = "", "", "conflicting_native_unit_mapping"
        else:
            raw_unit, unit_id, status = "", "", "native_unit_unavailable"
        output_rows.append({
            **row,
            "native_unit_id_raw": raw_unit,
            "biological_unit_id": unit_id,
            "unit_mapping_status": status,
        })
    unit_map = pd.DataFrame(output_rows)
    panel_rows = []
    for contrast_id, block in unit_map.groupby("contrast_id", sort=True):
        mapped = block.loc[block["unit_mapping_status"].eq("native_design_unit_mapped")]
        cross_level = mapped.groupby("biological_unit_id")["level"].nunique()
        units_per_level = mapped.groupby("level")["biological_unit_id"].nunique().to_dict()
        expected_levels = sorted(block["level"].unique())
        ready = (
            len(expected_levels) == 2
            and all(units_per_level.get(level, 0) >= args.min_units_per_level for level in expected_levels)
            and not (cross_level > 1).any()
            and len(mapped) == len(block)
        )
        panel_rows.append({
            "contrast_id": contrast_id,
            "dataset_accession": block["dataset_accession"].iloc[0],
            "preliminary_source_family": block["preliminary_source_family"].iloc[0],
            "comparison_stratum": block["comparison_stratum"].iloc[0],
            "n_files": len(block),
            "n_files_native_unit_mapped": len(mapped),
            "n_biological_units": mapped["biological_unit_id"].nunique(),
            "units_per_level": json.dumps(units_per_level, sort_keys=True),
            "units_crossing_contrast_levels": int((cross_level > 1).sum()),
            "native_unit_map_ready": bool(ready),
            "effect_model_requirement": (
                "cluster_or_mixed_model" if mapped["biological_unit_id"].duplicated().any()
                else "independent_unit_model"
            ),
        })
    panels = pd.DataFrame(panel_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unit_map.to_csv(args.output_dir / "file_to_biological_unit.tsv", sep="\t", index=False)
    pd.DataFrame(table_records).to_csv(args.output_dir / "unit_source_tables.tsv", sep="\t", index=False)
    panels.to_csv(args.output_dir / "panel_readiness.tsv", sep="\t", index=False)
    report = {
        "schema": "pan_redu_wave_native_unit_map_v1",
        "denominators": {
            "contrast_file_appearances": len(unit_map),
            "file_appearances_native_unit_mapped": int(unit_map["unit_mapping_status"].eq("native_design_unit_mapped").sum()),
            "contrast_panels": len(panels),
            "panels_with_complete_native_unit_map": int(panels["native_unit_map_ready"].sum()),
            "source_families_with_native_unit_table": int(pd.DataFrame(table_records)["preliminary_source_family"].nunique()) if table_records else 0,
        },
        "decision_boundary": (
            "Ready means identifier topology only. Covariates, batches, outcome coding, and source independence "
            "must still be frozen before abundance estimation."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
