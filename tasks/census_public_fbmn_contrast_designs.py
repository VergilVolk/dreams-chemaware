#!/usr/bin/env python3
"""Find reconstructable contrast designs in outcome-blind public FBMN metadata."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path

import pandas as pd
import requests


RESULT = "https://proteomics2.ucsd.edu/ProteoSAFe/DownloadResultFile"
TECHNICAL_TOKENS = {
    "filename", "file_name", "sample_name", "sampleid", "sample_id", "batch",
    "plate", "well", "injection", "replicate", "run_order", "instrument",
    "ionization", "polarity", "chromatography", "blank", "qc",
}


def fetch_metadata(task_id: str) -> tuple[str, pd.DataFrame | None, str]:
    try:
        response = requests.get(
            RESULT,
            params={"task": task_id, "file": "metadata_merged/", "block": "main"},
            timeout=45,
        )
        response.raise_for_status()
        if b"<html" in response.content[:200].lower():
            return task_id, None, "html_error"
        frame = pd.read_csv(io.BytesIO(response.content), sep=None, engine="python", dtype=str)
        return task_id, frame, ""
    except (requests.RequestException, pd.errors.ParserError, UnicodeError) as error:
        return task_id, None, type(error).__name__


def is_technical(column: str) -> bool:
    normalized = column.strip().lower().replace(" ", "_")
    return any(token in normalized for token in TECHNICAL_TOKENS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-census", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minimum-per-level", type=int, default=3)
    parser.add_argument("--maximum-levels", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    census = pd.read_csv(args.task_census, sep="\t", dtype=str, keep_default_na=False)
    truth = {"true", "1"}
    selected = census.loc[
        census.response_substrate_ready.str.lower().isin(truth)
        & census.metadata_readable.str.lower().isin(truth)
    ].copy()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        fetched = list(executor.map(fetch_metadata, selected.task_id))
    design_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    for task_id, frame, error in fetched:
        if frame is None:
            task_rows.append({"task_id": task_id, "status": "UNREADABLE", "error": error})
            continue
        task_rows.append({
            "task_id": task_id, "status": "METADATA_READABLE", "error": "",
            "samples": len(frame), "metadata_columns": len(frame.columns),
        })
        for column in frame.columns:
            if is_technical(column):
                continue
            values = frame[column].fillna("").astype(str).str.strip()
            values = values.loc[values.ne("")]
            counts = values.value_counts()
            if not (2 <= len(counts) <= args.maximum_levels):
                continue
            eligible = counts.loc[counts >= args.minimum_per_level]
            if len(eligible) < 2:
                continue
            design_rows.append({
                "task_id": task_id,
                "metadata_field": column,
                "samples_with_value": int(len(values)),
                "eligible_levels": int(len(eligible)),
                "minimum_level_n": int(eligible.min()),
                "maximum_level_n": int(eligible.max()),
                "levels_and_n": " | ".join(f"{level}::{int(count)}" for level, count in eligible.items()),
                "role": "candidate_design_requires_semantic_and_unit_adjudication",
            })
    tasks = pd.DataFrame(task_rows)
    designs = pd.DataFrame(design_rows)
    args.output_dir.mkdir(parents=True)
    tasks.to_csv(args.output_dir / "task_metadata_status.tsv", sep="\t", index=False)
    designs.to_csv(args.output_dir / "candidate_design_fields.tsv", sep="\t", index=False)
    report = {
        "schema": "public_fbmn_contrast_design_census_v1",
        "status": "outcome_blind_metadata_census_no_effects_or_biological_result",
        "parameters": {
            "minimum_samples_per_level": args.minimum_per_level,
            "maximum_levels_per_field": args.maximum_levels,
            "technical_fields_excluded_by_name": sorted(TECHNICAL_TOKENS),
        },
        "denominators": {
            "metadata_qualified_tasks_requested": int(len(selected)),
            "metadata_tasks_readable": int(tasks.status.eq("METADATA_READABLE").sum()),
            "tasks_with_candidate_design": int(designs.task_id.nunique()) if len(designs) else 0,
            "candidate_design_fields": int(len(designs)),
        },
        "claim_boundary": [
            "Low-cardinality metadata is only a candidate design, not a biological contrast or truth label.",
            "Replicates, batches, repeated measures, controls and intervention semantics require manual adjudication.",
            "No abundance values, feature effects or retrieval outcomes were read.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
