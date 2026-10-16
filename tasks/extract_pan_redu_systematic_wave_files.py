#!/usr/bin/env python3
"""Extract file/sample records for a frozen Pan-ReDU compute wave."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from build_pan_redu_contrast_universe import PANEL_FIELDS, clean, is_biological, safe_int

WAVE_PANEL_COLUMNS = (
    "dataset_accession", "taxonomy", "body_part", "instrument", "polarity", "chromatography",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pan-redu", required=True, type=Path)
    parser.add_argument("--wave", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    wave = pd.read_csv(args.wave, sep="\t", dtype=str, keep_default_na=False)
    required = set(WAVE_PANEL_COLUMNS) | {
        "contrast_id", "contrast_field", "level_a", "level_b",
        "comparison_stratum",
    }
    missing = required - set(wave)
    if missing:
        raise ValueError(f"wave lacks columns: {sorted(missing)}")
    source_family_fallback = "preliminary_source_family" not in wave.columns
    if source_family_fallback:
        # An accession is an acquisition container, not proof of an
        # independent biological source.  This fallback permits file
        # extraction from the broader registry but is explicitly provisional.
        wave["preliminary_source_family"] = wave["dataset_accession"].map(
            lambda value: f"accession_only_unadjudicated:{value}"
        )

    lookup: dict[tuple[tuple[str, ...], str], list[dict[str, str]]] = defaultdict(list)
    for row in wave.to_dict("records"):
        panel = tuple(row[column] for column in WAVE_PANEL_COLUMNS)
        lookup[(panel, row["contrast_field"])].append(row)

    sample_rows: list[dict[str, object]] = []
    unique_files: dict[tuple[str, str], dict[str, object]] = {}
    scanned = biological = 0
    with args.pan_redu.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            scanned += 1
            if not is_biological(raw):
                continue
            biological += 1
            panel = tuple(clean(raw.get(column)) for column in PANEL_FIELDS)
            for contrast_field in ("Diet", "HealthStatus", "DOIDCommonName", "LifeStage", "BiologicalSex"):
                candidates = lookup.get((panel, contrast_field), ())
                if not candidates:
                    continue
                level = clean(raw.get(contrast_field))
                for contrast in candidates:
                    if level not in {contrast["level_a"], contrast["level_b"]}:
                        continue
                    subject_recorded = clean(raw.get("SubjectIdentifierAsRecorded"))
                    unique_subject = clean(raw.get("UniqueSubjectID"))
                    filename = clean(raw.get("filename"))
                    relative = filename[2:] if filename.startswith("f.") else filename
                    remote_descriptor = f"f.{contrast['dataset_accession']}/{relative}"
                    file_key = (contrast["dataset_accession"], filename)
                    common = {
                        "dataset_accession": contrast["dataset_accession"],
                        "preliminary_source_family": contrast["preliminary_source_family"],
                        "filename": filename,
                        "remote_descriptor": remote_descriptor,
                        "massive_download_url": (
                            "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?"
                            + urllib.parse.urlencode({"file": remote_descriptor})
                        ),
                        "usi": clean(raw.get("USI")),
                        "reported_unique_subject_id": unique_subject,
                        "subject_identifier_as_recorded": subject_recorded,
                        "sample_collection_date_time": clean(raw.get("SampleCollectionDateandTime")),
                        "age_in_years": clean(raw.get("AgeInYears")),
                        "biological_sex": clean(raw.get("BiologicalSex")),
                        "health_status": clean(raw.get("HealthStatus")),
                        "diet": clean(raw.get("Diet")),
                        "life_stage": clean(raw.get("LifeStage")),
                        "body_part": clean(raw.get("UBERONBodyPartName")),
                        "taxonomy": clean(raw.get("NCBITaxonomy")),
                        "instrument": clean(raw.get("MassSpectrometer")),
                        "polarity": clean(raw.get("IonizationSourceAndPolarity")),
                        "chromatography": clean(raw.get("ChromatographyAndPhase")),
                        "ms2_spectra_count": safe_int(raw.get("MS2spectra_count")),
                        "data_source": clean(raw.get("DataSource")),
                    }
                    sample_rows.append({
                        "contrast_id": contrast["contrast_id"],
                        "comparison_stratum": contrast["comparison_stratum"],
                        "contrast_field": contrast_field,
                        "level": level,
                        **common,
                        "unit_status": "reported_identifier_not_yet_independence_adjudicated",
                    })
                    unique_files.setdefault(file_key, common)

    samples = pd.DataFrame(sample_rows)
    files = pd.DataFrame(unique_files.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.output_dir / "contrast_sample_manifest.tsv", sep="\t", index=False)
    files.to_csv(args.output_dir / "unique_file_manifest.tsv", sep="\t", index=False)
    extension_counts = Counter(Path(value).suffix.lower() or "none" for value in files["filename"])
    report = {
        "schema": "pan_redu_systematic_wave_file_manifest_v1",
        "denominators": {
            "pan_redu_rows_scanned": scanned,
            "biological_rows_scanned": biological,
            "contrast_file_appearances": len(samples),
            "unique_repository_files": len(files),
            "dataset_accessions": int(files["dataset_accession"].nunique()),
            "preliminary_source_families": int(files["preliminary_source_family"].nunique()),
            "files_with_reported_ms2": int((files["ms2_spectra_count"] > 0).sum()),
            "reported_ms2_spectra": int(files["ms2_spectra_count"].sum()),
            "file_extensions": dict(extension_counts.most_common()),
        },
        "leakage_status": "metadata_only_no_abundance_or_cross_study_truth_read",
        "source_family_status": (
            "accession_only_unadjudicated_fallback" if source_family_fallback
            else "provided_preliminary_source_family"
        ),
        "claim_boundary": [
            "Rows are acquisition/download instructions, not independent biological samples.",
            "True subject, visit, tissue-site, aliquot, and technical-replicate hierarchy remains to be reconstructed.",
            "Files lacking reported MS2 may still be quantitative MS1 files but cannot supply spectral evidence alone.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["denominators"], indent=2))


if __name__ == "__main__":
    main()
