#!/usr/bin/env python3
"""Build an outcome-blind universe of public study-level LC-MS/MS contrasts.

This program does not inspect feature abundances or biological outcomes.  It
streams the Pan-ReDU per-file metadata dump and identifies analytical panels in
which at least two biological condition levels have enough reported unit identifiers
and MS2-bearing files to support a within-study contrast.  Repeated condition
pairs across independent accessions are registered as contrast families for
later source-held-out response retrieval.

The output is a qualification registry, not evidence that any contrast is
biologically reproducible or that unresolved chemistry adds information.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable


MISSING = {"", "missing value", "missing", "na", "n/a", "none", "null"}
CONTRAST_FIELDS = (
    "Diet",
    "HealthStatus",
    "DOIDCommonName",
    "LifeStage",
    "BiologicalSex",
)
PANEL_FIELDS = (
    "ATTRIBUTE_DatasetAccession",
    "NCBITaxonomy",
    "UBERONBodyPartName",
    "MassSpectrometer",
    "IonizationSourceAndPolarity",
    "ChromatographyAndPhase",
)
NON_BIOLOGICAL_RE = re.compile(
    r"(?:^|[_\W])(blank|pool(?:ed)?|qc|quality.?control|standard|std|wash|"
    r"solvent|empty|calibration|calibrant)(?:$|[_\W])",
    flags=re.IGNORECASE,
)


def clean(value: str | None) -> str:
    value = (value or "").strip()
    return "" if value.lower() in MISSING else value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_int(value: str | None) -> int:
    try:
        x = float((value or "0").strip())
        return int(x) if math.isfinite(x) and x > 0 else 0
    except ValueError:
        return 0


def is_biological(row: dict[str, str]) -> bool:
    sample_type = clean(row.get("SampleType"))
    sample_subtype = clean(row.get("SampleTypeSub1"))
    filename = clean(row.get("filename"))
    joined = " ".join((sample_type, sample_subtype, Path(filename).name))
    if NON_BIOLOGICAL_RE.search(joined):
        return False
    return sample_type.lower() not in {
        "blank_analysis",
        "qc_analysis",
        "sample preparation blank",
        "instrument blank",
    }


def reported_unit(row: dict[str, str]) -> tuple[int, bool]:
    for column in ("UniqueSubjectID", "SubjectIdentifierAsRecorded"):
        value = clean(row.get(column))
        if value:
            token = f"{column}:{value}".encode("utf-8")
            return int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little"), True
    # The fallback is deliberately explicit: without a subject identifier a
    # file is only a putative biological unit and cannot support a sealed claim.
    token = f"file_fallback:{clean(row.get('filename'))}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little"), False


@dataclass
class LevelStats:
    units: set[int] = field(default_factory=set)
    subject_units: set[int] = field(default_factory=set)
    files: int = 0
    ms2_files: int = 0
    ms2_spectra: int = 0
    def add(self, row: dict[str, str], unit: int, has_subject: bool) -> None:
        self.units.add(unit)
        if has_subject:
            self.subject_units.add(unit)
        self.files += 1
        n_ms2 = safe_int(row.get("MS2spectra_count"))
        if n_ms2 > 0:
            self.ms2_files += 1
            self.ms2_spectra += n_ms2


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b), key=lambda x: x.casefold()))  # type: ignore[return-value]


def build_registry(
    input_path: Path,
    min_units: int,
    min_ms2_files: int,
    require_subject_ids: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    groups: dict[tuple[tuple[str, ...], str, str], LevelStats] = defaultdict(LevelStats)
    total_rows = biological_rows = 0
    accessions: set[str] = set()
    source_counts: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = set(PANEL_FIELDS) | set(CONTRAST_FIELDS) | {
            "filename",
            "SampleType",
            "SampleTypeSub1",
            "UniqueSubjectID",
            "SubjectIdentifierAsRecorded",
            "MS2spectra_count",
            "DataSource",
        }
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(f"Pan-ReDU dump lacks columns: {missing_columns}")

        for row in reader:
            total_rows += 1
            accession = clean(row.get("ATTRIBUTE_DatasetAccession"))
            if not accession or not is_biological(row):
                continue
            biological_rows += 1
            accessions.add(accession)
            source_counts[clean(row.get("DataSource")) or "unknown"] += 1
            panel = tuple(clean(row.get(column)) for column in PANEL_FIELDS)
            unit, has_subject = reported_unit(row)
            for contrast_field in CONTRAST_FIELDS:
                level = clean(row.get(contrast_field))
                if level:
                    groups[(panel, contrast_field, level)].add(row, unit, has_subject)

    by_panel_field: dict[tuple[tuple[str, ...], str], list[tuple[str, LevelStats]]] = defaultdict(list)
    for (panel, contrast_field, level), stats in groups.items():
        if len(stats.units) < min_units or stats.ms2_files < min_ms2_files:
            continue
        if require_subject_ids and len(stats.subject_units) < min_units:
            continue
        by_panel_field[(panel, contrast_field)].append((level, stats))

    rows: list[dict[str, object]] = []
    for (panel, contrast_field), levels in by_panel_field.items():
        if len(levels) < 2:
            continue
        for (level_a, stats_a), (level_b, stats_b) in combinations(sorted(levels), 2):
            pair = canonical_pair(level_a, level_b)
            stats_by_level = {level_a: stats_a, level_b: stats_b}
            pair_a = stats_by_level[pair[0]]
            pair_b = stats_by_level[pair[1]]
            accession, taxonomy, body_part, instrument, polarity, chromatography = panel
            unit_mode = (
                "subject_id"
                if min(len(pair_a.subject_units), len(pair_b.subject_units)) >= min_units
                else "file_fallback"
            )
            rows.append(
                {
                    "panel_id": stable_id(*panel),
                    "contrast_id": stable_id(*panel, contrast_field, pair[0], pair[1]),
                    "dataset_accession": accession,
                    "taxonomy": taxonomy,
                    "body_part": body_part,
                    "instrument": instrument,
                    "polarity": polarity,
                    "chromatography": chromatography,
                    "contrast_field": contrast_field,
                    "level_a": pair[0],
                    "level_b": pair[1],
                    "contrast_family": f"{contrast_field}::{pair[0]}__vs__{pair[1]}",
                    # This is a qualification stratum, not biological truth.
                    # Species and sampled body part are kept fixed; instrument,
                    # polarity and chromatography may vary and are measured
                    # transfer axes rather than silently pooled nuisance.
                    "comparison_stratum": (
                        f"{contrast_field}::{pair[0]}__vs__{pair[1]}"
                        f"::taxon={taxonomy or 'missing'}::body={body_part or 'missing'}"
                    ),
                    "unit_mode": unit_mode,
                    "n_units_a": len(pair_a.units),
                    "n_units_b": len(pair_b.units),
                    "n_subject_ids_a": len(pair_a.subject_units),
                    "n_subject_ids_b": len(pair_b.subject_units),
                    "n_ms2_files_a": pair_a.ms2_files,
                    "n_ms2_files_b": pair_b.ms2_files,
                    "n_ms2_spectra_a": pair_a.ms2_spectra,
                    "n_ms2_spectra_b": pair_b.ms2_spectra,
                    "metadata_role": "outcome_blind_candidate",
                    "abundance_opened": False,
                    "leakage_status": "metadata_only_no_feature_abundance_read",
                }
            )

    family_accessions: dict[str, set[str]] = defaultdict(set)
    stratum_accessions: dict[str, set[str]] = defaultdict(set)
    family_panels: Counter[str] = Counter()
    stratum_panels: Counter[str] = Counter()
    for row in rows:
        family = str(row["contrast_family"])
        stratum = str(row["comparison_stratum"])
        family_accessions[family].add(str(row["dataset_accession"]))
        stratum_accessions[stratum].add(str(row["dataset_accession"]))
        family_panels[family] += 1
        stratum_panels[stratum] += 1

    repeated = [
        {
            "contrast_family": family,
            "n_accessions": len(accessions_),
            "n_panels": family_panels[family],
        }
        for family, accessions_ in family_accessions.items()
        if len(accessions_) >= 2
    ]
    repeated.sort(key=lambda x: (-int(x["n_accessions"]), -int(x["n_panels"]), str(x["contrast_family"])))
    repeated_strata = [
        {
            "comparison_stratum": stratum,
            "n_accessions": len(accessions_),
            "n_panels": stratum_panels[stratum],
        }
        for stratum, accessions_ in stratum_accessions.items()
        if len(accessions_) >= 2
    ]
    repeated_strata.sort(
        key=lambda x: (-int(x["n_accessions"]), -int(x["n_panels"]), str(x["comparison_stratum"]))
    )

    report: dict[str, object] = {
        "schema": "pan_redu_contrast_universe_v1",
        "scientific_role": "outcome_blind_repository_scale_contrast_qualification",
        "input": {
            "path": str(input_path.resolve()),
            "bytes": input_path.stat().st_size,
            "sha256": sha256(input_path),
        },
        "parameters": {
            "contrast_fields": list(CONTRAST_FIELDS),
            "panel_fields": list(PANEL_FIELDS),
            "min_reported_unit_identifiers_per_level": min_units,
            "min_ms2_files_per_level": min_ms2_files,
            "require_subject_ids": require_subject_ids,
        },
        "denominators": {
            "all_metadata_rows": total_rows,
            "biological_rows": biological_rows,
            "dataset_accessions_with_biological_rows": len(accessions),
            "qualified_contrast_panels": len(rows),
            "qualified_contrast_families": len(family_accessions),
            "families_repeated_across_at_least_two_accessions": len(repeated),
            "species_body_strata_repeated_across_at_least_two_accessions": len(repeated_strata),
            "qualified_panels_with_subject_ids": sum(row["unit_mode"] == "subject_id" for row in rows),
            "qualified_panels_using_file_fallback": sum(row["unit_mode"] == "file_fallback" for row in rows),
        },
        "data_source_file_counts": dict(source_counts.most_common()),
        "top_repeated_families": repeated[:100],
        "top_repeated_species_body_strata": repeated_strata[:100],
        "claim_boundary": [
            "Qualification uses metadata and reported MS2 counts only; no feature abundance or response score was opened.",
            "A repeated condition pair is retrieval truth only after design, comparability, and independence review.",
            "ReDU subject fields are reported identifiers, not guaranteed independent persons or animals.",
            "File fallback is not a biological denominator and cannot support a sealed claim.",
            "Repeated measures and true biological units must be reconstructed from each study design before effect estimation.",
            "The registry does not establish unknown-chemistry increment or cross-study reproducibility.",
        ],
    }
    return rows, report


def write_outputs(rows: Iterable[dict[str, object]], report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    registry_path = output_dir / "contrast_registry.tsv"
    if rows:
        with registry_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    else:
        registry_path.write_text("", encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-units", type=int, default=5)
    parser.add_argument("--min-ms2-files", type=int, default=3)
    parser.add_argument("--require-subject-ids", action="store_true")
    args = parser.parse_args()
    rows, report = build_registry(
        input_path=args.input,
        min_units=args.min_units,
        min_ms2_files=args.min_ms2_files,
        require_subject_ids=args.require_subject_ids,
    )
    write_outputs(rows, report, args.output_dir)
    print(json.dumps(report["denominators"], indent=2))


if __name__ == "__main__":
    main()
