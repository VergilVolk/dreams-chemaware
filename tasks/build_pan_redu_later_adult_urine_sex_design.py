#!/usr/bin/env python3
"""Freeze the score-blind later-adult urine female-minus-male design.

This builder reads only the frozen Pan-ReDU metadata snapshot and three
phenotype-blind quantification manifests.  It never reads an abundance matrix,
an estimated effect, or a transport score.  Exact input hashes and denominators
are part of the contract so a silently replaced snapshot cannot be admitted.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


AXIS_ID = "human_urine_later_adulthood_female_minus_male"
GLASGOW_MREC_SOURCE = "source:glasgow_mrec_06_mre00_106"
GLASGOW_NGS_SOURCE = "source:glasgow_rec_14_lo_1887"
LATER_ADULTHOOD = "Later Adulthood (>65 yrs)"
HUMAN_TAXON = "9606|Homo sapiens"
URINE = "urine"
POSITIVE = "electrospray ionization (positive)"
HILIC = "normal phase (HILIC)"
MISSING_TOKENS = {"", "missing value", "na", "n/a", "nan", "none", "null"}

DEFAULT_SNAPSHOT = Path("data/reference/pan_redu_snapshot_20261014.tsv")
DEFAULT_SNAPSHOT_SHA256 = "6cda7e250ba645cace8b5e8cafd75a53ae13f617014673ecab8dbe973a2073eb"


@dataclass(frozen=True)
class PanelSpec:
    accession: str
    panel_id: str
    source_family: str
    manifest_path: Path
    manifest_sha256: str
    expected_panel_files: int
    expected_selected_units: int
    expected_female: int
    expected_male: int
    analysis_role: str = "primary_source_view"


DEFAULT_PANEL_SPECS = (
    PanelSpec(
        accession="MSV000083538",
        panel_id="498de29a8ab1f968",
        source_family=GLASGOW_MREC_SOURCE,
        manifest_path=Path(
            "data/validation/pan_redu_high_value_raw_panel_jobs_20261014/"
            "phenotype_blind_panels/498de29a8ab1f968/quantification_manifest.tsv"
        ),
        manifest_sha256="5c6b409ed5518d264389468d17e076e749d65c951657c926dd5e6777fe3ebbd3",
        expected_panel_files=22,
        expected_selected_units=17,
        expected_female=9,
        expected_male=8,
    ),
    PanelSpec(
        accession="MSV000083540",
        panel_id="d2c0b701d29c535d",
        # The MS2LDA+ paper says this is a different subset of the same
        # MREC 06/MRE00/106 cohort as the earlier antihypertensive study.
        source_family=GLASGOW_MREC_SOURCE,
        manifest_path=Path(
            "data/validation/pan_redu_high_value_raw_panel_jobs_20261014/"
            "phenotype_blind_panels/d2c0b701d29c535d/quantification_manifest.tsv"
        ),
        manifest_sha256="cf195f75bd4d3a40a1e1704a06b675ae220fa63e22a920b6250e003d7955b841",
        expected_panel_files=26,
        expected_selected_units=17,
        expected_female=11,
        expected_male=6,
        analysis_role="same_cohort_sensitivity_view",
    ),
    PanelSpec(
        accession="MSV000088255",
        panel_id="18d984665bd41c39",
        source_family=GLASGOW_NGS_SOURCE,
        manifest_path=Path(
            "data/validation/pan_redu_high_value_raw_panel_jobs_20261014/"
            "phenotype_blind_panels/18d984665bd41c39/quantification_manifest.tsv"
        ),
        manifest_sha256="2b072c5ccaf4b62980a6aa84d647c52ce75a17fee4d9d711974a3403f3b1f5b6",
        expected_panel_files=100,
        expected_selected_units=58,
        expected_female=30,
        expected_male=28,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed.lower() != expected.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {label}: expected {expected.lower()}, observed {observed.lower()}"
        )
    return observed.lower()


def is_missing(value: object) -> bool:
    return str(value).strip().lower() in MISSING_TOKENS


def _read_quantification_manifests(
    specs: Iterable[PanelSpec],
) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, str]]:
    by_file: dict[tuple[str, str], dict[str, str]] = {}
    hashes: dict[str, str] = {}
    required = {"sample_id", "biological_unit_id", "dataset_accession", "filename"}
    for spec in specs:
        hashes[spec.accession] = require_sha256(
            spec.manifest_path, spec.manifest_sha256, f"{spec.accession} quantification manifest"
        )
        table = pd.read_csv(spec.manifest_path, sep="\t", dtype=str, keep_default_na=False)
        missing_columns = sorted(required - set(table.columns))
        if missing_columns:
            raise ValueError(f"{spec.accession} manifest lacks columns: {missing_columns}")
        if len(table) != spec.expected_panel_files:
            raise ValueError(
                f"{spec.accession} panel file denominator changed: "
                f"expected {spec.expected_panel_files}, observed {len(table)}"
            )
        if set(table["dataset_accession"]) != {spec.accession}:
            raise ValueError(f"{spec.accession} manifest contains another accession")
        for column in ("sample_id", "filename"):
            if table[column].eq("").any() or table[column].duplicated().any():
                raise ValueError(f"{spec.accession} manifest has blank or duplicate {column}")
        if table["biological_unit_id"].eq("").any():
            raise ValueError(f"{spec.accession} manifest has blank biological_unit_id")
        for row in table.to_dict(orient="records"):
            key = (spec.accession, str(row["filename"]))
            if key in by_file:
                raise ValueError(f"duplicate accession/filename across manifests: {key}")
            row = {str(key_): str(value) for key_, value in row.items()}
            row["panel_id"] = spec.panel_id
            row["source_family"] = spec.source_family
            row["analysis_role"] = spec.analysis_role
            by_file[key] = row
    return by_file, hashes


def _stream_snapshot_join(
    snapshot_path: Path,
    manifest_by_file: dict[tuple[str, str], dict[str, str]],
) -> pd.DataFrame:
    required = {
        "ATTRIBUTE_DatasetAccession", "filename", "NCBITaxonomy",
        "IonizationSourceAndPolarity", "ChromatographyAndPhase",
        "SubjectIdentifierAsRecorded", "AgeInYears", "BiologicalSex",
        "UBERONBodyPartName", "SmokingStatus", "UniqueSubjectID", "LifeStage",
    }
    matched: dict[tuple[str, str], dict[str, object]] = {}
    relevant_accessions = {key[0] for key in manifest_by_file}
    with snapshot_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing_columns = sorted(required - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"Pan-ReDU snapshot lacks columns: {missing_columns}")
        for row in reader:
            accession = str(row["ATTRIBUTE_DatasetAccession"])
            if accession not in relevant_accessions:
                continue
            key = (accession, str(row["filename"]))
            if key not in manifest_by_file:
                continue
            if key in matched:
                raise ValueError(f"snapshot contains duplicate accession/filename row: {key}")
            quant = manifest_by_file[key]
            try:
                age = float(row["AgeInYears"])
            except (TypeError, ValueError) as error:
                age = np.nan
                if row["LifeStage"] == LATER_ADULTHOOD:
                    raise ValueError(f"selected row has invalid age: {key}") from error
            matched[key] = {
                "axis_id": AXIS_ID,
                "panel_id": quant["panel_id"],
                "source_family": quant["source_family"],
                "analysis_role": quant["analysis_role"],
                "dataset_accession": accession,
                "sample_id": quant["sample_id"],
                "biological_unit_id": quant["biological_unit_id"],
                "analysis_unit_id": f"{quant['panel_id']}:{quant['biological_unit_id']}",
                "filename": quant["filename"],
                "remote_descriptor": quant.get("remote_descriptor", ""),
                "local_path": quant.get("local_path", ""),
                "reported_unique_subject_id": str(row["UniqueSubjectID"]),
                "subject_identifier_as_recorded": str(row["SubjectIdentifierAsRecorded"]),
                "age_years": age,
                "biological_sex": str(row["BiologicalSex"]).strip().lower(),
                "level": str(row["BiologicalSex"]).strip().lower(),
                "female_indicator": int(str(row["BiologicalSex"]).strip().lower() == "female"),
                "life_stage": str(row["LifeStage"]),
                "body_part": str(row["UBERONBodyPartName"]),
                "taxonomy": str(row["NCBITaxonomy"]),
                "polarity": str(row["IonizationSourceAndPolarity"]),
                "chromatography": str(row["ChromatographyAndPhase"]),
                "smoking_status": "" if is_missing(row["SmokingStatus"]) else str(row["SmokingStatus"]),
            }
    missing_files = sorted(set(manifest_by_file) - set(matched))
    if missing_files:
        raise ValueError(f"quantification files absent from frozen snapshot: {missing_files[:5]}")
    return pd.DataFrame(matched.values())


def _validate_and_select(joined: pd.DataFrame, specs: Iterable[PanelSpec]) -> pd.DataFrame:
    selected = joined.loc[joined["life_stage"].eq(LATER_ADULTHOOD)].copy()
    categorical_contract = {
        "taxonomy": HUMAN_TAXON,
        "body_part": URINE,
        "polarity": POSITIVE,
        "chromatography": HILIC,
    }
    for column, expected in categorical_contract.items():
        observed = set(selected[column])
        if observed != {expected}:
            raise ValueError(f"selected {column} violates contract: {sorted(observed)}")
    if set(selected["biological_sex"]) != {"female", "male"}:
        raise ValueError("selected rows must contain only female and male")
    if not np.isfinite(selected["age_years"].to_numpy(float)).all():
        raise ValueError("age must be complete for all selected rows")
    if not (selected["age_years"].to_numpy(float) > 65.0).all():
        raise ValueError("later-adulthood selection contains age <= 65")
    if selected["smoking_status"].ne("").any():
        raise ValueError("SmokingStatus availability changed; adjudicate before using it")
    if selected["sample_id"].duplicated().any():
        raise ValueError("a selected sample_id occurs more than once")

    consistency_columns = [
        "dataset_accession", "source_family", "reported_unique_subject_id",
        "subject_identifier_as_recorded", "age_years", "biological_sex", "life_stage",
    ]
    inconsistent = []
    for unit, block in selected.groupby("analysis_unit_id", sort=False):
        if any(block[column].nunique(dropna=False) != 1 for column in consistency_columns):
            inconsistent.append(str(unit))
    if inconsistent:
        raise ValueError(f"repeated files disagree on unit metadata: {inconsistent[:5]}")

    per_unit_files = selected.groupby("analysis_unit_id")["sample_id"].nunique()
    if not per_unit_files.eq(1).all():
        raise ValueError("frozen 92-person contract requires exactly one file per participant")
    subject_to_units = selected.groupby(
        ["dataset_accession", "reported_unique_subject_id"], dropna=False
    )["analysis_unit_id"].nunique()
    if (subject_to_units > 1).any():
        raise ValueError("one reported participant maps to multiple biological units")

    for spec in specs:
        block = selected.loc[selected["dataset_accession"].eq(spec.accession)]
        counts = block["biological_sex"].value_counts().to_dict()
        observed = (block["analysis_unit_id"].nunique(), counts.get("female", 0), counts.get("male", 0))
        expected = (spec.expected_selected_units, spec.expected_female, spec.expected_male)
        if observed != expected:
            raise ValueError(
                f"{spec.accession} selected denominator changed: expected {expected}, observed {observed}"
            )
    total = (
        selected["analysis_unit_id"].nunique(),
        int(selected["biological_sex"].eq("female").sum()),
        int(selected["biological_sex"].eq("male").sum()),
        selected["source_family"].nunique(),
    )
    if total != (92, 50, 42, 2):
        raise ValueError(f"global 92-person denominator changed: observed {total}")
    primary = selected.loc[selected["analysis_role"].eq("primary_source_view")]
    primary_total = (
        primary["analysis_unit_id"].nunique(),
        int(primary["biological_sex"].eq("female").sum()),
        int(primary["biological_sex"].eq("male").sum()),
        primary["source_family"].nunique(),
    )
    if primary_total != (75, 39, 36, 2):
        raise ValueError(f"primary two-cohort denominator changed: observed {primary_total}")
    sensitivity = selected.loc[
        selected["analysis_role"].eq("same_cohort_sensitivity_view")
    ]
    if sensitivity["panel_id"].nunique() != 1:
        raise ValueError("exactly one same-cohort sensitivity panel is required")
    return selected.sort_values(["dataset_accession", "sample_id"], kind="stable").reset_index(drop=True)


def build_design(
    snapshot_path: Path,
    snapshot_sha256: str,
    specs: Iterable[PanelSpec],
    output_dir: Path,
) -> dict[str, object]:
    specs = tuple(specs)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if len({spec.accession for spec in specs}) != len(specs):
        raise ValueError("panel accessions must be unique")
    snapshot_hash = require_sha256(snapshot_path, snapshot_sha256, "frozen Pan-ReDU snapshot")
    manifests, manifest_hashes = _read_quantification_manifests(specs)
    joined = _stream_snapshot_join(snapshot_path, manifests)
    selected = _validate_and_select(joined, specs)
    primary = selected.loc[selected["analysis_role"].eq("primary_source_view")].copy()

    output_dir.mkdir(parents=True)
    design_path = output_dir / "design_table.tsv"
    selected.to_csv(design_path, sep="\t", index=False, lineterminator="\n")
    sources = []
    for spec in specs:
        block = selected.loc[selected["dataset_accession"].eq(spec.accession)]
        sources.append({
            "dataset_accession": spec.accession,
            "panel_id": spec.panel_id,
            "source_family": spec.source_family,
            "analysis_role": spec.analysis_role,
            "panel_files": spec.expected_panel_files,
            "selected_participants": int(block["analysis_unit_id"].nunique()),
            "female": int(block["biological_sex"].eq("female").sum()),
            "male": int(block["biological_sex"].eq("male").sum()),
            "age_min": float(block["age_years"].min()),
            "age_max": float(block["age_years"].max()),
        })
    report: dict[str, object] = {
        "schema": "pan_redu_later_adult_urine_sex_pretruth_design_v2_source_collapsed",
        "status": "development_pretruth_design_frozen_no_abundance_opened",
        "axis": {
            "axis_id": AXIS_ID,
            "contrast_field": "BiologicalSex",
            "level_a": "male",
            "level_b": "female",
            "effect_direction": "female_minus_male",
            "stratum": LATER_ADULTHOOD,
            "numeric_covariates": ["age_years"],
            "unavailable_covariates": ["SmokingStatus"],
        },
        "denominators": {
            "source_families": int(selected["source_family"].nunique()),
            "panel_files_before_stratum_selection": len(manifests),
            "selected_files": len(selected),
            "selected_participants": int(selected["analysis_unit_id"].nunique()),
            "female": int(selected["biological_sex"].eq("female").sum()),
            "male": int(selected["biological_sex"].eq("male").sum()),
            "age_observed": int(selected["age_years"].notna().sum()),
            "smoking_status_observed": int(selected["smoking_status"].ne("").sum()),
        },
        "primary_evaluation_subset": {
            "selection": "analysis_role == primary_source_view",
            "source_families": int(primary["source_family"].nunique()),
            "participants": int(primary["analysis_unit_id"].nunique()),
            "female": int(primary["biological_sex"].eq("female").sum()),
            "male": int(primary["biological_sex"].eq("male").sum()),
            "panels": sorted(primary["panel_id"].unique().tolist()),
        },
        "sources": sources,
        "unit_integrity": {
            "one_file_per_participant": True,
            "repeated_file_covariates_consistent": True,
            "reported_subject_to_biological_unit_one_to_one_within_source": True,
        },
        "source_provenance_adjudication": {
            "MSV000083538_and_MSV000083540": {
                "decision": "same_parent_cohort_not_independent_sources",
                "evidence": (
                    "MS2LDA+ reports MSV000083538 as a different subset of the same "
                    "MREC 06/MRE00/106 Glasgow clinical cohort used in the earlier "
                    "antihypertensive study represented by MSV000083540"
                ),
                "primary_panel": "498de29a8ab1f968",
                "sensitivity_panel": "d2c0b701d29c535d",
                "article_urls": [
                    "https://doi.org/10.1021/acs.analchem.7b01391",
                    "https://doi.org/10.1007/s11306-016-1064-z",
                ],
            },
            "MSV000088255": {
                "decision": "separate_parent_cohort",
                "evidence": (
                    "Next Generation Sequencing and Metabolomic Approaches in "
                    "Stratification of Resistant Hypertension, REC 14/LO/1887"
                ),
                "article_urls": [
                    "https://www.hra.nhs.uk/planning-and-improving-research/"
                    "application-summaries/research-summaries/"
                    "ngsmetabolomic-approaches-in-stratification-of-resistant-hypertension/"
                ],
            },
        },
        "input_sha256": {
            "pan_redu_snapshot": snapshot_hash,
            "quantification_manifests_by_accession": manifest_hashes,
        },
        "output_sha256": {"design_table": sha256(design_path)},
        "leakage_status": (
            "metadata and phenotype-blind quantification manifests only; no abundance, effect, "
            "retrieval, transport score or cross-study truth read"
        ),
        "claim_boundary": [
            "This is a development/pretruth adjustment design, not biological gold.",
            "The female-minus-male coefficient order is frozen; a shared cross-source response is not assumed.",
            "SmokingStatus is missing for all 92 participants and must not be imputed or silently adjusted.",
            "MSV000083539 is the same-participant negative-polarity acquisition and is not an independent source.",
            "MSV000083538 and MSV000083540 are different subsets of one parent cohort; only MSV000083538 enters the primary two-cohort evaluation and MSV000083540 is sensitivity-only.",
            "The superseded v1 design counted accessions rather than adjudicated parent cohorts and must not be used.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--manifest-msv000083538", type=Path, default=DEFAULT_PANEL_SPECS[0].manifest_path)
    parser.add_argument("--manifest-msv000083540", type=Path, default=DEFAULT_PANEL_SPECS[1].manifest_path)
    parser.add_argument("--manifest-msv000088255", type=Path, default=DEFAULT_PANEL_SPECS[2].manifest_path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = (
        args.manifest_msv000083538,
        args.manifest_msv000083540,
        args.manifest_msv000088255,
    )
    specs = tuple(replace(spec, manifest_path=path) for spec, path in zip(DEFAULT_PANEL_SPECS, paths))
    report = build_design(
        snapshot_path=args.snapshot,
        snapshot_sha256=DEFAULT_SNAPSHOT_SHA256,
        specs=specs,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
