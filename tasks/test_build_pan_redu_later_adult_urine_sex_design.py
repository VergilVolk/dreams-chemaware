from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from build_pan_redu_later_adult_urine_sex_design import (
    HILIC,
    HUMAN_TAXON,
    LATER_ADULTHOOD,
    POSITIVE,
    URINE,
    PanelSpec,
    build_design,
    sha256,
)


def _write_frozen_inputs(tmp_path: Path) -> tuple[Path, tuple[PanelSpec, ...]]:
    source_contracts = (
        ("MSV000083538", "P538", "F_MREC", "primary_source_view", 17, 9, 8),
        (
            "MSV000083540", "P540", "F_MREC", "same_cohort_sensitivity_view",
            17, 11, 6,
        ),
        ("MSV000088255", "P255", "F_NGS", "primary_source_view", 58, 30, 28),
    )
    snapshot_rows: list[dict[str, object]] = []
    specs: list[PanelSpec] = []
    for accession, panel, family, role, total, female, male in source_contracts:
        manifest_rows = []
        sexes = ["female"] * female + ["male"] * male
        for index, sex in enumerate(sexes):
            filename = f"f.peak/{accession}_{index}.mzML"
            manifest_rows.append({
                "sample_id": f"S{accession[-3:]}_{index}",
                "biological_unit_id": f"U{accession[-3:]}_{index}",
                "dataset_accession": accession,
                "filename": filename,
                "remote_descriptor": f"f.{accession}/peak/{accession}_{index}.mzML",
                "local_path": f"raw/{accession}_{index}.mzML",
            })
            snapshot_rows.append({
                "ATTRIBUTE_DatasetAccession": accession,
                "filename": filename,
                "NCBITaxonomy": HUMAN_TAXON,
                "IonizationSourceAndPolarity": POSITIVE,
                "ChromatographyAndPhase": HILIC,
                "SubjectIdentifierAsRecorded": f"subject-{index}",
                "AgeInYears": str(66 + index % 25),
                "BiologicalSex": sex,
                "UBERONBodyPartName": URINE,
                "SmokingStatus": "missing value",
                "UniqueSubjectID": f"{accession}-subject-{index}",
                "LifeStage": LATER_ADULTHOOD,
            })
        manifest_path = tmp_path / f"{accession}.tsv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)
        specs.append(PanelSpec(
            accession=accession,
            panel_id=panel,
            source_family=family,
            manifest_path=manifest_path,
            manifest_sha256=sha256(manifest_path),
            expected_panel_files=total,
            expected_selected_units=total,
            expected_female=female,
            expected_male=male,
            analysis_role=role,
        ))
    snapshot_path = tmp_path / "snapshot.tsv"
    pd.DataFrame(snapshot_rows).to_csv(snapshot_path, sep="\t", index=False)
    return snapshot_path, tuple(specs)


def test_freezes_exact_92_person_pretruth_design(tmp_path: Path) -> None:
    snapshot, specs = _write_frozen_inputs(tmp_path)
    output = tmp_path / "out"
    report = build_design(snapshot, sha256(snapshot), specs, output)
    design = pd.read_csv(output / "design_table.tsv", sep="\t", keep_default_na=False)
    assert len(design) == 92
    assert design.analysis_unit_id.nunique() == 92
    assert design.source_family.nunique() == 2
    assert design.loc[design.analysis_role.eq("primary_source_view")].shape[0] == 75
    assert design.biological_sex.value_counts().to_dict() == {"female": 50, "male": 42}
    assert design.age_years.notna().all()
    assert design.smoking_status.eq("").all()
    assert report["denominators"] == {
        "source_families": 2,
        "panel_files_before_stratum_selection": 92,
        "selected_files": 92,
        "selected_participants": 92,
        "female": 50,
        "male": 42,
        "age_observed": 92,
        "smoking_status_observed": 0,
    }
    assert report["primary_evaluation_subset"] == {
        "selection": "analysis_role == primary_source_view",
        "source_families": 2,
        "participants": 75,
        "female": 39,
        "male": 36,
        "panels": ["P255", "P538"],
    }
    assert report["source_provenance_adjudication"][
        "MSV000083538_and_MSV000083540"
    ]["decision"] == "same_parent_cohort_not_independent_sources"
    assert report["status"] == "development_pretruth_design_frozen_no_abundance_opened"
    assert "no abundance" in report["leakage_status"]
    assert report["output_sha256"]["design_table"] == sha256(output / "design_table.tsv")


def test_input_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    snapshot, specs = _write_frozen_inputs(tmp_path)
    bad_specs = (replace(specs[0], manifest_sha256="0" * 64), *specs[1:])
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        build_design(snapshot, sha256(snapshot), bad_specs, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_inconsistent_repeated_unit_metadata_fails_closed(tmp_path: Path) -> None:
    snapshot, specs = _write_frozen_inputs(tmp_path)
    table = pd.read_csv(specs[0].manifest_path, sep="\t", dtype=str)
    table.loc[1, "biological_unit_id"] = table.loc[0, "biological_unit_id"]
    table.to_csv(specs[0].manifest_path, sep="\t", index=False)
    specs = (replace(specs[0], manifest_sha256=sha256(specs[0].manifest_path)), *specs[1:])
    snapshot_table = pd.read_csv(snapshot, sep="\t", dtype=str)
    mask = snapshot_table["filename"].eq(table.loc[1, "filename"])
    snapshot_table.loc[mask, "AgeInYears"] = "91"
    snapshot_table.to_csv(snapshot, sep="\t", index=False)
    with pytest.raises(ValueError, match="repeated files disagree on unit metadata"):
        build_design(snapshot, sha256(snapshot), specs, tmp_path / "out")


def test_report_is_valid_json_and_does_not_open_results(tmp_path: Path) -> None:
    snapshot, specs = _write_frozen_inputs(tmp_path)
    output = tmp_path / "out"
    build_design(snapshot, sha256(snapshot), specs, output)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["axis"]["level_a"] == "male"
    assert report["axis"]["level_b"] == "female"
    assert report["axis"]["numeric_covariates"] == ["age_years"]
    assert report["axis"]["unavailable_covariates"] == ["SmokingStatus"]
