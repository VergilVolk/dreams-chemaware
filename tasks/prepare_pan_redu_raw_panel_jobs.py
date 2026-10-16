#!/usr/bin/env python3
"""Prepare phenotype-blind raw-panel jobs and sealed contrast memberships.

The quantification process receives only acquisition identifiers, hashed
biological-unit identifiers, paths, and byte envelopes.  Outcome fields are
written to a different tree and are opened only after the abundance matrix is
frozen.  This separation is a leakage boundary, not a claim that Pan-ReDU
metadata defines valid biological truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from download_pan_redu_wave_design_assets import safe_target


PANEL_KEYS = (
    "dataset_accession",
    "taxonomy",
    "body_part",
    "instrument",
    "polarity",
    "chromatography",
)
CONTRAST_COLUMNS = {
    "BiologicalSex": "biological_sex",
    "HealthStatus": "health_status",
    "Diet": "diet",
    "LifeStage": "life_stage",
}


def stable_id(*parts: str, length: int = 20) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:length]


def local_path(root: Path, accession: str, descriptor: str) -> Path:
    prefix = f"f.{accession}/"
    if not descriptor.startswith(prefix):
        raise ValueError(f"descriptor/accession mismatch: {descriptor}")
    target = safe_target(root, accession, descriptor[len(prefix):])
    # Manifests are generated on Windows and executed on Linux.  Keep paths
    # repository-relative and render them with POSIX separators.
    if root.is_absolute():
        raise ValueError("--raw-root must be repository-relative for portable server execution")
    return target.relative_to(Path.cwd().resolve())


def biological_unit(row: pd.Series) -> str:
    raw = str(row["reported_unique_subject_id"]).strip()
    source = "reported_unique_subject_id"
    if not raw:
        raw = str(row["subject_identifier_as_recorded"]).strip()
        source = "subject_identifier_as_recorded"
    if not raw:
        raise ValueError(f"strict panel file lacks subject identifier: {row['remote_descriptor']}")
    return f"U{stable_id(str(row['dataset_accession']), source, raw)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", required=True, type=Path)
    parser.add_argument("--contrast-panels", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_manifest, sep="\t", dtype=str, keep_default_na=False)
    contrasts = pd.read_csv(args.contrast_panels, sep="\t", dtype=str, keep_default_na=False)
    required_raw = set(PANEL_KEYS) | {
        "remote_descriptor", "filename", "deposited_bytes",
        "reported_unique_subject_id", "subject_identifier_as_recorded",
    }
    required_contrasts = set(PANEL_KEYS) | {
        "panel_id", "contrast_id", "contrast_field", "level_a", "level_b",
    }
    if missing := sorted(required_raw - set(raw.columns)):
        raise ValueError(f"raw manifest missing columns: {missing}")
    if missing := sorted(required_contrasts - set(contrasts.columns)):
        raise ValueError(f"contrast registry missing columns: {missing}")
    if raw["remote_descriptor"].duplicated().any():
        raise ValueError("raw acquisition manifest is not unique by remote_descriptor")

    panels = contrasts[["panel_id", *PANEL_KEYS]].drop_duplicates()
    if panels["panel_id"].duplicated().any():
        raise ValueError("one panel_id maps to multiple analytical strata")
    joined = raw.merge(panels, on=list(PANEL_KEYS), how="left", validate="many_to_one")
    if joined["panel_id"].isna().any():
        examples = joined.loc[joined["panel_id"].isna(), "remote_descriptor"].head().tolist()
        raise ValueError(f"raw files do not map to selected panels: {examples}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    blind_root = args.output_dir / "phenotype_blind_panels"
    sealed_root = args.output_dir / "sealed_contrast_membership"
    blind_root.mkdir(exist_ok=True)
    sealed_root.mkdir(exist_ok=True)
    job_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, str]] = []

    for panel in panels.sort_values("panel_id").itertuples(index=False):
        panel_id = str(panel.panel_id)
        selected = joined.loc[joined["panel_id"].eq(panel_id)].copy()
        selected["sample_id"] = selected["remote_descriptor"].map(lambda value: f"S{stable_id(value)}")
        selected["biological_unit_id"] = selected.apply(biological_unit, axis=1)
        selected["local_path"] = [
            local_path(args.raw_root, accession, descriptor).as_posix()
            for accession, descriptor in zip(selected["dataset_accession"], selected["remote_descriptor"])
        ]
        selected["format"] = selected["filename"].map(lambda value: Path(value).suffix.lower().lstrip("."))
        if not set(selected["format"]).issubset({"mzml", "mzxml"}):
            raise ValueError(f"unsupported raw format in panel {panel_id}")
        blind = selected[[
            "sample_id", "biological_unit_id", "dataset_accession", "filename",
            "remote_descriptor", "local_path", "deposited_bytes", "format",
        ]].sort_values("sample_id")
        panel_dir = blind_root / panel_id
        panel_dir.mkdir(exist_ok=True)
        manifest_path = panel_dir / "quantification_manifest.tsv"
        blind.to_csv(manifest_path, sep="\t", index=False)

        panel_contrasts = contrasts.loc[contrasts["panel_id"].eq(panel_id)]
        for contrast in panel_contrasts.itertuples(index=False):
            field = str(contrast.contrast_field)
            if field not in CONTRAST_COLUMNS:
                raise ValueError(f"no raw-manifest mapping for contrast field {field}")
            raw_column = CONTRAST_COLUMNS[field]
            admitted = selected.loc[selected[raw_column].isin((str(contrast.level_a), str(contrast.level_b)))]
            for row in admitted.itertuples(index=False):
                membership_rows.append({
                    "panel_id": panel_id,
                    "contrast_id": str(contrast.contrast_id),
                    "sample_id": str(row.sample_id),
                    "biological_unit_id": str(row.biological_unit_id),
                    "contrast_field": field,
                    "level": str(getattr(row, raw_column)),
                    "level_a": str(contrast.level_a),
                    "level_b": str(contrast.level_b),
                })
        job_rows.append({
            "job_index": len(job_rows),
            "panel_id": panel_id,
            "dataset_accession": str(panel.dataset_accession),
            "sample_files": len(blind),
            "biological_units": blind["biological_unit_id"].nunique(),
            "expected_bytes": int(pd.to_numeric(blind["deposited_bytes"], errors="raise").sum()),
            "quantification_manifest": manifest_path.as_posix(),
            "output_dir": (args.output_dir / "quantified_panels" / panel_id).as_posix(),
        })

    memberships = pd.DataFrame(membership_rows).sort_values(["panel_id", "contrast_id", "sample_id"])
    memberships.to_csv(sealed_root / "contrast_membership.tsv", sep="\t", index=False)
    jobs = pd.DataFrame(job_rows)
    jobs.to_csv(args.output_dir / "panel_jobs.tsv", sep="\t", index=False)
    crossing = (
        memberships.groupby(["contrast_id", "biological_unit_id"])["level"]
        .nunique().reset_index(name="n_levels")
    )
    crossing_counts = (
        crossing.loc[crossing["n_levels"] > 1].groupby("contrast_id").size().to_dict()
    )
    topology = contrasts[[
        "panel_id", "contrast_id", "dataset_accession", "preliminary_source_family",
        "comparison_stratum", "contrast_field", "level_a", "level_b",
    ]].drop_duplicates("contrast_id").copy()
    topology["reported_units_crossing_levels"] = topology["contrast_id"].map(crossing_counts).fillna(0).astype(int)
    topology["topology_status"] = np.where(
        topology["reported_units_crossing_levels"].gt(0),
        "FAIL_REPORTED_UNIT_CROSSES_LEVELS",
        "PASS_REPORTED_UNIT_TOPOLOGY_METADATA_ONLY",
    )
    topology.to_csv(args.output_dir / "contrast_topology_readiness.tsv", sep="\t", index=False)
    failed_topology = int(topology["reported_units_crossing_levels"].gt(0).sum())
    report = {
        "schema": "pan_redu_raw_panel_jobs_v1",
        "status": "development_inputs_prepared_no_abundance_opened",
        "denominators": {
            "panels": len(jobs),
            "contrasts": int(contrasts["contrast_id"].nunique()),
            "accessions": int(raw["dataset_accession"].nunique()),
            "sample_files": len(raw),
            "biological_units_across_panel_namespaces": int(jobs["biological_units"].sum()),
            "expected_bytes": int(pd.to_numeric(raw["deposited_bytes"], errors="raise").sum()),
            "sealed_membership_rows": len(memberships),
            "contrasts_passing_reported_unit_topology": len(topology) - failed_topology,
            "contrasts_failing_cross_level_units": failed_topology,
        },
        "leakage_status": (
            "quantification manifests exclude contrast fields and levels; sealed memberships "
            "must not be opened until each panel abundance matrix is frozen"
        ),
        "claim_boundary": [
            "Pan-ReDU labels qualify candidate contrasts but do not establish biological truth.",
            "Hashed subject identifiers preserve unit topology; they do not prove independence or remove confounding.",
            "Raw acquisition and quantification are pending until byte-verified files exist.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
