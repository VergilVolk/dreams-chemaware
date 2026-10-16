#!/usr/bin/env python3
"""Split the frozen phenotype-blind HFD manifest into assay-local jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    acquisition_path = args.wave_dir / "phenotype_blind" / "acquisition_manifest.tsv"
    registry_path = args.wave_dir / "panel_registry.tsv"
    acquisition = pd.read_csv(acquisition_path, sep="\t", dtype=str, keep_default_na=False)
    registry = pd.read_csv(registry_path, sep="\t", dtype=str, keep_default_na=False)
    required = {
        "panel_id", "dataset_accession", "sample_id", "biological_unit_id",
        "download_url", "local_path", "deposited_bytes",
    }
    if missing := sorted(required - set(acquisition)):
        raise ValueError(f"acquisition manifest lacks {missing}")
    if acquisition.duplicated(["panel_id", "sample_id"]).any():
        raise ValueError("duplicate panel/sample acquisition")
    if set(acquisition.panel_id) != set(registry.panel_id):
        raise ValueError("panel registry and acquisition manifest differ")

    blind_root = args.output_dir / "phenotype_blind_panels"
    blind_root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index, panel in enumerate(registry.sort_values(["dataset_accession", "panel_id"]).itertuples(index=False)):
        block = acquisition.loc[acquisition.panel_id.eq(str(panel.panel_id))].copy()
        blind = block[[
            "sample_id", "biological_unit_id", "dataset_accession", "filename",
            "download_url", "local_path", "deposited_bytes",
        ]].sort_values("sample_id")
        panel_dir = blind_root / str(panel.panel_id)
        panel_dir.mkdir()
        manifest_path = panel_dir / "quantification_manifest.tsv"
        blind.to_csv(manifest_path, sep="\t", index=False)
        rows.append({
            "job_index": index,
            "panel_id": str(panel.panel_id),
            "dataset_accession": str(panel.dataset_accession),
            "source_family": str(panel.source_family),
            "sample_files": len(blind),
            "biological_units": int(blind.biological_unit_id.nunique()),
            "expected_bytes": int(pd.to_numeric(blind.deposited_bytes, errors="raise").sum()),
            "quantification_manifest": manifest_path.as_posix(),
            "panel_object_dir": (args.output_dir / "quantified_panels" / str(panel.panel_id)).as_posix(),
        })
    jobs = pd.DataFrame(rows)
    jobs.to_csv(args.output_dir / "panel_jobs.tsv", sep="\t", index=False)
    report = {
        "schema": "factorial_hfd_panel_jobs_v1",
        "status": "phenotype_blind_panel_jobs_frozen",
        "denominators": {
            "panels": len(jobs),
            "sample_files": len(acquisition),
            "studies": int(acquisition.dataset_accession.nunique()),
            "expected_bytes": int(pd.to_numeric(acquisition.deposited_bytes).sum()),
        },
        "provenance": {
            "acquisition_manifest_sha256": sha256(acquisition_path),
            "panel_registry_sha256": sha256(registry_path),
        },
        "leakage_status": "no genotype_or_diet_fields_written_to_panel_quantification_manifests",
        "claim_boundary": "Acquisition jobs only; no abundance, response, or retrieval result.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
