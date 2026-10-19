#!/usr/bin/env python
"""Fail-closed local readiness audit for an MTBLS13729 Atlas contrast shard."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_summary(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "sample_columns": int(max(0, len(frame.columns) - 1)),
        "sha256": sha256(path),
    }


def audit(
    mzml_root: Path,
    manifest_path: Path,
    full_space_root: Path,
    ms2_link_root: Path,
    pilot_report_path: Path,
) -> dict[str, object]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    required = {"panel", "sample_id", "patient_id", "tissue", "histology", "hdf5_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(f"sample manifest missing columns: {sorted(missing)}")
    if manifest.duplicated(["panel", "sample_id"]).any():
        raise RuntimeError("duplicate panel/sample in manifest")
    pilot = json.loads(pilot_report_path.read_text(encoding="utf-8")) if pilot_report_path.is_file() else None
    pilot_by_panel = {
        str(item["panel"]): item for item in (pilot or {}).get("panels", [])
    }
    panels = []
    for panel, group in manifest.groupby("panel", sort=True):
        panel_dir = mzml_root / str(panel)
        expected = set(group["sample_id"].astype(str))
        mzml_samples = {path.stem for path in panel_dir.glob("*.mzML")}
        hdf5_samples = {path.stem for path in panel_dir.glob("*.hdf5")}
        paired = group.groupby("patient_id")["tissue"].agg(lambda values: set(values.astype(str)))
        complete_pairs = int(paired.map(lambda values: {"normal", "tumor"} <= values).sum())
        auc = matrix_summary(full_space_root / f"{panel}__eic_auc_matrix.csv.gz")
        detection = matrix_summary(full_space_root / f"{panel}__eic_detection_matrix.csv.gz")
        link_path = ms2_link_root / f"{panel}__linked_ms2.csv.gz"
        link_summary = None
        if link_path.is_file():
            links = pd.read_csv(link_path)
            link_summary = {
                "path": str(link_path),
                "rows": int(len(links)),
                "features": int(links["feature_id"].nunique()) if "feature_id" in links else None,
                "query_files": int(links["query_file"].nunique()) if "query_file" in links else None,
                "sha256": sha256(link_path),
            }
        pilot_panel = pilot_by_panel.get(str(panel))
        raw_complete = expected <= mzml_samples
        hdf5_complete = expected <= hdf5_samples
        formal_uniform_matrix = bool(
            pilot_panel
            and pilot_panel.get("formal") is True
            and pilot_panel.get("matrix_kind") == "uniform_requantified_eic"
        )
        pilot_only = bool(pilot_panel and pilot_panel.get("matrix_kind") == "discovery_missingness_pilot")
        panel_gates = {
            "registered_paired_design_complete": bool(complete_pairs == group["patient_id"].nunique()),
            "all_registered_mzml_local": raw_complete,
            "all_registered_hdf5_local": hdf5_complete,
            "formal_uniform_unbiased_matrix": formal_uniform_matrix,
            "ms1_ms2_links_present": link_summary is not None,
            "reproducible_quantitative_universe": bool(raw_complete or hdf5_complete or formal_uniform_matrix),
            "admissible_atlas_panel": bool(
                complete_pairs == group["patient_id"].nunique()
                and (raw_complete or hdf5_complete or formal_uniform_matrix)
                and link_summary is not None
                and not pilot_only
            ),
        }
        panels.append({
            "panel": str(panel),
            "registered_samples": int(len(group)),
            "registered_patients": int(group["patient_id"].nunique()),
            "complete_normal_tumor_pairs": complete_pairs,
            "local_mzml_samples": len(mzml_samples & expected),
            "local_hdf5_samples": len(hdf5_samples & expected),
            "unexpected_local_mzml": sorted(mzml_samples - expected),
            "unexpected_local_hdf5": sorted(hdf5_samples - expected),
            "full_space_auc": auc,
            "full_space_detection": detection,
            "ms2_links": link_summary,
            "global_peak_pilot": pilot_panel,
            "gates": panel_gates,
        })
    admitted = [item["panel"] for item in panels if item["gates"]["admissible_atlas_panel"]]
    return {
        "status": "MTBLS13729_CONTRAST_ATLAS_READINESS_AUDITED",
        "formal": True,
        "study_id": "MTBLS13729",
        "source_family": "MTBLS13729_original_study",
        "data_role": "development/consumed",
        "panels": panels,
        "admissible_panels": admitted,
        "pass_to_contrast_record": bool(admitted),
        "blocking_reason": None if admitted else (
            "No panel simultaneously has a complete reproducible quantitative universe, "
            "paired design, and MS1-MS2 linkage. Discovery-missingness pilots and targeted "
            "EIC matrices cannot substitute for an unbiased study-level contrast."
        ),
        "claim_limit": "Local file/provenance readiness only; no effect, retrieval, biology, or external validation is established.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mzml-root", type=Path, default=Path("data/mtbls13729/mzml"))
    parser.add_argument("--sample-manifest", type=Path, default=Path("data/mtbls13729/reverse_probe_manifest_v4/sample_manifest.csv"))
    parser.add_argument("--full-space-root", type=Path, default=Path("data/mtbls13729/full_space_eic_v1"))
    parser.add_argument("--ms2-link-root", type=Path, default=Path("data/mtbls13729/ms1_ms2_link"))
    parser.add_argument("--pilot-report", type=Path, default=Path("data/validation/mtbls13729_bioaware_global_peak_pilot_neg_20260829/report.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    report = audit(
        args.mzml_root, args.sample_manifest, args.full_space_root,
        args.ms2_link_root, args.pilot_report,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "data_role": report["data_role"],
        "admissible_panels": report["admissible_panels"],
        "pass_to_contrast_record": report["pass_to_contrast_record"],
        "blocking_reason": report["blocking_reason"],
    }, indent=2))


if __name__ == "__main__":
    main()
