#!/usr/bin/env python
"""Freeze a cost-controlled MTBLS13432 positive-mode MS/MS pilot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pyreadr


BASE = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS13432/"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_target_coordinates(mz_id: str) -> tuple[float, float]:
    match = re.match(r"^([0-9.]+)_([0-9.]+)", str(mz_id))
    if not match:
        raise ValueError(f"cannot parse target coordinates: {mz_id}")
    return float(match.group(1)), float(match.group(2))


def parse_apache_size(value: object) -> float:
    match = re.fullmatch(r"\s*([0-9.]+)\s*([KMGT]?)\s*", str(value))
    if not match:
        return float("nan")
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    return float(match.group(1)) * scale[match.group(2)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sgme-root",
        type=Path,
        default=Path("data/external/mtbls13432_sgme_hcc/source_snapshot"),
    )
    parser.add_argument(
        "--preflight-dir",
        type=Path,
        default=Path("data/validation/mtbls13432_sgme_preflight"),
    )
    args = parser.parse_args()

    peaks_path = args.sgme_root / "data/lcms/lcms_peaks.Rdata"
    abundance_path = args.sgme_root / "data/lcms/lcms_raw.rds"
    preflight_path = args.preflight_dir / "sgme_preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError(preflight_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight.get("non_probe_pass"):
        raise RuntimeError("SgME non-probe preflight did not pass")

    objects = pyreadr.read_r(str(peaks_path))
    raw = objects["lcms_raw_peaks"].copy()
    hit = objects["lcms_hit_peaks"].copy()
    unannotated = hit["MSMS_annotation"].fillna("").astype(str).str.strip().str.lower().isin(
        {"", "nan", "none", "unannotated", "no matching", "no match"}
    )
    retained_gap = hit.loc[unannotated].copy()
    retained_gap["gap_type"] = "retained_unannotated"
    selected_gap = raw[
        raw["is_tested"].fillna(False).astype(bool)
        & ~raw["mz_id"].isin(set(hit["mz_id"]))
    ].copy()
    selected_gap["gap_type"] = "selected_not_retained"
    targets = pd.concat([retained_gap, selected_gap], ignore_index=True, sort=False)
    coordinates = targets["mz_id"].map(parse_target_coordinates)
    targets["rt_min"] = coordinates.map(lambda value: value[0])
    targets["feature_mz"] = coordinates.map(lambda value: value[1])
    targets["public_tissue_ms2_mode"] = targets["lcms_mode"].isin(["HPOS", "LPOS"])
    target_columns = [
        "gap_type",
        "peak_id",
        "mz_id",
        "lcms_mode",
        "rt_min",
        "feature_mz",
        "is_msms",
        "MSMS_annotation",
        "Comment",
        "mean",
        "pct90",
        "public_tissue_ms2_mode",
    ]
    for column in target_columns:
        if column not in targets:
            targets[column] = pd.NA
    targets[target_columns].sort_values(
        ["public_tissue_ms2_mode", "lcms_mode", "pct90"], ascending=[False, True, False]
    ).to_csv(args.preflight_dir / "high_value_reannotation_targets.csv", index=False)

    abundance = pyreadr.read_r(str(abundance_path))[None]
    sections = abundance[["case_id", "section_id"]].drop_duplicates().copy()
    sections["case_id"] = sections["case_id"].astype(str).str.strip()
    sections["section_id"] = sections["section_id"].astype(str).str.strip()

    selected_sections: list[tuple[str, str, str]] = []
    for case_id, group in sections.groupby("case_id"):
        names = set(group["section_id"])
        if "N" not in names:
            continue
        tumors = sorted(name for name in names if name.startswith("T"))
        if not tumors:
            continue
        selected_sections.append((case_id, "N", "normal"))
        selected_sections.append((case_id, tumors[0], "tumor"))

    derived = pd.read_html(BASE + "FILES/DERIVED_FILES/")[0]
    derived = derived[
        derived["Name"].astype(str).str.startswith("LC-MS_")
        & derived["Name"].astype(str).str.endswith(".mzML")
    ].copy()
    available = set(derived["Name"].astype(str))
    size_by_name = dict(zip(derived["Name"].astype(str), derived["Size"].map(parse_apache_size)))
    rows: list[dict[str, str]] = []
    for case_id, section_id, tissue_role in selected_sections:
        for mode in ("HPOS", "LPOS"):
            filename = f"LC-MS_{case_id}_{section_id}_{mode}.mzML"
            if filename not in available:
                raise RuntimeError(f"missing public pilot file: {filename}")
            rows.append(
                {
                    "case_id": case_id,
                    "section_id": section_id,
                    "tissue_role": tissue_role,
                    "mode": mode,
                    "filename": filename,
                    "estimated_bytes": size_by_name[filename],
                    "url": BASE + "FILES/DERIVED_FILES/" + filename,
                }
            )
    pilot = pd.DataFrame(rows)
    pilot.to_csv(args.preflight_dir / "positive_mode_paired_pilot_files.csv", index=False)

    all_positive = derived[
        derived["Name"].astype(str).str.contains(r"_(?:HPOS|LPOS)\.mzML$", regex=True)
    ].copy()
    all_positive["url"] = BASE + "FILES/DERIVED_FILES/" + all_positive["Name"].astype(str)
    all_positive[["Name", "Size", "url"]].to_csv(
        args.preflight_dir / "all_positive_mode_files.csv", index=False
    )

    report = {
        "status": "mtbls13432_sgme_positive_mode_pilot_manifest_frozen",
        "high_value_targets": int(len(targets)),
        "targets_in_public_tissue_ms2_modes": int(targets["public_tissue_ms2_mode"].sum()),
        "targets_in_lneg_without_sample_ms2": int((targets["lcms_mode"] == "LNEG").sum()),
        "paired_patients": int(pilot["case_id"].nunique()),
        "paired_sections": int(pilot[["case_id", "section_id"]].drop_duplicates().shape[0]),
        "pilot_mzml_files": int(len(pilot)),
        "pilot_estimated_gib": float(pilot["estimated_bytes"].sum() / 1024**3),
        "full_positive_mode_mzml_files": int(len(all_positive)),
        "pilot_protocol": "one normal and one deterministic tumor section per paired patient; HPOS and LPOS",
        "coverage_gate": {
            "minimum_targets_with_ms2": 12,
            "minimum_targets_seen_in_three_patients": 6,
            "if_fail": "expand from paired pilot to all 218 positive-mode files once; if still failing, stop direct SgME reannotation",
        },
        "annotation_gate_after_coverage": {
            "minimum_auditable_level2_candidates": 12,
            "minimum_candidates_linkable_to_spatial_peak_pool": 6,
            "maximum_high_confidence_conflict_fraction": 0.05,
        },
        "missing_author_artifacts": [
            "pooled-QC multi-collision-energy MS/MS raw or mzML files used for the 230 retained records",
            "results/PM_groups.rds containing the published six-cluster membership of 175 spatial peaks",
        ],
        "provenance": {
            "preflight_sha256": file_hash(preflight_path),
            "peaks_sha256": file_hash(peaks_path),
            "abundance_sha256": file_hash(abundance_path),
        },
        "claim_limit": "This is a frozen acquisition and target manifest, not an annotation result.",
    }
    output = args.preflight_dir / "pilot_manifest.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
