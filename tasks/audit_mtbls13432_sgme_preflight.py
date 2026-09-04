#!/usr/bin/env python
"""Audit whether MTBLS13432/SgME-HCC is fit for an algorithm-to-biology study.

The audit deliberately separates four quantities that are easy to conflate:
all aligned LC-MS features, features selected for fragmentation, retained MS/MS
records, and records with a non-empty structural assignment.  It also rebuilds
the real section-level file manifest from the public directory because the ISA
assay tables expose only one representative section per patient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadr
import requests


BASE = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS13432/"
ASSAYS = {
    "LNEG": "a_MTBLS13432_LC-MS_negative_reverse-phase_metabolite_profiling.txt",
    "HPOS": "a_MTBLS13432_LC-MS_positive_hilic_metabolite_profiling.txt",
    "LPOS": "a_MTBLS13432_LC-MS_positive_reverse-phase_metabolite_profiling.txt",
}
PROBES = {
    mode: f"FILES/DERIVED_FILES/LC-MS_B002_N_{mode}.mzML" for mode in ASSAYS
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_apache_size(value: Any) -> float:
    match = re.fullmatch(r"\s*([0-9.]+)\s*([KMGT]?)\s*", str(value))
    if not match:
        return float("nan")
    multiplier = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    return float(match.group(1)) * multiplier[match.group(2)]


def is_unannotated(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip().str.lower()
    return values.isin({"", "nan", "none", "unannotated", "no matching", "no match"})


def probe_mzml(url: str) -> dict[str, Any]:
    patterns = {
        "ms1": re.compile(rb'name="ms level" value="1"'),
        "ms2": re.compile(rb'name="ms level" value="2"'),
    }
    counts = {key: 0 for key in patterns}
    total = 0
    carry = b""
    started = time.time()
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        for block in response.iter_content(1024 * 1024):
            if not block:
                continue
            payload = carry + block
            for key, pattern in patterns.items():
                counts[key] += len(pattern.findall(payload))
            carry = payload[-128:]
            total += len(block)
    return {
        "url": url,
        "bytes_streamed": total,
        "ms1_spectra": counts["ms1"],
        "ms2_spectra": counts["ms2"],
        "seconds": round(time.time() - started, 2),
        "has_ms2": counts["ms2"] > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sgme-root",
        type=Path,
        default=Path("data/external/mtbls13432_sgme_hcc/source_snapshot"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/mtbls13432_sgme_preflight"),
    )
    parser.add_argument("--probe-mzml", action="store_true")
    args = parser.parse_args()

    lcms_dir = args.sgme_root / "data" / "lcms"
    peaks_path = lcms_dir / "lcms_peaks.Rdata"
    raw_abundance_path = lcms_dir / "lcms_raw.rds"
    supplement_path = lcms_dir / "Supp_Tables_S1 to S4_v3.xlsx"
    for path in (peaks_path, raw_abundance_path, supplement_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    objects = pyreadr.read_r(str(peaks_path))
    feature_table = objects["lcms_raw_peaks"].copy()
    hit_table = objects["lcms_hit_peaks"].copy()
    abundance = pyreadr.read_r(str(raw_abundance_path))[None]
    sections = abundance[["case_id", "section_id"]].drop_duplicates().copy()
    sections["case_id"] = sections["case_id"].astype(str).str.strip()
    sections["section_id"] = sections["section_id"].astype(str).str.strip()

    clinical = pd.read_excel(supplement_path, sheet_name="Table_S1a_Clinical_info")
    clinical = clinical[
        clinical["Patient ID"].notna()
        & clinical["Histological diagnosis"].isin(["HCC", "CCA"])
    ].copy()
    clinical["patient_id"] = clinical["Patient ID"].astype(str).str.strip()
    processed_ids = set(sections["case_id"])
    processed_clinical = clinical[clinical["patient_id"].isin(processed_ids)]

    # Public listings are the authoritative section-level file inventory.
    raw_listing = pd.read_html(BASE + "FILES/RAW_FILES/")[0]
    raw_listing = raw_listing[raw_listing["Name"].astype(str).str.startswith("LC-MS_")].copy()
    raw_listing["bytes"] = raw_listing["Size"].map(parse_apache_size)
    raw_listing["mode"] = raw_listing["Name"].str.extract(r"_(HPOS|LPOS|LNEG)\.raw\.zip$")
    raw_listing["section_key"] = raw_listing["Name"].str.replace(
        r"_(HPOS|LPOS|LNEG)\.raw\.zip$", "", regex=True
    )
    mode_counts = raw_listing.groupby("mode").size().astype(int).to_dict()
    triplet_counts = raw_listing.groupby("section_key")["mode"].nunique()

    derived_listing = pd.read_html(BASE + "FILES/DERIVED_FILES/")[0]
    derived_listing = derived_listing[
        derived_listing["Name"].astype(str).str.startswith("LC-MS_")
        & derived_listing["Name"].astype(str).str.endswith(".mzML")
    ].copy()

    assay_reports: dict[str, Any] = {}
    for mode, filename in ASSAYS.items():
        table = pd.read_csv(BASE + filename, sep="\t", dtype=str)
        assay_reports[mode] = {
            "rows": int(len(table)),
            "unique_samples": int(table["Sample Name"].nunique()),
            "raw_files_referenced": int(table["Raw Spectral Data File"].nunique()),
        }

    unannotated_hits = is_unannotated(hit_table["MSMS_annotation"])
    tested = int(feature_table["is_tested"].fillna(False).astype(bool).sum())
    msms_flagged = int(feature_table["is_msms"].fillna(False).astype(bool).sum())
    retained = int(len(hit_table))
    assigned = int((~unannotated_hits).sum())
    conservative_headroom = tested - assigned

    headroom = hit_table.loc[
        unannotated_hits,
        ["peak_id", "mz_id", "lcms_mode", "MSMS_annotation", "Comment", "mean", "pct90"],
    ].copy()
    missing_fragmentation = feature_table[
        feature_table["is_tested"].fillna(False).astype(bool)
        & ~feature_table["mz_id"].isin(set(hit_table["mz_id"]))
    ][["mz_id", "lcms_mode", "mean", "pct90", "is_msms", "MSMS_annotation", "Comment"]]
    headroom.to_csv(args.output_dir / "retained_unannotated_high_abundance.csv", index=False)
    missing_fragmentation.to_csv(
        args.output_dir / "selected_but_not_retained_msms.csv", index=False
    )
    sections.sort_values(["case_id", "section_id"]).to_csv(
        args.output_dir / "processed_section_manifest.csv", index=False
    )
    raw_listing[["Name", "Size", "mode", "section_key"]].to_csv(
        args.output_dir / "public_lcms_raw_file_manifest.csv", index=False
    )

    probe_reports: dict[str, Any] = {}
    if args.probe_mzml:
        for mode, relative in PROBES.items():
            print(f"[probe] {mode}: {relative}", flush=True)
            probe_reports[mode] = probe_mzml(BASE + relative)

    biological_question = {
        "primary": (
            "Do corrected or newly assigned structures among high-abundance ions explain the "
            "ME-low-grade-positive / ME-necrotic-negative spatial metabolic program that the "
            "source study reported but left mechanistically unresolved?"
        ),
        "secondary": (
            "Can spatial localization distinguish tumor-cell metabolic reprogramming from "
            "signals arising in non-malignant metabolic regions?"
        ),
        "primary_endpoint_locked_before_new_annotation": True,
        "required_validation": [
            "patient/section leave-one-out spatial stability",
            "LC-MS retention-time and adduct consistency",
            "MS/MS candidate margin and decoy calibration",
            "coherent multi-metabolite family or reaction module",
        ],
    }

    gates = {
        "processed_patients_ge_20": len(processed_ids) >= 20,
        "processed_sections_ge_100": len(sections) >= 100,
        "three_lcms_modes_complete_for_every_section": bool(
            len(triplet_counts) >= 100 and (triplet_counts == 3).all()
        ),
        "aligned_features_ge_8000": len(feature_table) >= 8000,
        "selected_high_abundance_ge_250": tested >= 250,
        "retained_msms_records_ge_200": retained >= 200,
        "direct_annotation_headroom_ge_40": conservative_headroom >= 40,
        "public_mzml_triplets_complete": len(derived_listing) == len(raw_listing) == 327,
        "probe_each_mode_has_ms2": bool(
            probe_reports and all(item["has_ms2"] for item in probe_reports.values())
        ),
    }
    non_probe_gates = {key: value for key, value in gates.items() if not key.startswith("probe_")}

    report = {
        "status": "mtbls13432_sgme_preflight_complete",
        "formal": bool(args.probe_mzml),
        "author_repository_commit": "56683221c1fa5eb3d819248c9ab8f9cf7a7ff4a2",
        "biological_question": biological_question,
        "cohort": {
            "supplement_patients": int(len(clinical)),
            "processed_lcms_patients": int(len(processed_ids)),
            "processed_lcms_sections": int(len(sections)),
            "histology_in_processed_lcms": processed_clinical[
                "Histological diagnosis"
            ].value_counts().astype(int).to_dict(),
            "sections_per_patient": {
                "minimum": int(sections.groupby("case_id").size().min()),
                "median": float(sections.groupby("case_id").size().median()),
                "maximum": int(sections.groupby("case_id").size().max()),
            },
        },
        "data_scale": {
            "processed_aligned_features": int(len(feature_table)),
            "processed_abundance_rows": int(len(abundance)),
            "public_raw_lcms_files": int(len(raw_listing)),
            "public_derived_mzml_files": int(len(derived_listing)),
            "public_raw_lcms_gib": float(raw_listing["bytes"].sum() / 1024**3),
            "raw_files_by_mode": mode_counts,
            "complete_section_triplets": int((triplet_counts == 3).sum()),
        },
        "annotation_funnel": {
            "aligned_features": int(len(feature_table)),
            "selected_high_abundance_features": tested,
            "raw_is_msms_flagged": msms_flagged,
            "retained_msms_records": retained,
            "retained_nonempty_assignments": assigned,
            "retained_unannotated": int(unannotated_hits.sum()),
            "selected_but_not_retained": int(tested - retained),
            "conservative_high_value_headroom": int(conservative_headroom),
            "definition": "selected high-abundance features minus retained non-empty assignments",
            "important_boundary": (
                "8742 aligned features are not 8742 MS/MS-identifiable structures; the direct "
                "high-value reannotation pool is 59 under the retained-table definition."
            ),
        },
        "isa_metadata_audit": {
            "assays": assay_reports,
            "warning": (
                "Each ISA LC-MS assay exposes one representative section per patient, whereas "
                "the public file directory and processed abundance table contain 109 sections. "
                "Use the reconstructed manifest, not the ISA rows alone."
            ),
        },
        "mzml_probes": probe_reports,
        "gates": gates,
        "non_probe_pass": all(non_probe_gates.values()),
        "formal_pass": all(gates.values()),
        "decision": (
            "GO to a targeted annotation pilot" if all(non_probe_gates.values()) else "NO-GO"
        ),
        "claim_limit": (
            "Passing preflight establishes cohort size, multimodal structure, public file "
            "completeness and annotation headroom. It does not establish a new metabolite, "
            "mechanism, flux, or superiority of the DreaMS pipeline."
        ),
        "provenance": {
            "lcms_peaks_sha256": sha256(peaks_path),
            "lcms_raw_rds_sha256": sha256(raw_abundance_path),
            "supplement_sha256": sha256(supplement_path),
        },
    }
    output = args.output_dir / "sgme_preflight.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
