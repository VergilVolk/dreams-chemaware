#!/usr/bin/env python3
"""Repair IBD200 consensus precursor m/z from its phenotype-blind quant table.

The deposited consensus MGF uses GNPS row IDs in ``SCANS`` but writes 0.0 for
many precursor masses.  This builder changes only PEPMASS, using the matching
``row m/z`` from the frozen feature table.  Peaks and entity IDs are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_contrast_atlas_sparse_fragment_measure_baseline import (
    iter_mgf,
    spectrum_id,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mgf", required=True, type=Path)
    parser.add_argument("--quant", required=True, type=Path)
    parser.add_argument("--response-entities", required=True, type=Path)
    parser.add_argument("--study-id", default="IBD200_GROUP")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    quant = pd.read_csv(args.quant, usecols=["row ID", "row m/z"])
    quant["row ID"] = quant["row ID"].astype(int).astype(str)
    quant["row m/z"] = pd.to_numeric(quant["row m/z"], errors="raise")
    if quant["row ID"].duplicated().any() or not np.isfinite(quant["row m/z"]).all():
        raise RuntimeError("quant row ID/mz registry is invalid")
    mz_by_id = dict(zip(quant["row ID"], quant["row m/z"].astype(float)))
    required = pd.read_csv(
        args.response_entities,
        sep="\t",
        usecols=["study_id", "raw_entity_id"],
        dtype=str,
    ).drop_duplicates()
    required = set(
        required.loc[required["study_id"].eq(args.study_id), "raw_entity_id"].astype(str)
    )
    if not required:
        raise RuntimeError("response collection has no IBD200 entities")

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "representative_spectra.mgf"
    seen: set[str] = set()
    repaired = 0
    unchanged = 0
    missing_quant = 0
    insufficient_required = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for metadata, peaks in iter_mgf(args.input_mgf):
            entity = spectrum_id(metadata)
            if entity in seen:
                raise RuntimeError(f"duplicate consensus spectrum ID: {entity}")
            seen.add(entity)
            replacement = mz_by_id.get(entity)
            original_text = metadata.get("PEPMASS", "")
            original = float(original_text.split()[0]) if original_text else float("nan")
            if replacement is None:
                missing_quant += 1
                precursor = original
            else:
                precursor = float(replacement)
                if not np.isfinite(original) or abs(original - precursor) > 1e-12:
                    repaired += 1
                else:
                    unchanged += 1
            if entity in required:
                if replacement is None or not np.isfinite(precursor) or precursor <= 0:
                    raise RuntimeError(f"required IBD200 entity lacks a valid quant precursor: {entity}")
                if len(peaks) < 3:
                    insufficient_required += 1
            handle.write("BEGIN IONS\n")
            for key, value in metadata.items():
                if key == "PEPMASS":
                    continue
                handle.write(f"{key}={value}\n")
            handle.write(f"PEPMASS={precursor:.10f}\n")
            for mz, intensity in peaks:
                handle.write(f"{mz:.10g} {intensity:.10g}\n")
            handle.write("END IONS\n")
    if not required.issubset(seen):
        raise RuntimeError("repaired MGF does not contain every required IBD200 entity")

    report = {
        "schema": "contrast_atlas_ibd200_precursor_repair_v1",
        "status": "PHENOTYPE_BLIND_IBD200_PRECURSOR_REPAIR_COMPLETE",
        "denominators": {
            "input_spectra": len(seen),
            "quant_features": len(mz_by_id),
            "required_response_entities": len(required),
            "precursors_repaired": repaired,
            "precursors_already_equal": unchanged,
            "spectra_without_quant_row": missing_quant,
            "required_entities_with_fewer_than_3_peaks": insufficient_required,
        },
        "repair_rule": "SCANS equals GNPS row ID; replace PEPMASS by exact matching quant-table row m/z",
        "leakage_status": (
            "only spectrum IDs, peaks, precursor metadata, quant row ID/mz and frozen response entity IDs "
            "were read; no beta, SE, phenotype, contrast, relation or retrieval result was loaded"
        ),
        "provenance": {
            "input_mgf_sha256": sha256(args.input_mgf),
            "quant_sha256": sha256(args.quant),
            "response_entities_sha256": sha256(args.response_entities),
            "output_mgf_sha256": sha256(output),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": (
            "This is metadata repair, not re-annotation. Sparse spectra remain explicit method-level "
            "abstentions and stay in the fixed response denominator."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
