#!/usr/bin/env python3
"""Freeze a phenotype-blind strict-library partition for one Pan-ReDU panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from analyze_msv000088023_hfd_unknown_response import library_partition, parse_mgf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-object", required=True, type=Path)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--library-manifest", required=True, type=Path)
    parser.add_argument("--library-mgf", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    parser.add_argument("--fragment-da", type=float, default=0.02)
    parser.add_argument("--minimum-wse", type=float, default=0.70)
    parser.add_argument("--minimum-matched-peaks", type=int, default=6)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    panel_report_path = args.panel_object / "report.json"
    panel_report = json.loads(panel_report_path.read_text(encoding="utf-8"))
    if panel_report.get("status") != "development_consumed_abundance_matrix_frozen":
        raise RuntimeError("panel abundance matrix is not frozen")
    mgf_path = args.panel_object / "representative_spectra.mgf"
    manifest_path = args.panel_object / "anchor_manifest.tsv"
    anchors = pd.read_csv(manifest_path, sep="\t", dtype={"anchor_id": str})
    spectra = parse_mgf(mgf_path)
    anchor_ids = anchors.anchor_id.astype(str).tolist()
    if len(anchor_ids) != len(set(anchor_ids)) or set(anchor_ids) != set(spectra):
        raise RuntimeError("representative spectra do not exactly cover panel anchors")
    partition = library_partition(
        spectra, anchor_ids, args.library_manifest, args.library_mgf,
        args.precursor_ppm, args.fragment_da, args.minimum_wse, args.minimum_matched_peaks,
    ).rename(columns={"feature_id": "anchor_id"})
    partition.insert(0, "panel_id", args.panel_id)
    partition["identity_status"] = "unresolved_entity"
    partition.loc[partition.library_supported.astype(bool), "identity_status"] = "library_supported"
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "identity_partition.tsv.gz"
    partition.to_csv(output, sep="\t", index=False, compression="gzip")
    supported = int(partition.library_supported.astype(bool).sum())
    report = {
        "schema": "pan_redu_panel_strict_library_partition_v1",
        "status": "PHENOTYPE_BLIND_STRICT_LIBRARY_PARTITION_FROZEN",
        "panel_id": args.panel_id,
        "denominators": {
            "panel_anchors": len(partition), "library_supported": supported,
            "unresolved_entity": len(partition) - supported,
            "library_supported_fraction": supported / len(partition),
        },
        "parameters": {
            "precursor_ppm": args.precursor_ppm, "fragment_da": args.fragment_da,
            "minimum_wse": args.minimum_wse, "minimum_matched_peaks": args.minimum_matched_peaks,
            "reference_library": "frozen GNPS Gold/Silver positive [M+H]+ development library",
        },
        "leakage_status": "representative MS/MS and frozen library only; no sample label, contrast effect, retrieval truth or pathway read",
        "interpretation": {
            "library_supported": "strict spectral-library control, not MSI level-1 confirmation",
            "unresolved_entity": "no qualifying hit under this frozen library contract; not proof of novelty",
        },
        "provenance": {
            "panel_report_sha256": sha256(panel_report_path), "panel_mgf_sha256": sha256(mgf_path),
            "anchor_manifest_sha256": sha256(manifest_path), "library_manifest_sha256": sha256(args.library_manifest),
            "library_mgf_sha256": sha256(args.library_mgf), "partition_sha256": sha256(output),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
