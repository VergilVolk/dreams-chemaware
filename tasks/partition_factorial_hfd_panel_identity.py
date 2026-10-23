#!/usr/bin/env python3
"""Freeze a polarity-matched strict-library partition for one factorial-HFD panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from ms_entropy import calculate_entropy_similarity

from analyze_msv000088023_hfd_unknown_response import (
    cleaned_peaks,
    library_partition,
    matched_peak_count,
    parse_mgf,
)
from partition_msv000101150_unified_negative_library import parse_library


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_polarity(assay_file: str) -> str:
    upper = assay_file.upper()
    positive, negative = "POS" in upper, "NEG" in upper
    if positive == negative:
        raise ValueError(f"cannot infer a unique polarity from {assay_file}")
    return "positive" if positive else "negative"


def negative_library_partition(
    queries: dict[str, dict[str, object]],
    anchor_ids: list[str],
    library_path: Path,
    precursor_ppm: float,
    fragment_da: float,
    minimum_wse: float,
    minimum_matches: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    all_records = parse_library(library_path)
    library = [
        record for record in all_records
        if str(record.get("adduct", "")).strip() == "[M-H]-"
        and bool(str(record.get("inchikey", "")).strip())
    ]
    if not library:
        raise RuntimeError("negative reference contains no strict [M-H]- records")
    library_mz = np.asarray([record["precursor_mz"] for record in library], dtype=float)
    order = np.argsort(library_mz)
    sorted_mz = library_mz[order]
    rows: list[dict[str, object]] = []
    for anchor_id in anchor_ids:
        query = queries[anchor_id]
        precursor = float(query["precursor_mz"])
        delta = precursor * precursor_ppm * 1e-6
        candidates = order[
            np.searchsorted(sorted_mz, precursor - delta, side="left"):
            np.searchsorted(sorted_mz, precursor + delta, side="right")
        ]
        query_peaks = cleaned_peaks(query, fragment_da)
        best: dict[str, object] | None = None
        for library_index in candidates:
            candidate = library[int(library_index)]
            candidate_peaks = cleaned_peaks(candidate, fragment_da)
            matches = matched_peak_count(query_peaks, candidate_peaks, fragment_da)
            wse = float(calculate_entropy_similarity(
                query_peaks, candidate_peaks,
                ms2_tolerance_in_da=fragment_da, clean_spectra=False,
            ))
            if best is None or (wse, matches) > (float(best["wse"]), int(best["matched_peaks"])):
                best = {
                    "wse": wse,
                    "matched_peaks": matches,
                    "library_name": candidate.get("name", ""),
                    "library_smiles": candidate.get("smiles", ""),
                    "library_inchikey": candidate.get("inchikey", ""),
                    "library_adduct": candidate.get("adduct", ""),
                    "library_source": candidate.get("source", ""),
                }
        row: dict[str, object] = {
            "anchor_id": anchor_id,
            "precursor_mz": precursor,
            "library_candidates_10ppm": int(len(candidates)),
        }
        if best is None:
            row.update({
                "wse": math.nan, "matched_peaks": 0, "library_name": "",
                "library_smiles": "", "library_inchikey": "",
                "library_adduct": "", "library_source": "",
            })
        else:
            row.update(best)
        row["library_supported"] = bool(
            np.isfinite(float(row["wse"]))
            and float(row["wse"]) >= minimum_wse
            and int(row["matched_peaks"]) >= minimum_matches
            and bool(str(row["library_inchikey"]).strip())
        )
        rows.append(row)
    return pd.DataFrame(rows), {
        "all_negative_records": len(all_records),
        "strict_m_h_records": len(library),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-object", required=True, type=Path)
    parser.add_argument("--panel-registry", required=True, type=Path)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--positive-library-manifest", required=True, type=Path)
    parser.add_argument("--positive-library-mgf", required=True, type=Path)
    parser.add_argument("--negative-library-mgf", required=True, type=Path)
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
    registry = pd.read_csv(args.panel_registry, sep="\t", dtype=str).set_index("panel_id", verify_integrity=True)
    if args.panel_id not in registry.index:
        raise ValueError(f"panel absent from registry: {args.panel_id}")
    polarity = infer_polarity(str(registry.loc[args.panel_id, "assay_file"]))

    mgf_path = args.panel_object / "representative_spectra.mgf"
    anchor_path = args.panel_object / "anchor_manifest.tsv"
    anchors = pd.read_csv(anchor_path, sep="\t", dtype={"anchor_id": str})
    spectra = parse_mgf(mgf_path)
    anchor_ids = anchors.anchor_id.astype(str).tolist()
    if len(anchor_ids) != len(set(anchor_ids)) or set(anchor_ids) != set(spectra):
        raise RuntimeError("representative spectra do not exactly cover panel anchors")

    reference_counts: dict[str, int] = {}
    if polarity == "positive":
        partition = library_partition(
            spectra, anchor_ids, args.positive_library_manifest,
            args.positive_library_mgf, args.precursor_ppm, args.fragment_da,
            args.minimum_wse, args.minimum_matched_peaks,
        ).rename(columns={"feature_id": "anchor_id"})
        reference_name = "GNPS Gold/Silver positive [M+H]+ development library"
        reference_path = args.positive_library_mgf
    else:
        partition, reference_counts = negative_library_partition(
            spectra, anchor_ids, args.negative_library_mgf,
            args.precursor_ppm, args.fragment_da,
            args.minimum_wse, args.minimum_matched_peaks,
        )
        reference_name = "GNPS-source unified negative strict [M-H]- development library"
        reference_path = args.negative_library_mgf

    partition.insert(0, "panel_id", args.panel_id)
    partition.insert(1, "polarity", polarity)
    partition["identity_status"] = "unresolved_entity"
    partition.loc[partition.library_supported.astype(bool), "identity_status"] = "library_supported"
    if polarity == "negative" and partition.loc[partition.library_supported.astype(bool), "library_adduct"].ne("[M-H]-").any():
        raise RuntimeError("negative supported hit escaped the [M-H]- policy")

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "identity_partition.tsv.gz"
    partition.to_csv(
        output, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    supported = int(partition.library_supported.astype(bool).sum())
    report = {
        "schema": "factorial_hfd_polarity_matched_identity_partition_v1",
        "status": "PHENOTYPE_BLIND_POLARITY_MATCHED_LIBRARY_PARTITION_FROZEN",
        "panel_id": args.panel_id,
        "polarity": polarity,
        "denominators": {
            "panel_anchors": len(partition),
            "library_supported": supported,
            "unresolved_entity": len(partition) - supported,
            "reference_records": reference_counts,
        },
        "parameters": {
            "precursor_ppm": args.precursor_ppm,
            "fragment_da": args.fragment_da,
            "minimum_wse": args.minimum_wse,
            "minimum_matched_peaks": args.minimum_matched_peaks,
            "reference_library": reference_name,
        },
        "leakage_status": "polarity, representative MS/MS and frozen library only; no sample label, contrast effect, rank or response truth was read",
        "interpretation": {
            "library_supported": "strict same-polarity spectral-library control, not MSI level-1 confirmation",
            "unresolved_entity": "no qualifying same-polarity hit under this policy; not proof of novelty",
        },
        "provenance": {
            "panel_report_sha256": sha256(panel_report_path),
            "panel_registry_sha256": sha256(args.panel_registry),
            "panel_mgf_sha256": sha256(mgf_path),
            "anchor_manifest_sha256": sha256(anchor_path),
            "reference_library_sha256": sha256(reference_path),
            "positive_manifest_sha256": sha256(args.positive_library_manifest),
            "partition_sha256": sha256(output),
            "implementation_sha256": sha256(Path(__file__).resolve()),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
