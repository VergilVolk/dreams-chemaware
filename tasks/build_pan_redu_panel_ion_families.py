#!/usr/bin/env python3
"""Build phenotype-blind ion-family capacity controls for one Pan-ReDU panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from build_ad_phenotype_blind_ion_families import ION_FORM_OFFSETS, build_families
from build_ad_measurement_covariates import measurement_covariates
from encode_pan_redu_panel_spectra import load_mgf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-object", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rt-tolerance-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-codetected", type=int, default=20)
    parser.add_argument("--minimum-codetected-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-spearman", type=float, default=0.95)
    parser.add_argument("--minimum-adduct-spectral-similarity", type=float, default=0.80)
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    parser.add_argument("--constant-delta-tolerance-da", type=float, default=0.01)
    parser.add_argument("--maximum-family-size", type=int, default=6)
    parser.add_argument("--maximum-component-mz-span", type=float, default=40.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not (0 < args.minimum_codetected_fraction <= 1):
        raise ValueError("minimum codetected fraction must lie in (0,1]")

    panel_report_path = args.panel_object / "report.json"
    panel_report = json.loads(panel_report_path.read_text(encoding="utf-8"))
    if panel_report.get("status") != "development_consumed_abundance_matrix_frozen":
        raise RuntimeError("panel abundance matrix is not frozen")
    matrix_path = args.panel_object / "ms1_peak_height.npz"
    manifest_path = args.panel_object / "anchor_manifest.tsv"
    with np.load(matrix_path, allow_pickle=False) as packed:
        values = np.asarray(packed["values"], dtype=np.float64)
        anchor_ids = np.asarray(packed["anchor_id"], dtype=str)
        sample_ids = np.asarray(packed["sample_id"], dtype=str)
    anchors = pd.read_csv(manifest_path, sep="\t", dtype={"anchor_id": str})
    if values.shape != (len(anchor_ids), len(sample_ids)):
        raise RuntimeError("panel matrix dimensions are invalid")
    if not np.array_equal(anchor_ids, anchors.anchor_id.to_numpy(str)):
        raise RuntimeError("anchor order differs between matrix and manifest")
    with np.load(args.embeddings, allow_pickle=False) as packed:
        embedded_ids = np.asarray(packed["anchor_id"], dtype=str)
        embeddings = np.asarray(packed["embeddings"], dtype=np.float32)
    if len(embedded_ids) != len(set(embedded_ids)) or embeddings.shape[0] != len(embedded_ids):
        raise RuntimeError("embedding archive is invalid")
    embedding_lookup = dict(zip(embedded_ids, embeddings))
    minimum_codetected = max(
        20, args.minimum_codetected,
        int(math.ceil(args.minimum_codetected_fraction * len(sample_ids))),
    )
    assignments, edges = build_families(
        anchor_ids,
        anchors.precursor_mz.to_numpy(float),
        anchors.rt_seconds.to_numpy(float),
        values,
        rt_tolerance=args.rt_tolerance_seconds,
        ppm=args.precursor_ppm,
        delta_da=args.constant_delta_tolerance_da,
        minimum_codetected=minimum_codetected,
        minimum_spearman=args.minimum_spearman,
        embedding_lookup=embedding_lookup,
        minimum_adduct_spectral_similarity=args.minimum_adduct_spectral_similarity,
        maximum_family_size=args.maximum_family_size,
        maximum_component_mz_span=args.maximum_component_mz_span,
    )
    if len(assignments) != len(anchor_ids):
        raise RuntimeError("ion-family assignments do not cover the complete panel")
    spectrum_ids, spectra = load_mgf(args.panel_object / "representative_spectra.mgf")
    peak_count = {entity: int(np.asarray(record["peaks"]).shape[1]) for entity, record in zip(spectrum_ids, spectra)}
    covariates = pd.DataFrame({
        "entity_id": anchor_ids,
        "precursor_mz": anchors.precursor_mz.to_numpy(float),
        "retention_time": anchors.rt_seconds.to_numpy(float),
        **measurement_covariates(values),
        "ms2_peak_count": [peak_count.get(entity, 0) for entity in anchor_ids],
    })
    required_covariates = ["median_log2_intensity", "detection_rate", "log2_intensity_iqr", "ms2_peak_count"]
    if covariates[required_covariates].isna().any().any():
        raise RuntimeError("panel contains incomplete measurement-matching covariates")
    args.output_dir.mkdir(parents=True)
    assignment_path = args.output_dir / "ion_family_assignments.tsv.gz"
    edge_path = args.output_dir / "ion_family_edges.tsv.gz"
    covariate_path = args.output_dir / "measurement_covariates.tsv.gz"
    assignments.to_csv(assignment_path, sep="\t", index=False, compression="gzip")
    edges.to_csv(edge_path, sep="\t", index=False, compression="gzip")
    covariates.to_csv(covariate_path, sep="\t", index=False, compression="gzip")
    sizes = assignments.drop_duplicates("ion_family_id").ion_family_size
    report = {
        "schema": "pan_redu_panel_phenotype_blind_ion_families_v2_latent_mass_coherent",
        "status": "PHENOTYPE_BLIND_PANEL_ION_FAMILIES_FROZEN",
        "panel_id": args.panel_id,
        "denominators": {
            "anchors": len(anchor_ids), "samples": len(sample_ids), "family_edges": len(edges),
            "ion_families": int(assignments.ion_family_id.nunique()),
            "multi_entity_families": int((sizes > 1).sum()),
            "entities_in_multi_entity_families": int(assignments.ion_family_size.gt(1).sum()),
            "maximum_family_size": int(sizes.max()),
        },
        "parameters": {
            "rt_tolerance_seconds": args.rt_tolerance_seconds,
            "minimum_codetected_injections": minimum_codetected,
            "minimum_codetected_fraction": args.minimum_codetected_fraction,
            "minimum_spearman": args.minimum_spearman,
            "minimum_adduct_official_dreams_similarity": args.minimum_adduct_spectral_similarity,
            "precursor_ppm": args.precursor_ppm,
            "constant_delta_tolerance_da": args.constant_delta_tolerance_da,
            "maximum_family_size": args.maximum_family_size,
            "maximum_component_rt_span_seconds": args.rt_tolerance_seconds,
            "maximum_component_mz_span": args.maximum_component_mz_span,
            "allowed_latent_ion_form_offsets_da": ION_FORM_OFFSETS,
            "component_requires_one_shared_latent_protonated_monoisotopic_mass": True,
        },
        "capacity_rule": "total family capacity one is divided among members eligible in each scored evidence world",
        "leakage_status": "m/z, RT, phenotype-blind abundance and frozen embeddings only; no contrast membership, effect or retrieval truth",
        "provenance": {
            "panel_report_sha256": sha256(panel_report_path), "matrix_sha256": sha256(matrix_path),
            "anchor_manifest_sha256": sha256(manifest_path), "embeddings_sha256": sha256(args.embeddings),
            "assignments_sha256": sha256(assignment_path), "edges_sha256": sha256(edge_path),
            "measurement_covariates_sha256": sha256(covariate_path),
        },
        "claim_boundary": "Ion families control redundant capacity; they are not metabolite identities.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
