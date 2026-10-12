#!/usr/bin/env python
"""Genotype-blind robustness panel for the MSV000096214 MS1 positive-control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_msv000096214_genetic_ms1_baseline import run_world


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    abundance_path = args.input / "structure_hidden_abundance.csv.gz"
    qc_path = args.input / "pooled_qc_abundance.csv.gz"
    manifest_path = args.input / "sample_manifest.csv.gz"
    abundance = pd.read_csv(abundance_path)
    qc = pd.read_csv(qc_path).set_index("feature_id").loc[abundance["feature_id"]]
    manifest = pd.read_csv(manifest_path)
    sample_ids = manifest["sample_id"].tolist()
    values = abundance[sample_ids].to_numpy(float)
    modes = abundance["ion_mode"].astype(str).to_numpy()
    groups = manifest["genotype"].astype(str).to_numpy()
    qc_values = qc[["pooled_qc_1", "pooled_qc_2", "pooled_qc_3"]].to_numpy(float)
    qc_cv = np.std(qc_values, axis=1, ddof=1) / np.maximum(np.mean(qc_values, axis=1), 1e-12)

    masks = {
        "all_920": np.ones(len(abundance), dtype=bool),
        "pooled_qc_cv_le_20pct": qc_cv <= 0.20,
        "pooled_qc_cv_le_30pct": qc_cv <= 0.30,
        "positive_mode_only": modes == "ESI+",
        "negative_mode_only": modes == "ESI-",
    }
    worlds = {}
    for name, mask in masks.items():
        result = run_world(values[mask], modes[mask], groups, median_center=True)
        worlds[name] = {
            "features": int(mask.sum()),
            "positive_features": int(np.count_nonzero(mask & (modes == "ESI+"))),
            "negative_features": int(np.count_nonzero(mask & (modes == "ESI-"))),
            "exact_test": result["exact_cluster_assignment_test"],
            "gates": result["gates"],
            "g3": result["g3"],
        }
    core_gate_names = list(worlds["all_920"]["gates"])
    consistent = all(
        all(bool(world["gates"][gate]) for gate in core_gate_names)
        for world in worlds.values()
    )
    report = {
        "status": "MSV000096214_MS1_ROBUSTNESS_AUDITED",
        "formal": True,
        "dataset": "MSV000096214",
        "data_role": "development/consumed",
        "sealed": False,
        "structure_or_annotation_used": False,
        "pooled_qc_cv": {
            "features": int(len(qc_cv)),
            "median": float(np.median(qc_cv)),
            "q90": float(np.quantile(qc_cv, 0.90)),
            "q95": float(np.quantile(qc_cv, 0.95)),
            "le_20pct": int(np.count_nonzero(qc_cv <= 0.20)),
            "le_30pct": int(np.count_nonzero(qc_cv <= 0.30)),
        },
        "worlds": worlds,
        "decision": {
            "all_five_prespecified_worlds_pass_same_gates": consistent,
            "pass_ms1_positive_control_robustness": consistent,
            "pass_to_spectral_increment": consistent,
        },
        "unresolved_design_risk": (
            "Supplement columns and raw filenames are genotype-blocked. The deposited metadata do not "
            "establish randomized injection order, and Table S1 is an author-processed MZmine table "
            "normalized by internal standard and seed weight. Robustness here does not remove possible "
            "run-order/batch confounding or author feature-universe selection; raw truth-blind reprocessing "
            "or acquisition-order evidence is required before calling the task unbiased discovery."
        ),
        "provenance": {
            "abundance_sha256": sha256(abundance_path),
            "pooled_qc_sha256": sha256(qc_path),
            "sample_manifest_sha256": sha256(manifest_path),
        },
        "claim_limit": (
            "Genotype-blind QC-threshold and polarity sensitivity of the structure-hidden MS1 positive "
            "control only. No MS/MS relationship or unknown-annotation increment was tested."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "pooled_qc_cv": report["pooled_qc_cv"],
        "worlds": {
            name: {"features": value["features"], "exact_test": value["exact_test"], "gates": value["gates"]}
            for name, value in worlds.items()
        },
        "decision": report["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
