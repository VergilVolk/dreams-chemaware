#!/usr/bin/env python3
"""Patient-level raw-UMI audit of host sialic-acid source hypotheses.

The frozen contract is in
docs/GSE178341_SIALIC_CELL_SOURCE_PREREGISTRATION_20260831.md. Cells are used
only to construct patient pseudobulks; every test is performed at PID level.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
TASKS = Path(__file__).resolve().parent
sys.path.insert(0, str(TASKS))
import audit_gse178341_nxpe1_mucinous_v1 as core  # noqa: E402


BASE = ROOT / "data/external/GSE178341_mucinous_secretory_audit"
DEFAULT_OUT = BASE / "sialic_cell_source_patient_pseudobulk_v1"
PREREG = ROOT / "docs/GSE178341_SIALIC_CELL_SOURCE_PREREGISTRATION_20260831.md"

MODULES = {
    "secretory_carrier": ("MUC2", "TFF3", "SPDEF", "FCGBP", "AGR2"),
    "cmp_neu5ac_capacity": ("GNE", "NANS", "CMAS", "SLC35A1"),
    "glycoconjugate_release": ("NEU1", "NEU3"),
    "salvage_catabolism": ("SLC17A5", "NPL"),
}
ENDPOINTS = [
    ("Epi", "secretory_carrier"),
    ("Epi", "cmp_neu5ac_capacity"),
    ("Myeloid", "cmp_neu5ac_capacity"),
    ("Epi", "glycoconjugate_release"),
    ("Myeloid", "glycoconjugate_release"),
    ("Epi", "salvage_catabolism"),
    ("Myeloid", "salvage_catabolism"),
]
GENES = tuple(dict.fromkeys(gene for genes in MODULES.values() for gene in genes))
SEED = 20260831


def build_profiles(frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    profiles = core.patient_profiles(frame, mask)
    if profiles["PID"].duplicated().any():
        raise RuntimeError("patient profiles are not unique")
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=core.DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    for path in (args.h5, core.META, core.CLUSTERS, core.PREFLIGHT, core.MATCHES, PREREG):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.h5.stat().st_size != core.EXPECTED_H5_SIZE:
        raise RuntimeError("H5 size mismatch")
    h5_hash = core.sha256(args.h5)
    if h5_hash != core.EXPECTED_H5_SHA256:
        raise RuntimeError("H5 SHA256 mismatch")

    preflight = json.loads(core.PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != "gse178341_nxpe1_mucinous_metadata_preflight_passed":
        raise RuntimeError("metadata preflight did not pass")
    if preflight["metadata_blind_matching"]["matches_sha256"] != core.sha256(core.MATCHES):
        raise RuntimeError("frozen match file hash mismatch")

    meta = pd.read_csv(core.META)
    clusters = pd.read_csv(core.CLUSTERS)
    if not meta["cellID"].astype(str).equals(clusters["sampleID"].astype(str)):
        raise RuntimeError("official metadata tables are not row-aligned")
    frame = pd.concat(
        [meta.reset_index(drop=True), clusters.drop(columns=["sampleID"]).reset_index(drop=True)],
        axis=1,
    )

    with h5py.File(args.h5, "r") as handle:
        group = handle["matrix"]
        barcodes = core.decode(group["barcodes"][:])
        expected_barcodes = frame["cellID"].astype(str).tolist()
        if barcodes != expected_barcodes:
            mismatches = sum(a != b for a, b in zip(barcodes, expected_barcodes))
            raise RuntimeError(f"H5 barcode order differs from metadata: mismatches={mismatches}")
        names = core.decode(group["features"]["name"][:])
        shape = tuple(int(value) for value in group["shape"][:])
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]), shape=shape
        )
    total_umi = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    if np.any(total_umi <= 0):
        raise RuntimeError("non-positive cell library size")
    upper = np.asarray([name.upper() for name in names], dtype=object)
    gene_rows = {gene: np.flatnonzero(upper == gene).tolist() for gene in GENES}
    available = [gene for gene in GENES if gene_rows[gene]]
    missing = [gene for gene in GENES if not gene_rows[gene]]
    if len(available) < 10:
        raise RuntimeError(f"too few fixed genes available: {available}")
    # sparse.sum(axis=0) is a dense np.matrix in current SciPy. Convert each
    # result explicitly so the fixed-gene cache has stable 2-D semantics.
    target = np.vstack(
        [np.asarray(matrix[gene_rows[gene], :].sum(axis=0)).ravel() for gene in available]
    ).astype(float, copy=False)
    del matrix

    pure = frame["SPECIMEN_TYPE"].eq("T") & frame["HistologicTypeSimple"].isin(
        (core.PURE_ADENO, core.PURE_MUC)
    )
    masks = {
        compartment: pure.to_numpy() & frame["clTopLevel"].eq(compartment).to_numpy()
        for compartment in ("Epi", "Myeloid")
    }
    patient_tables: dict[str, pd.DataFrame] = {}
    pseudobulk_rows: list[dict[str, object]] = []
    for compartment, mask in masks.items():
        profiles = build_profiles(frame, mask)
        rows: list[dict[str, object]] = []
        for profile in profiles.to_dict("records"):
            cell_mask = mask & frame["PID"].astype(str).eq(str(profile["PID"])).to_numpy()
            cells = int(cell_mask.sum())
            library = float(total_umi[cell_mask].sum())
            if cells < 30 or library <= 0:
                continue
            row = dict(profile)
            row.update({"compartment": compartment, "library_umi": library, "cells": cells})
            for index, gene in enumerate(available):
                count = float(target[index, cell_mask].sum())
                row[gene] = float(np.log2(count * 1_000_000.0 / library + 1.0))
                pseudobulk_rows.append(
                    {
                        "PID": profile["PID"],
                        "histology": profile["histology"],
                        "compartment": compartment,
                        "gene": gene,
                        "counts": count,
                        "library_umi": library,
                        "cells": cells,
                        "log2_cpm_plus1": row[gene],
                    }
                )
            rows.append(row)
        patient = pd.DataFrame(rows).sort_values("PID").reset_index(drop=True)
        expected = {core.PURE_MUC: 6, core.PURE_ADENO: 53}
        observed = patient["histology"].value_counts().to_dict()
        if observed != expected:
            raise RuntimeError(f"{compartment} patient counts changed: {observed}")
        for module, genes in MODULES.items():
            present = [gene for gene in genes if gene in patient.columns]
            if len(present) < max(1, len(genes) - 1):
                raise RuntimeError(f"{compartment}/{module} has insufficient genes: {present}")
            z = []
            for gene in present:
                value = patient[gene].to_numpy(float)
                z.append((value - value.mean()) / max(value.std(), 1e-12))
            patient[module] = np.column_stack(z).mean(axis=1)
        patient_tables[compartment] = patient

    matches = pd.read_csv(core.MATCHES, dtype=str)
    endpoint_rows = []
    matched_rows = []
    loo_rows = []
    for endpoint_index, (compartment, module) in enumerate(ENDPOINTS):
        patient = patient_tables[compartment]
        primary = core.group_result(
            patient, module, "all_pure_tumours", False, SEED + endpoint_index
        )
        right = patient[patient["site"].eq("right")].reset_index(drop=True)
        right_result = core.group_result(
            right, module, "right_colon_MMR_stratified", True, SEED + 100 + endpoint_index
        )
        endpoint_rows.append(
            {
                "compartment": compartment,
                "module": module,
                **{f"primary_{key}": value for key, value in primary.items() if key != "cohort"},
                **{f"right_{key}": value for key, value in right_result.items() if key != "cohort"},
            }
        )
        lookup = patient.set_index("PID")
        contrasts = []
        cases = []
        for case, group in matches.groupby("case_PID", sort=True):
            controls = group["control_PID"].tolist()
            if case not in lookup.index or not set(controls) <= set(lookup.index):
                raise RuntimeError(f"missing frozen matched PID for {compartment}/{module}")
            contrast = float(lookup.loc[case, module] - lookup.loc[controls, module].mean())
            contrasts.append(contrast)
            cases.append(case)
        matched_rows.append(
            {
                "compartment": compartment,
                "module": module,
                "n_cases": len(contrasts),
                "positive_cases": int(np.sum(np.asarray(contrasts) > 0)),
                "mean_contrast": float(np.mean(contrasts)),
                "median_contrast": float(np.median(contrasts)),
                "exact_sign_flip_p": core.exact_sign_flip(np.asarray(contrasts)),
            }
        )
        for omitted in cases:
            retained = [value for case, value in zip(cases, contrasts) if case != omitted]
            loo_rows.append(
                {
                    "compartment": compartment,
                    "module": module,
                    "omitted_case": omitted,
                    "mean_matched_contrast": float(np.mean(retained)),
                }
            )

    q_values = core.bh([row["primary_permutation_p"] for row in endpoint_rows])
    for row, q_value in zip(endpoint_rows, q_values):
        row["primary_BH_q_across_7"] = q_value
        matching = next(
            item for item in matched_rows
            if item["compartment"] == row["compartment"] and item["module"] == row["module"]
        )
        loo = [
            item["mean_matched_contrast"] for item in loo_rows
            if item["compartment"] == row["compartment"] and item["module"] == row["module"]
        ]
        row["leave_one_case_out_min_matched_contrast"] = float(min(loo))
        row["support_gate"] = bool(
            row["primary_bootstrap_95ci"][0] > 0
            and q_value < 0.10
            and min(loo) > 0
            and matching["positive_cases"] >= 5
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(pseudobulk_rows).to_csv(args.output_dir / "patient_gene_pseudobulk.csv", index=False)
    for compartment, table in patient_tables.items():
        table.to_csv(args.output_dir / f"{compartment.lower()}_patient_matrix.csv", index=False)
    pd.DataFrame(endpoint_rows).to_csv(args.output_dir / "fixed_endpoint_results.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(args.output_dir / "matched_endpoint_results.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(args.output_dir / "leave_one_case_out.csv", index=False)

    report = {
        "status": "gse178341_sialic_cell_source_patient_pseudobulk_complete",
        "formal": True,
        "patients": {"pure_mucinous": 6, "pure_adenocarcinoma": 53},
        "compartments": ["Epi", "Myeloid"],
        "available_fixed_genes": available,
        "missing_fixed_genes": missing,
        "fixed_endpoints": endpoint_rows,
        "matched_endpoints": matched_rows,
        "supporting_endpoints": [
            f"{row['compartment']}|{row['module']}" for row in endpoint_rows if row["support_gate"]
        ],
        "provenance": {
            "h5_sha256": h5_hash,
            "metadata_sha256": core.sha256(core.META),
            "clusters_sha256": core.sha256(core.CLUSTERS),
            "matches_sha256": core.sha256(core.MATCHES),
            "preregistration_sha256": core.sha256(PREREG),
            "script_sha256": core.sha256(Path(__file__)),
        },
        "claim_limit": (
            "Patient-level host transcriptional source context only. The audit cannot establish Neu5Ac "
            "biochemical source, enzyme activity, microbial contribution, glycan destination, or flux."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    plot = pd.DataFrame(endpoint_rows)
    labels = [f"{row.compartment}\n{row.module}" for row in plot.itertuples()]
    effects = plot["primary_mean_difference"].to_numpy(float)
    lower = np.asarray([value[0] for value in plot["primary_bootstrap_95ci"]], dtype=float)
    upper_ci = np.asarray([value[1] for value in plot["primary_bootstrap_95ci"]], dtype=float)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    positions = np.arange(len(plot))
    axis.axhline(0, color="black", lw=1)
    axis.errorbar(
        positions, effects, yerr=np.vstack([effects - lower, upper_ci - effects]),
        fmt="o", color="#326891", ecolor="#8AA6B8", capsize=4,
    )
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel("Mucinous - conventional module z score")
    axis.set_title("GSE178341 patient-level host sialic-source audit")
    figure.tight_layout()
    figure.savefig(args.output_dir / "sialic_cell_source_audit.png", dpi=220)
    plt.close(figure)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
