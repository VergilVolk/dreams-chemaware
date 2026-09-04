#!/usr/bin/env python
"""Preregistered patient-level GSE178341 mucinous/NXPE1 expression audit.

The script reads the official 10x HDF5 count matrix and author-provided cell
metadata. Cells are used only to construct patient pseudobulks; every inferential
unit is a PID. The analysis follows
docs/GSE178341_NXPE1_MUCINOUS_PREREGISTRATION_20260831.md.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external/GSE178341_mucinous_secretory_audit"
DEFAULT_H5 = BASE / "GSE178341_crc10x_full_c295v4_submit.h5"
META = BASE / "GSE178341_crc10x_full_c295v4_submit_metatables.csv.gz"
CLUSTERS = BASE / "GSE178341_crc10x_full_c295v4_submit_cluster.csv.gz"
PREFLIGHT = BASE / "metadata_preflight_v1.json"
MATCHES = BASE / "metadata_matches_v1.csv"
DEFAULT_OUT = BASE / "nxpe1_mucinous_patient_pseudobulk_v1"

PURE_ADENO = "Adenocarcinoma"
PURE_MUC = "Adenocarcinoma;Mucinous"
GOBLET_CODES = ("cE02", "cE06", "cE07", "cE08")
PRIMARY_GENE = "NXPE1"
SECRETORY = ("MUC2", "TFF3", "SPDEF", "FCGBP", "AGR2")
SIALIC_BACKGROUND = ("GNE", "NANS", "CMAS", "SLC35A1")
OAC_CONTEXT = ("CASD1", "SIAE")
GENES = (PRIMARY_GENE,) + SECRETORY + SIALIC_BACKGROUND + OAC_CONTEXT
EXPECTED_H5_SIZE = 1_203_550_558
EXPECTED_H5_SHA256 = "f435bb2651ff5297d0c24a99daf58850ed67ae1ed6c5ef05fad48fa3f0186670"
SEED = 20260831


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]


def bootstrap_group_difference(case: np.ndarray, control: np.ndarray, resamples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=float)
    for start in range(0, resamples, 2000):
        stop = min(start + 2000, resamples)
        n = stop - start
        a = case[rng.integers(0, len(case), size=(n, len(case)))].mean(axis=1)
        b = control[rng.integers(0, len(control), size=(n, len(control)))].mean(axis=1)
        values[start:stop] = a - b
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def permutation_group_difference(
    frame: pd.DataFrame,
    value_column: str,
    resamples: int,
    seed: int,
    strata: tuple[str, ...] = (),
) -> float:
    observed = float(
        frame.loc[frame["histology"].eq(PURE_MUC), value_column].mean()
        - frame.loc[frame["histology"].eq(PURE_ADENO), value_column].mean()
    )
    labels = frame["histology"].eq(PURE_MUC).to_numpy()
    values = frame[value_column].to_numpy(float)
    rng = np.random.default_rng(seed)
    exceed = 1
    if strata:
        groups = [group.index.to_numpy() for _, group in frame.groupby(list(strata), dropna=False, sort=True)]
        case_counts = [int(labels[index].sum()) for index in groups]
    else:
        groups = [np.arange(len(frame))]
        case_counts = [int(labels.sum())]
    for _ in range(resamples):
        permuted = np.zeros(len(frame), dtype=bool)
        for index, count in zip(groups, case_counts):
            if count:
                permuted[rng.choice(index, size=count, replace=False)] = True
        statistic = float(values[permuted].mean() - values[~permuted].mean())
        exceed += abs(statistic) >= abs(observed) - 1e-12
    return exceed / (resamples + 1)


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        exceed += statistic >= observed - 1e-12
        total += 1
    return exceed / total


def bh(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    adjusted = np.empty_like(array)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(array) - reverse_rank + 1
        running = min(running, float(array[index]) * len(array) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def hc3_group_coefficient(frame: pd.DataFrame, outcome: str, include_secretory: bool) -> dict[str, object]:
    work = frame.dropna(subset=[outcome]).copy()
    columns = [np.ones(len(work)), work["histology"].eq(PURE_MUC).to_numpy(float)]
    names = ["intercept", "mucinous"]
    fixed = [
        ("MMRd", work["MMRStatus"].eq("MMRd").to_numpy(float)),
        ("right", work["site"].eq("right").to_numpy(float)),
        ("SC3Pv3_fraction", work["frac_sc3pv3"].to_numpy(float)),
        ("unsorted_fraction", work["frac_unsorted"].to_numpy(float)),
        ("CD45pMACS_fraction", work["frac_cd45pmacs"].to_numpy(float)),
        ("CCPM_Regev_team", work["team"].eq("CCPM_Regev").to_numpy(float)),
        ("MGH", work["hospital"].eq("MGH").to_numpy(float)),
    ]
    for name, value in fixed:
        if np.std(value) > 1e-12:
            columns.append(value)
            names.append(name)
    for name in ("age", "stage"):
        value = work[name].to_numpy(float)
        value = (value - np.mean(value)) / max(np.std(value), 1e-12)
        columns.append(value)
        names.append(name)
    if include_secretory:
        value = work["SECRETORY_COMPOSITE"].to_numpy(float)
        columns.append(value)
        names.append("SECRETORY_COMPOSITE")
    x = np.column_stack(columns)
    y = work[outcome].to_numpy(float)
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    scaled = residual / np.maximum(1.0 - leverage, 1e-6)
    meat = x.T @ ((scaled * scaled)[:, None] * x)
    covariance = xtx_inv @ meat @ xtx_inv
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    target = names.index("mucinous")
    statistic = float(beta[target] / max(standard_error[target], 1e-12))
    degrees = max(len(y) - x.shape[1], 1)
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees))
    return {
        "n_patients": len(work),
        "parameters": names,
        "mucinous_beta": float(beta[target]),
        "HC3_se": float(standard_error[target]),
        "t": statistic,
        "df": degrees,
        "p": p_value,
        "condition_number": float(np.linalg.cond(x)),
    }


def patient_profiles(frame: pd.DataFrame, compartment_mask: np.ndarray) -> pd.DataFrame:
    rows = []
    subset = frame.loc[compartment_mask]
    for patient, group in subset.groupby("PID", sort=True, observed=True):
        first = group.sort_values("PatientTypeID").iloc[0]
        process = group["PROCESSING_TYPE"].value_counts(normalize=True)
        chemistry = group["SINGLECELL_TYPE"].value_counts(normalize=True)
        stage_text = str(first["TumorStage"]).lower()
        stage = next((value for token, value in (("t4", 4.0), ("t3", 3.0), ("t2", 2.0), ("t1", 1.0)) if token in stage_text), 3.0)
        rows.append({
            "PID": str(patient),
            "histology": first["HistologicTypeSimple"],
            "MMRStatus": first["MMRStatus"],
            "site": first["TissueSiteSimple"],
            "team": first["TISSUE_PROCESSING_TEAM"],
            "hospital": first["SOURCE_HOSPITAL"],
            "age": float(first["Age"]),
            "stage": stage,
            "frac_sc3pv3": float(chemistry.get("SC3Pv3", 0.0)),
            "frac_unsorted": float(process.get("unsorted", 0.0)),
            "frac_cd45pmacs": float(process.get("CD45pMACS", 0.0)),
            "frac_livemacs": float(process.get("LiveMACS", 0.0)),
            "frac_mixed": float(process.get("mixUnsortCD45MACS", 0.0)),
            "cells": int(len(group)),
        })
    return pd.DataFrame(rows)


def group_result(frame: pd.DataFrame, value: str, label: str, stratified: bool, seed: int) -> dict[str, object]:
    case = frame.loc[frame["histology"].eq(PURE_MUC), value].to_numpy(float)
    control = frame.loc[frame["histology"].eq(PURE_ADENO), value].to_numpy(float)
    return {
        "cohort": label,
        "n_mucinous": len(case),
        "n_adenocarcinoma": len(control),
        "mucinous_mean": float(np.mean(case)),
        "adenocarcinoma_mean": float(np.mean(control)),
        "mean_difference": float(np.mean(case) - np.mean(control)),
        "median_difference": float(np.median(case) - np.median(control)),
        "bootstrap_95ci": bootstrap_group_difference(case, control, 20000, seed),
        "permutation_p": permutation_group_difference(
            frame.reset_index(drop=True), value, 100000, seed + 17, ("MMRStatus",) if stratified else ()
        ),
        "permutation_strata": ["MMRStatus"] if stratified else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    for path in (args.h5, META, CLUSTERS, PREFLIGHT, MATCHES):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.h5.stat().st_size != EXPECTED_H5_SIZE:
        raise RuntimeError(f"H5 size mismatch: {args.h5.stat().st_size} != {EXPECTED_H5_SIZE}")
    h5_sha256 = sha256(args.h5)
    if h5_sha256 != EXPECTED_H5_SHA256:
        raise RuntimeError(f"H5 SHA256 mismatch: {h5_sha256} != {EXPECTED_H5_SHA256}")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != "gse178341_nxpe1_mucinous_metadata_preflight_passed":
        raise RuntimeError("metadata preflight did not pass")
    if preflight["metadata_blind_matching"]["matches_sha256"] != sha256(MATCHES):
        raise RuntimeError("frozen match file hash mismatch")

    meta = pd.read_csv(META)
    clusters = pd.read_csv(CLUSTERS)
    if not meta["cellID"].astype(str).equals(clusters["sampleID"].astype(str)):
        raise RuntimeError("official metadata tables are not row-aligned")
    frame = pd.concat([meta.reset_index(drop=True), clusters.drop(columns=["sampleID"]).reset_index(drop=True)], axis=1)

    with h5py.File(args.h5, "r") as handle:
        if "matrix" not in handle:
            raise RuntimeError(f"not a 10x HDF5 matrix; top-level keys={list(handle.keys())}")
        matrix_group = handle["matrix"]
        barcodes = decode(matrix_group["barcodes"][:])
        if barcodes != frame["cellID"].astype(str).tolist():
            mismatches = sum(a != b for a, b in zip(barcodes, frame["cellID"].astype(str)))
            raise RuntimeError(f"H5 barcode order differs from official metadata: mismatches={mismatches}")
        names = decode(matrix_group["features"]["name"][:])
        shape = tuple(int(value) for value in matrix_group["shape"][:])
        if shape != (len(names), len(barcodes)):
            raise RuntimeError(f"H5 shape mismatch: {shape}")
        data = matrix_group["data"][:]
        indices = matrix_group["indices"][:]
        indptr = matrix_group["indptr"][:]
    matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
    total_umi = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    if np.any(total_umi <= 0):
        raise RuntimeError(f"non-positive library sizes: {int(np.sum(total_umi <= 0))}")
    upper_names = np.asarray([name.upper() for name in names], dtype=object)
    gene_rows = {gene: np.flatnonzero(upper_names == gene).tolist() for gene in GENES}
    if not gene_rows[PRIMARY_GENE]:
        raise RuntimeError("NXPE1 is absent from the official feature names")
    available = [gene for gene in GENES if gene_rows[gene]]
    # scipy returns a dense np.matrix for sparse.sum(axis=0); make every row an
    # explicit 1-D ndarray before stacking. Passing those matrices to
    # scipy.sparse.vstack is version-dependent and fails on current SciPy.
    target_matrix = np.vstack(
        [np.asarray(matrix[gene_rows[gene], :].sum(axis=0)).ravel() for gene in available]
    ).astype(float, copy=False)
    del matrix, data, indices, indptr

    pure = frame["SPECIMEN_TYPE"].eq("T") & frame["HistologicTypeSimple"].isin((PURE_ADENO, PURE_MUC))
    compartments = {
        "broad_epithelial": pure.to_numpy() & frame["clTopLevel"].eq("Epi").to_numpy(),
        "goblet_family": pure.to_numpy() & frame["clTopLevel"].eq("Epi").to_numpy()
        & frame["cl295v11SubShort"].isin(GOBLET_CODES).to_numpy(),
    }
    pseudobulk = []
    profile_tables: dict[str, pd.DataFrame] = {}
    for compartment, mask in compartments.items():
        profiles = patient_profiles(frame, mask)
        profile_tables[compartment] = profiles
        for _, profile in profiles.iterrows():
            cell_mask = mask & frame["PID"].astype(str).eq(profile["PID"]).to_numpy()
            library = float(total_umi[cell_mask].sum())
            if int(cell_mask.sum()) < 30 or library <= 0:
                continue
            for gene_index, gene in enumerate(available):
                count = float(target_matrix[gene_index, cell_mask].sum())
                positive_cells = int(np.sum(target_matrix[gene_index, cell_mask] > 0))
                row = profile.to_dict()
                row.update({
                    "compartment": compartment,
                    "gene": gene,
                    "counts": count,
                    "library_umi": library,
                    "positive_cells": positive_cells,
                    "detection_fraction": positive_cells / int(cell_mask.sum()),
                    "log2_cpm_plus1": float(np.log2(count * 1_000_000.0 / library + 1.0)),
                })
                pseudobulk.append(row)
    pseudobulk_table = pd.DataFrame(pseudobulk)

    patient_tables: dict[str, pd.DataFrame] = {}
    for compartment in compartments:
        subset = pseudobulk_table[pseudobulk_table["compartment"].eq(compartment)]
        values = subset.pivot(index="PID", columns="gene", values="log2_cpm_plus1")
        profiles = subset.drop_duplicates("PID").set_index("PID")[
            ["histology", "MMRStatus", "site", "team", "hospital", "age", "stage", "frac_sc3pv3", "frac_unsorted", "frac_cd45pmacs", "frac_livemacs", "frac_mixed", "cells"]
        ]
        patient = profiles.join(values)
        for axis_name, genes in (("SECRETORY_COMPOSITE", SECRETORY), ("SIALIC_BACKGROUND", SIALIC_BACKGROUND)):
            present = [gene for gene in genes if gene in patient.columns]
            standardized = []
            for gene in present:
                value = patient[gene].to_numpy(float)
                standardized.append((value - np.mean(value)) / max(np.std(value), 1e-12))
            patient[axis_name] = np.mean(np.column_stack(standardized), axis=1) if standardized else np.nan
        patient_tables[compartment] = patient.reset_index()

    matches = pd.read_csv(MATCHES, dtype=str)
    results = []
    matched_results = []
    regressions = []
    leave_one_out = []
    for compartment, patient in patient_tables.items():
        for gene_index, gene in enumerate(available):
            all_result = group_result(patient, gene, "all_pure_tumours", False, SEED + gene_index)
            right = patient[patient["site"].eq("right")].reset_index(drop=True)
            right_result = group_result(right, gene, "right_colon_MMR_stratified", True, SEED + 100 + gene_index)
            for item in (all_result, right_result):
                item.update({"compartment": compartment, "gene": gene})
                results.append(item)

        # The frozen 18 controls were selected for the primary broad-epithelial analysis.
        # Several have fewer than 30 cells in the narrower goblet-family gate, so re-matching
        # that sensitivity subset would be post hoc. It is intentionally not done.
        if compartment == "broad_epithelial":
            lookup = patient.set_index("PID")
            for outcome in (PRIMARY_GENE, "SECRETORY_COMPOSITE", "SIALIC_BACKGROUND"):
                contrasts = []
                per_case = []
                for case, rows in matches.groupby("case_PID", sort=True):
                    controls = rows["control_PID"].tolist()
                    if case not in lookup.index or not set(controls) <= set(lookup.index):
                        raise RuntimeError(f"missing frozen matched patient in {compartment}: {case} / {controls}")
                    contrast = float(lookup.loc[case, outcome] - lookup.loc[controls, outcome].mean())
                    contrasts.append(contrast)
                    per_case.append({"case_PID": case, "controls": controls, "contrast": contrast})
                matched_results.append({
                    "compartment": compartment,
                    "outcome": outcome,
                    "n_cases": len(contrasts),
                    "positive_cases": int(np.sum(np.asarray(contrasts) > 0)),
                    "mean_contrast": float(np.mean(contrasts)),
                    "median_contrast": float(np.median(contrasts)),
                    "exact_sign_flip_p": exact_sign_flip(np.asarray(contrasts)),
                    "per_case": per_case,
                })
                if outcome == PRIMARY_GENE:
                    for omitted in sorted(matches["case_PID"].unique()):
                        retained = [row["contrast"] for row in per_case if row["case_PID"] != omitted]
                        leave_one_out.append({
                            "compartment": compartment,
                            "omitted_case": omitted,
                            "mean_matched_contrast": float(np.mean(retained)),
                        })
        regressions.append({
            "compartment": compartment,
            "model": "fixed_clinicotechnical",
            **hc3_group_coefficient(patient, PRIMARY_GENE, False),
        })
        regressions.append({
            "compartment": compartment,
            "model": "fixed_clinicotechnical_plus_secretory",
            **hc3_group_coefficient(patient, PRIMARY_GENE, True),
        })

    # Multiplicity is applied separately within each compartment/cohort across the fixed gene panel.
    result_frame = pd.DataFrame(results)
    result_frame["BH_q_within_fixed_panel"] = np.nan
    for _, index in result_frame.groupby(["compartment", "cohort"]).groups.items():
        positions = list(index)
        result_frame.loc[positions, "BH_q_within_fixed_panel"] = bh(result_frame.loc[positions, "permutation_p"].tolist())
    results = result_frame.to_dict(orient="records")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    pseudobulk_table.to_csv(args.output_dir / "patient_gene_pseudobulk.csv", index=False)
    for compartment, patient in patient_tables.items():
        patient.to_csv(args.output_dir / f"{compartment}_patient_matrix.csv", index=False)
    pd.DataFrame(results).to_csv(args.output_dir / "fixed_gene_results.csv", index=False)
    pd.DataFrame([
        {key: value for key, value in row.items() if key != "per_case"}
        for row in matched_results
    ]).to_csv(args.output_dir / "matched_results.csv", index=False)
    pd.DataFrame(leave_one_out).to_csv(args.output_dir / "nxpe1_leave_one_case_out.csv", index=False)
    pd.DataFrame(regressions).to_csv(args.output_dir / "nxpe1_regression_sensitivity.csv", index=False)

    broad = patient_tables["broad_epithelial"]
    broad_primary = [row for row in results if row["compartment"] == "broad_epithelial" and row["gene"] == PRIMARY_GENE]
    broad_matched = next(row for row in matched_results if row["compartment"] == "broad_epithelial" and row["outcome"] == PRIMARY_GENE)
    broad_loo = [row["mean_matched_contrast"] for row in leave_one_out if row["compartment"] == "broad_epithelial"]
    gates = {
        "nxpe1_all_pure_mean_positive": broad_primary[0]["mean_difference"] > 0,
        "nxpe1_right_MMR_stratified_mean_positive": broad_primary[1]["mean_difference"] > 0,
        "nxpe1_matched_mean_positive": broad_matched["mean_contrast"] > 0,
        "nxpe1_matched_at_least_5_of_6_positive": broad_matched["positive_cases"] >= 5,
        "nxpe1_matched_leave_one_out_sign_stable": min(broad_loo) > 0,
        "nxpe1_primary_support": (
            broad_primary[1]["bootstrap_95ci"][0] > 0
            and broad_matched["exact_sign_flip_p"] <= 0.05
            and min(broad_loo) > 0
        ),
    }
    report = {
        "status": "gse178341_nxpe1_mucinous_patient_pseudobulk_complete",
        "formal": True,
        "patients": {
            "pure_mucinous": int((broad["histology"] == PURE_MUC).sum()),
            "pure_adenocarcinoma": int((broad["histology"] == PURE_ADENO).sum()),
        },
        "available_fixed_genes": available,
        "feature_rows_per_gene": {gene: len(rows) for gene, rows in gene_rows.items()},
        "primary_NXPE1": broad_primary,
        "matched_NXPE1": broad_matched,
        "regression_sensitivity": [row for row in regressions if row["compartment"] == "broad_epithelial"],
        "gates": gates,
        "all_fixed_gene_results": results,
        "all_matched_results": matched_results,
        "provenance": {
            "h5_sha256": h5_sha256,
            "metadata_sha256": sha256(META),
            "clusters_sha256": sha256(CLUSTERS),
            "preflight_sha256": sha256(PREFLIGHT),
            "matches_sha256": sha256(MATCHES),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Independent patient-level tumour-epithelial transcript context only. A positive association does not "
            "identify the source of Neu5Ac, establish an NXPE1 substrate, prove O-acetylation position, enzyme "
            "activity, causal mediation, or metabolic flux."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    colors = {PURE_ADENO: "#4C78A8", PURE_MUC: "#E45756"}
    for axis, outcome, title in (
        (axes[0, 0], PRIMARY_GENE, "NXPE1: all tumour epithelial cells"),
        (axes[0, 1], "SECRETORY_COMPOSITE", "Fixed secretory-state composite"),
    ):
        for position, group in enumerate((PURE_ADENO, PURE_MUC)):
            values = broad.loc[broad["histology"].eq(group), outcome].to_numpy(float)
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(1)
            axis.scatter(np.full(len(values), position) + jitter, values, s=28, alpha=0.8, color=colors[group])
            axis.plot([position - 0.18, position + 0.18], [np.mean(values)] * 2, color="black", lw=2)
        axis.set_xticks([0, 1], ["Adenocarcinoma", "Mucinous\nadenocarcinoma"])
        axis.set_title(title)
        axis.set_ylabel("patient pseudobulk" if outcome != PRIMARY_GENE else "log2(CPM + 1)")
    contrast = pd.DataFrame(broad_matched["per_case"])
    axes[1, 0].axhline(0, color="black", lw=1)
    axes[1, 0].bar(contrast["case_PID"], contrast["contrast"], color=np.where(contrast["contrast"] > 0, "#59A14F", "#E45756"))
    axes[1, 0].set_title("NXPE1: each mucinous patient vs 3 frozen controls")
    axes[1, 0].set_ylabel("matched log2(CPM+1) contrast")
    goblet = patient_tables["goblet_family"]
    for position, group in enumerate((PURE_ADENO, PURE_MUC)):
        values = goblet.loc[goblet["histology"].eq(group), PRIMARY_GENE].to_numpy(float)
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(1)
        axes[1, 1].scatter(np.full(len(values), position) + jitter, values, s=28, alpha=0.8, color=colors[group])
        axes[1, 1].plot([position - 0.18, position + 0.18], [np.mean(values)] * 2, color="black", lw=2)
    axes[1, 1].set_xticks([0, 1], ["Adenocarcinoma", "Mucinous\nadenocarcinoma"])
    axes[1, 1].set_title("NXPE1: predefined goblet-family sensitivity")
    axes[1, 1].set_ylabel("log2(CPM + 1)")
    fig.suptitle("GSE178341 patient-level independent mucinous CRC audit", fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.output_dir / "nxpe1_mucinous_patient_audit.png", dpi=220)
    plt.close(fig)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
