#!/usr/bin/env python
"""Composition-sensitivity audit for fixed proline/sialic CRC axes.

This is a sensitivity analysis of the already-defined TCGA COAD/READ
mucinous-versus-conventional contrast.  For each metabolic axis, broad-lineage
scores are recomputed after excluding any genes that also define that outcome.
That exclusion prevents mechanical over-adjustment (for example COL1A1/2 are
both collagen-axis genes and common fibroblast markers).

The analysis is deliberately not described as cell-type deconvolution.  The
lineage scores are coarse expression proxies used to ask whether a bulk-tumour
histology association is highly sensitive to broad tissue composition.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from analyze_gse236696_mucinous_axes_by_lineage import LINEAGE_MARKERS
from analyze_tcga_coadread_mucinous_axes import GENE_ALIASES, bh, side, stage_group
from analyze_tcga_coadread_proline_sialic_axes import AXES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_expression(path: Path, requested: set[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    values: dict[str, np.ndarray] = {}
    source_to_current = {GENE_ALIASES.get(gene, gene): gene for gene in requested}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        samples = header[1:]
        for row in reader:
            source_gene = row[0].upper()
            if source_gene in source_to_current:
                values[source_to_current[source_gene]] = np.asarray(row[1:], dtype=float)
    return samples, values


def standardize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    scale = float(np.std(vector, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(vector)
    return (vector - float(np.mean(vector))) / scale


def fit_hc3(
    frame: pd.DataFrame,
    outcome: str,
    continuous: list[str],
    categorical: list[str],
) -> dict:
    columns = [outcome, "mucinous", *continuous, *categorical]
    work = frame[columns].copy()
    for column in continuous:
        work[column] = pd.to_numeric(work[column], errors="coerce")
        work[column] = work[column].fillna(work[column].median())
    for column in categorical:
        work[column] = work[column].fillna("unknown").replace("", "unknown")
    if "msi" in categorical:
        work = work[work["msi"].isin(["MSS", "MSI-L", "MSI-H"])].copy()
    design = pd.get_dummies(
        work[["mucinous", *continuous, *categorical]],
        columns=categorical,
        drop_first=True,
        dtype=float,
    )
    design.insert(0, "intercept", 1.0)
    x = design.to_numpy(float)
    y = work[outcome].to_numpy(float)
    rank = int(np.linalg.matrix_rank(x))
    group_counts = work["mucinous"].value_counts().to_dict()
    singular = np.linalg.svd(x, compute_uv=False)
    condition_number = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    base = {
        "n": int(len(y)), "parameters": int(x.shape[1]), "rank": rank,
        "condition_number": condition_number,
        "n_mucinous": int(group_counts.get(1, 0)),
        "n_conventional": int(group_counts.get(0, 0)),
    }
    if (
        len(y) <= x.shape[1] + 10
        or rank < x.shape[1]
        or min(group_counts.get(0, 0), group_counts.get(1, 0)) < 10
    ):
        return {**base, "estimable": False, "beta": None, "hc3_se": None,
                "p": None, "ci_low": None, "ci_high": None}
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.sum((x @ inverse) * x, axis=1)
    scaled = residual / np.maximum(1.0 - leverage, 1e-8)
    covariance = inverse @ (x.T @ ((scaled ** 2)[:, None] * x)) @ inverse
    position = list(design.columns).index("mucinous")
    standard_error = float(np.sqrt(max(float(covariance[position, position]), 0.0)))
    coefficient = float(beta[position])
    degrees = max(1, len(y) - x.shape[1])
    statistic = coefficient / standard_error if standard_error > 0 else float("nan")
    p_value = float(2 * student_t.sf(abs(statistic), degrees)) if np.isfinite(statistic) else float("nan")
    critical = float(student_t.ppf(0.975, degrees))
    return {
        **base, "estimable": True, "beta": coefficient, "hc3_se": standard_error,
        "p": p_value, "ci_low": coefficient - critical * standard_error,
        "ci_high": coefficient + critical * standard_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical", type=Path, default=Path(
        "data/external/TCGA_COADREAD_Xena_20260830/COADREAD_clinicalMatrix.tsv"))
    parser.add_argument("--expression", type=Path, default=Path(
        "data/external/TCGA_COADREAD_Xena_20260830/HiSeqV2.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_lineage_sensitivity_v1"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested = (
        {gene for genes in AXES.values() for gene in genes}
        | {gene for genes in LINEAGE_MARKERS.values() for gene in genes}
    )
    samples, expression = read_expression(args.expression, requested)
    missing_axes = sorted({gene for genes in AXES.values() for gene in genes} - set(expression))
    if missing_axes:
        raise RuntimeError(f"missing metabolic-axis genes: {missing_axes}")
    sample_index = {sample: index for index, sample in enumerate(samples)}

    clinical = pd.read_csv(args.clinical, sep="\t", dtype=str, keep_default_na=False)
    conventional = {"Colon Adenocarcinoma", "Rectal Adenocarcinoma"}
    mucinous = {"Colon Mucinous Adenocarcinoma", "Rectal Mucinous Adenocarcinoma"}
    clinical = clinical[
        (clinical["sample_type"] == "Primary Tumor")
        & clinical["histological_type"].isin(conventional | mucinous)
        & clinical["sampleID"].isin(sample_index)
    ].copy()
    clinical["patient"] = clinical["sampleID"].str.slice(0, 12)
    clinical = clinical.sort_values("sampleID").drop_duplicates("patient", keep="first")
    clinical["mucinous"] = clinical["histological_type"].isin(mucinous).astype(int)
    clinical["side"] = clinical["anatomic_neoplasm_subdivision"].map(side)
    clinical["stage_group"] = clinical["pathologic_stage"].map(stage_group)
    clinical["age"] = pd.to_numeric(clinical["age_at_initial_pathologic_diagnosis"], errors="coerce")
    standardized_msi = clinical["CDE_ID_3226963"].replace({"": "unknown", "Indeterminate": "unknown"})
    updated_msi = clinical["MSI_updated_Oct62011"].replace("", "unknown")
    legacy_msi = clinical["microsatellite_instability"].map({"YES": "MSI-H", "NO": "MSS"}).fillna("unknown")
    clinical["msi"] = np.where(
        standardized_msi != "unknown", standardized_msi,
        np.where(updated_msi != "unknown", updated_msi, legacy_msi),
    )
    if clinical["mucinous"].sum() < 30 or (1 - clinical["mucinous"]).sum() < 100:
        raise RuntimeError("insufficient histology groups after expression matching")

    positions = np.asarray([sample_index[sample] for sample in clinical["sampleID"]])
    z = {gene: standardize(values[positions]) for gene, values in expression.items()}
    rows: list[dict] = []
    lineage_coverage: dict[str, dict] = {}
    for lineage, genes in LINEAGE_MARKERS.items():
        present = sorted(set(genes) & set(z))
        lineage_coverage[lineage] = {
            "expected": len(genes), "present": len(present),
            "missing": sorted(set(genes) - set(present)),
        }
        if len(present) < 3:
            raise RuntimeError(f"insufficient observable {lineage} markers: {present}")

    base_continuous = ["age"]
    base_categorical = ["side", "stage_group", "gender"]
    for axis, genes in AXES.items():
        outcome = f"axis__{axis}"
        clinical[outcome] = np.mean(np.vstack([z[gene] for gene in genes]), axis=0)
        lineage_columns: list[str] = []
        excluded_overlap: dict[str, list[str]] = {}
        for lineage, markers in LINEAGE_MARKERS.items():
            usable = sorted((set(markers) & set(z)) - set(genes))
            overlap = sorted(set(markers) & set(genes))
            if len(usable) < 3:
                raise RuntimeError(f"{axis}: fewer than three non-overlapping {lineage} markers")
            column = f"lineage__{lineage}"
            clinical[column] = np.mean(np.vstack([z[gene] for gene in usable]), axis=0)
            lineage_columns.append(column)
            excluded_overlap[lineage] = overlap

        clinical_model = fit_hc3(clinical, outcome, base_continuous, base_categorical)
        lineage_model = fit_hc3(
            clinical, outcome, [*base_continuous, *lineage_columns], base_categorical,
        )
        msi_lineage_model = fit_hc3(
            clinical, outcome, [*base_continuous, *lineage_columns], [*base_categorical, "msi"],
        )
        beta_base = clinical_model["beta"]
        beta_lineage = lineage_model["beta"]
        attenuation = None
        if beta_base is not None and beta_lineage is not None and abs(beta_base) > 1e-12:
            attenuation = float(1.0 - beta_lineage / beta_base)
        rows.append({
            "axis": axis,
            "clinical_beta": beta_base,
            "clinical_p": clinical_model["p"],
            "clinical_ci_low": clinical_model["ci_low"],
            "clinical_ci_high": clinical_model["ci_high"],
            "lineage_adjusted_beta": beta_lineage,
            "lineage_adjusted_p": lineage_model["p"],
            "lineage_adjusted_ci_low": lineage_model["ci_low"],
            "lineage_adjusted_ci_high": lineage_model["ci_high"],
            "lineage_adjusted_condition_number": lineage_model["condition_number"],
            "beta_attenuation_fraction": attenuation,
            "direction_preserved": bool(
                beta_base is not None and beta_lineage is not None
                and np.sign(beta_base) == np.sign(beta_lineage)
            ),
            "msi_lineage_n": msi_lineage_model["n"],
            "msi_lineage_estimable": msi_lineage_model["estimable"],
            "msi_lineage_beta": msi_lineage_model["beta"],
            "msi_lineage_p": msi_lineage_model["p"],
            "axis_lineage_gene_overlap_excluded": json.dumps(excluded_overlap, sort_keys=True),
        })

    lineage_q = bh([row["lineage_adjusted_p"] for row in rows])
    for row, qvalue in zip(rows, lineage_q):
        row["lineage_adjusted_bh_q_all_axes"] = qvalue
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "axis_lineage_sensitivity.csv", index=False)

    keep = [
        "sampleID", "patient", "histological_type", "mucinous", "side",
        "stage_group", "age", "gender", "msi",
        *[f"axis__{axis}" for axis in AXES],
    ]
    clinical[keep].to_csv(args.output_dir / "analysis_samples.csv", index=False)
    report = {
        "status": "tcga_proline_sialic_lineage_sensitivity_complete",
        "formal": False,
        "analysis_role": "composition sensitivity of previously analysed TCGA axes; not a new blind validation",
        "samples": {
            "total": int(len(clinical)), "mucinous": int(clinical["mucinous"].sum()),
            "conventional": int((1 - clinical["mucinous"]).sum()),
            "msi_complete": int(clinical["msi"].isin(["MSS", "MSI-L", "MSI-H"]).sum()),
        },
        "models": {
            "clinical": "HC3 OLS: axis ~ mucinous + age + side + stage + sex",
            "lineage_sensitivity": "clinical model + six broad-lineage expression scores",
            "msi_lineage_sensitivity": "lineage model + MSI, complete cases only",
            "anti_circularity": "for each outcome axis, overlapping genes are removed from lineage scores before fitting",
        },
        "lineage_marker_coverage": lineage_coverage,
        "results": rows,
        "provenance": {
            "clinical_sha256": sha256(args.clinical),
            "expression_sha256": sha256(args.expression),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Broad-lineage expression scores are composition proxies, not measured cell fractions. "
            "Persistence supports robustness to coarse bulk composition; attenuation does not prove "
            "that a signal is non-biological. No transcriptomic result establishes metabolite identity, "
            "glycan linkage, enzyme activity, flux or causality."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
