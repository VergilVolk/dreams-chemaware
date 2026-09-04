#!/usr/bin/env python
"""Post-hoc bulk-composition sensitivity for GSE281917 metabolic axes.

This audit reuses the broad-lineage marker sets frozen before the present
cohort analysis.  It asks whether associations with MuC23 risk remain after
residualizing stage, age, sex and six bulk lineage scores.  It is explicitly a
confounder sensitivity analysis, not cell-type deconvolution.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze_gse236696_mucinous_axes_by_lineage import LINEAGE_MARKERS
from analyze_gse281917_mucinous_metabolic_axes import (
    AXES,
    axis_scores,
    bh,
    load_risk_coefficients,
    load_tar_expression,
    parse_series_matrix,
    parse_stage,
    partial_rank_association,
    sha256,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty output: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data/external/GSE281917/source"))
    parser.add_argument("--primary-dir", type=Path, default=Path("data/external/GSE281917/mucinous_metabolic_axes_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/GSE281917/mucinous_axis_composition_audit_v1"))
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = args.source_dir
    tar_path = source / "GSE281917_RAW.tar"
    matrix_path = source / "GSE281917_series_matrix.txt.gz"
    supplement_path = source / "41416_2025_3104_MOESM2_ESM.xlsx"
    primary_report = args.primary_dir / "report.json"
    for path in [tar_path, matrix_path, supplement_path, primary_report]:
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = parse_series_matrix(matrix_path, "MuC").set_index("sample")
    expression = load_tar_expression(tar_path, set(metadata.index))
    axes, _ = axis_scores(expression)
    risk = load_risk_coefficients(supplement_path)
    risk_score = sum(
        float(row.coefficient) * expression.loc[row.gene]
        for row in risk.itertuples(index=False)
    )
    z = expression.sub(expression.mean(axis=1), axis=0).div(
        expression.std(axis=1, ddof=1).replace(0, np.nan), axis=0
    )
    lineage_scores = {}
    lineage_coverage = []
    for lineage, genes in LINEAGE_MARKERS.items():
        present = [gene for gene in genes if gene in z.index]
        if len(present) < 3:
            raise RuntimeError(f"insufficient {lineage} markers: {present}")
        lineage_scores[lineage] = z.loc[present].mean(axis=0)
        lineage_coverage.append({
            "lineage": lineage,
            "genes_expected": len(genes),
            "genes_present": len(present),
            "present_genes": ";".join(present),
        })

    metadata = metadata.loc[axes.index]
    clinical_covariates = np.column_stack([
        metadata["tumor_stage"].map(parse_stage).to_numpy(float),
        pd.to_numeric(metadata["age"], errors="coerce").to_numpy(float),
        metadata["sex"].astype(str).str.lower().map({"female": 0.0, "male": 1.0}).to_numpy(float),
    ])
    composition_covariates = np.column_stack([
        clinical_covariates,
        *[lineage_scores[lineage].loc[metadata.index].to_numpy(float) for lineage in LINEAGE_MARKERS],
    ])

    association_rows = []
    for number, axis in enumerate(AXES):
        x = axes.loc[metadata.index, axis].to_numpy(float)
        y = risk_score.loc[metadata.index].to_numpy(float)
        clinical = partial_rank_association(
            x, y, clinical_covariates, args.seed + number, args.bootstrap_resamples
        )
        composition = partial_rank_association(
            x, y, composition_covariates, args.seed + 100 + number, args.bootstrap_resamples
        )
        association_rows.append({
            "axis": axis,
            "clinical_adjusted_rho": clinical["rho"],
            "clinical_adjusted_p": clinical["pvalue"],
            "clinical_adjusted_ci_low": clinical["ci_low"],
            "clinical_adjusted_ci_high": clinical["ci_high"],
            "clinical_and_lineage_adjusted_rho": composition["rho"],
            "clinical_and_lineage_adjusted_p": composition["pvalue"],
            "clinical_and_lineage_adjusted_ci_low": composition["ci_low"],
            "clinical_and_lineage_adjusted_ci_high": composition["ci_high"],
            "clinical_and_lineage_adjusted_leave_one_out_min": composition["leave_one_out_min"],
            "clinical_and_lineage_adjusted_leave_one_out_max": composition["leave_one_out_max"],
        })
    qvalues = bh([row["clinical_and_lineage_adjusted_p"] for row in association_rows])
    for row, qvalue in zip(association_rows, qvalues):
        row["clinical_and_lineage_adjusted_q"] = qvalue

    lineage_rows = []
    for lineage in LINEAGE_MARKERS:
        item = partial_rank_association(
            lineage_scores[lineage].loc[metadata.index].to_numpy(float),
            risk_score.loc[metadata.index].to_numpy(float),
            clinical_covariates,
            args.seed + 200 + len(lineage_rows),
            args.bootstrap_resamples,
        )
        lineage_rows.append({
            "lineage": lineage,
            "rho": item["rho"],
            "pvalue": item["pvalue"],
            "ci_low": item["ci_low"],
            "ci_high": item["ci_high"],
        })
    for row, qvalue in zip(lineage_rows, bh([row["pvalue"] for row in lineage_rows])):
        row["qvalue"] = qvalue

    gene_rows = []
    for axis, genes in AXES.items():
        for gene in genes:
            values = z.loc[gene, metadata.index].to_numpy(float)
            item = partial_rank_association(
                values,
                risk_score.loc[metadata.index].to_numpy(float),
                composition_covariates,
                args.seed + 300 + len(gene_rows),
                max(1000, args.bootstrap_resamples // 2),
            )
            gene_rows.append({
                "axis": axis,
                "gene": gene,
                "composition_adjusted_rho": item["rho"],
                "composition_adjusted_p": item["pvalue"],
                "composition_adjusted_ci_low": item["ci_low"],
                "composition_adjusted_ci_high": item["ci_high"],
            })
    for row, qvalue in zip(gene_rows, bh([row["composition_adjusted_p"] for row in gene_rows])):
        row["composition_adjusted_q_all_axis_genes"] = qvalue

    leave_gene_out_rows = []
    for axis, genes in AXES.items():
        for omitted in genes:
            retained = [gene for gene in genes if gene != omitted]
            if len(retained) < 2:
                continue
            score = z.loc[retained, metadata.index].mean(axis=0).to_numpy(float)
            item = partial_rank_association(
                score,
                risk_score.loc[metadata.index].to_numpy(float),
                composition_covariates,
                args.seed + 500 + len(leave_gene_out_rows),
                max(1000, args.bootstrap_resamples // 2),
            )
            leave_gene_out_rows.append({
                "axis": axis,
                "omitted_gene": omitted,
                "rho": item["rho"],
                "pvalue": item["pvalue"],
                "ci_low": item["ci_low"],
                "ci_high": item["ci_high"],
            })

    write_csv(args.output_dir / "axis_composition_sensitivity.csv", association_rows)
    write_csv(args.output_dir / "lineage_risk_associations.csv", lineage_rows)
    write_csv(args.output_dir / "axis_gene_risk_associations.csv", gene_rows)
    write_csv(args.output_dir / "axis_leave_one_gene_out.csv", leave_gene_out_rows)
    write_csv(args.output_dir / "lineage_marker_coverage.csv", lineage_coverage)

    report = {
        "status": "gse281917_metabolic_axis_composition_audit_complete",
        "formal": False,
        "analysis_type": "post-hoc confounder sensitivity using previously frozen broad-lineage markers",
        "n_samples": len(metadata),
        "axis_associations": association_rows,
        "lineage_risk_associations": lineage_rows,
        "surviving_axes_at_q10": [
            row["axis"] for row in association_rows
            if row["clinical_and_lineage_adjusted_q"] < 0.10
            and row["clinical_and_lineage_adjusted_ci_low"] * row["clinical_and_lineage_adjusted_ci_high"] > 0
        ],
        "claim_limit": (
            "Broad marker residualization is a composition sensitivity analysis, not formal deconvolution or malignant-cell proof. "
            "Surviving association supports transcript-program alignment with MuC23 risk, not metabolite flux, recurrence causality or enzyme activity."
        ),
        "provenance": {
            "muc_tar": sha256(tar_path),
            "series_matrix": sha256(matrix_path),
            "supplement": sha256(supplement_path),
            "primary_report": sha256(primary_report),
        },
        "parameters": {"bootstrap_resamples": args.bootstrap_resamples, "seed": args.seed},
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
