#!/usr/bin/env python
"""Targeted TCGA replication of the GSE281917 MuC23-purine association.

The primary hypothesis is fixed before this script reads TCGA risk outcomes:
within mucinous COAD/READ, higher MuC23 score associates with a lower purine
synthesis/salvage axis after clinical and broad-composition adjustment.  Other
metabolic axes are secondary.  TCGA was previously used for related histology
analyses, so this is independent-sample targeted replication, not a pristine
blind test.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_gse236696_mucinous_axes_by_lineage import LINEAGE_MARKERS
from analyze_gse281917_mucinous_metabolic_axes import (
    AXES,
    bh,
    load_risk_coefficients,
    parse_stage,
    partial_rank_association,
    sha256,
)
from analyze_tcga_coadread_mucinous_axes import GENE_ALIASES, side


def read_expression_flexible(path: Path, requested: set[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    values = {}
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical", type=Path, default=Path("data/external/TCGA_COADREAD_Xena_20260830/COADREAD_clinicalMatrix.tsv"))
    parser.add_argument("--expression", type=Path, default=Path("data/external/TCGA_COADREAD_Xena_20260830/HiSeqV2.gz"))
    parser.add_argument("--supplement", type=Path, default=Path("data/external/GSE281917/source/41416_2025_3104_MOESM2_ESM.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/TCGA_COADREAD_Xena_20260830/mucinous_risk_axis_replication_v1"))
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    risk = load_risk_coefficients(args.supplement)
    requested = sorted(
        {gene for genes in AXES.values() for gene in genes}
        | {gene for genes in LINEAGE_MARKERS.values() for gene in genes}
        | set(risk["gene"])
    )
    samples, expression = read_expression_flexible(args.expression, set(requested))
    required = set(risk["gene"]) | {gene for genes in AXES.values() for gene in genes}
    missing_required = sorted(required - set(expression))
    if missing_required:
        raise RuntimeError(f"missing required risk/axis genes: {missing_required}")
    expression_index = {sample: index for index, sample in enumerate(samples)}

    clinical = pd.read_csv(args.clinical, sep="\t", dtype=str, keep_default_na=False)
    mucinous = {"Colon Mucinous Adenocarcinoma", "Rectal Mucinous Adenocarcinoma"}
    clinical = clinical[
        (clinical["sample_type"] == "Primary Tumor")
        & clinical["histological_type"].isin(mucinous)
        & clinical["sampleID"].isin(expression_index)
    ].copy()
    clinical["patient"] = clinical["sampleID"].str.slice(0, 12)
    clinical = clinical.sort_values("sampleID").drop_duplicates("patient", keep="first")
    if len(clinical) < 35:
        raise RuntimeError(f"insufficient TCGA mucinous samples: {len(clinical)}")
    positions = np.asarray([expression_index[sample] for sample in clinical["sampleID"]])
    matrix = pd.DataFrame(
        {gene: expression[gene][positions] for gene in sorted(expression)},
        index=clinical["sampleID"],
    ).T
    z = matrix.sub(matrix.mean(axis=1), axis=0).div(matrix.std(axis=1, ddof=1).replace(0, np.nan), axis=0)

    axis_scores = {
        axis: z.loc[genes].mean(axis=0).to_numpy(float) for axis, genes in AXES.items()
    }
    lineage_scores = {}
    lineage_coverage = {}
    for lineage, genes in LINEAGE_MARKERS.items():
        present = [gene for gene in genes if gene in z.index]
        if len(present) < 3:
            raise RuntimeError(f"insufficient observable {lineage} markers: {present}")
        lineage_scores[lineage] = z.loc[present].mean(axis=0).to_numpy(float)
        lineage_coverage[lineage] = {
            "expected": len(genes), "present": len(present),
            "missing": sorted(set(genes) - set(present)),
        }
    risk_score = sum(
        float(row.coefficient) * matrix.loc[row.gene].to_numpy(float)
        for row in risk.itertuples(index=False)
    )

    clinical_covariates = np.column_stack([
        clinical["pathologic_stage"].str.upper().str.replace(r"^STAGE\s+", "", regex=True).map(parse_stage).to_numpy(float),
        pd.to_numeric(clinical["age_at_initial_pathologic_diagnosis"], errors="coerce").to_numpy(float),
        clinical["gender"].str.lower().map({"female": 0.0, "male": 1.0}).to_numpy(float),
        clinical["anatomic_neoplasm_subdivision"].map(side).eq("right_colon").astype(float).to_numpy(),
    ])
    full_covariates = np.column_stack([
        clinical_covariates,
        *[lineage_scores[lineage] for lineage in LINEAGE_MARKERS],
    ])

    rows = []
    ordered_axes = ["purine_synthesis_salvage", *[axis for axis in AXES if axis != "purine_synthesis_salvage"]]
    for number, axis in enumerate(ordered_axes):
        clinical_result = partial_rank_association(
            axis_scores[axis], risk_score, clinical_covariates,
            args.seed + number, args.bootstrap_resamples,
        )
        composition_result = partial_rank_association(
            axis_scores[axis], risk_score, full_covariates,
            args.seed + 100 + number, args.bootstrap_resamples,
        )
        rows.append({
            "axis": axis,
            "endpoint": "primary" if axis == "purine_synthesis_salvage" else "secondary",
            "clinical_adjusted_rho": clinical_result["rho"],
            "clinical_adjusted_p": clinical_result["pvalue"],
            "clinical_adjusted_ci_low": clinical_result["ci_low"],
            "clinical_adjusted_ci_high": clinical_result["ci_high"],
            "clinical_and_lineage_adjusted_rho": composition_result["rho"],
            "clinical_and_lineage_adjusted_p": composition_result["pvalue"],
            "clinical_and_lineage_adjusted_ci_low": composition_result["ci_low"],
            "clinical_and_lineage_adjusted_ci_high": composition_result["ci_high"],
            "leave_one_out_min": composition_result["leave_one_out_min"],
            "leave_one_out_max": composition_result["leave_one_out_max"],
        })
    secondary = [row for row in rows if row["endpoint"] == "secondary"]
    for row, qvalue in zip(secondary, bh([row["clinical_and_lineage_adjusted_p"] for row in secondary])):
        row["secondary_bh_q"] = qvalue
    rows[0]["secondary_bh_q"] = None

    result_table = args.output_dir / "tcga_mucinous_risk_axis_results.csv"
    pd.DataFrame(rows).to_csv(result_table, index=False)
    sample_table = clinical[[
        "sampleID", "patient", "histological_type", "pathologic_stage",
        "age_at_initial_pathologic_diagnosis", "gender", "anatomic_neoplasm_subdivision",
    ]].copy()
    sample_table["muc23_risk_score"] = risk_score
    for axis in AXES:
        sample_table[f"axis__{axis}"] = axis_scores[axis]
    sample_table.to_csv(args.output_dir / "analysis_samples.csv", index=False)

    primary = rows[0]
    report = {
        "status": "tcga_mucinous_risk_axis_replication_complete",
        "formal": False,
        "analysis_type": "targeted independent-sample replication; TCGA previously consumed for related axis analyses",
        "n_mucinous_samples": len(clinical),
        "primary_hypothesis": "MuC23 risk score is inversely associated with purine synthesis/salvage after clinical and broad-lineage adjustment",
        "primary_result": primary,
        "primary_direction_replicated": bool(
            primary["clinical_and_lineage_adjusted_rho"] < 0
            and primary["clinical_and_lineage_adjusted_ci_high"] < 0
        ),
        "secondary_results": rows[1:],
        "muc23_axis_gene_overlap": {
            axis: sorted(set(genes) & set(risk["gene"])) for axis, genes in AXES.items()
        },
        "lineage_marker_coverage": lineage_coverage,
        "claim_limit": (
            "This tests reproducibility of a transcript-program association in an independent mucinous cohort. "
            "It is not an independent survival analysis and does not establish metabolite abundance, flux, enzyme activity or causality."
        ),
        "parameters": {"bootstrap_resamples": args.bootstrap_resamples, "seed": args.seed},
        "provenance": {
            "clinical": sha256(args.clinical),
            "expression": sha256(args.expression),
            "supplement": sha256(args.supplement),
        },
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
