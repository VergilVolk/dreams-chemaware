#!/usr/bin/env python
"""Gene-level TCGA audit for the targeted proline/sialic-axis interpretation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

import analyze_tcga_coadread_mucinous_axes as engine
from analyze_tcga_coadread_proline_sialic_axes import AXES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical", type=Path, default=Path(
        "data/external/TCGA_COADREAD_Xena_20260830/COADREAD_clinicalMatrix.tsv"))
    parser.add_argument("--expression", type=Path, default=Path(
        "data/external/TCGA_COADREAD_Xena_20260830/HiSeqV2.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_gene_audit_v1"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    genes = sorted({gene for members in AXES.values() for gene in members})
    samples, expression = engine.read_expression(args.expression, genes)
    sample_index = {sample: index for index, sample in enumerate(samples)}

    clinical = pd.read_csv(args.clinical, sep="\t", dtype=str, keep_default_na=False)
    conventional = {"Colon Adenocarcinoma", "Rectal Adenocarcinoma"}
    mucinous = {"Colon Mucinous Adenocarcinoma", "Rectal Mucinous Adenocarcinoma"}
    clinical = clinical[
        (clinical["sample_type"] == "Primary Tumor")
        & clinical["histological_type"].isin(conventional | mucinous)
    ].copy()
    clinical["patient"] = clinical["sampleID"].str.slice(0, 12)
    clinical = clinical.sort_values("sampleID").drop_duplicates("patient", keep="first")
    clinical["mucinous"] = clinical["histological_type"].isin(mucinous).astype(int)
    clinical["side"] = clinical["anatomic_neoplasm_subdivision"].map(engine.side)
    clinical["stage_group"] = clinical["pathologic_stage"].map(engine.stage_group)
    clinical["age"] = pd.to_numeric(clinical["age_at_initial_pathologic_diagnosis"], errors="coerce")
    clinical = clinical[clinical["sampleID"].isin(sample_index)].copy()

    patient_types: dict[str, dict[str, str]] = {}
    for sample in sorted(samples):
        if len(sample) >= 15 and sample[13:15] in {"01", "11"}:
            patient_types.setdefault(sample[:12], {}).setdefault(sample[13:15], sample)
    paired = sorted(patient for patient, members in patient_types.items() if {"01", "11"} <= set(members))

    gene_to_axes = {
        gene: ";".join(axis for axis, members in AXES.items() if gene in members)
        for gene in genes
    }
    rows = []
    tumour_positions = [sample_index[sample] for sample in clinical["sampleID"]]
    for gene in genes:
        all_values = expression[gene]
        mean = float(np.mean(all_values))
        sd = float(np.std(all_values, ddof=1))
        standardized = (all_values - mean) / (sd if sd > 0 else 1.0)
        clinical[f"gene__{gene}"] = standardized[tumour_positions]
        model = engine.hc3(clinical, f"gene__{gene}", include_msi=False)
        deltas = np.asarray([
            standardized[sample_index[patient_types[patient]["01"]]]
            - standardized[sample_index[patient_types[patient]["11"]]]
            for patient in paired
        ])
        nonzero = deltas[deltas != 0]
        sign_p = float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), .5).pvalue) if len(nonzero) else 1.0
        paired_p = float(wilcoxon(deltas, zero_method="wilcox").pvalue) if np.any(deltas != 0) else 1.0
        rows.append({
            "gene": gene,
            "axes": gene_to_axes[gene],
            "paired_patients": len(deltas),
            "paired_mean_tumour_minus_normal_z": float(np.mean(deltas)),
            "paired_median_tumour_minus_normal_z": float(np.median(deltas)),
            "paired_tumour_higher": int(np.sum(deltas > 0)),
            "paired_tumour_lower": int(np.sum(deltas < 0)),
            "paired_exact_sign_p": sign_p,
            "paired_wilcoxon_p": paired_p,
            "mucinous_adjusted_beta": model["coefficient"],
            "mucinous_adjusted_hc3_p": model["p"],
            "mucinous_adjusted_ci_low": model["ci_low"],
            "mucinous_adjusted_ci_high": model["ci_high"],
        })

    paired_q = engine.bh([row["paired_wilcoxon_p"] for row in rows])
    histology_q = engine.bh([row["mucinous_adjusted_hc3_p"] for row in rows])
    for row, q_paired, q_histology in zip(rows, paired_q, histology_q):
        row["paired_wilcoxon_bh_q"] = q_paired
        row["mucinous_adjusted_hc3_bh_q"] = q_histology
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "gene_results.csv", index=False)

    replicated = frame[
        (frame["paired_wilcoxon_bh_q"] < .05)
        | (frame["mucinous_adjusted_hc3_bh_q"] < .05)
    ].copy()
    replicated.to_csv(args.output_dir / "genes_passing_either_context.csv", index=False)
    report = {
        "status": "tcga_coadread_proline_sialic_gene_audit_complete",
        "formal": True,
        "genes": len(rows),
        "paired_patients": len(paired),
        "primary_tumours": len(clinical),
        "mucinous_tumours": int(clinical["mucinous"].sum()),
        "genes_passing_either_bh05": int(len(replicated)),
        "provenance": {
            "clinical_sha256": engine.sha256(args.clinical),
            "expression_sha256": engine.sha256(args.expression),
            "script_sha256": engine.sha256(Path(__file__)),
        },
        "claim_limit": (
            "Gene-level bulk RNA context. Multiple testing is controlled across all prespecified genes. "
            "Expression does not establish enzyme activity, metabolite identity, glycan linkage, or flux."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
