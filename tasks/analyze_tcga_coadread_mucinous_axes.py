#!/usr/bin/env python
"""Test whether fixed metabolic axes are enriched in mucinous vs conventional CRC.

The analysis uses primary-tumour TCGA COAD/READ RNA-seq from UCSC Xena.
Axis genes are frozen to the same sets used in the GSE236696 sensitivity
analysis.  The primary contrast is mucinous adenocarcinoma versus conventional
adenocarcinoma.  HC3 linear models adjust for anatomic side, stage, age and
sex; an MSI-complete-case model is reported separately because MSI is missing
for most samples.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu, t as student_t, ttest_ind, wilcoxon


AXES = {
    "modified_nucleoside_processing": [
        "METTL1", "WDR4", "RNMT", "CMTR1", "CMTR2", "TRMT1", "TRMT5",
        "TRMT10C", "TGS1", "THUMPD3", "NUDT16", "DCP2",
    ],
    "purine_synthesis_salvage": [
        "HPRT1", "PNP", "GMPS", "IMPDH1", "IMPDH2", "GDA", "APRT",
        "XDH", "ADA", "ADK",
    ],
    "methionine_sah_cycle": ["AHCY", "MAT2A", "MAT2B", "MTAP", "MTR", "MTHFR", "BHMT"],
    "carnitine_long_chain_fao": [
        "CPT1A", "CPT2", "SLC25A20", "ACADVL", "ACADM", "ACADS", "HADHA",
        "HADHB", "ETFA", "ETFB", "ETFDH", "CRAT", "CROT",
    ],
    "sphingolipid_metabolism": [
        "CERS2", "CERS4", "CERS5", "CERS6", "SPTLC1", "SPTLC2", "SPTLC3",
        "SGMS1", "SGMS2", "SPHK1", "SPHK2", "ASAH1", "ASAH2", "SMPD1",
        "SMPD2", "SMPD3", "SMPD4", "UGCG",
    ],
}

# The legacy TCGA HiSeqV2 matrix predates several HGNC symbol updates.
# These mappings are symbol aliases only; no phenotype information is used.
GENE_ALIASES = {
    "CERS2": "LASS2",
    "CERS4": "LASS4",
    "CERS5": "LASS5",
    "CERS6": "LASS6",
    "CMTR1": "FTSJD2",
    "CMTR2": "FTSJD1",
    "TRMT10C": "RG9MTD1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values):
    values = list(values)
    order = np.argsort(values)[::-1]
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order):
        rank = len(values) - reverse_rank
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def side(value):
    value = str(value or "").lower()
    if any(token in value for token in ("cecum", "ascending", "hepatic", "transverse")):
        return "right_colon"
    if any(token in value for token in ("descending", "splenic", "sigmoid")):
        return "left_colon"
    if "rect" in value:
        return "rectum"
    return "unknown"


def stage_group(value):
    match = re.search(r"stage\s+(iv|iii|ii|i)\b", str(value or "").lower())
    return match.group(1).upper() if match else "unknown"


def hc3(frame: pd.DataFrame, outcome: str, include_msi: bool = False):
    columns = ["mucinous", "age"]
    categorical = ["side", "stage_group", "gender"]
    if include_msi:
        categorical.append("msi")
    work = frame[[outcome] + columns + categorical].copy()
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    work["age"] = work["age"].fillna(work["age"].median())
    if include_msi:
        work = work[work["msi"].isin(["MSS", "MSI-L", "MSI-H"])].copy()
    for column in categorical:
        work[column] = work[column].fillna("unknown").replace("", "unknown")
    design = pd.get_dummies(work[columns + categorical], columns=categorical, drop_first=True, dtype=float)
    design.insert(0, "intercept", 1.0)
    x = design.to_numpy(dtype=float)
    y = work[outcome].to_numpy(dtype=float)
    group_counts = work["mucinous"].value_counts().to_dict()
    rank = int(np.linalg.matrix_rank(x))
    if len(y) <= x.shape[1] + 10 or rank < x.shape[1] or min(group_counts.get(0, 0), group_counts.get(1, 0)) < 10:
        return {
            "estimable": False, "n": int(len(y)), "parameters": int(x.shape[1]),
            "rank": rank, "coefficient": None, "hc3_se": None, "t": None,
            "p": None, "ci_low": None, "ci_high": None,
        }
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    leverage = np.sum((x @ inverse) * x, axis=1)
    scaled = residual / np.maximum(1.0 - leverage, 1e-8)
    meat = x.T @ ((scaled ** 2)[:, None] * x)
    covariance = inverse @ meat @ inverse
    position = list(design.columns).index("mucinous")
    standard_error = float(np.sqrt(max(covariance[position, position], 0.0)))
    coefficient = float(beta[position])
    degrees = max(1, len(y) - x.shape[1])
    statistic = coefficient / standard_error if standard_error > 0 else float("nan")
    p_value = float(2 * student_t.sf(abs(statistic), degrees)) if np.isfinite(statistic) else float("nan")
    return {
        "estimable": True, "n": int(len(y)), "parameters": int(x.shape[1]), "rank": rank,
        "coefficient": coefficient,
        "hc3_se": standard_error, "t": statistic, "p": p_value,
        "ci_low": coefficient - float(student_t.ppf(0.975, degrees)) * standard_error,
        "ci_high": coefficient + float(student_t.ppf(0.975, degrees)) * standard_error,
    }


def read_expression(path: Path, requested):
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
    missing = sorted(set(requested) - set(values))
    if missing:
        raise RuntimeError(f"missing target genes in expression matrix: {missing}")
    return samples, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical", type=Path, default=Path(
        "data/external/TCGA_COADREAD_Xena_20260830/COADREAD_clinicalMatrix.tsv"))
    parser.add_argument("--expression", type=Path, default=Path(
        "data/external/TCGA_COADREAD_Xena_20260830/HiSeqV2.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/external/TCGA_COADREAD_Xena_20260830/mucinous_axis_analysis_v4"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    requested = sorted({gene for genes in AXES.values() for gene in genes})
    expression_samples, expression = read_expression(args.expression, requested)
    expression_index = {sample: index for index, sample in enumerate(expression_samples)}
    clinical = clinical[clinical["sampleID"].isin(expression_index)].copy()
    if clinical["mucinous"].sum() < 30 or (1 - clinical["mucinous"]).sum() < 100:
        raise RuntimeError("insufficient histology groups after expression matching")

    positions = [expression_index[sample] for sample in clinical["sampleID"]]
    gene_matrix = np.column_stack([expression[gene][positions] for gene in requested])
    means = gene_matrix.mean(axis=0)
    standard = gene_matrix.std(axis=0, ddof=1)
    z = (gene_matrix - means) / np.where(standard > 0, standard, 1.0)
    gene_position = {gene: index for index, gene in enumerate(requested)}
    for axis, genes in AXES.items():
        clinical[f"axis__{axis}"] = np.mean(z[:, [gene_position[gene] for gene in genes]], axis=1)

    rows = []
    for axis in AXES:
        outcome = f"axis__{axis}"
        a = clinical.loc[clinical["mucinous"] == 1, outcome].to_numpy(float)
        b = clinical.loc[clinical["mucinous"] == 0, outcome].to_numpy(float)
        welch = ttest_ind(a, b, equal_var=False)
        mann = mannwhitneyu(a, b, alternative="two-sided")
        base_model = hc3(clinical, outcome, include_msi=False)
        msi_model = hc3(clinical, outcome, include_msi=True)
        colon_only = clinical[clinical["side"].isin(["left_colon", "right_colon"])].copy()
        colon_model = hc3(colon_only, outcome, include_msi=False)
        rows.append({
            "axis": axis,
            "n_mucinous": int(len(a)), "n_conventional": int(len(b)),
            "mucinous_mean_z": float(np.mean(a)), "conventional_mean_z": float(np.mean(b)),
            "unadjusted_delta": float(np.mean(a) - np.mean(b)),
            "welch_p": float(welch.pvalue), "mann_whitney_p": float(mann.pvalue),
            "adjusted_beta": base_model["coefficient"], "adjusted_hc3_p": base_model["p"],
            "adjusted_ci_low": base_model["ci_low"], "adjusted_ci_high": base_model["ci_high"],
            "colon_only_beta": colon_model["coefficient"], "colon_only_hc3_p": colon_model["p"],
            "msi_complete_n": msi_model["n"], "msi_model_estimable": msi_model["estimable"],
            "msi_adjusted_beta": msi_model["coefficient"], "msi_adjusted_hc3_p": msi_model["p"],
        })
    adjusted = bh([row["adjusted_hc3_p"] for row in rows])
    for row, q_value in zip(rows, adjusted):
        row["adjusted_hc3_bh_q_all_axes"] = q_value

    # Independent question: do these axes change in paired CRC tumour vs normal,
    # regardless of mucinous histology?  This uses only patients with both an
    # RNA-seq primary tumour and adjacent-solid-tissue-normal sample.
    sample_by_patient_type = {}
    for sample in sorted(expression_samples):
        if len(sample) < 15:
            continue
        sample_type = sample[13:15]
        if sample_type not in {"01", "11"}:
            continue
        sample_by_patient_type.setdefault(sample[:12], {}).setdefault(sample_type, sample)
    paired_patients = sorted(
        patient for patient, members in sample_by_patient_type.items()
        if "01" in members and "11" in members
    )
    all_positions = np.arange(len(expression_samples))
    full_matrix = np.column_stack([expression[gene][all_positions] for gene in requested])
    full_mean = full_matrix.mean(axis=0)
    full_sd = full_matrix.std(axis=0, ddof=1)
    full_z = (full_matrix - full_mean) / np.where(full_sd > 0, full_sd, 1.0)
    all_index = {sample: index for index, sample in enumerate(expression_samples)}
    paired_results = []
    for axis, genes in AXES.items():
        positions_axis = [gene_position[gene] for gene in genes]
        deltas = []
        for patient in paired_patients:
            members = sample_by_patient_type[patient]
            tumor_score = float(np.mean(full_z[all_index[members["01"]], positions_axis]))
            normal_score = float(np.mean(full_z[all_index[members["11"]], positions_axis]))
            deltas.append(tumor_score - normal_score)
        values = np.asarray(deltas)
        nonzero = values[values != 0]
        sign_p = float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue) if len(nonzero) else 1.0
        wilcoxon_p = float(wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)
        paired_results.append({
            "axis": axis, "patients": int(len(values)),
            "mean_tumor_minus_normal": float(np.mean(values)),
            "median_tumor_minus_normal": float(np.median(values)),
            "tumor_higher_pairs": int(np.sum(values > 0)),
            "tumor_lower_pairs": int(np.sum(values < 0)),
            "exact_sign_p": sign_p, "wilcoxon_p": wilcoxon_p,
        })
    paired_q = bh([row["wilcoxon_p"] for row in paired_results])
    for row, q_value in zip(paired_results, paired_q):
        row["wilcoxon_bh_q_all_axes"] = q_value

    output_table = args.output_dir / "axis_results.csv"
    pd.DataFrame(rows).to_csv(output_table, index=False)
    sample_table = args.output_dir / "analysis_samples.csv"
    keep = ["sampleID", "patient", "histological_type", "mucinous", "side", "stage_group", "age", "gender", "msi"]
    clinical[keep].to_csv(sample_table, index=False)
    pd.DataFrame(paired_results).to_csv(args.output_dir / "paired_tumor_normal_axis_results.csv", index=False)

    figure, axis_plot = plt.subplots(figsize=(9.5, 5.5))
    labels = [row["axis"].replace("_", "\n") for row in rows]
    values = [row["adjusted_beta"] for row in rows]
    low = [row["adjusted_beta"] - row["adjusted_ci_low"] for row in rows]
    high = [row["adjusted_ci_high"] - row["adjusted_beta"] for row in rows]
    palette = ["#7b3294", "#008837", "#7f7f7f", "#2166ac", "#c51b7d", "#d95f02", "#1b9e77", "#7570b3"]
    colors = [palette[index % len(palette)] for index in range(len(rows))]
    axis_plot.errorbar(range(len(rows)), values, yerr=[low, high], fmt="none", ecolor="black", capsize=4)
    axis_plot.scatter(range(len(rows)), values, color=colors, s=60, zorder=3)
    axis_plot.axhline(0, color="black", linewidth=0.8)
    axis_plot.set_xticks(range(len(rows)), labels, fontsize=8)
    axis_plot.set_ylabel("adjusted mucinous - conventional axis score (HC3 95% CI)")
    axis_plot.set_title("TCGA COAD/READ: histology-specific fixed metabolic axes")
    figure.tight_layout()
    figure.savefig(args.output_dir / "adjusted_axis_effects.png", dpi=220)
    plt.close(figure)

    report = {
        "status": "tcga_coadread_mucinous_axis_analysis_complete",
        "formal": True,
        "samples": {
            "total": int(len(clinical)), "mucinous": int(clinical["mucinous"].sum()),
            "conventional": int((1 - clinical["mucinous"]).sum()),
            "msi_complete": int(clinical["msi"].isin(["MSS", "MSI-L", "MSI-H"]).sum()),
        },
        "primary_model": "HC3 OLS adjusted for anatomic side, stage, age and sex",
        "msi_sensitivity": "standardized CDE_ID_3226963 MSI field, supplemented by updated and legacy YES/NO fields; reported only when full-rank with >=10 samples per histology group",
        "results": rows,
        "paired_tumor_normal": {
            "patients": len(paired_patients),
            "score": "mean gene z-score; each gene standardized across the full COADREAD RNA-seq matrix",
            "results": paired_results,
        },
        "provenance": {
            "clinical_sha256": sha256(args.clinical), "expression_sha256": sha256(args.expression),
            "script_sha256": sha256(Path(__file__)),
            "xena_datasets": ["TCGA.COADREAD.sampleMap/COADREAD_clinicalMatrix", "TCGA.COADREAD.sampleMap/HiSeqV2"],
            "legacy_symbol_aliases": GENE_ALIASES,
        },
        "claim_limit": (
            "Bulk tumour histology contrast. It tests mucinous enrichment, not tumour-vs-normal change, "
            "cellular origin, metabolite identity, enzyme activity or flux. BRAF/KRAS assay results are too sparse for stable multivariable adjustment."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
