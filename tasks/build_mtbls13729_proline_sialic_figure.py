#!/usr/bin/env python
"""Build the frozen cross-cohort proline/sialic evidence figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    output = Path("data/mtbls13729/proline_sialic_summary_figure_v1")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    ledger = pd.read_csv("data/mtbls13729/integrated_biology_ledger_v2/integrated_candidate_ledger_v2.csv")
    selected = ledger[ledger["feature_id"].isin([345, 374, 703])].copy()
    selected = selected.set_index("feature_id").loc[[345, 374, 703]].reset_index()
    labels = ["Proline", "Glutamate", "Neu5Ac"]

    tcga_axes = pd.read_csv(
        "data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_axes_v1/axis_results.csv"
    ).set_index("axis")
    tcga_paired = pd.read_csv(
        "data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_axes_v1/paired_tumor_normal_axis_results.csv"
    ).set_index("axis")
    gse = pd.read_csv(
        "data/external/GSE236696/proline_sialic_by_lineage_v1/lineage_axis_paired_results.csv"
    )
    epithelial = gse[gse["lineage"] == "epithelial"].set_index("axis")
    proteomics = pd.read_csv(
        "data/external/mucinous_crc_proteomics_2021/proline_sialic_reanalysis_v1/axis_summary.csv"
    ).set_index("axis")
    cross = pd.read_csv(
        "data/mtbls13729/expanded_crosspanel_audit_v1/expanded_crosspanel_summary.csv"
    ).set_index("feature_id")

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), constrained_layout=True)

    ax = axes[0, 0]
    values = selected["mean_log2fc"].to_numpy(float)
    low = selected["abundance_bootstrap_ci_low"].to_numpy(float)
    high = selected["abundance_bootstrap_ci_high"].to_numpy(float)
    ax.errorbar(np.arange(3), values, yerr=[values - low, high - values], fmt="none",
                ecolor="#333333", capsize=5, linewidth=1.6)
    ax.scatter(np.arange(3), values, s=85, color=["#1b9e77", "#7570b3", "#d95f02"], zorder=3)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Rmu tumour-normal mean log2 fold change")
    ax.set_title("A  Orthogonally recovered metabolites (10 paired patients)", loc="left", fontweight="bold")

    ax = axes[0, 1]
    sample_rho = [cross.loc[index, "sample_spearman"] for index in [345, 374, 703]]
    delta_rho = [cross.loc[index, "paired_delta_spearman"] for index in [345, 374, 703]]
    x = np.arange(3)
    ax.bar(x - .18, sample_rho, .36, color="#80cdc1", label="sample-level")
    ax.bar(x + .18, delta_rho, .36, color="#018571", label="paired tumour-normal delta")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Spearman correlation to source Level-1 feature")
    ax.set_title("B  Cross-panel identity concordance", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    axes_to_show = [
        "proline_synthesis", "sialic_acid_synthesis_transport",
        "mucin_sialylation", "secretory_mucin_program", "collagen_proline_context",
    ]
    compact = ["Proline\nsynthesis", "Sialic acid\nsynthesis/transport", "Mucin\nsialylation",
               "Secretory\nmucin", "Collagen/proline\ncontext"]
    ax = axes[1, 0]
    x = np.arange(len(axes_to_show))
    paired_values = [tcga_paired.loc[name, "mean_tumor_minus_normal"] for name in axes_to_show]
    mucinous_values = [tcga_axes.loc[name, "adjusted_beta"] for name in axes_to_show]
    ax.bar(x - .19, paired_values, .38, color="#2166ac", label="CRC tumour-normal (32 pairs)")
    ax.bar(x + .19, mucinous_values, .38, color="#b2182b", label="mucinous-conventional (adjusted)")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(x, compact, fontsize=8)
    ax.set_ylabel("standardized RNA axis effect")
    ax.set_title("C  TCGA separates general CRC from subtype context", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    rows = ["GSE epithelial\npaired RNA", "Independent LMC\npooled protein", "Independent RMC\npooled protein"]
    columns = ["Proline synthesis", "Sialic synthesis/transport", "Mucin sialylation", "Collagen context"]
    mechanisms = ["proline_synthesis", "sialic_acid_synthesis_transport", "mucin_sialylation", "collagen_proline_context"]
    matrix = np.asarray([
        [epithelial.loc[name, "mean_paired_delta"] for name in mechanisms],
        [proteomics.loc[name, "LMC_vs_NC__median_log2"] for name in mechanisms],
        [proteomics.loc[name, "RMC_vs_NC__median_log2"] for name in mechanisms],
    ])
    limit = max(.5, float(np.nanmax(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), columns, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(rows)), rows, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(matrix[i, j]) > .62 * limit else "black")
    figure.colorbar(image, ax=ax, shrink=.78, label="within-dataset signed effect (descriptive)")
    ax.set_title("D  External context is axis- and compartment-dependent", loc="left", fontweight="bold")

    figure.suptitle(
        "MTBLS13729: orthogonal annotation reveals a robust proline program and a subtype-relative sialic program",
        fontsize=14, fontweight="bold",
    )
    png = output / "proline_sialic_crosscohort_summary.png"
    pdf = output / "proline_sialic_crosscohort_summary.pdf"
    figure.savefig(png, dpi=250)
    figure.savefig(pdf)
    plt.close(figure)

    report = {
        "status": "mtbls13729_proline_sialic_figure_complete",
        "formal": False,
        "outputs": {"png": str(png), "pdf": str(pdf)},
        "claim_limit": (
            "Panels combine different measurement scales for directional synthesis only. "
            "They do not establish metabolite flux, enzyme activity, glycan linkage, or causality."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
