"""Create a claim-bounded three-cohort LCNEC mechanism summary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_three_cohort_mechanism_v1"
METABOLITE = ROOT / "data/validation/lcnec_hsst3n_multicohort_triangulation_v1/candidate_triangulation.csv"
PROTEIN = ROOT / "data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1/protein_results.csv"
GENE = ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_genomic_audit_v1/gene_genomic_results.csv"
AXIS = ROOT / "data/external/LCNEC_George2018_transcriptome/frozen_axis_genomic_audit_v1/axis_genomic_results.csv"


AXIS_COLORS = {
    "quinolinate_de_novo_nad": "#7a5195",
    "adp_ribose_turnover": "#ef5675",
    "ascorbate_redox": "#ffa600",
    "phosphorylated_nucleotide": "#2f4b7c",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metabolite = pd.read_csv(METABOLITE)
    protein = pd.read_csv(PROTEIN)
    gene = pd.read_csv(GENE)
    axis = pd.read_csv(AXIS)

    if len(metabolite) != 4 or protein["primary_protein_gate"].astype(bool).sum() != 13:
        raise RuntimeError("frozen metabolite/protein inputs changed")
    if gene["secondary_gene_gate"].astype(bool).sum() != 4 or not axis["fixed_axis_gate"].astype(bool).all():
        raise RuntimeError("frozen external genomic inputs changed")

    fig = plt.figure(figsize=(15.2, 7.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.35, 1.15], wspace=0.42)

    # Panel A: direct paired metabolite abundance, which is the only metabolite-level outcome.
    ax = fig.add_subplot(grid[0, 0])
    metabolite = metabolite.set_index("priority_name").loc[[
        "adenosine_diphosphate_family",
        "adenosine_diphosphoribose_family",
        "quinolinate",
        "ascorbate",
    ]].reset_index()
    labels = ["ADP family", "ADP-ribose family", "Quinolinate", "Ascorbate"]
    colors = [AXIS_COLORS["phosphorylated_nucleotide"], AXIS_COLORS["adp_ribose_turnover"],
              AXIS_COLORS["quinolinate_de_novo_nad"], AXIS_COLORS["ascorbate_redox"]]
    y = np.arange(4)
    ax.barh(y, metabolite["local_mean_log2fc"], color=colors, alpha=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Tumor - adjacent mean log2 abundance")
    ax.set_title("A  Local metabolite hypotheses\n34 paired LCNEC tissues", loc="left", fontweight="bold")
    for i, row in metabolite.iterrows():
        ax.text(float(row.local_mean_log2fc) + 0.10, i,
                f"{int(row.local_concordant_pairs)}/34", va="center", fontsize=9)
    ax.text(0.0, -0.13, "Level 2 / connectivity family; no exact identity claim",
            transform=ax.transAxes, fontsize=8.5, color="#555555")

    # Panel B: independent paired protein context. Show every protein passing the frozen primary gate.
    ax = fig.add_subplot(grid[0, 1])
    passed = protein.loc[protein["primary_protein_gate"].astype(bool)].copy()
    axis_order = ["quinolinate_de_novo_nad", "adp_ribose_turnover", "ascorbate_redox"]
    passed["axis_order"] = passed["axis"].map({name: i for i, name in enumerate(axis_order)})
    passed = passed.sort_values(["axis_order", "mean_effect"])
    y = np.arange(len(passed))
    ax.barh(y, passed["mean_effect"], color=[AXIS_COLORS[name] for name in passed["axis"]], alpha=0.9)
    ax.set_yticks(y, passed["gene"])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Tumor - NAT mean log2 protein abundance")
    ax.set_title("B  Independent protein context\n80 paired pure-LCNEC cases", loc="left", fontweight="bold")
    ax.set_xlim(-1.35, 1.65)
    for i, row in enumerate(passed.itertuples()):
        ax.text(row.mean_effect + 0.045, i, f"q={row.primary_bh_q_22:.2g}",
                ha="left", va="center", fontsize=7.3)
    ax.text(0.0, -0.13, "Protein context is not metabolite replication or flux",
            transform=ax.transAxes, fontsize=8.5, color="#555555")

    # Panel C: expression-independent genomic contrast in a second external LCNEC cohort.
    ax = fig.add_subplot(grid[0, 2])
    selected = gene.loc[gene["secondary_gene_gate"].astype(bool)].copy()
    selected = selected.set_index("gene").loc[["NMNAT1", "NMNAT3", "PARP1", "TKT"]].reset_index()
    y = np.arange(len(selected))
    ax.barh(y, selected["median_difference_stk11_keap1_minus_rb1"],
            color=[AXIS_COLORS[name] for name in selected["axis"]], alpha=0.9)
    ax.set_yticks(y, selected["gene"])
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("STK11/KEAP1 - RB1 median log2 RSEM")
    ax.set_title("C  External genomic-stratum context\n22 vs 17 LCNEC tumors", loc="left", fontweight="bold")
    ax.set_xlim(-1.78, 1.58)
    for i, row in selected.iterrows():
        ax.text(row.median_difference_stk11_keap1_minus_rb1 + 0.045, i,
                f"q={row.bh_q_22:.2g}",
                ha="left", va="center", fontsize=8)
    axis_text = "Axis R2: NAD 0.111; ADP-ribose 0.104; redox 0.137\nLOO: redox 8/8; NAD 8/9; ADP-ribose 4/5 omissions pass"
    ax.text(0.0, -0.13, axis_text, transform=ax.transAxes, fontsize=8.5, color="#555555")

    for panel in fig.axes:
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="x", alpha=0.18)
    fig.suptitle("Three-cohort triangulation of LCNEC metabolite hypotheses and pathway context",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.text(0.5, 0.015,
             "The panels answer different questions: paired metabolite abundance, paired protein context, and tumor-only genomic heterogeneity.",
             ha="center", fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(bottom=0.20, top=0.88)
    fig.savefig(OUT / "lcnec_three_cohort_mechanism.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "lcnec_three_cohort_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "lcnec_three_cohort_mechanism_figure_complete",
        "formal": True,
        "local_metabolite_pairs": 34,
        "independent_protein_pairs": 80,
        "external_clean_genomic_tumors": 39,
        "metabolite_priorities": 4,
        "proteins_passing": 13,
        "genomic_axes_passing": 3,
        "genomic_genes_passing": 4,
        "claim_limit": "Three orthogonal contexts; no cross-panel substitution of identity, abundance replication, flux or causality.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
