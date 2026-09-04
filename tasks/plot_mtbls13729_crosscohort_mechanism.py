"""Create a manuscript-style evidence synthesis figure for MTBLS13729 biology."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/crosscohort_mechanism_figure_v1"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    local = pd.read_csv(
        ROOT / "data/mtbls13729/biology_closure_analysis_v1/candidate_identity_and_abundance.csv"
    ).set_index("feature_id")
    st = pd.read_csv(
        ROOT / "data/mtbls13729/external_st001087_axis_validation_v1/external_axis_metabolite_results.csv"
    ).set_index("metabolite")
    tcga = load_json(
        ROOT / "data/external/TCGA_COADREAD_Xena_20260830/mucinous_axis_analysis_v4/summary.json"
    )
    gse = pd.read_csv(
        ROOT / "data/external/GSE236696/epithelial_axis_adversarial_audit_v1/score_method_results.csv"
    )
    prot = pd.read_csv(
        ROOT / "data/external/mucinous_crc_proteomics_2021/axis_reanalysis_v1/axis_summary.csv"
    ).set_index("axis")
    oep = load_json(
        ROOT / "data/external/OEP00006137_support/modified_guanosine_reanalysis/report.json"
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(16.5, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1.05, 0.95])

    # A. Discovery-cohort paired effects.
    ax = fig.add_subplot(grid[0, 0])
    feature_order = [1597, 3019, 4966, 1717, 3222]
    labels = ["Me-guanosine\n1597", "diMe-guanosine\n3019", "purine-like\n4966", "diacetyl-polyamine\n1717", "C20:4 acylcarnitine\n3222"]
    effects = local.loc[feature_order, "rmu_mean_log2fc"].to_numpy(float)
    colors = ["#5145CD", "#6D5CE7", "#8A74E8", "#D97706", "#0F766E"]
    bars = ax.bar(np.arange(len(effects)), effects, color=colors, width=0.72)
    ax.axhline(0, color="#222", lw=0.8)
    ax.set_xticks(np.arange(len(effects)), labels)
    ax.tick_params(axis="x", labelsize=8.3)
    ax.set_ylabel("Mean paired log2 fold change, Rmu vs RN")
    ax.set_title("A  MTBLS13729 discovery cohort (10 mucinous pairs)", loc="left", weight="bold")
    ax.set_ylim(0, max(effects) + 0.55)
    for bar, value in zip(bars, effects):
        if value > 3.5:
            ax.text(bar.get_x() + bar.get_width() / 2, value - 0.12, f"{value:+.2f}", ha="center", va="top", color="white", weight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:+.2f}", ha="center", va="bottom")
    ax.text(
        0.01,
        0.91,
        "All five mean effects remain positive under leave-one-patient analysis;\npatient-level concordance varies (feature 3222: 8/10). Identities remain family-level without standards.",
        transform=ax.transAxes,
        va="top",
        color="#444",
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )

    # B. Independent metabolite observations, including discordance.
    ax = fig.add_subplot(grid[0, 1])
    oep_n2 = oep["metabolites"]["N2,N2-Dimethylguanosine"]
    names = ["ST001087\ndiMe-guanosine", "ST001087\ndiacetylspermine", "OEP MSI-H\ndiMe-guanosine", "OEP MSS\ndiMe-guanosine"]
    vals = [
        float(st.loc["N2,N2-Dimethylguanosine", "all_pairs_mean_log2fc"]),
        float(st.loc["N1,N12-Diacetylspermine", "all_pairs_mean_log2fc"]),
        float(oep_n2["MSI"]["mean_log2fc"]),
        float(oep_n2["MSS"]["mean_log2fc"]),
    ]
    bcolors = ["#6D5CE7", "#D97706", "#9CA3AF", "#9CA3AF"]
    bars = ax.bar(np.arange(len(vals)), vals, color=bcolors, width=0.68)
    ax.axhline(0, color="#222", lw=0.8)
    ax.set_xticks(np.arange(len(vals)), names)
    ax.tick_params(axis="x", labelsize=8.1)
    ax.set_ylabel("Paired mean log2 fold change, tumour vs normal")
    ax.set_title("B  Independent tissue metabolomics: support and heterogeneity", loc="left", weight="bold")
    ax.set_ylim(min(vals) - 0.27, max(vals) + 0.48)
    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.07 if value >= 0 else -0.07),
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    ax.text(
        0.01,
        0.91,
        "ST001087 is FindByFormula and sparse; OEP00006137 is Level 1.\nOpposite cohort directions rule out a universal CRC increase.",
        transform=ax.transAxes,
        va="top",
        color="#444",
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )

    # C. Multi-omic program matrix.
    ax = fig.add_subplot(grid[1, 0])
    axes = ["modified_nucleoside_processing", "purine_synthesis_salvage", "carnitine_long_chain_fao"]
    tcga_pairs = {x["axis"]: x["mean_tumor_minus_normal"] for x in tcga["paired_tumor_normal"]["results"]}
    tcga_mucinous = {x["axis"]: x["adjusted_beta"] for x in tcga["results"]}
    gse_diff = gse[gse["score_method"] == "difference_of_gene_medians"].set_index("axis")["mean"].to_dict()
    matrix = np.array(
        [
            [tcga_pairs[a] for a in axes],
            [gse_diff[a] for a in axes],
            [float(prot.loc[a, "LMC_vs_NC__median_log2"]) for a in axes],
            [float(prot.loc[a, "LMC_vs_LNMC__median_log2"]) for a in axes],
            [tcga_mucinous[a] for a in axes],
        ]
    )
    vmax = float(np.max(np.abs(matrix)))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    rows = [
        "TCGA tumour-normal (32 pairs)",
        "GSE236696 epithelial (6 pairs)",
        "Pooled proteomics LMC-normal",
        "Pooled proteomics LMC-LNMC",
        "TCGA mucinous-conventional (adjusted)",
    ]
    cols = ["Modified\nnucleosides", "Purine\nsalvage", "Long-chain\nFAO"]
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_xticks(np.arange(len(cols)), cols)
    ax.set_title("C  Orthogonal evidence: tumour program versus subtype specificity", loc="left", weight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if abs(matrix[i, j]) > 0.55 * vmax else "#111"
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", color=color, weight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="Reported axis effect")

    # D. Evidence/claim ladder.
    ax = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    ax.set_title("D  Defensible mechanism and the remaining causal gap", loc="left", weight="bold")
    boxes = [
        (0.03, 0.78, 0.94, 0.17, "1. Discovery metabolite phenotype", "Rmu shows strong accumulation of modified-guanosine, acetylated-polyamine and long-chain acylcarnitine families."),
        (0.03, 0.56, 0.94, 0.17, "2. Independent context", "Public tissue metabolomics and multi-omics support context-dependent nucleoside/polyamine remodeling. Some CRC expression cohorts show lower long-chain FAO programs, but metabolite cohorts are heterogeneous."),
        (0.03, 0.34, 0.94, 0.17, "3. Mechanistic interpretation", "Best-supported model: three parallel abundance programs. Long-chain acylcarnitines indicate carnitine-shuttle imbalance; increased entry, incomplete oxidation and impaired utilization remain competing explanations."),
        (0.03, 0.08, 0.94, 0.20, "4. Not yet established", "No external mucinous metabolomics replication, no authentic-standard positional-isomer confirmation, no isotope flux, and no enzyme perturbation. Therefore do not claim METTL1/SAT1 causality or increased FAO flux."),
    ]
    face = ["#EEF2FF", "#ECFDF5", "#FFF7ED", "#FEF2F2"]
    edge = ["#6366F1", "#10B981", "#F59E0B", "#EF4444"]
    for (x, y, w, h, title, body), fc, ec in zip(boxes, face, edge):
        rect = plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=fc, edgecolor=ec, lw=1.4)
        ax.add_patch(rect)
        ax.text(x + 0.025, y + h - 0.04, title, transform=ax.transAxes, va="top", weight="bold", color=ec)
        ax.text(
            x + 0.025,
            y + h - 0.085,
            fill(body, width=72),
            transform=ax.transAxes,
            va="top",
            fontsize=7.55,
            color="#333",
            linespacing=1.15,
        )

    fig.suptitle(
        "Rmu discovery phenotype within CRC-wide programs: convergent pathways, heterogeneous metabolites",
        fontsize=16,
        weight="bold",
    )
    png = OUT / "crosscohort_mechanism_evidence.png"
    pdf = OUT / "crosscohort_mechanism_evidence.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)


if __name__ == "__main__":
    main()
