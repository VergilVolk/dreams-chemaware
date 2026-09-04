from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/integrated_biology_evidence_v1"
AUTHOR = ROOT / "data/mtbls13729/original_vs_dreams_biology_delta_v1/author_relevant_differential_rows.csv"
CANDIDATES = ROOT / "data/mtbls13729/candidate_evidence_ledger_v1/candidate_evidence_ledger.csv"
PROTEOMICS = ROOT / "data/external/mucinous_crc_proteomics_2021/axis_reanalysis_v1/axis_summary.csv"
SCRNA = ROOT / "data/external/GSE236696/paired_axis_by_lineage_v2/lineage_axis_paired_results.csv"


COLORS = {
    "modified_guanosine": "#376AA0",
    "polyamine": "#A9574F",
    "acylcarnitine": "#D28E2D",
    "purine": "#6B55A3",
    "other": "#777777",
}


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidate = pd.read_csv(CANDIDATES)
    author = pd.read_csv(AUTHOR)
    proteomics = pd.read_csv(PROTEOMICS)
    scrna = pd.read_csv(SCRNA)

    selected = candidate.loc[candidate["feature_id"].isin([1597, 1717, 3019, 3222, 4966])].copy()
    selected["short_label"] = selected["feature_id"].map(
        {
            1597: "methylguanosine-like\n1597",
            1717: "acetylated-polyamine\n1717",
            3019: "dimethylguanosine-like\n3019",
            3222: "C20:4-acylcarnitine-like\n3222",
            4966: "purine-like C7H9N5O\n4966",
        }
    )
    selected["axis"] = selected["feature_id"].map(
        {1597: "modified_guanosine", 1717: "polyamine", 3019: "modified_guanosine", 3222: "acylcarnitine", 4966: "purine"}
    )

    author_carnitine = author.loc[
        author["comparison"].eq("rmu_vs_normal")
        & author["metabolites"].astype(str).str.contains("carnitine", case=False, na=False)
    ].copy()
    author_carnitine["FC [log2]"] = pd.to_numeric(author_carnitine["FC [log2]"], errors="coerce")
    author_carnitine = author_carnitine.sort_values("FC [log2]")

    axes_keep = ["modified_nucleoside_processing", "purine_synthesis_salvage", "carnitine_long_chain_fao"]
    proteomics = proteomics.loc[proteomics["axis"].isin(axes_keep)].copy()
    scrna = scrna.loc[scrna["lineage"].eq("epithelial") & scrna["axis"].isin(axes_keep)].copy()
    axis_labels = {
        "modified_nucleoside_processing": "modified-nucleoside\nprocessing",
        "purine_synthesis_salvage": "purine synthesis/\nsalvage",
        "carnitine_long_chain_fao": "carnitine shuttle/\nlong-chain FAO",
    }

    evidence_rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        evidence_rows.append(
            {
                "evidence_layer": "MTBLS13729_DreaMS_candidate",
                "entity": row.short_label.replace("\n", " "),
                "effect": row.rmu_mean_log2fc,
                "direction": "up",
                "unit": "paired mean log2 fold change",
                "claim_level": "discovery candidate; exact structure unresolved",
            }
        )
    for _, row in author_carnitine.iterrows():
        effect = float(row["FC [log2]"])
        evidence_rows.append(
            {
                "evidence_layer": "original_author_Rmu_vs_normal",
                "entity": row["metabolites"],
                "effect": effect,
                "direction": "up" if effect > 0 else "down",
                "unit": "author FC log2",
                "claim_level": "original author significant metabolite",
            }
        )
    for row in proteomics.itertuples(index=False):
        evidence_rows.append(
            {
                "evidence_layer": "independent_pooled_mucinous_proteomics",
                "entity": row.axis,
                "effect": row.RMC_vs_NC__median_log2,
                "direction": "up" if row.RMC_vs_NC__median_log2 > 0 else "down",
                "unit": "median protein log2 ratio",
                "claim_level": "pooled direction; not patient-level inference",
            }
        )
    for row in scrna.itertuples(index=False):
        evidence_rows.append(
            {
                "evidence_layer": "GSE236696_paired_epithelial_pseudobulk",
                "entity": row.axis,
                "effect": row.mean_paired_delta,
                "direction": "up" if row.mean_paired_delta > 0 else "down",
                "unit": "mean paired log2CPM axis delta",
                "claim_level": "six-patient direction; not mucinous specificity",
            }
        )
    pd.DataFrame(evidence_rows).to_csv(OUT / "integrated_evidence_rows.csv", index=False)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 9.7))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.11, top=0.89, hspace=0.36, wspace=0.48)

    ax = axs[0, 0]
    order = selected.sort_values("rmu_mean_log2fc")
    ax.barh(
        order["short_label"],
        order["rmu_mean_log2fc"],
        color=[COLORS[a] for a in order["axis"]],
    )
    for i, value in enumerate(order["rmu_mean_log2fc"]):
        ax.text(value + 0.05, i, f"{value:+.2f}", va="center", fontsize=8)
    ax.set_title("A  DreaMS-recovered Rmu-vs-normal candidate features")
    ax.set_xlabel("paired mean log2 fold change")
    style_axes(ax)

    ax = axs[0, 1]
    ax.barh(author_carnitine["metabolites"], author_carnitine["FC [log2]"], color="#D9A441")
    feature_3222 = float(candidate.loc[candidate["feature_id"].eq(3222), "rmu_mean_log2fc"].iloc[0])
    ax.axvline(feature_3222, color="#9B5C00", linestyle="--", linewidth=1.4)
    ax.text(
        feature_3222,
        len(author_carnitine) - 0.2,
        " DreaMS C20:4-like 3222",
        color="#7B4700",
        fontsize=8,
        ha="left",
        va="top",
    )
    ax.set_title("B  Original paper already reported a broad carnitine program")
    ax.set_xlabel("author Rmu-vs-normal log2 fold change")
    style_axes(ax)

    ax = axs[1, 0]
    proteomics = proteomics.set_index("axis").loc[axes_keep].reset_index()
    y = np.arange(len(proteomics))
    width = 0.35
    ax.barh(y - width / 2, proteomics["LMC_vs_NC__median_log2"], height=width, label="LMC/NC", color="#7AA6C2")
    ax.barh(y + width / 2, proteomics["RMC_vs_NC__median_log2"], height=width, label="RMC/NC", color="#356A8A")
    ax.set_yticks(y, [axis_labels[x] for x in proteomics["axis"]])
    ax.set_title("C  Independent pooled mucinous CRC proteomics")
    ax.set_xlabel("median protein log2 ratio")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)

    ax = axs[1, 1]
    scrna = scrna.set_index("axis").loc[axes_keep].reset_index()
    y = np.arange(len(scrna))
    values = scrna["mean_paired_delta"].to_numpy(float)
    lower = values - scrna["patient_bootstrap_ci_low"].to_numpy(float)
    upper = scrna["patient_bootstrap_ci_high"].to_numpy(float) - values
    ax.errorbar(values, y, xerr=np.vstack([lower, upper]), fmt="o", color="#6B55A3", capsize=4)
    ax.set_yticks(y, [axis_labels[x] for x in scrna["axis"]])
    for i, row in enumerate(scrna.itertuples(index=False)):
        ax.text(
            max(row.patient_bootstrap_ci_high, row.mean_paired_delta) + 0.05,
            i,
            f"{row.tumor_higher_pairs}/6 up",
            va="center",
            fontsize=8,
        )
    ax.set_title("D  Six paired mucinous CRC epithelial pseudobulks")
    ax.set_xlabel("tumor-normal axis delta (95% patient bootstrap CI)")
    style_axes(ax)

    fig.suptitle(
        "Evidence-calibrated reanalysis supports parallel abundance programs, not a causal chain",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.025,
        (
            "Boundaries: exact isomers require standards; pooled proteomics is not patient-level; "
            "static abundance/transcript/protein does not establish flux or mucinous specificity."
        ),
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.savefig(OUT / "integrated_biology_evidence.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "mtbls13729_integrated_biology_evidence_complete",
        "dreams_candidate_features": int(len(selected)),
        "author_rmu_normal_carnitines": int(len(author_carnitine)),
        "proteomics_axes": int(len(proteomics)),
        "scrna_axes": int(len(scrna)),
        "interpretation": (
            "DreaMS adds candidate ion-family resolution to author-covered pathway contexts. "
            "Acylcarnitine accumulation plus lower downstream FAO transcript/protein programs "
            "supports an FAO-utilization bottleneck hypothesis but does not prove flux."
        ),
        "claim_limit": (
            "The panels summarize heterogeneous discovery and external evidence. They are not "
            "independent replications of exact metabolite identities or mucinous specificity."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
