#!/usr/bin/env python3
"""Plot pathway-level LCFA context without implying identity replication."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compact_name(feature: str) -> str:
    chain = re.match(r"^(C\d+\.\d+)_", feature)
    rt = feature.rsplit("_", 1)[-1]
    hydroxy = "OH-" if "_Hydroxy." in feature else ""
    return f"{hydroxy}{chain.group(1) if chain else feature}  RT {rt}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-results", type=Path, required=True)
    parser.add_argument("--age-results", type=Path, required=True)
    parser.add_argument("--local-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    external = pd.read_csv(arguments.external_results)
    age = pd.read_csv(arguments.age_results)
    ledger = pd.read_csv(arguments.local_ledger)
    local = ledger[ledger["feature_id"] == 3222]
    if len(local) != 1:
        raise RuntimeError("feature 3222 does not map uniquely in the local ledger")
    local_effect = float(local.iloc[0]["rmu_mean_log2fc"])

    long_significant = external[
        external["long_chain_c20_c24"] & external["paper_protocol_significant"]
    ].copy()
    if len(long_significant) != 17:
        raise RuntimeError(f"expected 17 significant C20-C24 features, got {len(long_significant)}")
    long_significant = long_significant.sort_values("log2_tumour_to_adjacent_mean_ratio")
    long_significant["label"] = long_significant["feature"].map(compact_name)

    early = age[(age["crc_age_group"] == "early") & age["long_chain_c20_c24"]].set_index(
        "feature"
    )
    late = age[(age["crc_age_group"] == "late") & age["long_chain_c20_c24"]].set_index(
        "feature"
    )
    common = early.index.intersection(late.index)
    if len(common) != 59:
        raise RuntimeError(f"expected 59 shared C20-C24 features, got {len(common)}")
    early = early.loc[common]
    late = late.loc[common]

    c204 = external[external["feature"].str.startswith("C20.4_")].copy()
    c204["label"] = c204["feature"].map(compact_name)
    c204 = c204.sort_values("log2_tumour_to_adjacent_mean_ratio")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
        }
    )
    figure = plt.figure(figsize=(15.6, 11.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1.25, 1])

    ax_a = figure.add_subplot(grid[0, 0])
    effects = long_significant["log2_tumour_to_adjacent_mean_ratio"].to_numpy()
    colors = np.where(effects >= 0, "#ba3c50", "#2d6e9f")
    y = np.arange(len(long_significant))
    ax_a.barh(y, effects, color=colors, alpha=0.88)
    ax_a.axvline(0, color="#333333", linewidth=0.8)
    ax_a.set_yticks(y, long_significant["label"])
    ax_a.set_xlabel("Tumour / adjacent log$_2$ mean ratio")
    ax_a.set_title("A  MTBLS7387: 17 FDR-significant C20-C24 features (251 pairs)", loc="left")
    for position, (_, row) in enumerate(long_significant.iterrows()):
        if bool(row["paper_standard_validated"]):
            x = float(row["log2_tumour_to_adjacent_mean_ratio"])
            ax_a.scatter(x, position, marker="*", s=90, color="#f2c14e", edgecolor="black", zorder=3)
    ax_a.text(
        0.99,
        0.02,
        "★ paper-validated standard (22:4, 22:5, 22:6)",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    ax_b = figure.add_subplot(grid[0, 1])
    # Subgroup mean ratios can explode when one subgroup mean is close to zero.
    # Paired Cohen's dz is a bounded, patient-level standardized comparison and
    # is therefore the honest effect scale for the early-versus-late panel.
    early_effect = early["paired_cohens_dz"].to_numpy()
    late_effect = late["paired_cohens_dz"].to_numpy()
    early_sig = early["paper_protocol_significant"].to_numpy(bool)
    late_sig = late["paper_protocol_significant"].to_numpy(bool)
    categories = np.select(
        [early_sig & late_sig, early_sig & ~late_sig, ~early_sig & late_sig],
        ["both", "early only", "late only"],
        default="neither",
    )
    palette = {
        "both": "#6f3c8d",
        "early only": "#df8f2d",
        "late only": "#2f8f7b",
        "neither": "#c3c7cc",
    }
    for category in ["neither", "late only", "early only", "both"]:
        mask = categories == category
        ax_b.scatter(
            early_effect[mask],
            late_effect[mask],
            s=38 if category != "neither" else 24,
            color=palette[category],
            alpha=0.9 if category != "neither" else 0.65,
            edgecolor="white",
            linewidth=0.4,
            label=f"{category} (n={int(mask.sum())})",
        )
    limits = [min(early_effect.min(), late_effect.min()) - 0.2, max(early_effect.max(), late_effect.max()) + 0.2]
    ax_b.plot(limits, limits, linestyle="--", color="#555555", linewidth=0.8)
    ax_b.axhline(0, color="#999999", linewidth=0.6)
    ax_b.axvline(0, color="#999999", linewidth=0.6)
    ax_b.set_xlim(limits)
    ax_b.set_ylim(limits)
    ax_b.set_xlabel("Early CRC paired Cohen's d$_z$ (99 pairs)")
    ax_b.set_ylabel("Late CRC paired Cohen's d$_z$ (152 pairs)")
    ax_b.set_title("B  Age-stratified long-chain remodeling is not identical", loc="left")
    ax_b.legend(frameon=False, fontsize=8, loc="lower right")

    ax_c = figure.add_subplot(grid[1, 0])
    c_labels = c204["label"].tolist() + ["Local feature 3222\nC20:4-acylcarnitine-like"]
    c_effects = c204["log2_tumour_to_adjacent_mean_ratio"].tolist() + [local_effect]
    c_colors = ["#ba3c50" if value >= 0 else "#2d6e9f" for value in c_effects[:-1]] + ["#6f3c8d"]
    positions = np.arange(len(c_labels))
    ax_c.barh(positions, c_effects, color=c_colors, alpha=0.9)
    ax_c.axvline(0, color="#333333", linewidth=0.8)
    ax_c.axhline(len(c204) - 0.5, color="#777777", linestyle=":", linewidth=1)
    ax_c.set_yticks(positions, c_labels)
    ax_c.set_xlabel("Within-cohort paired log$_2$ effect")
    ax_c.set_title("C  C20:4 context: external fatty acids versus local acylcarnitine-like ion", loc="left")
    for position, (_, row) in enumerate(c204.iterrows()):
        q = float(row["paired_t_q_bh"])
        ax_c.text(
            float(row["log2_tumour_to_adjacent_mean_ratio"]),
            position,
            f"  q={q:.3g}",
            va="center",
            ha="left" if float(row["log2_tumour_to_adjacent_mean_ratio"]) >= 0 else "right",
            fontsize=8,
        )
    ax_c.text(
        0.99,
        0.02,
        "Different analyte classes and cohorts; pathway context only, not identity replication",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#6f3c8d",
    )

    ax_d = figure.add_subplot(grid[1, 1])
    ax_d.axis("off")
    ax_d.set_title("D  Evidence ladder and the remaining causal gap", loc="left")
    boxes = [
        (
            0.01,
            0.70,
            0.28,
            0.22,
            "MTBLS13729\n10 Rmu pairs",
            "Feature 3222 +1.78 log$_2$\nacylcarnitine-class MS2\nidentity unresolved",
            "#e8dcf0",
        ),
        (
            0.36,
            0.70,
            0.28,
            0.22,
            "MTBLS7387\n251 CRC pairs",
            "17 significant C20-C24 ions\n14 increased; 3 decreased\npaired human replication",
            "#e0efe9",
        ),
        (
            0.71,
            0.70,
            0.28,
            0.22,
            "ATF6 study\ncausal tier",
            "Standards + D$_3$ tracing\nFASN perturbation + GF/FMT\nmicrobial growth/function",
            "#f5e8ce",
        ),
    ]
    for x, y0, width, height, title, body, color in boxes:
        rectangle = plt.Rectangle((x, y0), width, height, transform=ax_d.transAxes, color=color, ec="#666666")
        ax_d.add_patch(rectangle)
        ax_d.text(x + 0.013, y0 + height - 0.04, title, transform=ax_d.transAxes, va="top", fontweight="bold", fontsize=8.8)
        ax_d.text(x + 0.013, y0 + height - 0.115, body, transform=ax_d.transAxes, va="top", fontsize=7.7)
    ax_d.annotate("", xy=(0.36, 0.81), xytext=(0.29, 0.81), xycoords=ax_d.transAxes, textcoords=ax_d.transAxes, arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax_d.annotate("", xy=(0.71, 0.81), xytext=(0.64, 0.81), xycoords=ax_d.transAxes, textcoords=ax_d.transAxes, arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax_d.text(0.325, 0.94, "external context", transform=ax_d.transAxes, ha="center", fontsize=7.3)
    ax_d.text(0.675, 0.94, "causal benchmark", transform=ax_d.transAxes, ha="center", fontsize=7.3)
    boundary = (
        "Defensible synthesis\n"
        "• Rmu contains a long-chain acylcarnitine-like abundance signal.\n"
        "• Independent paired CRC tissue supports long-chain fatty-acid remodeling.\n"
        "• The ATF6 study shows how such a lipid phenotype can be made causal.\n\n"
        "Not established here\n"
        "• feature 3222 exact isomer or Level 1 identity\n"
        "• ATF6 activity, FAO flux, or mucinous specificity in MTBLS13729\n"
        "• equivalence between free/hydroxy fatty acids and acylcarnitines"
    )
    ax_d.text(0.03, 0.58, boundary, transform=ax_d.transAxes, va="top", linespacing=1.35)

    figure.suptitle(
        "External paired-human evidence supports a long-chain lipid context, not direct feature identity",
        fontsize=15,
        fontweight="bold",
    )
    png = output_dir / "mtbls13729_mtbls7387_lcfa_context.png"
    pdf = output_dir / "mtbls13729_mtbls7387_lcfa_context.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    report = {
        "status": "mtbls13729_mtbls7387_lcfa_context_figure_complete",
        "external_pairs": 251,
        "external_significant_c20_c24": int(len(long_significant)),
        "external_increased_c20_c24": int((effects > 0).sum()),
        "external_decreased_c20_c24": int((effects < 0).sum()),
        "local_feature_3222_log2fc": local_effect,
        "identity_boundary": "pathway-level long-chain lipid context; no direct identity or flux replication",
        "png": str(png),
        "pdf": str(pdf),
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
