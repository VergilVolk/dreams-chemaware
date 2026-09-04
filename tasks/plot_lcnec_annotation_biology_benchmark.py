"""Plot the denominator-safe LCNEC annotation and discovery funnel."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_annotation_benchmark_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    src = report["source_paper"]
    qual = report["reconstructed_qualified_universe"]
    dark = report["frozen_dark_universe"]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10.5})
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.8), gridspec_kw={"width_ratios": [0.9, 1.0, 1.35]})

    ax = axes[0]
    values = [src["msi_level_1_2_3"][0], src["msi_level_1_2_3"][1], src["msi_level_1_2_3"][2]]
    labels = ["Level 1", "Level 2", "Level 3"]
    colors = ["#59a14f", "#4e79a7", "#bab0ac"]
    bottom = 0
    for value, label, color in zip(values, labels, colors):
        ax.bar([0], [value], bottom=[bottom], color=color, edgecolor="white", width=0.6, label=f"{label}: {value}")
        bottom += value
    ax.set_xticks([0], ["Source atlas\n1,052 declared"])
    ax.set_ylabel("Annotated statistical rows")
    ax.set_title("Source paper")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.5, -0.30, "Detected-feature denominator absent\n→ annotation rate unavailable", transform=ax.transAxes, ha="center", va="top", color="#9c2f2f", fontweight="bold", fontsize=9)

    ax = axes[1]
    values = [qual["source_hsst3n_overlap"], qual["source_table_absent"]]
    labels = ["Matches source\nHSST3n table", "Source-table-absent\nanalytical headroom"]
    bars = ax.bar(np.arange(2), values, color=["#59a14f", "#f28e2b"], edgecolor="black", linewidth=0.55)
    ax.set_xticks(np.arange(2), labels)
    ax.set_ylim(0, 250)
    ax.set_title("Reconstructed QC-qualified universe\n(n=263)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 5, f"{value}\n({100*value/263:.1f}%)", ha="center", fontsize=9)
    ax.text(0.5, -0.30, "Absence from the source table\n≠ a novel metabolite", transform=ax.transAxes, ha="center", va="top", color="#555555", fontsize=9)

    ax = axes[2]
    labels = ["DreaMS\ncandidate", "DreaMS–P2b\nagreement", "Multi-evidence\nretained", "Cross-platform\nreproduced", "Author-\nunreported", "Priority"]
    values = [
        dark["official_dreams_candidates"],
        dark["dreams_p2b_agreement"],
        dark["high_or_moderate_consistency_features"],
        dark["cross_platform_reproductions"],
        dark["author_unreported_hypotheses"],
        dark["priority_hypotheses"],
    ]
    colors = ["#4e79a7", "#76b7b2", "#edc948", "#59a14f", "#f28e2b", "#e15759"]
    bars = ax.bar(np.arange(len(values)), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(values)), labels, rotation=20, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 60)
    ax.set_title("Frozen dark-module universe\n(n=81)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value}\n{100*value/81:.1f}%", ha="center", fontsize=8.5)
    ax.text(0.5, -0.38, "Coverage → agreement → evidence calibration\nnot successive accuracy estimates", transform=ax.transAxes, ha="center", va="top", color="#555555", fontsize=9)

    fig.suptitle("LCNEC annotation recovery with explicit denominators", fontsize=15, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.30, wspace=0.28)
    fig.savefig(OUT / "lcnec_annotation_biology_benchmark.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "lcnec_annotation_biology_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)
    print(OUT / "lcnec_annotation_biology_benchmark.png")


if __name__ == "__main__":
    main()
