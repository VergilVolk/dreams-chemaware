"""Render the frozen MTBLS13729 annotation benchmark as a paper-ready figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/mtbls13729/annotation_biology_benchmark_v1/report.json"
OUTPUT = ROOT / "data/mtbls13729/annotation_biology_benchmark_v1/annotation_benchmark.png"


def main() -> None:
    report = json.loads(INPUT.read_text(encoding="utf-8"))
    native = report["source_paper_native"]
    systems = report["shared_rplc_target_universe"]["systems"]

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    labels_a = ["All detected\nfeatures", "MS2-bearing\nfeatures"]
    values_a = [
        100 * native["annotation_rate_all_detected"],
        100 * native["annotation_rate_ms2_eligible"],
    ]
    bars = axes[0].bar(labels_a, values_a, color=["#7f8c8d", "#34495e"], width=0.62)
    axes[0].set_title("A  Source-paper native annotation")
    axes[0].set_ylabel("Annotation rate (%)")
    axes[0].set_ylim(0, max(values_a) * 1.35)
    axes[0].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[0].text(
        0.5, -0.24, "345 annotated metabolites; RPLC and HILIC combined",
        transform=axes[0].transAxes, ha="center", va="top", fontsize=9,
    )

    order = [
        "author_shared_rplc_coordinates", "official_dreams", "experimental_e6",
        "frozen_p2b", "threeway_consensus", "threeway_union",
    ]
    labels_b = ["Author\ncoordinates", "Official\nDreaMS", "E6\nembedding", "Frozen\nP2b", "3-way\nconsensus", "3-way\nunion"]
    values_b = [100 * systems[key]["rate"] for key in order]
    colors_b = ["#7f8c8d", "#4c78a8", "#2ca02c", "#f28e2b", "#9467bd", "#d62728"]
    bars = axes[1].bar(labels_b, values_b, color=colors_b, width=0.72)
    axes[1].set_title("B  Shared RPLC target universe")
    axes[1].set_ylabel("Coverage of 16,953 targets (%)")
    axes[1].set_ylim(0, max(values_b) * 1.22)
    axes[1].tick_params(axis="x", labelsize=8.5)
    axes[1].bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8.5)

    labels_c = ["Official\nDreaMS", "E6\nembedding", "Frozen\nP2b"]
    supported = [
        systems["official_dreams"]["level2a_supported"],
        systems["experimental_e6"]["level2a_supported"],
        systems["frozen_p2b"]["level2a_supported"],
    ]
    colors_c = ["#4c78a8", "#2ca02c", "#f28e2b"]
    bars = axes[2].bar(labels_c, supported, color=colors_c, width=0.65)
    axes[2].set_title("C  Strong spectral-evidence tier")
    axes[2].set_ylabel("Level 2a-supported features")
    axes[2].set_ylim(0, max(supported) * 1.25)
    axes[2].bar_label(bars, padding=3)
    axes[2].text(
        0.5, -0.24, "Coverage and evidence quality are separate endpoints",
        transform=axes[2].transAxes, ha="center", va="top", fontsize=9,
    )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("MTBLS13729: source annotation, DreaMS and the integrated tool", fontsize=14, fontweight="bold")
    figure.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
