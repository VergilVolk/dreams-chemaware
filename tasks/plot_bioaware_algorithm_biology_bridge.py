"""Plot version-specific BioAware evidence and its primary failure bottlenecks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/bioaware_algorithm_biology_bridge_v1"


def main() -> None:
    ledger = pd.read_csv(OUT / "bioaware_benchmark_ledger.csv")
    failures = pd.read_csv(OUT / "bioaware_failure_decomposition.csv")
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), gridspec_kw={"width_ratios": [1.05, 1.2]})

    ax = axes[0]
    x = np.arange(len(ledger))
    delta = 100 * ledger.delta_recall1.to_numpy()
    low = 100 * ledger.ci_low.to_numpy()
    high = 100 * ledger.ci_high.to_numpy()
    colors = ["#9c755f", "#4e79a7", "#e15759", "#59a14f"]
    ax.bar(x, delta, color=colors, width=0.68, edgecolor="black", linewidth=0.55)
    ax.errorbar(x, delta, yerr=np.vstack([delta - low, high - delta]), fmt="none", ecolor="black", capsize=4, lw=1.1)
    ax.axhline(0, color="#222222", lw=0.9)
    ax.set_xticks(x, ["v1\nMTBLS", "V3\ninternal", "V4\n7-panel", "V6\n5-panel"])
    ax.set_ylabel("Recall@1 difference (percentage points)")
    ax.set_title("Version-specific BioAware evidence")
    for index, row in ledger.iterrows():
        y = delta[index]
        offset = 0.5 if y >= 0 else -0.7
        va = "bottom" if y >= 0 else "top"
        ax.text(index, y + offset, f"{y:+.2f}\n{int(row.corrected)}/{int(row.introduced)}", ha="center", va=va, fontsize=9)
    ax.text(0.02, 0.02, "Labels: corrected / introduced\nAll intervals cross zero", transform=ax.transAxes, fontsize=8.5, color="#444444")

    ax = axes[1]
    labels = {
        "A_truth_absent_from_rhea": "Truth absent\nfrom Rhea",
        "F_raw_ms2_edge_favors_wrong_or_ties": "Raw network MS2\nfavors wrong/ties",
        "B_no_eligible_level1_seed_neighbor": "No eligible\nseed neighbor",
        "E_raw_truth_path_without_competing_raw_edge": "Truth path only;\nno discriminative edge",
        "D_network_path_but_no_raw_seed_spectrum": "Network path;\nno raw seed spectrum",
        "H_rescue_headroom_exists": "Residual rescue\nheadroom",
    }
    ordered = failures.sort_values(["queries", "bottleneck"], ascending=[True, True])
    y = np.arange(len(ordered))
    bars = ax.barh(y, ordered.queries, color="#76b7b2", edgecolor="black", linewidth=0.55)
    ax.set_yticks(y, [labels.get(value, value) for value in ordered.bottleneck])
    ax.set_xlabel("Official DreaMS error queries (development; n=22)")
    ax.set_title("Why one-hop network tuning saturates")
    ax.set_xlim(0, max(12, int(ordered.queries.max()) + 1))
    for bar, value in zip(bars, ordered.queries):
        ax.text(bar.get_width() + 0.18, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontsize=9)
    ax.text(
        0.98,
        0.03,
        f"Largest bottleneck: {100*report['primary_failure_bottleneck']['fraction']:.0f}% of errors",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color="#444444",
    )

    fig.suptitle("BioAware is a conservative context expert, not a confirmed identity upgrade", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = OUT / "bioaware_algorithm_biology_bridge.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "bioaware_algorithm_biology_bridge.pdf", bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
