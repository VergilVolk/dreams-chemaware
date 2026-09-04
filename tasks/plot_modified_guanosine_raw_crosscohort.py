#!/usr/bin/env python
"""Publication-style cross-cohort raw-data figure for the biology application."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local",
        type=Path,
        default=Path("data/mtbls13729/biology_axes_analysis_v1/raw_patient_axes.csv"),
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=Path(
            "data/external/OEP00006137_raw/methyl_purine_coupling_v1/patient_deltas.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/mtbls13729/modified_guanosine_raw_crosscohort_20260830.png"
        ),
    )
    return parser.parse_args()


def strip(ax, values, position, color, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    x = position + rng.uniform(-0.09, 0.09, len(values))
    ax.scatter(x, values, s=35, color=color, alpha=0.82, edgecolor="white", linewidth=0.5)
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    ax.errorbar(position, mean, yerr=1.96 * sem, color="black", marker="D", ms=5, capsize=4)
    return mean


def main() -> None:
    args = parse_args()
    local = pd.read_csv(args.local)
    external = pd.read_csv(args.external)
    colors = {"local": "#8A3FFC", "MSI-H": "#D1495B", "MSS": "#2F6690"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)

    local_values = local["modified_guanosine_module"].dropna().to_numpy()
    strip(axes[0], local_values, 0, colors["local"], 1)
    axes[0].axhline(0, color="#777777", lw=1)
    axes[0].set_xticks([0], ["MTBLS13729\nRmu (n=10)"])
    axes[0].set_ylabel("Tumor vs matched normal log2FC")
    axes[0].set_title("A  Local modified-guanosine module", loc="left", fontweight="bold")
    axes[0].text(
        0.97, 0.93, "10/10 increased", transform=axes[0].transAxes,
        ha="right", va="top", fontsize=10,
    )

    for position, subtype in enumerate(("MSI-H", "MSS")):
        values = external.loc[
            external["subtype"].eq(subtype), "modified_guanosine_3peak_mean"
        ]
        strip(axes[1], values, position, colors[subtype], 10 + position)
    axes[1].axhline(0, color="#777777", lw=1)
    axes[1].set_xticks([0, 1], ["MSI-H", "MSS"])
    axes[1].set_title("B  External raw 3-peak module", loc="left", fontweight="bold")

    for position, subtype in enumerate(("MSI-H", "MSS")):
        values = external.loc[external["subtype"].eq(subtype), "M385T405"]
        strip(axes[2], values, position, colors[subtype], 20 + position)
    axes[2].axhline(0, color="#777777", lw=1)
    axes[2].set_xticks([0, 1], ["MSI-H", "MSS"])
    axes[2].set_title("C  External raw SAH", loc="left", fontweight="bold")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E6E6E6", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Context-dependent modified-guanosine remodeling and an independent SAH pool increase",
        fontsize=14,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
