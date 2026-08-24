#!/usr/bin/env python
"""Plot the phenotype-blind MTBLS13729 long-chain acylcarnitine result.

This script only visualizes already-computed paired statistics. It does not
reselect features or use phenotype labels to redefine the class panel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "data" / "mtbls13729" / "acylcarnitine_panel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_PANEL / "acylcarnitine_biology_result",
    )
    return parser.parse_args()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    args = parse_args()
    panel_dir = args.panel_dir.resolve()
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    pair = pd.read_csv(panel_dir / "acylcarnitine_chain_collapsed_pair_scores.csv")
    feat = pd.read_csv(panel_dir / "acylcarnitine_class_score_features.csv")
    null = pd.read_csv(panel_dir / "acylcarnitine_matched_background_null.csv.gz")
    report = json.loads((panel_dir / "class_score_report.json").read_text(encoding="utf-8"))
    bg_report = json.loads(
        (panel_dir / "matched_background_report.json").read_text(encoding="utf-8")
    )

    pair = pair[pair["normalization"].eq("pqn")].copy()
    feat = feat[feat["used_in_class_score"].astype(str).str.lower().eq("true")].copy()
    # Some isobaric Cn:u/adduct hypotheses map to the same quantified MS1
    # feature. The class score is defined on unique MS1 features, so the plot
    # must use the same unit of analysis. The most widely MS2-supported mass
    # hypothesis is retained only as a label; it is not treated as a confirmed
    # structural assignment.
    feat = (
        feat.sort_values(
            ["n_samples_with_ms2", "n_ms2_spectra"], ascending=[False, False]
        )
        .drop_duplicates("feature_id", keep="first")
        .copy()
    )
    if len(feat) != int(report["n_unique_ms1_features"]):
        raise ValueError(
            f"Expected {report['n_unique_ms1_features']} selected features, found {len(feat)}"
        )

    style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 4.1),
        gridspec_kw={"width_ratios": [0.85, 1.6, 1.15]},
        constrained_layout=True,
    )

    # A: patient-level class score.
    ax = axes[0]
    group_order = ["Rmu", "Rtu"]
    colors = {"Rmu": "#9B2C6B", "Rtu": "#4C78A8"}
    rng = np.random.default_rng(20260820)
    for x, group in enumerate(group_order):
        values = pair.loc[pair["tumour_suffix"].eq(group), "class_median_log2fc"].to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=35,
            color=colors[group],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        mean = values.mean()
        ax.plot([x - 0.20, x + 0.20], [mean, mean], color="black", lw=2.0, zorder=4)
    ax.axhline(0, color="#666666", lw=0.8, ls="--")
    ax.set_xticks(range(2), ["Rmu–RN", "Rtu–RN"])
    ax.set_ylabel("Patient class score\nmedian paired log$_2$ fold change")
    ax.set_title("A  Patient-level class accumulation", loc="left", fontweight="bold")
    ax.text(
        0.03,
        0.98,
        "Rmu exact p=0.0078\nSubtype interaction p=0.0694",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    # B: all selected MS1 features; labels retain the mass-defined chain hypothesis.
    ax = axes[1]
    feat = feat.sort_values(
        ["pqn__rmu_vs_rn_mean_log2fc", "carbon", "unsaturation"], ascending=[True, True, True]
    ).reset_index(drop=True)
    labels = [
        f"{chain}  RT {rt / 60:.1f}"
        for chain, rt in zip(feat["acyl_chain"], feat["median_rt_sec"])
    ]
    y = np.arange(len(feat))
    rmu = feat["pqn__rmu_vs_rn_mean_log2fc"].to_numpy()
    rtu = feat["pqn__rtu_vs_rn_mean_log2fc"].to_numpy()
    for yi, a, b in zip(y, rtu, rmu):
        ax.plot([a, b], [yi, yi], color="#C7C7C7", lw=1.0, zorder=1)
    ax.scatter(rtu, y, s=22, color=colors["Rtu"], label="Rtu–RN", zorder=2)
    ax.scatter(rmu, y, s=25, color=colors["Rmu"], label="Rmu–RN", zorder=3)
    ax.axvline(0, color="#666666", lw=0.8, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean paired log$_2$ fold change")
    ax.set_title("B  Phenotype-blind long-chain panel", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    # C: competitive matched-feature null.
    ax = axes[2]
    rmu_col = next(c for c in null.columns if "rmu" in c.lower())
    int_col = next(c for c in null.columns if "interaction" in c.lower())
    bins = np.linspace(
        min(null[rmu_col].min(), null[int_col].min()),
        max(null[rmu_col].max(), null[int_col].max()),
        35,
    )
    ax.hist(null[rmu_col], bins=bins, color=colors["Rmu"], alpha=0.45, label="Matched null: Rmu")
    ax.hist(null[int_col], bins=bins, color="#E0A42B", alpha=0.45, label="Matched null: interaction")
    observed_rmu = float(bg_report["observed_rmu_mean_class_log2fc"])
    observed_int = float(bg_report["observed_interaction_difference"])
    ax.axvline(observed_rmu, color=colors["Rmu"], lw=2.0)
    ax.axvline(observed_int, color="#B36B00", lw=2.0)
    ax.set_xlabel("Mean class effect")
    ax.set_ylabel("Matched random panels")
    ax.set_title("C  Feature-level matched background", loc="left", fontweight="bold")
    ax.text(
        0.97,
        0.97,
        "2,000 matched panels\nempirical p=0.00050 (both)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.83), fontsize=7.5)

    fig.suptitle(
        "Long-chain acylcarnitines accumulate in right-sided mucinous CRC",
        fontsize=13,
        fontweight="bold",
    )

    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
