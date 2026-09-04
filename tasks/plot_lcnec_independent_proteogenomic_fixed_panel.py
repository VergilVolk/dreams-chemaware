"""Plot all frozen proteins from the independent LCNEC proteogenomic audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


AXIS_COLORS = {
    "quinolinate_de_novo_nad": "#2F6B9A",
    "adp_ribose_turnover": "#8F4E8B",
    "ascorbate_redox": "#2F7D5A",
}


def bootstrap_mean_ci(values: np.ndarray, seed: int, repeats: int = 5000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(repeats, values.size))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(
            "data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1/figures"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    proteins = pd.read_csv(args.result_dir / "protein_results.csv")
    pairs = pd.read_csv(args.result_dir / "pure_lcnec_patient_pair_differences.csv")
    proteins["primary_ci_low"] = np.nan
    proteins["primary_ci_high"] = np.nan
    for index, row in proteins.iterrows():
        if not bool(row["measured"]):
            continue
        values = pairs.loc[pairs["gene"] == row["gene"], "tumor_minus_normal"].to_numpy(float)
        low, high = bootstrap_mean_ci(values, seed=20260901 + index)
        proteins.loc[index, ["primary_ci_low", "primary_ci_high"]] = [low, high]

    proteins.to_csv(args.output_dir / "protein_results_with_bootstrap_ci.csv", index=False)
    y = np.arange(len(proteins))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 8.4), gridspec_kw={"width_ratios": [1.12, 1.0]})

    ax = axes[0]
    for position, (_, row) in zip(y, proteins.iterrows()):
        color = AXIS_COLORS[row["axis"]]
        if bool(row["measured"]):
            mean = float(row["mean_effect"])
            low = float(row["primary_ci_low"])
            high = float(row["primary_ci_high"])
            ax.errorbar(
                mean,
                position,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                markersize=7 if bool(row["primary_protein_gate"]) else 5,
                markerfacecolor=color if bool(row["primary_protein_gate"]) else "white",
                markeredgecolor=color,
                ecolor=color,
                capsize=2.5,
                linewidth=1.3,
            )
        else:
            ax.scatter(0, position, marker="x", color="#9A9A9A", s=34)
    ax.axvline(0, color="#444444", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{row.gene}{' *' if bool(row.primary_protein_gate) else ''}"
            for row in proteins.itertuples(index=False)
        ]
    )
    ax.set_xlabel("Pure LCNEC protein abundance: tumor − NAT (log2)")
    ax.set_title("A  Frozen paired endpoint (80 patients)", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.7)

    ax = axes[1]
    for position, (_, row) in zip(y, proteins.iterrows()):
        color = AXIS_COLORS[row["axis"]]
        if bool(row["measured"]):
            value = float(row["secondary_combined_minus_pure_median"])
            secondary_pass = float(row["secondary_bh_q_22"]) < 0.10
            ax.scatter(
                value,
                position,
                s=58 if secondary_pass else 32,
                facecolor=color if secondary_pass else "white",
                edgecolor=color,
                linewidth=1.2,
            )
        else:
            ax.scatter(0, position, marker="x", color="#9A9A9A", s=34)
    ax.axvline(0, color="#444444", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("Combined − pure tumor abundance (median log2; exploratory)")
    ax.set_title("B  Histology-context positive control", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.7)

    fig.suptitle(
        "Independent LCNEC proteogenomics: 107-patient cohort, 103 quantified pairs",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                   markeredgecolor=color, markersize=7, label=axis.replace("_", " "))
            for axis, color in AXIS_COLORS.items()
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        fontsize=8.8,
    )
    fig.text(
        0.5,
        0.018,
        "Filled markers: fixed-panel BH q<0.10 (primary also requires all direction-stability gates). "
        "×: not measured; no substitution. Protein abundance supplies context only—not metabolite identity, flux, or causality.",
        ha="center",
        va="bottom",
        fontsize=9.2,
    )
    fig.tight_layout(rect=(0.03, 0.055, 0.99, 0.925))
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"independent_fixed_panel.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
