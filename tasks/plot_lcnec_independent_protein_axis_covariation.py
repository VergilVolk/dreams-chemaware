"""Plot exploratory within-axis protein covariation without hiding failed pairs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external/LCNEC_proteogenomic_2026/protein_axis_covariation_exploratory_v1"


def main() -> None:
    pairs = pd.read_csv(BASE / "all_within_axis_pairwise_covariation.csv")
    axes = list(pairs["axis"].drop_duplicates())
    fig, plots = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    image = None
    for ax, axis in zip(plots, axes):
        block = pairs[pairs["axis"] == axis]
        genes = sorted(set(block["gene_a"]) | set(block["gene_b"]))
        matrix = np.eye(len(genes))
        passing = np.zeros((len(genes), len(genes)), dtype=bool)
        index = {gene: i for i, gene in enumerate(genes)}
        for row in block.itertuples(index=False):
            i, j = index[row.gene_a], index[row.gene_b]
            matrix[i, j] = matrix[j, i] = row.rho
            passing[i, j] = passing[j, i] = row.exploratory_gate
        image = ax.imshow(matrix, vmin=-0.6, vmax=0.6, cmap="RdBu_r")
        ax.set_xticks(range(len(genes)), genes, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(len(genes)), genes, fontsize=8)
        ax.set_title(axis.replace("_", " "), fontsize=11)
        for i in range(len(genes)):
            for j in range(len(genes)):
                if i == j:
                    text = "1"
                else:
                    text = f"{matrix[i, j]:.2f}" + ("*" if passing[i, j] else "")
                ax.text(j, i, text, ha="center", va="center", fontsize=7,
                        color="white" if abs(matrix[i, j]) > 0.42 else "black")
    if image is not None:
        cbar = fig.colorbar(image, ax=plots, shrink=0.78, pad=0.02)
        cbar.set_label("Spearman rho of tumor-minus-NAT protein effects")
    fig.suptitle("Independent pure-LCNEC protein-axis covariation (80 paired patients)\n"
                 "* exploratory gate: |rho|>=0.30, BH46 q<0.05, bootstrap CI excludes 0, LOO sign stable",
                 fontsize=13)
    fig.savefig(BASE / "protein_axis_covariation.png", dpi=220, bbox_inches="tight")
    fig.savefig(BASE / "protein_axis_covariation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_lcnec_independent_protein_axis_covariation] PASS axes={len(axes)}")


if __name__ == "__main__":
    main()

