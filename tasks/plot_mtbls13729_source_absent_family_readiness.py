"""Plot the evidence gates for the source-table-absent family readiness audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/source_absent_family_readiness_v1"


def main() -> None:
    data = pd.read_csv(OUT / "source_absent_family_readiness.csv")
    data = data.set_index("feature_id").loc[[1717, 1597, 3019, 150, 3222]].reset_index()
    gates = [
        "abundance_gate",
        "ms2_recurrence_gate",
        "diagnostic_transition_gate",
        "candidate_panel_primary_fdr10",
        "full_untargeted_exact_fdr10",
        "exact_metabolite_claim_permitted",
    ]
    labels = ["paired abundance", "raw MS2 recurrence", "diagnostic transition", "selected-panel FDR10", "full-space FDR10", "exact identity"]
    matrix = data[gates].astype(bool).to_numpy(dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.imshow(matrix, vmin=0, vmax=1, cmap=plt.get_cmap("RdYlGn", 2), aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(data)), [f"{r.feature_id}: {r.label}" for r in data.itertuples()])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "PASS" if matrix[i, j] else "NO", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_title("Evidence gates (no composite score)")

    ax = axes[1]
    colors = ["#7b3294", "#008837", "#008837", "#5e81ac", "#5e81ac"]
    y = np.arange(len(data))
    ax.barh(y, data.primary_mean_log2fc, color=colors, alpha=0.85)
    ax.set_yticks(y, [str(value) for value in data.feature_id])
    ax.invert_yaxis()
    ax.set_xlabel("Rmu vs matched normal mean log2 fold change")
    ax.set_title("Primary protocol effect and paired direction")
    for i, row in data.iterrows():
        text = f"{int(row.primary_positive_pairs)}/{int(row.primary_rmu_pairs)} positive"
        ax.text(row.primary_mean_log2fc + 0.05, i, text, va="center", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)

    fig.suptitle("MTBLS13729: source-table-absent signals collapse to three family-level modules", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "source_absent_family_readiness.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
