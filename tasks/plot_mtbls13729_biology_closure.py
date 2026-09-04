#!/usr/bin/env python
"""Create compact publication-draft figures for MTBLS13729 biology closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    4966: "C7H9N5O\npurine-like",
    3019: "dimethyl-\nguanosine",
    1597: "methylguanosine\n[M+H]+",
    7489: "methylguanosine\n[M+Na]+",
    1717: "N1,N8-diacetyl-\nspermidine",
    3222: "C20:4\nacylcarnitine",
    3180: "interpretation\ncontrol",
    16425: "LPE-like",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/mtbls13729/biology_closure_analysis_v1"))
    args = parser.parse_args()
    out = args.analysis_dir

    effects = pd.read_csv(out / "paired_abundance_by_normalization.csv")
    patients = pd.read_csv(out / "fully_ion_family_collapsed_module_patient_effects.csv")
    # Figure 1: individual patient effects and normalization sensitivity.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    ax = axes[0]
    raw = patients[(patients.normalization == "raw") & patients.cohort.isin(["Rmu", "Rtu"])].copy()
    colors = {"Rmu": "#8c2d6f", "Rtu": "#2878b5"}
    for position, cohort in enumerate(["Rmu", "Rtu"]):
        values = raw.loc[raw.cohort.eq(cohort), "module_log2fc"].dropna().to_numpy(float)
        jitter = np.linspace(-0.09, 0.09, len(values))
        ax.scatter(np.full(len(values), position) + jitter, values, s=48, color=colors[cohort], alpha=0.9, edgecolor="white", linewidth=0.6)
        ax.plot([position - 0.22, position + 0.22], [np.mean(values), np.mean(values)], color="black", lw=2)
    ax.axhline(0, color="#555555", lw=1, ls="--")
    ax.set_xticks([0, 1], ["Rmu vs matched RN", "Rtu vs matched RN"])
    ax.set_ylabel("Ion-family-collapsed log2 fold change")
    ax.set_title("Modified-guanosine module: patient-level effects")
    ax.text(0.02, 0.98, "Rmu: all complete pairs > 0", transform=ax.transAxes, va="top", fontsize=9)

    ax = axes[1]
    family = json.loads((out / "report.json").read_text(encoding="utf-8"))["dimethylguanosine_ion_family_support"]
    module = pd.DataFrame(family["fully_ion_family_collapsed_modified_guanosine"])
    order = ["raw", "global_pqn_prev60", "global_pqn_prev80", "global_pqn_prev90"]
    module = module.set_index("normalization").loc[order]
    ax.bar(np.arange(len(order)), module.mean_log2fc, color=["#5b4b8a", "#6c6db2", "#4c8fc4", "#4ba3a5"])
    ax.set_xticks(np.arange(len(order)), ["Raw", "PQN\nprev>=60%", "PQN\nprev>=80%", "PQN\nprev>=90%"])
    ax.set_ylabel("Mean paired log2 fold change")
    ax.set_title("Robust to phenotype-blind background normalization")
    ax.set_ylim(0, max(module.mean_log2fc) * 1.25)
    for i, row in enumerate(module.itertuples()):
        ax.text(i, row.mean_log2fc + 0.08, f"p={row.exact_signflip_p:.4f}", ha="center", fontsize=8)
    fig.savefig(out / "modified_guanosine_module_summary.png", dpi=220)
    plt.close(fig)

    # Figure 2: candidate effects under the primary local sensitivity view.
    view = effects[effects.normalization.eq("global_pqn_prev80")].set_index("feature_id").loc[list(LABELS)]
    matrix = view[["rmu_mean_log2fc", "rtu_mean_log2fc", "interaction_log2fc"]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.7, 6.2), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(matrix)))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(view)), [LABELS[int(x)] for x in view.index])
    ax.set_xticks([0, 1, 2], ["Rmu-RN", "Rtu-RN", "Rmu-Rtu\neffect difference"])
    ax.set_title("Frozen candidate abundance effects (global PQN, prevalence >=80%)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(matrix[row, col]) > 0.58 * vmax else "black")
    fig.colorbar(image, ax=ax, label="log2 fold change", shrink=0.78)
    fig.savefig(out / "biology_candidate_effect_heatmap.png", dpi=220)
    plt.close(fig)

    print(out / "modified_guanosine_module_summary.png")
    print(out / "biology_candidate_effect_heatmap.png")


if __name__ == "__main__":
    main()
