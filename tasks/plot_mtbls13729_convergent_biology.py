"""Plot the evidence-calibrated MTBLS13729 convergent biology summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/convergent_biology_figure_v1"


MODULE_LABELS = {
    "long_chain_acylcarnitine_accumulation": "Long-chain\nacylcarnitines",
    "acetylated_polyamine_mta_turnover": "Acetylated\npolyamine + MTA",
    "purine_modified_nucleoside_pool": "Purine / modified\nnucleosides",
    "large_neutral_amino_acid_pool": "Large neutral\namino acids",
}

MODULE_COLORS = {
    "long_chain_acylcarnitine_accumulation": "#4C78A8",
    "acetylated_polyamine_mta_turnover": "#F58518",
    "purine_modified_nucleoside_pool": "#72B7B2",
    "large_neutral_amino_acid_pool": "#B279A2",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    modules = pd.read_csv(ROOT / "data/mtbls13729/convergent_metabolic_modules_v1/module_summary.csv")
    features = pd.read_csv(ROOT / "data/mtbls13729/integrated_biology_ledger_v1/integrated_candidate_ledger.csv")
    coordination = pd.read_csv(ROOT / "data/mtbls13729/module_coordination_v1/module_pairwise_coordination.csv")

    selected = features[features["module"].isin(MODULE_LABELS)].copy()
    selected["feature_label"] = selected["label"].str.replace("_", " ", regex=False)
    selected["crosspanel_pass"] = (
        (selected["crosspanel_within_tissue_spearman"] > 0.3)
        & (selected["crosspanel_tissue_permutation_p"] < 0.01)
    )
    selected["source_assignment"] = selected["published_source_msi"].notna()
    selected["dreams_consensus"] = selected["dreams_supporting_spectra"].fillna(0) >= 10
    selected["raw_ms2"] = selected["peak_resolved_ms2_spectra"].fillna(0) > 0
    selected["paired_direction"] = selected["positive_pairs"].fillna(0) >= 8

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.35), height_ratios=(0.9, 1.1))

    ax = fig.add_subplot(grid[0, 0])
    modules = modules.copy()
    modules["label"] = modules["module"].map(MODULE_LABELS)
    modules = modules.sort_values("mean_module_log2fc")
    y = np.arange(len(modules))
    means = modules["mean_module_log2fc"].to_numpy(float)
    errors = np.vstack(
        [means - modules["bootstrap_ci_low"].to_numpy(float), modules["bootstrap_ci_high"].to_numpy(float) - means]
    )
    ax.barh(y, means, xerr=errors, color=modules["module"].map(MODULE_COLORS), alpha=0.9, capsize=4)
    ax.set_yticks(y, modules["label"])
    ax.set_xlabel("Rmu tumour vs matched normal (mean log2FC)")
    ax.set_title("A  Four post-selection modules are directionally stable")
    ax.axvline(0, color="black", lw=0.8)
    for position, (_, row) in enumerate(modules.iterrows()):
        ax.text(row["bootstrap_ci_high"] + 0.08, position, f"{int(row['positive_patients'])}/{int(row['patients'])}", va="center", fontsize=9)

    ax = fig.add_subplot(grid[0, 1])
    selected = selected.sort_values(["module", "mean_log2fc"])
    colors = selected["module"].map(MODULE_COLORS)
    y = np.arange(len(selected))
    means = selected["mean_log2fc"].to_numpy(float)
    low = selected["abundance_bootstrap_ci_low"].to_numpy(float)
    high = selected["abundance_bootstrap_ci_high"].to_numpy(float)
    ax.barh(y, means, xerr=np.vstack([means - low, high - means]), color=colors, alpha=0.88, capsize=3)
    ax.set_yticks(y, selected["feature_label"])
    ax.set_xlabel("Rmu tumour vs matched normal (mean log2FC)")
    ax.set_title("B  Named nodes and family-level candidates")
    ax.axvline(0, color="black", lw=0.8)

    ax = fig.add_subplot(grid[1, 0])
    evidence_columns = ["source_assignment", "dreams_consensus", "raw_ms2", "crosspanel_pass", "paired_direction"]
    evidence_labels = ["Published\nsource ID", "DreaMS\nconsensus", "Peak-resolved\nMS2", "Cross-panel\nconcordance", "Paired\ndirection"]
    matrix = selected[evidence_columns].astype(float).to_numpy()
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(selected)), selected["feature_label"], fontsize=8)
    ax.set_xticks(np.arange(len(evidence_labels)), evidence_labels, fontsize=8)
    ax.set_title("C  Evidence layers are complementary, not interchangeable")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "●" if matrix[i, j] else "–", ha="center", va="center", color="white" if matrix[i, j] else "#555555", fontsize=10)

    ax = fig.add_subplot(grid[1, 1])
    module_order = list(MODULE_LABELS)
    corr = pd.DataFrame(np.eye(len(module_order)), index=module_order, columns=module_order)
    qvals = pd.DataFrame(np.nan, index=module_order, columns=module_order)
    for _, row in coordination.iterrows():
        left, right = row["module_left"], row["module_right"]
        corr.loc[left, right] = corr.loc[right, left] = row["spearman_rho"]
        qvals.loc[left, right] = qvals.loc[right, left] = row["bh_q_across_six_module_pairs"]
    image = ax.imshow(corr.to_numpy(float), cmap="coolwarm", vmin=-1, vmax=1)
    short = [MODULE_LABELS[x].replace("\n", " ") for x in module_order]
    ax.set_xticks(np.arange(len(short)), short, rotation=32, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(short)), short, fontsize=8)
    ax.set_title("D  Patient-level module coordination (Spearman ρ)")
    for i in range(len(short)):
        for j in range(len(short)):
            if i == j:
                text = "1.00"
            else:
                q = qvals.iloc[i, j]
                text = f"{corr.iloc[i, j]:.2f}\nq={q:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if abs(corr.iloc[i, j]) > 0.55 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "MTBLS13729: convergent Rmu abundance programs with evidence-calibrated identities\n"
        "Post-selection discovery; abundance does not establish flux, enzyme activity, causality or mucinous specificity",
        fontsize=16,
        fontweight="bold",
    )
    png = OUT / "mtbls13729_convergent_biology.png"
    pdf = OUT / "mtbls13729_convergent_biology.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
