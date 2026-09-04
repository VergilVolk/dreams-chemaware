"""Plot the frozen subtype effects and candidate claim gates for the manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATIENT = ROOT / "data/mtbls13729/module_subtype_interactions_v1/module_patient_paired_effects.csv"
SUMMARY = ROOT / "data/mtbls13729/module_subtype_interactions_v1/module_subtype_interaction_summary.csv"
SCORECARD = ROOT / "data/mtbls13729/candidate_claim_scorecard_v3/candidate_claim_scorecard_v3.csv"
OUT = ROOT / "data/mtbls13729/subtype_claim_figure_v1"

MODULE_ORDER = [
    "neu5ac",
    "acetylated_polyamine_mta",
    "long_chain_acylcarnitine",
    "purine_modified_guanosine",
    "expanded_amino_acid",
]
MODULE_LABELS = {
    "neu5ac": "Neu5Ac",
    "acetylated_polyamine_mta": "Acetylated\npolyamine–MTA",
    "long_chain_acylcarnitine": "Long-chain\nacylcarnitine",
    "purine_modified_guanosine": "Purine / modified\nguanosine",
    "expanded_amino_acid": "Expanded\namino-acid",
}
COHORT_COLORS = {"Rmu": "#9C2F45", "Rtu": "#2E6F9E"}


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    patient = pd.read_csv(PATIENT)
    summary = pd.read_csv(SUMMARY)
    score = pd.read_csv(SCORECARD)

    patient = patient[(patient.normalization == "log_raw") & patient.cohort.isin(["Rmu", "Rtu"])].copy()
    summary = summary[summary.normalization == "log_raw"].set_index("module")

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0])
    ax = fig.add_subplot(grid[0])
    rng = np.random.default_rng(20260830)
    for x, module in enumerate(MODULE_ORDER):
        for offset, cohort in [(-0.16, "Rmu"), (0.16, "Rtu")]:
            values = patient.loc[
                (patient.module == module) & (patient.cohort == cohort), "paired_tumour_normal_log2fc"
            ].dropna().to_numpy(float)
            jitter = rng.uniform(-0.045, 0.045, len(values))
            ax.scatter(
                np.full(len(values), x + offset) + jitter,
                values,
                s=34,
                alpha=0.72,
                color=COHORT_COLORS[cohort],
                edgecolor="white",
                linewidth=0.5,
                label=cohort if x == 0 else None,
                zorder=3,
            )
            mean = float(np.mean(values))
            sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            ax.errorbar(
                x + offset,
                mean,
                yerr=1.96 * sem,
                fmt="_",
                markersize=18,
                markeredgewidth=3,
                color="black",
                capsize=4,
                linewidth=1.2,
                zorder=4,
            )
        q = summary.loc[module, "rmu_vs_rtu_bh_q_five_modules"]
        label = f"q={q:.3g}" if q < 0.1 else f"q={q:.2f}"
        ymax = patient.loc[patient.module == module, "paired_tumour_normal_log2fc"].max()
        ax.text(x, ymax + 0.35, label, ha="center", va="bottom", fontsize=9)
    ax.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_xticks(range(len(MODULE_ORDER)), [MODULE_LABELS[value] for value in MODULE_ORDER])
    ax.set_ylabel("Patient-level tumour − matched-normal log2 abundance")
    ax.set_title("A. Only Neu5Ac shows candidate-panel mucinous-relative subtype sensitivity")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(grid[1])
    positive = score[score.feature_id != 9900175].copy()
    class_order = {
        "PRIMARY_SUBTYPE_ANCHOR": 0,
        "FAMILY_VALIDATION_PRIORITY": 1,
        "GENERAL_TUMOUR_SUPPORT": 2,
        "LOW_COVERAGE_IDENTITY_VALIDATION": 3,
        "CONTEXT_ONLY": 4,
        "NEGATIVE_CONTROL": 5,
    }
    positive["class_order"] = positive.claim_class.map(class_order)
    positive = positive.sort_values(["class_order", "rmu_vs_rtu_bh_q_max", "feature_id"])
    gates = [
        ("identity_anchor", "Identity\nanchor"),
        ("adequate_rmu_coverage", "Rmu n≥8"),
        ("candidate_panel_primary_fdr10", "Candidate-panel\nRmu FDR10"),
        ("candidate_panel_subtype_fdr10", "Candidate-panel\nsubtype FDR10"),
        ("background_robust_three_specs", "Matched-background\n3/3"),
        ("full_untargeted_exact_fdr10", "Full 13,155-target\nexact FDR10"),
    ]
    values = positive[[column for column, _ in gates]].fillna(False).astype(int).to_numpy()
    ax2.imshow(values, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["#ECECEC", "#2B8C6B"]), vmin=0, vmax=1)
    ax2.set_xticks(range(len(gates)), [label for _, label in gates])
    ax2.set_yticks(
        range(len(positive)),
        [f"{int(row.feature_id)}  {row.label}" for row in positive.itertuples()],
        fontsize=8.5,
    )
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            ax2.text(x, y, "✓" if values[y, x] else "–", ha="center", va="center", color="white" if values[y, x] else "#777777", fontsize=11)
    ax2.set_title("B. Frozen evidence gates prevent candidate-panel signals from being presented as full-space discoveries")
    ax2.tick_params(axis="x", bottom=False, top=False)
    ax2.tick_params(axis="y", left=False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    fig.suptitle("MTBLS13729 biological claim audit", fontsize=16, fontweight="bold")
    png = OUT / "mtbls13729_subtype_claim_audit_v1.png"
    svg = OUT / "mtbls13729_subtype_claim_audit_v1.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "mtbls13729_subtype_claim_figure_v1_complete",
        "module_panel": "log_raw patient-level tumour-minus-matched-normal effects; five-module BH q",
        "candidate_panel": "17 positive-RP candidates; binary evidence gates",
        "primary_subtype_anchor": "feature 703 Neu5Ac",
        "full_space_exact_fdr10_positive_candidates": int(positive.full_untargeted_exact_fdr10.sum()),
        "outputs": [str(png.relative_to(ROOT)), str(svg.relative_to(ROOT))],
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
