from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("data/mtbls13729")
EXTERNAL = Path("data/external/OEP00006137_support/modified_guanosine_reanalysis/report.json")
OUTPUT = ROOT / "modified_guanosine_biology_evidence_20260830.png"


def main() -> None:
    effects = pd.read_csv(
        ROOT / "biology_closure_analysis_v1/fully_ion_family_collapsed_module_patient_effects.csv"
    )
    effects = effects[effects["normalization"] == "raw"].copy()
    ms2 = pd.read_csv(ROOT / "modified_guanosine_ms2_audit_v1/modified_guanosine_ms2_summary.csv")
    external = json.loads(EXTERNAL.read_text(encoding="utf-8"))["metabolites"]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    ax = axes[0, 0]
    colors = {"Rmu": "#8E2C7D", "Rtu": "#4C78A8"}
    for x, cohort in enumerate(["Rmu", "Rtu"]):
        vals = effects.loc[effects["cohort"] == cohort, "module_log2fc"].to_numpy()
        jitter = np.linspace(-0.10, 0.10, len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, color=colors[cohort], s=45, alpha=0.85)
        ax.hlines(np.mean(vals), x - 0.22, x + 0.22, color="black", linewidth=2.5)
        annotation_y = 5.45 if cohort == "Rmu" else 3.70
        ax.text(x, annotation_y, f"mean={np.mean(vals):.2f}\n{np.sum(vals>0)}/{len(vals)} positive",
                ha="center", va="bottom")
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks([0, 1], ["Mucinous (Rmu)", "Tubular (Rtu)"])
    ax.set_ylabel("Paired tumor-normal log2 fold change")
    ax.set_title("A. Local collapsed modified-guanosine module", pad=12)

    ax = axes[0, 1]
    labels = [f"{int(v)}" for v in ms2["feature_id"]]
    fractions = ms2["ribose_loss_support_fraction"].to_numpy()
    bars = ax.bar(labels, fractions, color=["#8E2C7D", "#C06C9B", "#4C78A8", "#79A7D3"])
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("MS2 spectra supporting neutral loss 132.042")
    ax.set_xlabel("MTBLS13729 feature ID")
    for bar, n in zip(bars, ms2["n_ms2_spectra"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04, f"n={int(n)}", ha="center")
    ax.set_title("B. Peak-resolved raw MS2 class evidence")

    ax = axes[1, 0]
    names = ["1-Methylguanosine", "2-Methylguanosine", "N2,N2-Dimethylguanosine", "7-Methylguanosine"]
    short = ["1-MeG", "2-MeG", "m2,2G", "7-MeG"]
    x = np.arange(len(names))
    width = 0.36
    msi = [external[n]["MSI"]["mean_log2fc"] for n in names]
    mss = [external[n]["MSS"]["mean_log2fc"] for n in names]
    ax.bar(x - width / 2, msi, width, label="MSI", color="#E45756")
    ax.bar(x + width / 2, mss, width, label="MSS", color="#4C78A8")
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(x, short)
    ax.set_ylabel("Paired tumor-normal mean log2 fold change")
    ax.legend(frameon=False)
    ax.set_title("C. Independent 40-pair Level-1 cohort (OEP00006137)")

    ax = axes[1, 1]
    names = ["Methionine", "S-Adenosylmethionine", "S-Adenosylhomocysteine"]
    short = ["Methionine", "SAM", "SAH"]
    x = np.arange(len(names))
    msi = [external[n]["MSI"]["mean_log2fc"] for n in names]
    mss = [external[n]["MSS"]["mean_log2fc"] for n in names]
    ax.bar(x - width / 2, msi, width, label="MSI", color="#E45756")
    ax.bar(x + width / 2, mss, width, label="MSS", color="#4C78A8")
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(x, short)
    ax.set_ylabel("Paired tumor-normal mean log2 fold change")
    ax.legend(frameon=False)
    ax.set_title("D. Independent one-carbon / methyl-donor context")

    fig.suptitle(
        "Modified-guanosine evidence: strong local mucinous signal, class-level MS2 support,\n"
        "but no universal cross-cohort direction",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUTPUT, dpi=220)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
