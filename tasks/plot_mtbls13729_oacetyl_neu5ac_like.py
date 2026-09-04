#!/usr/bin/env python
"""Plot the phenotype-blind mono-O-acetyl-Neu5Ac-like audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/mtbls13729/oacetyl_neu5ac_like_v2"
DONOR = ROOT / (
    "data/mtbls13729/sialic_donor_decoupling_v1/"
    "rmu_patient_sialic_donor_deltas.csv"
)
OUT = ROOT / "data/mtbls13729/oacetyl_neu5ac_like_figure_v1"


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "mtbls13729_oacetyl_neu5ac_like_audit_complete":
        raise RuntimeError("O-acetyl-Neu5Ac-like audit is incomplete")
    trace = pd.read_csv(SOURCE / "phenotype_blind_consensus_trace.csv.gz")
    features = pd.read_csv(SOURCE / "frozen_rt_features.csv")
    pairs = pd.read_csv(SOURCE / "paired_patient_values.csv")
    ms2 = pd.read_csv(SOURCE / "rt_resolved_ms2_spectra.csv.gz")
    donor = pd.read_csv(DONOR)
    free = donor.loc[
        donor["node"].eq("free_neu5ac"), ["patient", "paired_log2_delta"]
    ].rename(columns={"paired_log2_delta": "free Neu5Ac"})
    rmu = pairs.loc[pairs["contrast"].eq("Rmu_vs_RN")].pivot(
        index="patient", columns="feature_id", values="paired_log2_delta_floor"
    )
    matrix = free.set_index("patient").join(rmu).sort_index()

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11})
    fig = plt.figure(figsize=(12.5, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])

    ax = fig.add_subplot(grid[0, 0])
    visible = trace.loc[trace["rt_sec"].between(220, 365)]
    ax.plot(
        visible["rt_sec"] / 60,
        visible["mean_unit_scaled_intensity"],
        color="#2d6a9f",
        linewidth=1.8,
        label="mean unit-scaled EIC",
    )
    ax.plot(
        visible["rt_sec"] / 60,
        visible["sample_median_unit_scaled_intensity"],
        color="#91bfd8",
        linewidth=1.2,
        label="sample median",
    )
    for row in features.itertuples(index=False):
        ax.axvline(row.median_rt_sec / 60, color="#b4442e", linestyle="--", alpha=0.8)
        ax.text(
            row.median_rt_sec / 60,
            ax.get_ylim()[1] * 0.92,
            f"{row.feature_id}\n{row.median_rt_sec / 60:.2f} min",
            ha="center",
            va="top",
            color="#7f2a1e",
        )
    ax.set_title("A  Phenotype-blind exact-mass chromatographic discovery", loc="left", fontweight="bold")
    ax.set_xlabel("Retention time (min)")
    ax.set_ylabel("Population consensus intensity")
    ax.legend(frameon=False, loc="upper left")

    ax = fig.add_subplot(grid[0, 1])
    values = matrix.to_numpy(float)
    limit = max(4.0, float(np.nanmax(np.abs(values))))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=20, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_title("B  Same-patient Rmu–RN log2 changes", loc="left", fontweight="bold")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:+.1f}", ha="center", va="center", fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("paired log2 change")

    ax = fig.add_subplot(grid[1, 0])
    x = np.arange(len(matrix))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(x, matrix["free Neu5Ac"], marker="o", color="#b4442e", label="free Neu5Ac (Level 1)")
    ax.plot(x, matrix["OAc-like-01"], marker="s", color="#2d6a9f", label="OAc-like-01, 4.29 min")
    ax.plot(x, matrix["OAc-like-02"], marker="^", color="#5a8f4f", label="OAc-like-02, 5.55 min")
    ax.set_xticks(x, matrix.index, rotation=45, ha="right")
    ax.set_ylabel("Rmu–RN paired log2 change")
    ax.set_title("C  Expanded free pool does not track the two exact-mass features", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(grid[1, 1])
    fragments = {
        "87.0088": "frag_87_0088_present",
        "128.0350": "frag_128_0350_present",
        "170.0459": "frag_170_0459_present",
        "308.0987": "frag_308_0987_present",
        "332.0987": "frag_332_0987_present",
    }
    width = 0.34
    positions = np.arange(len(fragments))
    for offset, feature_id in zip([-width / 2, width / 2], features["feature_id"]):
        subset = ms2.loc[ms2["feature_id"].eq(feature_id)]
        fractions = [float(subset[column].mean()) for column in fragments.values()]
        ax.bar(positions + offset, fractions, width=width, label=f"{feature_id} (n={len(subset)})")
    ax.set_xticks(positions, fragments.keys())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of RT-resolved MS2 spectra")
    ax.set_xlabel("Fragment m/z (>=1% base peak)")
    ax.set_title("D  Fragment support is strong for m/z 87 but not site-diagnostic", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    fig.suptitle(
        "MTBLS13729 mono-O-acetyl-Neu5Ac-like exact-mass audit",
        fontsize=15,
        fontweight="bold",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "oacetyl_neu5ac_like_audit.png"
    pdf = OUT / "oacetyl_neu5ac_like_audit.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    matrix.to_csv(OUT / "rmu_patient_delta_matrix.csv")
    (OUT / "report.json").write_text(
        json.dumps(
            {
                "status": "mtbls13729_oacetyl_neu5ac_like_figure_complete",
                "png": str(png.relative_to(ROOT)),
                "pdf": str(pdf.relative_to(ROOT)),
                "interpretation": (
                    "Two phenotype-blind exact-mass RT features have reproducible m/z 87-containing MS2, "
                    "but neither shows a consistent Rmu abundance increase or patient-level coupling to free Neu5Ac."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
