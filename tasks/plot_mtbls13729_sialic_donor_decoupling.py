#!/usr/bin/env python
"""Create a publication-oriented figure for same-patient sialic-pool decoupling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/mtbls13729/sialic_donor_decoupling_v1"
MECHANISM = ROOT / (
    "data/external/TCGA_COADREAD_Xena_20260830/sialic_pool_mechanisms_v1/report.json"
)
OUT = ROOT / "data/mtbls13729/sialic_donor_decoupling_figure_v1"

COLORS = {
    "free_neu5ac": "#C83E4D",
    "cmp_neu5ac": "#3F7CAC",
    "udp_glcnac": "#5B8C5A",
}
LABELS = {
    "free_neu5ac": "Free Neu5Ac\n(Level 1)",
    "cmp_neu5ac": "CMP-Neu5Ac\n(Level 2)",
    "udp_glcnac": "UDP-GlcNAc\n(Level 1)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    patient_path = INPUT / "rmu_patient_sialic_donor_deltas.csv"
    report_path = INPUT / "report.json"
    for path in (patient_path, report_path, MECHANISM):
        if not path.is_file():
            raise FileNotFoundError(path)
    frame = pd.read_csv(patient_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mechanism = json.loads(MECHANISM.read_text(encoding="utf-8"))
    order = ["free_neu5ac", "cmp_neu5ac", "udp_glcnac"]
    pivot = frame.pivot(index="patient", columns="node", values="paired_log2_delta")[order]
    if pivot.shape != (10, 3) or pivot.isna().any().any():
        raise RuntimeError("expected complete 10-patient by 3-node matrix")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(13.2, 4.7), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.3])

    ax = fig.add_subplot(grid[0, 0])
    xs = np.arange(3)
    for _, row in pivot.iterrows():
        ax.plot(xs, row.values, color="#9A9A9A", alpha=0.42, linewidth=1.0, zorder=1)
        ax.scatter(xs, row.values, color=[COLORS[node] for node in order], s=22, alpha=0.82, zorder=2)
    means = pivot.mean(axis=0).values
    ax.scatter(xs, means, color=[COLORS[node] for node in order], edgecolor="black", linewidth=0.8,
               s=95, marker="D", zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs, [LABELS[node] for node in order])
    ax.set_ylabel("Tumour − matched normal (log2)")
    ax.set_title("A  Same-patient pool changes", loc="left", fontweight="bold")
    ax.text(0.02, 0.98, "Neu5Ac: 10/10 increased", transform=ax.transAxes,
            va="top", ha="left", fontsize=9, fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    contrast_names = [
        ("CMP-Neu5Ac", pivot["free_neu5ac"] - pivot["cmp_neu5ac"]),
        ("UDP-GlcNAc", pivot["free_neu5ac"] - pivot["udp_glcnac"]),
    ]
    rng = np.random.default_rng(20260831)
    for index, (label, values) in enumerate(contrast_names):
        jitter = rng.normal(0, 0.035, len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, color="#7057A3", alpha=0.8, s=30)
        ax.scatter(index, values.mean(), marker="D", s=95, color="#F4A261",
                   edgecolor="black", linewidth=0.8, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0, 1], ["Free −\nCMP-Neu5Ac", "Free −\nUDP-GlcNAc"])
    ax.set_ylabel("Within-patient difference (log2)")
    ax.set_title("B  Prespecified contrasts", loc="left", fontweight="bold")
    ax.text(0, ax.get_ylim()[1] * 0.92, "Holm p=0.027", ha="center", fontsize=9)
    ax.text(1, ax.get_ylim()[1] * 0.92, "Holm p=0.027", ha="center", fontsize=9)

    ax = fig.add_subplot(grid[0, 2])
    ax.axis("off")
    axes = {row["outcome"]: row for row in mechanism["results"] if row["outcome_type"] == "axis"}
    boxes = [
        (0.03, 0.76, 0.94, 0.17, "Measured free pool", "Free Neu5Ac +2.249 log2\n10/10 Rmu pairs", "#FCE8E9"),
        (0.03, 0.52, 0.94, 0.17, "Measured donor/precursor", "CMP-Neu5Ac +0.556; UDP-GlcNAc +0.327\nnot nominally increased", "#E8F0F8"),
        (0.03, 0.28, 0.94, 0.17, "Mucinous-relative RNA capacity", "CMP activation/transport β=+0.449\nBH q=1.61×10⁻⁴", "#E9F3E8"),
        (0.03, 0.04, 0.94, 0.17, "Mucinous-relative release proxy", "NEU1/NEU3 β=−0.691\nBH q=5.58×10⁻⁶", "#F4EAF6"),
    ]
    for x, y, width, height, title, body, color in boxes:
        patch = plt.Rectangle((x, y), width, height, transform=ax.transAxes,
                              facecolor=color, edgecolor="#333333", linewidth=0.8)
        ax.add_patch(patch)
        ax.text(x + 0.03, y + height - 0.035, title, transform=ax.transAxes,
                fontweight="bold", va="top", fontsize=9.5)
        ax.text(x + 0.03, y + 0.035, body, transform=ax.transAxes, va="bottom", fontsize=9)
    ax.text(0.0, 1.01, "C  Capacity–pool mismatch", transform=ax.transAxes,
            fontweight="bold", fontsize=10.5)
    ax.text(0.5, -0.02, "Static abundance + bulk RNA: no flux or causal direction",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#555555")

    fig.suptitle("Mucinous CRC: free Neu5Ac expands without matched activated-donor expansion",
                 fontsize=14, fontweight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "sialic_donor_decoupling.png"
    pdf = OUT / "sialic_donor_decoupling.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    pivot.reset_index().to_csv(OUT / "patient_node_matrix.csv", index=False)
    figure_report = {
        "status": "mtbls13729_sialic_donor_decoupling_figure_complete",
        "panels": 3,
        "source_patient_report_sha256": sha256(report_path),
        "source_patient_csv_sha256": sha256(patient_path),
        "source_mechanism_report_sha256": sha256(MECHANISM),
        "png_sha256": sha256(png),
        "pdf_sha256": sha256(pdf),
        "claim_limit": report["claim_limit"],
    }
    (OUT / "report.json").write_text(json.dumps(figure_report, indent=2), encoding="utf-8")
    print(json.dumps(figure_report, indent=2))


if __name__ == "__main__":
    main()
