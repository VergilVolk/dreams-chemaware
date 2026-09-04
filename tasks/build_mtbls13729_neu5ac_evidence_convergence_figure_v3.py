#!/usr/bin/env python3
"""Build the integrated Neu5Ac biology evidence figure (v3).

The figure keeps metabolite abundance, independent transcript state,
independent protein context, and mechanistic inference in separate panels.
Arrows in the final panel denote evidence convergence, not causality.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DONOR = ROOT / "data/mtbls13729/sialic_donor_decoupling_v1/rmu_patient_sialic_donor_deltas.csv"
DONOR_REPORT = ROOT / "data/mtbls13729/sialic_donor_decoupling_v1/report.json"
COMPOSITION = ROOT / "data/external/GSE178341_mucinous_secretory_audit/epithelial_composition_diagnostic_v1/composition_adjusted_models.csv"
COMPOSITION_REPORT = ROOT / "data/external/GSE178341_mucinous_secretory_audit/epithelial_composition_diagnostic_v1/report.json"
PROTEIN = ROOT / "data/external/GSE178341_mucinous_secretory_audit/independent_proteomics_fixed_panel_v1/protein_summary.csv"
PROTEIN_REPORT = ROOT / "data/external/GSE178341_mucinous_secretory_audit/independent_proteomics_fixed_panel_v1/result.json"
OUT = ROOT / "data/mtbls13729/neu5ac_evidence_convergence_figure_v3_final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    for path in (DONOR, DONOR_REPORT, COMPOSITION, COMPOSITION_REPORT, PROTEIN, PROTEIN_REPORT):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    donor = pd.read_csv(DONOR)
    composition = pd.read_csv(COMPOSITION)
    protein = pd.read_csv(PROTEIN)
    if set(donor["node"]) != {"free_neu5ac", "cmp_neu5ac", "udp_glcnac"}:
        raise RuntimeError("donor nodes changed")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.titlesize": 10.7,
        "axes.labelsize": 9.2,
        "pdf.fonttype": 42,
    })
    figure = plt.figure(figsize=(13.8, 9.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.02])

    # A. Same-patient free pool and donor/precursor abundance.
    ax = figure.add_subplot(grid[0, 0])
    order = ["free_neu5ac", "cmp_neu5ac", "udp_glcnac"]
    labels = ["Free Neu5Ac\nLevel 1", "CMP-Neu5Ac\nLevel 2", "UDP-GlcNAc\nLevel 1"]
    wide = donor.pivot(index="patient", columns="node", values="paired_log2_delta").loc[:, order]
    for _, row in wide.iterrows():
        ax.plot(range(3), row.to_numpy(float), color="#b8b8b8", alpha=0.62, lw=0.9, zorder=1)
        ax.scatter(range(3), row.to_numpy(float), color=["#a63232", "#4477aa", "#5a8f55"], s=20, zorder=2)
    means = wide.mean(axis=0).to_numpy(float)
    ax.scatter(range(3), means, marker="D", color="black", s=46, zorder=3, label="patient mean")
    for x, value in enumerate(means):
        ax.text(x, value + 0.23, f"{value:+.2f}", ha="center", fontweight="bold")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Rmu − matched normal (log2 abundance)")
    ax.set_title("A  Same-patient free-pool/donor decoupling", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    # B. Independent patient raw-UMI state after composition adjustment.
    ax = figure.add_subplot(grid[0, 1])
    endpoints = ["SECRETORY_COMPOSITE", "AGR2", "SLC35A1", "MUC2", "SPDEF", "NXPE1"]
    names = ["Secretory composite", "AGR2", "SLC35A1", "MUC2", "SPDEF", "NXPE1"]
    y = np.arange(len(endpoints))[::-1]
    for offset, model, label, color in (
        (0.11, "histology_only", "unadjusted", "#9a9a9a"),
        (-0.11, "plus_goblet_fraction", "+ goblet fraction", "#2f6f8f"),
    ):
        subset = composition.set_index(["endpoint", "model"])
        values = [subset.loc[(endpoint, model)] for endpoint in endpoints]
        beta = np.asarray([row.mucinous_beta for row in values], dtype=float)
        low = np.asarray([row.ci_low for row in values], dtype=float)
        high = np.asarray([row.ci_high for row in values], dtype=float)
        ax.errorbar(beta, y + offset, xerr=np.vstack([beta - low, high - beta]), fmt="o", color=color,
                    ecolor=color, capsize=2.3, ms=4.7, label=label)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y, names)
    ax.set_xlabel("Mucinous coefficient (HC3 95% CI)")
    ax.set_title("B  Independent raw-UMI epithelial state", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # C. Independent patient proteomics, with all null intervals visible.
    ax = figure.add_subplot(grid[1, 0])
    protein_names = ["AGR2", "GNE", "NANS", "CMAS", "SIAE"]
    subset = protein.set_index("name").loc[protein_names]
    beta = subset["mc_minus_ac"].to_numpy(float)
    intervals = np.asarray([ast.literal_eval(value) for value in subset["bootstrap_95ci"]], dtype=float)
    y = np.arange(len(protein_names))[::-1]
    ax.errorbar(beta, y, xerr=np.vstack([beta - intervals[:, 0], intervals[:, 1] - beta]), fmt="o",
                color="#775a88", ecolor="#775a88", capsize=2.5, ms=5.5)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y, protein_names)
    ax.set_xlabel("MC − conventional CRC (log2 protein; bootstrap 95% CI)")
    ax.set_title("C  Independent proteomics: directional, not confirmed", loc="left", fontweight="bold")
    ax.text(0.02, 0.02, "Frozen 8-protein BH q: all shown genes ≥ 0.643\nAGR2/GNE/NANS: 15/15 leave-one-MC-out positive",
            transform=ax.transAxes, va="bottom", fontsize=7.8, color="#604768")
    ax.spines[["top", "right"]].set_visible(False)

    # D. Evidence-calibrated, explicitly non-causal synthesis.
    ax = figure.add_subplot(grid[1, 1])
    ax.axis("off")
    ax.set_title("D  Evidence-calibrated model and remaining hard gaps", loc="left", fontweight="bold")
    boxes = [
        (0.03, 0.71, 0.27, 0.19, "Measured pool\nFree Neu5Ac ↑\n10/10 Rmu pairs", "#f5e6e2"),
        (0.365, 0.71, 0.27, 0.19, "Epithelial capacity\nAGR2 + SLC35A1 ↑\nafter composition", "#e2eef4"),
        (0.70, 0.71, 0.27, 0.19, "Carrier/linkage context\nMUC2 glycoforms\ncore/linkage remodelling", "#e8f1e6"),
    ]
    for x0, y0, width, height, text, face in boxes:
        ax.add_patch(plt.Rectangle((x0, y0), width, height, facecolor=face, edgecolor="#565656", lw=1.05))
        ax.text(x0 + width / 2, y0 + height / 2, text, ha="center", va="center", fontsize=8.2)
        ax.annotate("", xy=(0.50, 0.48), xytext=(x0 + width / 2, y0),
                    arrowprops=dict(arrowstyle="->", linestyle="--", color="#777777", lw=1.0))
    ax.add_patch(plt.Rectangle((0.21, 0.34), 0.58, 0.14, facecolor="#f2eee2", edgecolor="#665f4d", lw=1.25))
    ax.text(0.50, 0.41, "Selective secretory/transport capacity\nwith pool-to-donor/destination decoupling",
            ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(0.04, 0.19, "Negative constraints", fontweight="bold", color="#8c2e27", fontsize=9)
    ax.text(0.04, 0.06,
            "• CMP-Neu5Ac pool not proportionally increased\n"
            "• host NEU1/NEU3 release not supported\n"
            "• NXPE1 not independent of secretory/goblet state\n"
            "• protein modules not multiplicity-confirmed",
            fontsize=8.0, color="#7e312b", va="bottom")
    ax.text(0.57, 0.06,
            "Still missing\n"
            "• independent Rmu metabolomics replication\n"
            "• same-method standard/spike-in\n"
            "• same-sample linkage-aware glycomics\n"
            "• isotope tracing / perturbation / rescue",
            fontsize=8.0, color="#333333", va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    figure.suptitle("Free Neu5Ac expansion is coupled to selective epithelial capacity, not uniform pathway activation",
                    fontsize=14.2, fontweight="bold")
    OUT.mkdir(parents=True, exist_ok=False)
    png = OUT / "neu5ac_evidence_convergence_figure_v3.png"
    pdf = OUT / "neu5ac_evidence_convergence_figure_v3.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    donor.to_csv(OUT / "same_patient_donor_deltas.csv", index=False)
    composition.to_csv(OUT / "independent_transcript_composition_models.csv", index=False)
    subset.reset_index().to_csv(OUT / "independent_proteomics_context.csv", index=False)
    report = {
        "status": "mtbls13729_neu5ac_evidence_convergence_figure_v3_complete",
        "formal": False,
        "figure_role": "integrated evidence display; not a causal pathway",
        "same_patient_metabolite_patients": int(wide.shape[0]),
        "independent_transcript_patients": {"mucinous": 6, "conventional": 53},
        "independent_proteomics_patients": {"mucinous": 15, "conventional": 15},
        "provenance": {
            "donor_sha256": sha256(DONOR),
            "donor_report_sha256": sha256(DONOR_REPORT),
            "composition_sha256": sha256(COMPOSITION),
            "composition_report_sha256": sha256(COMPOSITION_REPORT),
            "protein_sha256": sha256(PROTEIN),
            "protein_report_sha256": sha256(PROTEIN_REPORT),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_limit": "The figure supports evidence convergence for selective epithelial capacity and pool-to-donor/destination decoupling; it does not establish independent Neu5Ac replication, biochemical source, flux, enzyme causality or glycan destination.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Neu5Ac evidence convergence figure v3\n\n"
        "This publication figure separates same-patient metabolite abundance, independent patient raw-UMI state, independent proteomics context, and the non-causal synthesis. Negative evidence and missing validation are intentionally shown in the main figure.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
