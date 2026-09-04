"""Build a publication figure for the Neu5Ac pool-carrier-destination model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DONOR = ROOT / "data/mtbls13729/sialic_donor_decoupling_v1"
OAC = ROOT / "data/mtbls13729/oacetyl_neu5ac_like_v2"
NXPE1 = ROOT / "data/external/TCGA_COADREAD_Xena_20260830/nxpe1_free_donor_v3_secretory"
OUT = ROOT / "data/mtbls13729/nxpe1_pool_carrier_figure_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    donor_csv = DONOR / "rmu_patient_sialic_donor_deltas.csv"
    donor_json = DONOR / "report.json"
    oac_csv = OAC / "paired_patient_values.csv"
    oac_json = OAC / "report.json"
    nxpe1_json = NXPE1 / "report.json"
    required = [donor_csv, donor_json, oac_csv, oac_json, nxpe1_json]
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(source)

    donor = pd.read_csv(donor_csv)
    oac = pd.read_csv(oac_csv)
    donor_report = json.loads(donor_json.read_text(encoding="utf-8"))
    oac_report = json.loads(oac_json.read_text(encoding="utf-8"))
    nx_report = json.loads(nxpe1_json.read_text(encoding="utf-8"))

    if donor["patient"].nunique() != 10:
        raise RuntimeError("expected exactly 10 Rmu pairs")
    if nx_report["units"]["tpm"]["locked_legacy_371"]["primary_tumours"] != 371:
        raise RuntimeError("NXPE1 report is not the locked 371-tumour cohort")
    if nx_report["primary_nxpe1"]["secretory_lineage_p"] <= 0.05:
        raise RuntimeError("NXPE1 secretory-adjusted result changed")

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2), constrained_layout=True)
    fig.suptitle(
        "Mucinous CRC: free Neu5Ac expansion is decoupled from donor activation and final glycan destination",
        fontsize=16,
        fontweight="bold",
    )

    # A: same-patient free-pool/donor decomposition.
    ax = axes[0, 0]
    node_order = ["free_neu5ac", "cmp_neu5ac", "udp_glcnac"]
    labels = ["Free Neu5Ac\nLevel 1", "CMP-Neu5Ac\nLevel 2", "UDP-GlcNAc\nLevel 1"]
    colors = ["#b64e3b", "#507aa3", "#6f8f69"]
    wide = donor.pivot(index="patient", columns="node", values="paired_log2_delta")
    for _, row in wide[node_order].iterrows():
        ax.plot(range(3), row.values, color="#b9bec3", linewidth=1, alpha=0.65)
        ax.scatter(range(3), row.values, color=colors, s=27, zorder=3)
    ax.scatter(range(3), wide[node_order].mean(), marker="D", color="#111111", s=80, zorder=4)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Rmu tumour - matched normal (log2)")
    ax.set_title("A  Same-patient pool-to-donor decoupling", loc="left", fontweight="bold")
    ax.text(
        0.02, 0.97,
        "Free-CMP: +1.69, Holm p=0.027\nFree-UDP: +1.92, Holm p=0.027",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#888888"},
    )

    # B: NXPE1 coefficient through the adjustment hierarchy.
    ax = axes[0, 1]
    tpm = nx_report["primary_nxpe1"]
    model_rows = [
        ("Clinical", tpm["clinical_beta"], tpm["clinical_ci_low"], tpm["clinical_ci_high"], tpm["clinical_p"]),
        ("+ lineage", tpm["lineage_beta"], tpm["lineage_ci_low"], tpm["lineage_ci_high"], tpm["lineage_p"]),
        ("+ lineage + MSI", tpm["msi_lineage_beta"], tpm["msi_lineage_ci_low"], tpm["msi_lineage_ci_high"], tpm["msi_lineage_p"]),
        ("+ lineage + secretory", tpm["secretory_lineage_beta"], tpm["secretory_lineage_ci_low"], tpm["secretory_lineage_ci_high"], tpm["secretory_lineage_p"]),
        ("+ lineage + MSI + secretory", tpm["secretory_msi_lineage_beta"], tpm["secretory_msi_lineage_ci_low"], tpm["secretory_msi_lineage_ci_high"], tpm["secretory_msi_lineage_p"]),
    ]
    y = np.arange(len(model_rows))[::-1]
    beta = np.array([row[1] for row in model_rows])
    low = np.array([row[2] for row in model_rows])
    high = np.array([row[3] for row in model_rows])
    significant = np.array([row[4] < 0.05 for row in model_rows])
    ax.errorbar(beta, y, xerr=[beta - low, high - beta], fmt="none", ecolor="#4c5661", capsize=3)
    ax.scatter(beta[significant], y[significant], color="#b64e3b", s=62, zorder=3, label="p < 0.05")
    ax.scatter(beta[~significant], y[~significant], facecolor="white", edgecolor="#4c5661", s=62, zorder=3, label="p >= 0.05")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y, [row[0] for row in model_rows])
    ax.set_xlabel("Mucinous vs conventional NXPE1 coefficient (log2 expression)")
    ax.set_title("B  NXPE1 enrichment is carried by the secretory-mucin state", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98, 0.03, "Locked TCGA: 42 mucinous / 329 conventional\nTPM; FPKM-UQ gives the same conclusion",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#888888"},
    )

    # C: local O-acetyl-like exact-mass counter-evidence.
    ax = axes[1, 0]
    rmu = oac[oac["contrast"].eq("Rmu_vs_RN")].copy()
    rng = np.random.default_rng(20260831)
    for x, feature in enumerate(["OAc-like-01", "OAc-like-02"]):
        values = rmu.loc[rmu["feature_id"].eq(feature), "paired_log2_delta_floor"].to_numpy()
        ax.scatter(np.full(len(values), x) + rng.normal(0, 0.035, len(values)), values, s=32, alpha=0.85, color="#8a6997")
        ax.hlines(values.mean(), x - 0.18, x + 0.18, color="#111111", linewidth=3)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks([0, 1], ["4.29 min\n50/60 samples", "5.55 min\n54/60 samples"])
    ax.set_ylabel("Rmu tumour - matched normal (log2; floor sensitivity)")
    ax.set_title("C  Bulk mono-OAc-Neu5Ac-like features do not follow the pool", loc="left", fontweight="bold")
    ax.text(
        0.03, 0.97,
        "103 RT-resolved MS2 in total\nBoth paired BH q=0.930\nPositional isomer unresolved",
        transform=ax.transAxes, va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#888888"},
    )

    # D: evidence-calibrated mechanism map.
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("D  Minimal sufficient mechanism", loc="left", fontweight="bold")
    boxes = [
        (0.02, 0.62, 0.27, 0.22, "FREE POOL", "Neu5Ac +2.25 log2\n10/10 Rmu pairs", "#f5ddd7"),
        (0.365, 0.62, 0.27, 0.22, "CARRIER STATE", "MUC2/secretory programme\nNXPE1 embedded within it", "#dce9e5"),
        (0.71, 0.62, 0.27, 0.22, "DESTINATION", "core/linkage/O-acetyl\nheterogeneous remodelling", "#e2e6f1"),
        (0.16, 0.19, 0.32, 0.20, "ACTIVATED DONOR", "CMP-Neu5Ac does not\nexpand in parallel", "#e4edf5"),
        (0.54, 0.19, 0.42, 0.20, "LOCAL PRODUCT SCREEN", "two OAc-like peaks do not\ntrack free Neu5Ac", "#eee5f1"),
    ]
    for x0, y0, w, h, heading, body, color in boxes:
        patch = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.015", facecolor=color, edgecolor="#666666", linewidth=1.1, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(x0 + 0.018, y0 + h - 0.05, heading, transform=ax.transAxes, fontweight="bold", va="top", fontsize=9.5)
        ax.text(x0 + 0.018, y0 + h - 0.105, body, transform=ax.transAxes, va="top", fontsize=8.2)
    for start, end in [((0.29, 0.73), (0.365, 0.73)), ((0.635, 0.73), (0.71, 0.73)), ((0.25, 0.62), (0.31, 0.39)), ((0.82, 0.62), (0.76, 0.39))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#666666", transform=ax.transAxes))
    ax.text(
        0.5, 0.04,
        "Static abundance + expression context; not flux, enzyme activity or same-sample glycan fate",
        ha="center", va="bottom", transform=ax.transAxes, fontsize=9.5, color="#7a2d24",
    )

    png = OUT / "nxpe1_pool_carrier_figure_v2.png"
    pdf = OUT / "nxpe1_pool_carrier_figure_v2.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "mtbls13729_nxpe1_pool_carrier_figure_v2_complete",
        "formal": False,
        "locked_tcga_tumours": 371,
        "rmu_pairs": 10,
        "claim_limit": "Separates direct paired abundance, bulk RNA context and product-side counter-evidence; does not establish flux, enzyme activity or causal substrate flow.",
        "provenance": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "outputs": {"png": png.name, "pdf": pdf.name},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
