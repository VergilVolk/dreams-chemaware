"""Build an Extended Data figure separating pool, donor and carrier evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DONOR = ROOT / "data/mtbls13729/sialic_donor_decoupling_v1"
OAC = ROOT / "data/mtbls13729/oacetyl_neu5ac_like_v2"
PXD = ROOT / "data/external/PXD055865_2026_MUC2/audit_v1"
OUT = ROOT / "data/mtbls13729/pool_carrier_boundary_figure_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    donor_path = DONOR / "rmu_patient_sialic_donor_deltas.csv"
    donor_report_path = DONOR / "report.json"
    oac_values_path = OAC / "paired_patient_values.csv"
    oac_report_path = OAC / "report.json"
    pxd_summary_path = PXD / "specimen_summary.csv"
    pxd_report_path = PXD / "report.json"
    required = [
        donor_path,
        donor_report_path,
        oac_values_path,
        oac_report_path,
        pxd_summary_path,
        pxd_report_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    donor = pd.read_csv(donor_path)
    donor_report = json.loads(donor_report_path.read_text(encoding="utf-8"))
    oac = pd.read_csv(oac_values_path)
    oac_report = json.loads(oac_report_path.read_text(encoding="utf-8"))
    pxd = pd.read_csv(pxd_summary_path)
    pxd_report = json.loads(pxd_report_path.read_text(encoding="utf-8"))

    if donor["patient"].nunique() != 10 or set(donor["node"]) != {
        "free_neu5ac",
        "cmp_neu5ac",
        "udp_glcnac",
    }:
        raise RuntimeError("unexpected donor-pool input")
    if pxd_report["dataset"]["design"].find("two patients") < 0:
        raise RuntimeError("PXD patient-independence boundary missing")

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        "Free-pool expansion is not equivalent to activated-donor or MUC2-carrier remodeling",
        fontsize=17,
        fontweight="bold",
    )

    # A: same-patient pool-to-donor decomposition.
    ax = axes[0, 0]
    node_order = ["free_neu5ac", "cmp_neu5ac", "udp_glcnac"]
    node_labels = ["Free Neu5Ac\n(Level 1)", "CMP-Neu5Ac\n(Level 2)", "UDP-GlcNAc\n(Level 1)"]
    wide = donor.pivot(index="patient", columns="node", values="paired_log2_delta")
    for _, row in wide[node_order].iterrows():
        ax.plot(range(3), row.values, color="#aeb8c2", alpha=0.65, linewidth=1)
        ax.scatter(range(3), row.values, color=["#b94a3b", "#557da7", "#6d8d68"], s=26, zorder=3)
    means = wide[node_order].mean().values
    ax.scatter(range(3), means, marker="D", s=85, color="#111111", zorder=4, label="mean")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(range(3), node_labels)
    ax.set_ylabel("Rmu tumour - matched normal (log2)")
    ax.set_title("A  Same-patient free-pool-to-donor decoupling", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.97,
        "Free minus CMP: +1.69, Holm p=0.027\nFree minus UDP: +1.92, Holm p=0.027",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#888888"},
    )

    # B: phenotype-blind O-acetyl-like free-pool negative result.
    ax = axes[0, 1]
    oac_rmu = oac[oac["contrast"].eq("Rmu_vs_RN")].copy()
    features = ["OAc-like-01", "OAc-like-02"]
    rng = np.random.default_rng(20260831)
    for x, feature in enumerate(features):
        values = oac_rmu.loc[oac_rmu["feature_id"].eq(feature), "paired_log2_delta_floor"].to_numpy()
        jitter = rng.normal(0, 0.035, len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=32, alpha=0.85, color="#8e6a9c")
        ax.hlines(values.mean(), x - 0.18, x + 0.18, color="#111111", linewidth=3)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks([0, 1], ["4.29 min", "5.55 min"])
    ax.set_ylabel("Rmu tumour - matched normal (log2; floor sensitivity)")
    ax.set_title("B  Bulk mono-O-acetyl-Neu5Ac-like pool is not increased", loc="left", fontweight="bold")
    ax.text(
        0.03,
        0.97,
        "Phenotype-blind RT features\nBH q=0.930 for both\nPosition isomer unresolved",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#888888"},
    )

    # C: carrier-resolved identification presence. Explicitly not abundance.
    ax = axes[1, 0]
    categories = [
        ("unique_sialylated_muc2", "Sialylated MUC2", "#3b7a78"),
        ("unique_o_acetyl_neu5ac_muc2", "OAc-Neu5Ac MUC2", "#d0793f"),
        ("unique_o_acetyl_galnac_muc2", "putative OAc-GalNAc MUC2", "#7b5aa6"),
    ]
    x = np.arange(len(pxd))
    width = 0.24
    for offset, (column, label, color) in enumerate(categories):
        ax.bar(x + (offset - 1) * width, pxd[column], width=width, label=label, color=color)
    ax.set_xticks(x, ["Colon1a\nPatient 1", "Colon1b\nPatient 1", "Colon2\nPatient 2", "Healthy\ndonor"])
    ax.set_ylabel("Unique manually reviewed MUC2 identifications")
    ax.set_title("C  PXD055865 carrier-resolved structural presence", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.02,
        0.96,
        "Presence/coverage only - NOT abundance\n3 tumour specimens = 2 independent patients",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff4df", "edgecolor": "#aa7b2d"},
    )

    # D: evidence and claim boundary.
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("D  Evidence-calibrated interpretation", loc="left", fontweight="bold")
    boxes = [
        (0.05, 0.70, 0.90, 0.18, "Supported in MTBLS13729", "Free Neu5Ac rises in 10/10 Rmu pairs and exceeds\nCMP-Neu5Ac/UDP-GlcNAc changes", "#e7f1ed"),
        (0.05, 0.43, 0.90, 0.18, "Externally compatible", "MUC2 carrier/destination is heterogeneous; O-acetylated\nsialic-acid structures are not a uniform tumour-wide endpoint", "#e8edf5"),
        (0.05, 0.16, 0.90, 0.18, "Not established", "Free-pool flux, specific MUC2 incorporation, O-acetyl position,\nenzyme causality or independent mucinous abundance replication", "#f7e9e5"),
    ]
    for x0, y0, w, h, heading, body, color in boxes:
        patch = FancyBboxPatch(
            (x0, y0), w, h, boxstyle="round,pad=0.015", linewidth=1.2,
            edgecolor="#666666", facecolor=color, transform=ax.transAxes
        )
        ax.add_patch(patch)
        ax.text(x0 + 0.025, y0 + h - 0.045, heading, transform=ax.transAxes, fontweight="bold", va="top")
        ax.text(x0 + 0.025, y0 + h - 0.095, body, transform=ax.transAxes, va="top", fontsize=9.5)

    png = OUT / "pool_carrier_boundary_figure_v1.png"
    pdf = OUT / "pool_carrier_boundary_figure_v1.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "mtbls13729_pool_carrier_boundary_figure_v1_complete",
        "formal": False,
        "rmu_pairs": int(wide.shape[0]),
        "pdx_mucinous_specimens": 3,
        "pdx_independent_mucinous_patients": 2,
        "pdx_abundance_interpretation_permitted": False,
        "claim_limit": (
            "The figure separates same-patient free-pool/donor abundance from external MUC2 "
            "identification presence. It does not establish flux, positional isomers, carrier "
            "incorporation or independent abundance replication."
        ),
        "provenance": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "outputs": {"png": png.name, "pdf": pdf.name},
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Pool-to-carrier boundary figure v1\n\n"
        "Extended Data candidate separating free-pool/donor abundance, the negative bulk "
        "mono-O-acetyl-Neu5Ac-like result, and external MUC2 identification presence. Panel C "
        "is explicitly not an abundance comparison.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
