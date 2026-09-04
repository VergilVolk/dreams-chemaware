#!/usr/bin/env python
"""Build the publication v2 figure for the hybrid mucin glycome model.

The panels keep abundance, transcriptomic branching and structural glycomics
as distinct evidence layers.  The final panel is a convergence model, not a
causal pathway diagram.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_mtbls13729_full_requantifiable_space import sample_pairs


ROOT = Path(__file__).resolve().parents[1]
EIC_DIR = ROOT / "data/mtbls13729/full_space_eic_v1"
TCGA = ROOT / "data/external/TCGA_COADREAD_Xena_20260830/glycan_branching_v2/glycan_branch_results.csv"
OGLY = ROOT / "data/external/CRC_Oglycomics_PMC9254241_20260830/mucinous_structural_audit_values.csv"
OGLY_XLSX = ROOT / "data/external/CRC_Oglycomics_PMC9254241_20260830/supplementary_tables.xlsx"
OUT = ROOT / "data/mtbls13729/neu5ac_glycan_publication_figure_v2_final"
FEATURE = 703


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_deltas() -> pd.DataFrame:
    auc_path = EIC_DIR / "pos_rp__eic_auc_matrix.csv.gz"
    det_path = EIC_DIR / "pos_rp__eic_detection_matrix.csv.gz"
    auc = pd.read_csv(auc_path).set_index("feature_id")
    detected = pd.read_csv(det_path).set_index("feature_id")
    detected = detected.apply(lambda c: c.astype(str).str.lower().isin({"true", "1"}))
    if FEATURE not in auc.index or FEATURE not in detected.index:
        raise RuntimeError("feature 703 missing from locked targeted-EIC matrices")
    masked = auc.where(detected & auc.gt(0.0))
    pseudo = float(np.percentile(masked.stack().to_numpy(float), 1) / 2.0)
    log_auc = np.log2(masked + pseudo)
    rows: list[dict[str, object]] = []
    for cohort, tumour, normal in (("Ltu", "Ltu", "LN"), ("Rtu", "Rtu", "RN"), ("Rmu", "Rmu", "RN")):
        for tumour_sample, normal_sample in sample_pairs(list(log_auc.columns), tumour, normal):
            t = log_auc.loc[FEATURE, tumour_sample]
            n = log_auc.loc[FEATURE, normal_sample]
            rows.append({
                "cohort": cohort,
                "patient": tumour_sample.split("-")[0],
                "tumour_sample": tumour_sample,
                "normal_sample": normal_sample,
                "log2_tumour_minus_normal": float(t - n) if np.isfinite(t) and np.isfinite(n) else np.nan,
            })
    return pd.DataFrame(rows)


def tcga_branches() -> pd.DataFrame:
    data = pd.read_csv(TCGA)
    wanted = [
        ("neu5ac_donor_supply_transport", "Neu5Ac donor/transport"),
        ("secretory_mucin_program", "Secretory mucin carrier"),
        ("normal_mucosal_core3_sda", "Core-3/Sda lineage"),
        ("core2_slex_biosynthesis", "Core-2/sLeX transcript axis"),
        ("alpha23_o_glycan_sialylation", "α2-3 O-glycan axis"),
        ("ST6GAL1", "ST6GAL1 (α2-6 N-glycan)"),
        ("ST6GALNAC1", "ST6GALNAC1"),
        ("GCNT3", "GCNT3"),
    ]
    rows = []
    for outcome, label in wanted:
        match = data.loc[data.outcome.eq(outcome)]
        if len(match) != 1:
            raise RuntimeError(f"expected exactly one TCGA row for {outcome}; found {len(match)}")
        row = match.iloc[0]
        rows.append({
            "outcome": outcome,
            "label": label,
            "beta": float(row.lineage_beta),
            "ci_low": float(row.lineage_ci_low),
            "ci_high": float(row.lineage_ci_high),
            "bh_q": float(row.lineage_bh_q),
            "msi_beta": float(row.msi_lineage_beta),
            "msi_bh_q": float(row.msi_lineage_bh_q),
        })
    return pd.DataFrame(rows)


def external_structures() -> pd.DataFrame:
    data = pd.read_csv(OGLY)
    wanted = [
        ("core_2", "Core 2"),
        ("sialyl_Lewis_X_A", "sialyl-Lewis X/A"),
        ("core_2_and_alpha2_3_sialylation", "Core 2 + α2-3"),
        ("core_3", "Core 3"),
        ("alpha2_6_sialylation", "α2-6 sialylation"),
    ]
    rows = []
    for feature, label in wanted:
        match = data.loc[data.feature.eq(feature)]
        if len(match) != 1:
            raise RuntimeError(f"expected exactly one O-glycomics row for {feature}")
        row = match.iloc[0].to_dict()
        row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    patient = patient_deltas()
    tcga = tcga_branches()
    ogly = external_structures()
    patient.to_csv(OUT / "neu5ac_targeted_eic_patient_deltas.csv", index=False)
    tcga.to_csv(OUT / "tcga_glycan_branch_effects.csv", index=False)
    ogly.to_csv(OUT / "external_mucinous_oglycan_structures.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.8,
        "axes.titlesize": 10.7,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(13.4, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.02, 1.0])

    # A. Patient-level abundance.
    ax = fig.add_subplot(grid[0, 0])
    order = ["Ltu", "Rtu", "Rmu"]
    labels = ["Left conventional", "Right conventional", "Right mucinous"]
    colors = {"Ltu": "#8a8f98", "Rtu": "#4c78a8", "Rmu": "#b33c2e"}
    rng = np.random.default_rng(20260831)
    for x, cohort in enumerate(order):
        values = patient.loc[patient.cohort.eq(cohort), "log2_tumour_minus_normal"].dropna().to_numpy(float)
        ax.scatter(np.full(len(values), x) + rng.uniform(-0.10, 0.10, len(values)), values,
                   s=31, color=colors[cohort], alpha=0.88, edgecolor="white", linewidth=0.4)
        ax.plot([x - 0.22, x + 0.22], [values.mean()] * 2, color="black", lw=2)
        ax.text(x, 4.38, f"{int((values > 0).sum())}/{len(values)} positive\nmean {values.mean():+.2f}",
                ha="center", va="bottom", fontsize=7.8)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_ylim(-3.0, 5.05)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Neu5Ac log2(tumour / matched normal)")
    ax.set_title("A  Locked targeted-EIC abundance", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # B. Branch-resolved TCGA forest.
    ax = fig.add_subplot(grid[0, 1])
    y = np.arange(len(tcga))[::-1]
    xerr = np.vstack([tcga.beta - tcga.ci_low, tcga.ci_high - tcga.beta])
    point_colors = ["#356859" if beta > 0 else "#a33b31" for beta in tcga.beta]
    for yi, (_, row), color in zip(y, tcga.iterrows(), point_colors):
        ax.errorbar(row.beta, yi, xerr=[[row.beta - row.ci_low], [row.ci_high - row.beta]],
                    fmt="o", color=color, ecolor=color, capsize=2.5, ms=5.5)
        q_label = f"q={row.bh_q:.2g}" if row.bh_q < 0.01 else f"q={row.bh_q:.3f}"
        ax.text(max(row.ci_high, 0.0) + 0.04, yi, q_label, va="center", fontsize=7.2)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y, tcga.label)
    ax.set_xlabel("Mucinous vs conventional adjusted beta (95% CI)")
    ax.set_title("B  Donor–carrier–core–linkage transcript branches", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # C. Patient-level structural O-glycomics.
    ax = fig.add_subplot(grid[1, 0])
    y = np.arange(len(ogly))[::-1]
    ax.axvline(0, color="#333333", lw=0.8)
    ax.scatter(ogly.T2_delta.astype(float), y + 0.10, color="#d95f02", label="T2 tumour − C2", s=48)
    ax.scatter(ogly.T3_delta.astype(float), y - 0.10, color="#1b9e77", label="T3 tumour − C3", s=48, marker="D")
    for yi, (_, row) in zip(y, ogly.iterrows()):
        ax.plot([float(row.T2_delta), float(row.T3_delta)], [yi + 0.10, yi - 0.10], color="#b8b8b8", lw=1)
        ax.text(73, yi, f"tumour ranks {int(row.T2_rank_desc)}/{int(row.T3_rank_desc)}", va="center", fontsize=7.1)
    ax.set_xlim(-80, 115)
    ax.set_yticks(y, ogly.label)
    ax.set_xlabel("Tumour − matched normal structural abundance")
    ax.set_title("C  Independent patient O-glycomics (MUC n=2)", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.50, -0.24), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    # D. Non-causal convergence model.
    ax = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    ax.set_title("D  Hybrid mucin glycome: convergent, non-causal model", loc="left", fontweight="bold")
    boxes = [
        (0.04, 0.65, 0.27, 0.22, "Resource/donor\nFree Neu5Ac ↑\nGNE/NANS/SLC35A1 ↑", "#e8f1ec"),
        (0.365, 0.65, 0.27, 0.22, "Carrier/lineage\nMUC2 program ↑\nCore-3 relative retention", "#edf0f7"),
        (0.69, 0.65, 0.27, 0.22, "Structure/linkage\nCore-2/sLeX acquired\nα2-6 lost", "#f7ece8"),
    ]
    for x0, y0, width, height, text, face in boxes:
        ax.add_patch(plt.Rectangle((x0, y0), width, height, facecolor=face, edgecolor="#5d5d5d", lw=1.1))
        ax.text(x0 + width / 2, y0 + height / 2, text, ha="center", va="center", fontsize=8.1)
        ax.annotate("", xy=(0.50, 0.42), xytext=(x0 + width / 2, y0),
                    arrowprops=dict(arrowstyle="->", linestyle="--", lw=1.1, color="#6b6b6b"))
    ax.add_patch(plt.Rectangle((0.26, 0.25), 0.48, 0.17, facecolor="#f4efe4", edgecolor="#6a5d4d", lw=1.3))
    ax.text(0.50, 0.335, "Hybrid mucin glycome\nDonor–carrier–core–linkage decoupling",
            ha="center", va="center", fontsize=10.2, fontweight="bold")
    ax.text(0.04, 0.09, "Not established: global hypersialylation, ST6GAL1–PD-L1 activation,\n"
            "free-Neu5Ac flux, specific MUC2 glycoform, enzyme causality or independent abundance replication",
            fontsize=8.2, color="#8d3027")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("A hybrid mucin glycome in mucinous colorectal cancer", fontsize=15, fontweight="bold")
    png = OUT / "neu5ac_hybrid_glycome_figure_v2.png"
    pdf = OUT / "neu5ac_hybrid_glycome_figure_v2.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    rmu = patient.loc[patient.cohort.eq("Rmu"), "log2_tumour_minus_normal"].dropna()
    report = {
        "status": "mtbls13729_neu5ac_hybrid_glycome_figure_v2_complete",
        "formal": False,
        "neu5ac_targeted_eic": {
            "rmu_n": int(len(rmu)),
            "rmu_positive": int(rmu.gt(0).sum()),
            "rmu_mean_log2fc": float(rmu.mean()),
        },
        "tcga": {
            "samples": {"mucinous": 42, "conventional": 329, "msi_complete": 364},
            "model": "HC3 OLS with clinical covariates and non-overlapping broad-lineage scores; MSI sensitivity reported separately",
        },
        "external_oglycomics": {
            "mucinous_cases": ["T2", "T3"],
            "role": "independent structural support, not free-Neu5Ac replication",
            "source_workbook_sha256": sha256(OGLY_XLSX),
        },
        "claim_limit": "Evidence convergence supports a hybrid mucin glycome and branch decoupling, not flux, carrier-specific incorporation, enzyme causality or independent abundance replication.",
        "provenance": {
            "eic_auc_sha256": sha256(EIC_DIR / "pos_rp__eic_auc_matrix.csv.gz"),
            "eic_detection_sha256": sha256(EIC_DIR / "pos_rp__eic_detection_matrix.csv.gz"),
            "tcga_results_sha256": sha256(TCGA),
            "ogly_values_sha256": sha256(OGLY),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
