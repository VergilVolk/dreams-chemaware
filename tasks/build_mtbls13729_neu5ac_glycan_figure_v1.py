#!/usr/bin/env python
"""Build a manuscript-grade Neu5Ac/mucin-glycan evidence figure.

The figure combines three non-equivalent evidence layers without treating any
of them as causal proof: locked targeted-EIC abundance, TCGA bulk-expression
context, and an independent CRC O-glycomics structural audit.
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
TCGA = ROOT / "data/external/TCGA_COADREAD_Xena_20260830/proline_sialic_lineage_sensitivity_v1/report.json"
OGLY = ROOT / "data/external/CRC_Oglycomics_PMC9254241_20260830/mucinous_structural_audit_values.csv"
OGLY_XLSX = ROOT / "data/external/CRC_Oglycomics_PMC9254241_20260830/supplementary_tables.xlsx"
OUT = ROOT / "data/mtbls13729/neu5ac_glycan_publication_figure_v1"
FEATURE = 703


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def neu5ac_patient_deltas() -> pd.DataFrame:
    auc_path = EIC_DIR / "pos_rp__eic_auc_matrix.csv.gz"
    det_path = EIC_DIR / "pos_rp__eic_detection_matrix.csv.gz"
    auc = pd.read_csv(auc_path).set_index("feature_id")
    detected = pd.read_csv(det_path).set_index("feature_id")
    detected = detected.apply(lambda column: column.astype(str).str.lower().isin({"true", "1"}))
    if FEATURE not in auc.index or FEATURE not in detected.index:
        raise RuntimeError("feature 703 missing from locked targeted-EIC matrices")
    masked = auc.where(detected & (auc > 0.0))
    positive = masked.stack().to_numpy(float)
    pseudo = float(np.percentile(positive, 1) / 2.0)
    log_auc = np.log2(masked + pseudo)
    rows = []
    for cohort, tumour, normal in (("Ltu", "Ltu", "LN"), ("Rtu", "Rtu", "RN"), ("Rmu", "Rmu", "RN")):
        for tumour_sample, normal_sample in sample_pairs(list(log_auc.columns), tumour, normal):
            patient = tumour_sample.split("-")[0]
            t = log_auc.loc[FEATURE, tumour_sample]
            n = log_auc.loc[FEATURE, normal_sample]
            rows.append({
                "cohort": cohort,
                "patient": patient,
                "tumour_sample": tumour_sample,
                "normal_sample": normal_sample,
                "log2_tumour_minus_normal": float(t - n) if np.isfinite(t) and np.isfinite(n) else np.nan,
            })
    return pd.DataFrame(rows)


def tcga_axes() -> pd.DataFrame:
    report = json.loads(TCGA.read_text(encoding="utf-8"))
    wanted = {
        "sialic_acid_synthesis_transport": "Sialic synthesis/transport",
        "mucin_sialylation": "Mucin sialylation",
        "secretory_mucin_program": "Secretory mucin program",
    }
    rows = []
    by_axis = {row["axis"]: row for row in report["results"]}
    for axis, label in wanted.items():
        row = by_axis[axis]
        rows.append({
            "axis": axis,
            "label": label,
            "beta": row["lineage_adjusted_beta"],
            "ci_low": row["lineage_adjusted_ci_low"],
            "ci_high": row["lineage_adjusted_ci_high"],
            "p": row["lineage_adjusted_p"],
            "msi_lineage_beta": row["msi_lineage_beta"],
            "msi_lineage_p": row["msi_lineage_p"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    patient = neu5ac_patient_deltas()
    tcga = tcga_axes()
    ogly = pd.read_csv(OGLY)
    patient.to_csv(OUT / "neu5ac_targeted_eic_patient_deltas.csv", index=False)
    tcga.to_csv(OUT / "tcga_mucinous_axis_effects.csv", index=False)
    ogly.to_csv(OUT / "external_oglycomics_structural_values.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(12.6, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08])

    # A. Locked targeted-EIC paired effects.
    ax = fig.add_subplot(grid[0, 0])
    order = ["Ltu", "Rtu", "Rmu"]
    colors = {"Ltu": "#8a8f98", "Rtu": "#4c78a8", "Rmu": "#b33c2e"}
    rng = np.random.default_rng(20260831)
    for x, cohort in enumerate(order):
        values = patient.loc[patient.cohort.eq(cohort), "log2_tumour_minus_normal"].dropna().to_numpy(float)
        jitter = rng.uniform(-0.11, 0.11, len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=28, color=colors[cohort], alpha=0.84, edgecolor="white", linewidth=0.4)
        ax.plot([x - 0.22, x + 0.22], [values.mean(), values.mean()], color="black", lw=2)
        ax.text(x, max(values) + 0.25, f"n={len(values)}\nmean={values.mean():+.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(range(3), ["Left conventional", "Right conventional", "Right mucinous"])
    ax.set_ylabel("Neu5Ac log2(tumour / matched normal)")
    ax.set_title("A  Locked targeted-EIC abundance (feature 703)", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # B. TCGA histology contrast after broad-lineage adjustment.
    ax = fig.add_subplot(grid[0, 1])
    y = np.arange(len(tcga))[::-1]
    xerr = np.vstack([tcga.beta - tcga.ci_low, tcga.ci_high - tcga.beta])
    ax.errorbar(tcga.beta, y, xerr=xerr, fmt="o", color="#5b3f8c", ecolor="#5b3f8c", capsize=3, ms=6)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y, tcga.label)
    ax.set_xlabel("Mucinous vs conventional adjusted beta (95% CI)")
    ax.set_title("B  TCGA COADREAD transcriptional context", loc="left", fontweight="bold")
    for yi, (_, row) in zip(y, tcga.iterrows()):
        ax.text(row.ci_high + 0.025, yi, f"p={row.p:.2g}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # C. Independent glycomics paired structural changes.
    ax = fig.add_subplot(grid[1, 0])
    selected = ogly[ogly.feature.isin([
        "core_2", "sialyl_Lewis_X_A", "core_2_and_sialyl_Lewis_X_A",
        "core_2_and_alpha2_3_sialylation", "alpha2_6_sialylation",
    ])].copy()
    labels = {
        "core_2": "Core 2",
        "sialyl_Lewis_X_A": "sialyl-Lewis X/A",
        "core_2_and_sialyl_Lewis_X_A": "Core 2 + sLeX/A",
        "core_2_and_alpha2_3_sialylation": "Core 2 + α2-3",
        "alpha2_6_sialylation": "α2-6 sialylation",
    }
    selected["label"] = selected.feature.map(labels)
    selected = selected.set_index("feature").loc[list(labels)].reset_index()
    y = np.arange(len(selected))[::-1]
    ax.axvline(0, color="#333333", lw=0.8)
    ax.scatter(selected.T2_delta, y + 0.10, color="#d95f02", label="T2−C2", s=48)
    ax.scatter(selected.T3_delta, y - 0.10, color="#1b9e77", label="T3−C3", s=48, marker="D")
    for yi, (_, row) in zip(y, selected.iterrows()):
        ax.plot([row.T2_delta, row.T3_delta], [yi + 0.10, yi - 0.10], color="#b8b8b8", lw=1)
    ax.set_yticks(y, selected.label)
    ax.set_xlabel("Tumour − matched normal structural abundance")
    ax.set_title("C  Independent CRC O-glycomics (two MUC cases)", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # D. Interpretation and boundaries.
    ax = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    ax.set_title("D  Evidence-calibrated biological model", loc="left", fontweight="bold")
    boxes = [
        (0.05, 0.72, 0.26, 0.17, "Free Neu5Ac pool\nRmu ↑"),
        (0.37, 0.72, 0.26, 0.17, "Mucin/sialic program\nTCGA ↑"),
        (0.69, 0.72, 0.26, 0.17, "Core-2/sLeX/A ↑\nα2-6 ↓"),
    ]
    for x0, y0, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor="#f3efe8", edgecolor="#6a5d4d", lw=1.2))
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=10)
    for x0 in (0.31, 0.63):
        ax.annotate("", xy=(x0 + 0.05, 0.805), xytext=(x0, 0.805), arrowprops=dict(arrowstyle="->", lw=1.5, color="#6a5d4d"))
    ax.text(0.05, 0.53, "Supported model", fontweight="bold", color="#285943", fontsize=10)
    ax.text(0.05, 0.43, "Selective mucinous-relative sialic/mucin-glycan remodeling", fontsize=10)
    ax.text(0.05, 0.28, "Not established", fontweight="bold", color="#9f2f22", fontsize=10)
    ax.text(0.05, 0.11, "Free-Neu5Ac flux, glycosyltransferase causality,\ncell of origin, or population-level external replication", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("Mucinous-relative Neu5Ac and O-glycan remodeling in colorectal cancer", fontsize=15, fontweight="bold")
    png = OUT / "neu5ac_glycan_integrated_figure.png"
    pdf = OUT / "neu5ac_glycan_integrated_figure.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    rmu = patient[patient.cohort.eq("Rmu")].log2_tumour_minus_normal.dropna()
    report = {
        "status": "mtbls13729_neu5ac_glycan_publication_figure_complete",
        "formal": False,
        "neu5ac_targeted_eic": {
            "rmu_n": int(len(rmu)),
            "rmu_positive": int((rmu > 0).sum()),
            "rmu_mean_log2fc": float(rmu.mean()),
            "protocol": "locked targeted EIC, detection-masked, same pseudo-count rule as full_space_eic_analysis_v1",
        },
        "tcga": {
            "samples": "42 mucinous, 329 conventional",
            "model": "HC3 OLS with clinical covariates plus six broad-lineage expression scores",
        },
        "external_oglycomics": {
            "mucinous_cases": ["T2", "T3"],
            "comparison": "descriptive position versus nine AC primary tumours and paired C2/C3 controls",
            "source_workbook_sha256": sha256(OGLY_XLSX),
            "metadata_warning": "T.S8 mislabels T6 as MUC; T.S2 is authoritative",
        },
        "claim_limit": "Orthogonal convergence supports selective remodeling, not flux, glycosyltransferase causality, cell of origin, or independent free-Neu5Ac replication.",
        "provenance": {
            "eic_auc_sha256": sha256(EIC_DIR / "pos_rp__eic_auc_matrix.csv.gz"),
            "eic_detection_sha256": sha256(EIC_DIR / "pos_rp__eic_detection_matrix.csv.gz"),
            "tcga_report_sha256": sha256(TCGA),
            "ogly_values_sha256": sha256(OGLY),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
