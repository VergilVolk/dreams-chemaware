"""Plot the local metabolite phenotype beside mucinous-risk transcript contexts.

The panels intentionally keep abundance, transcript association, and causal
interpretation separate.  This is a manuscript asset, not a new hypothesis test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/mucinous_risk_context_figure_v1"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def errorbar_panel(ax, records: list[dict], title: str, cohort_note: str) -> None:
    axes = [
        "modified_nucleoside_processing",
        "purine_synthesis_salvage",
        "carnitine_long_chain_fao",
        "polyamine_acetylation_catabolism",
    ]
    labels = ["Modified\nnucleosides", "Purine\nsalvage", "LCFA / FAO", "Polyamine\nacetylation"]
    lookup = {row["axis"]: row for row in records}
    y = np.arange(len(axes))
    for offset, prefix, label, color, marker in [
        (-0.10, "clinical_adjusted", "clinical-adjusted", "#64748B", "o"),
        (+0.10, "clinical_and_lineage_adjusted", "+ broad-lineage adjusted", "#7C3AED", "s"),
    ]:
        values = np.array([float(lookup[a][f"{prefix}_rho"]) for a in axes])
        lo = np.array([float(lookup[a][f"{prefix}_ci_low"]) for a in axes])
        hi = np.array([float(lookup[a][f"{prefix}_ci_high"]) for a in axes])
        ax.errorbar(
            values,
            y + offset,
            xerr=np.vstack([values - lo, hi - values]),
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=3,
            lw=1.4,
            ms=5.5,
            label=label,
        )
    ax.axvline(0, color="#111827", lw=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.85, 0.85)
    ax.set_xlabel("Partial-rank association with MuC23 risk (rho, 95% CI)")
    ax.set_title(title, loc="left", weight="bold")
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.text(
        0.01,
        0.02,
        cohort_note,
        transform=ax.transAxes,
        va="bottom",
        fontsize=7.7,
        color="#475569",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    local_path = ROOT / "data/mtbls13729/biology_closure_analysis_v1/candidate_identity_and_abundance.csv"
    gse_path = ROOT / "data/external/GSE281917/mucinous_axis_composition_audit_v1/report.json"
    tcga_path = ROOT / "data/external/TCGA_COADREAD_Xena_20260830/mucinous_risk_axis_replication_v1/report.json"

    local = pd.read_csv(local_path).set_index("feature_id")
    gse = load_json(gse_path)
    tcga = load_json(tcga_path)
    tcga_records = [tcga["primary_result"], *tcga["secondary_results"]]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig = plt.figure(figsize=(16.5, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.08])

    # A. Direct metabolite abundance discovery.
    ax = fig.add_subplot(grid[0, :])
    features = [1597, 3019, 4966, 1717, 3222]
    labels = [
        "Me-guanosine\n1597",
        "diMe-guanosine\n3019",
        "purine-like\n4966",
        "diacetyl-polyamine\n1717",
        "long-chain acylcarnitine\n3222",
    ]
    values = local.loc[features, "rmu_mean_log2fc"].to_numpy(float)
    colors = ["#5B4BC4", "#7766D7", "#998AE6", "#D97706", "#0F766E"]
    bars = ax.bar(np.arange(len(values)), values, color=colors, width=0.68)
    ax.set_xticks(np.arange(len(values)), labels)
    ax.set_ylabel("Mean paired log2 fold change, Rmu vs RN")
    ax.set_title("A  Direct tissue abundance phenotype in MTBLS13729 (10 Rmu pairs)", loc="left", weight="bold")
    ax.set_ylim(0, 4.35)
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    for bar, value, feature in zip(bars, values, features):
        row = local.loc[feature]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.07,
            f"{value:+.2f}\n{int(row.rmu_n)} pairs",
            ha="center",
            va="bottom",
            fontsize=8.1,
        )
    ax.text(
        0.99,
        0.93,
        "Family-level identities; paired abundance only\n(no flux or enzyme-activity inference)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#475569",
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.9, "pad": 3},
    )

    # B/C. Risk-context association, with composition sensitivity shown explicitly.
    ax_gse = fig.add_subplot(grid[1, 0])
    errorbar_panel(
        ax_gse,
        gse["axis_associations"],
        "B  GSE281917 mucinous cohort (n=140)",
        "Purine association survives broad-lineage adjustment;\nother axes are composition-sensitive.",
    )
    ax_tcga = fig.add_subplot(grid[1, 1])
    errorbar_panel(
        ax_tcga,
        tcga_records,
        "C  TCGA mucinous cohort (n=42; targeted replication)",
        "Clinical-only directions recur, but purine loses significance\nafter lineage adjustment; polyamine is exploratory.",
    )

    fig.suptitle(
        "Mucinous CRC metabolite abundance and risk-associated transcript context are related but non-equivalent",
        fontsize=15.5,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.018,
        fill(
            "Interpretation: MTBLS13729 establishes a strong paired abundance phenotype in a 10-patient discovery subgroup. "
            "GSE281917 and TCGA provide risk-associated bulk transcript context, not metabolite replication, cell-autonomous "
            "mechanism, subtype specificity, isotope flux, or independent prognostic validation.",
            width=175,
        ),
        ha="center",
        va="bottom",
        fontsize=8.3,
        color="#334155",
    )

    png = OUT / "mtbls13729_mucinous_risk_context.png"
    pdf = OUT / "mtbls13729_mucinous_risk_context.pdf"
    fig.savefig(png, dpi=230, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "mtbls13729_mucinous_risk_context_figure_complete",
        "formal": False,
        "panels": {
            "A": "direct paired Rmu-vs-RN metabolite abundance",
            "B": "within-GSE281917 MuC23 transcript-axis association with composition sensitivity",
            "C": "targeted TCGA mucinous transcript-axis replication with composition sensitivity",
        },
        "claim_limit": "Risk-associated transcript contexts are not independent metabolite replication, flux, enzyme activity, cell-autonomous mechanism, or independent survival validation.",
        "provenance": {
            "local_abundance_sha256": sha256(local_path),
            "gse_composition_audit_sha256": sha256(gse_path),
            "tcga_replication_sha256": sha256(tcga_path),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(png)


if __name__ == "__main__":
    main()
