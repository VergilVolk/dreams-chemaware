"""Test frozen metabolic-context genes against expression-independent genomic strata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analyze_lcnec_george2018_frozen_axes import bh, permutation_p, stage_bin


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external/LCNEC_George2018_transcriptome"
EXPRESSION = BASE / "Supplementary_Data_11.clean.xlsx"
ANNOTATION = BASE / "Supplementary_Data_12.xlsx"
PREREG = BASE / "frozen_axis_genomic_preregistration_v1.json"
OUT = BASE / "frozen_axis_genomic_audit_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_for(row: pd.Series) -> str | None:
    rb1, stk11, keap1 = int(row["RB1"]), int(row["STK11"]), int(row["KEAP1"])
    altered = {3, 4}
    if rb1 == 2 and (stk11 in altered or keap1 in altered):
        return "STK11/KEAP1-altered"
    if rb1 in altered and stk11 == 2 and keap1 == 2:
        return "RB1-altered"
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    axes: dict[str, list[str]] = prereg["axes"]
    genes = [gene for members in axes.values() for gene in members]
    if len(genes) != 22 or len(set(genes)) != 22:
        raise RuntimeError("genomic-strata preregistration must contain the frozen 22-gene panel")

    expression = pd.read_excel(EXPRESSION, header=2).set_index("Gene").loc[genes]
    expression.columns = [str(column).removeprefix("LCNEC_") for column in expression.columns]
    annotation = pd.read_excel(ANNOTATION, header=8)
    annotation = annotation.loc[annotation["Tumor_Type"].astype(str).eq("LCNEC")].copy()
    annotation["Sample_ID"] = annotation["Sample_ID"].astype(str)
    annotation = annotation.set_index("Sample_ID").loc[expression.columns]
    annotation["genomic_group"] = annotation.apply(group_for, axis=1)
    retained = annotation["genomic_group"].notna()
    group_counts = annotation.loc[retained, "genomic_group"].value_counts().to_dict()
    if set(group_counts) != {"STK11/KEAP1-altered", "RB1-altered"}:
        raise RuntimeError(f"both clean genomic strata are required: {group_counts}")
    if min(group_counts.values()) < int(prereg["minimum_group_size"]):
        raise RuntimeError(f"clean genomic group below minimum size: {group_counts}")

    selected_ids = annotation.index[retained]
    values = np.log2(expression[selected_ids].astype(float).T + 1.0)
    standard_deviation = values.std(axis=0, ddof=0)
    if (standard_deviation <= 0).any():
        raise RuntimeError("constant frozen gene in genomic contrast")
    z = (values - values.mean(axis=0)) / standard_deviation
    labels = annotation.loc[selected_ids, "genomic_group"].to_numpy(str)
    stage = annotation.loc[selected_ids, "tumor stage (LCNEC)"].map(stage_bin)
    groups = ["STK11/KEAP1-altered", "RB1-altered"]

    rng = np.random.default_rng(int(prereg["seed"]))
    axis_rows: list[dict[str, object]] = []
    gene_rows: list[dict[str, object]] = []
    for axis, members in axes.items():
        matrix = z[members].to_numpy(float)
        primary_p, r2 = permutation_p(matrix, labels, rng, 10000)
        known = stage.notna().to_numpy()
        stage_p, stage_r2 = permutation_p(
            matrix[known], labels[known], rng, 5000, stage.loc[known].to_numpy(str)
        )
        centroids = {group: matrix[labels == group].mean(axis=0) for group in groups}
        distances = np.asarray([
            np.linalg.norm(row - centroids[label]) for row, label in zip(matrix, labels, strict=True)
        ])
        dispersion_p = float(mannwhitneyu(
            distances[labels == groups[0]], distances[labels == groups[1]], alternative="two-sided"
        ).pvalue)
        axis_rows.append({
            "axis": axis,
            "genes": ";".join(members),
            "n_genes": len(members),
            "n_samples": len(matrix),
            "n_stk11_keap1_altered": int(np.sum(labels == groups[0])),
            "n_rb1_altered": int(np.sum(labels == groups[1])),
            "primary_permutation_p": primary_p,
            "multivariate_r2": r2,
            "stage_known_samples": int(known.sum()),
            "stage_stratified_permutation_p": stage_p,
            "stage_restricted_multivariate_r2": stage_r2,
            "dispersion_mannwhitney_p": dispersion_p,
        })
        for gene in members:
            left = values.loc[labels == groups[0], gene].to_numpy(float)
            right = values.loc[labels == groups[1], gene].to_numpy(float)
            test = mannwhitneyu(left, right, alternative="two-sided")
            gene_rows.append({
                "axis": axis,
                "gene": gene,
                "mannwhitney_u": float(test.statistic),
                "mannwhitney_p": float(test.pvalue),
                "stk11_keap1_median_log2_rsem": float(np.median(left)),
                "rb1_median_log2_rsem": float(np.median(right)),
                "median_difference_stk11_keap1_minus_rb1": float(np.median(left) - np.median(right)),
            })

    axis_result = pd.DataFrame(axis_rows)
    axis_result["primary_bh_q_3"] = bh(axis_result["primary_permutation_p"].tolist())
    axis_result["dispersion_bh_q_3"] = bh(axis_result["dispersion_mannwhitney_p"].tolist())
    axis_result["fixed_axis_gate"] = (
        (axis_result["primary_bh_q_3"] < 0.05)
        & (axis_result["multivariate_r2"] >= 0.10)
        & (axis_result["stage_stratified_permutation_p"] < 0.05)
        & (axis_result["dispersion_bh_q_3"] >= 0.05)
    )
    axis_result.to_csv(OUT / "axis_genomic_results.csv", index=False)

    gene_result = pd.DataFrame(gene_rows)
    gene_result["bh_q_22"] = bh(gene_result["mannwhitney_p"].tolist())
    gene_result["secondary_gene_gate"] = gene_result["bh_q_22"] < 0.05
    gene_result.to_csv(OUT / "gene_genomic_results.csv", index=False)
    annotation.loc[selected_ids, ["RB1", "STK11", "KEAP1", "tumor stage (LCNEC)", "genomic_group"]].to_csv(
        OUT / "clean_genomic_strata.csv", index_label="sample_id"
    )

    medians = pd.DataFrame(index=genes, columns=groups, dtype=float)
    for gene in genes:
        for group in groups:
            medians.loc[gene, group] = z.loc[labels == group, gene].median()
    fig, (ax_heat, ax_axis) = plt.subplots(1, 2, figsize=(12.6, 8.4), gridspec_kw={"width_ratios": [1.25, 1.0]})
    image = ax_heat.imshow(medians.to_numpy(float), cmap="RdBu_r", vmin=-1.25, vmax=1.25, aspect="auto")
    ax_heat.set_xticks(range(2), [f"STK11/KEAP1 altered\n(n={group_counts[groups[0]]})", f"RB1 altered\n(n={group_counts[groups[1]]})"])
    ax_heat.set_yticks(range(22), genes, fontsize=8.5)
    for boundary in np.cumsum([len(members) for members in axes.values()])[:-1] - 0.5:
        ax_heat.axhline(boundary, color="black", linewidth=1.2)
    ax_heat.set_title("Frozen genes across clean genomic strata", fontweight="bold")
    cbar = fig.colorbar(image, ax=ax_heat, shrink=0.82)
    cbar.set_label("Median gene-wise z score")

    ordered = axis_result.set_index("axis").loc[list(axes)]
    y = np.arange(3)
    colors = ["#1764ab" if value else "#9aa0a6" for value in ordered["fixed_axis_gate"]]
    ax_axis.barh(y, ordered["multivariate_r2"], color=colors)
    ax_axis.set_yticks(y, ["Quinolinate / de novo NAD", "ADP-ribose turnover", "Ascorbate / redox"])
    ax_axis.invert_yaxis()
    ax_axis.set_xlabel("Genomic-stratum multivariate R2")
    ax_axis.set_title("Expression-independent genomic contrast", fontweight="bold")
    for i, row in enumerate(ordered.itertuples()):
        ax_axis.text(row.multivariate_r2 + 0.005, i, f"q={row.primary_bh_q_3:.3g}\nstage p={row.stage_stratified_permutation_p:.3g}", va="center", fontsize=9)
    ax_axis.set_xlim(0, max(0.25, float(ordered["multivariate_r2"].max()) + 0.14))
    ax_axis.grid(axis="x", alpha=0.25)
    fig.suptitle("External LCNEC genomic-stratum audit of frozen metabolic-context axes", fontsize=14.5, fontweight="bold")
    fig.text(0.5, 0.015, "Groups are defined without expression outcomes; no matched normal tissue or metabolite measurements.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"frozen_axis_genomic_audit.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    passed = axis_result.loc[axis_result["fixed_axis_gate"], "axis"].tolist()
    report = {
        "status": "lcnec_george2018_frozen_axis_genomic_audit_complete",
        "formal": True,
        "clean_genomic_group_counts": group_counts,
        "excluded_lcnec_samples": int((~retained).sum()),
        "known_stage_in_clean_groups": int(stage.notna().sum()),
        "axes_passing_fixed_gate": passed,
        "genes_passing_secondary_bh22": gene_result.loc[
            gene_result["secondary_gene_gate"], ["axis", "gene", "bh_q_22", "median_difference_stk11_keap1_minus_rb1"]
        ].to_dict("records"),
        "provenance": {
            "expression_sha256": sha256(EXPRESSION),
            "annotation_sha256": sha256(ANNOTATION),
            "preregistration_sha256": sha256(PREREG),
            "script_sha256": sha256(Path(__file__)),
            "article_doi": "10.1038/s41467-018-03099-x"
        },
        "claim_limit": prereg["claim_limit"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
