"""Test three frozen LCNEC metabolic-context axes in George et al. 2018.

The external cohort contains 66 LCNEC tumors but no matched normal tissue.
Accordingly, this analysis tests subtype heterogeneity only.  The 22 genes,
author subtype labels, multiplicity families and decision gates are read from a
pre-result preregistration file.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external/LCNEC_George2018_transcriptome"
EXPRESSION = BASE / "Supplementary_Data_11.clean.xlsx"
ANNOTATION = BASE / "Supplementary_Data_12.xlsx"
PREREG = BASE / "frozen_axis_subtype_preregistration_v1.json"
OUT = BASE / "frozen_axis_subtype_audit_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bh(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def pseudo_f(matrix: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    groups = np.unique(labels)
    if len(groups) < 2:
        raise RuntimeError("pseudo-F requires at least two groups")
    grand = matrix.mean(axis=0)
    between = 0.0
    within = 0.0
    for group in groups:
        block = matrix[labels == group]
        center = block.mean(axis=0)
        between += len(block) * float(np.sum((center - grand) ** 2))
        within += float(np.sum((block - center) ** 2))
    df_between = len(groups) - 1
    df_within = len(matrix) - len(groups)
    statistic = (between / df_between) / (within / df_within) if within > 0 else float("inf")
    r2 = between / (between + within) if between + within > 0 else 0.0
    return float(statistic), float(r2)


def permutation_p(
    matrix: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
    strata: np.ndarray | None = None,
) -> tuple[float, float]:
    observed, r2 = pseudo_f(matrix, labels)
    exceed = 0
    for _ in range(repeats):
        permuted = labels.copy()
        if strata is None:
            rng.shuffle(permuted)
        else:
            for stratum in np.unique(strata):
                idx = np.flatnonzero(strata == stratum)
                permuted[idx] = rng.permutation(permuted[idx])
        statistic, _ = pseudo_f(matrix, permuted)
        exceed += statistic >= observed - 1e-12
    return (exceed + 1) / (repeats + 1), r2


def stage_bin(value: object) -> str | None:
    value = str(value)
    if value in {"Ia", "Ib", "IIa", "IIb"}:
        return "early"
    if value in {"III", "IIIa", "IIIb", "IV"}:
        return "advanced"
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if not prereg["formal"] or prereg["outcome_values_inspected_before_freeze"]:
        raise RuntimeError("invalid preregistration state")
    axes: dict[str, list[str]] = prereg["axes"]
    frozen_genes = [gene for genes in axes.values() for gene in genes]
    if len(frozen_genes) != 22 or len(set(frozen_genes)) != 22:
        raise RuntimeError("frozen axis panel must contain exactly 22 unique genes")

    expression = pd.read_excel(EXPRESSION, header=2)
    annotation = pd.read_excel(ANNOTATION, header=8)
    if expression.columns[0] != "Gene" or expression.shape[1] != 67:
        raise RuntimeError(f"unexpected expression table shape: {expression.shape}")
    if expression["Gene"].duplicated().any():
        duplicates = expression.loc[expression["Gene"].duplicated(), "Gene"].tolist()
        raise RuntimeError(f"duplicate expression gene symbols: {duplicates[:10]}")
    panel = expression.set_index("Gene").loc[frozen_genes]
    sample_ids = [str(column).removeprefix("LCNEC_") for column in panel.columns]
    panel.columns = sample_ids

    annotation = annotation.loc[annotation["Tumor_Type"].astype(str).eq("LCNEC")].copy()
    subtype_column = "LCNEC_SCLC_classification_subtype (Figure 3a)"
    stage_column = "tumor stage (LCNEC)"
    annotation["Sample_ID"] = annotation["Sample_ID"].astype(str)
    annotation = annotation.set_index("Sample_ID").loc[sample_ids]
    groups = prereg["frozen_groups"]
    if annotation[subtype_column].value_counts().to_dict() != {
        "type 1 LCNEC": 30,
        "type 2 LCNEC": 25,
        "SCLC/SCLC-like": 11,
    }:
        raise RuntimeError(f"author subtype counts changed: {annotation[subtype_column].value_counts().to_dict()}")
    if set(annotation[subtype_column]) != set(groups):
        raise RuntimeError("author subtype labels changed")

    values = np.log2(panel.astype(float).T + 1.0)
    standard_deviation = values.std(axis=0, ddof=0)
    if (standard_deviation <= 0).any():
        raise RuntimeError(f"constant frozen genes: {standard_deviation[standard_deviation <= 0].index.tolist()}")
    z = (values - values.mean(axis=0)) / standard_deviation
    labels = annotation[subtype_column].to_numpy(str)
    stage = annotation[stage_column].map(stage_bin)

    rng = np.random.default_rng(int(prereg["seed"]))
    axis_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for axis, genes in axes.items():
        matrix = z[genes].to_numpy(float)
        primary_p, r2 = permutation_p(matrix, labels, rng, 10000)

        known = stage.notna().to_numpy()
        stage_p, stage_r2 = permutation_p(
            matrix[known], labels[known], rng, 5000, stage.loc[known].to_numpy(str)
        )

        centroids = {group: matrix[labels == group].mean(axis=0) for group in groups}
        distances = np.asarray([
            np.linalg.norm(row - centroids[label]) for row, label in zip(matrix, labels, strict=True)
        ])
        dispersion_p = float(kruskal(*(distances[labels == group] for group in groups)).pvalue)
        axis_rows.append({
            "axis": axis,
            "genes": ";".join(genes),
            "n_genes": len(genes),
            "n_samples": len(matrix),
            "primary_permutation_p": primary_p,
            "multivariate_r2": r2,
            "stage_known_samples": int(known.sum()),
            "stage_stratified_permutation_p": stage_p,
            "stage_restricted_multivariate_r2": stage_r2,
            "dispersion_kruskal_p": dispersion_p,
        })

        for left, right in combinations(groups, 2):
            keep = np.isin(labels, [left, right])
            p, pair_r2 = permutation_p(matrix[keep], labels[keep], rng, 10000)
            pair_rows.append({
                "axis": axis,
                "left_subtype": left,
                "right_subtype": right,
                "n_left": int(np.sum(labels == left)),
                "n_right": int(np.sum(labels == right)),
                "permutation_p": p,
                "multivariate_r2": pair_r2,
            })

    axis_result = pd.DataFrame(axis_rows)
    axis_result["primary_bh_q_3"] = bh(axis_result["primary_permutation_p"].tolist())
    axis_result["dispersion_bh_q_3"] = bh(axis_result["dispersion_kruskal_p"].tolist())
    axis_result["fixed_axis_gate"] = (
        (axis_result["primary_bh_q_3"] < 0.05)
        & (axis_result["multivariate_r2"] >= 0.10)
        & (axis_result["stage_stratified_permutation_p"] < 0.05)
        & (axis_result["dispersion_bh_q_3"] >= 0.05)
    )
    axis_result.to_csv(OUT / "axis_subtype_results.csv", index=False)

    pair_result = pd.DataFrame(pair_rows)
    pair_result["bh_q_9"] = bh(pair_result["permutation_p"].tolist())
    pair_result.to_csv(OUT / "pairwise_axis_subtype_results.csv", index=False)

    gene_rows: list[dict[str, object]] = []
    for axis, genes in axes.items():
        for gene in genes:
            arrays = [z.loc[labels == group, gene].to_numpy(float) for group in groups]
            test = kruskal(*arrays)
            row: dict[str, object] = {
                "axis": axis,
                "gene": gene,
                "kruskal_h": float(test.statistic),
                "kruskal_p": float(test.pvalue),
            }
            for group in groups:
                key = group.lower().replace(" ", "_").replace("/", "_")
                row[f"{key}_median_log2_rsem"] = float(values.loc[labels == group, gene].median())
                row[f"{key}_median_z"] = float(z.loc[labels == group, gene].median())
            gene_rows.append(row)
    gene_result = pd.DataFrame(gene_rows)
    gene_result["bh_q_22"] = bh(gene_result["kruskal_p"].tolist())
    gene_result["fixed_gene_secondary_gate"] = gene_result["bh_q_22"] < 0.05
    gene_result.to_csv(OUT / "gene_subtype_results.csv", index=False)

    axis_scores = pd.DataFrame(index=z.index)
    for axis, genes in axes.items():
        axis_scores[axis] = z[genes].mean(axis=1)
    axis_scores.insert(0, "author_subtype", labels)
    axis_scores.insert(1, "tumor_stage", annotation[stage_column].to_numpy())
    axis_scores.to_csv(OUT / "sample_axis_mean_z_scores.csv", index_label="sample_id")

    median_z = pd.DataFrame(index=frozen_genes, columns=groups, dtype=float)
    for gene in frozen_genes:
        for group in groups:
            median_z.loc[gene, group] = z.loc[labels == group, gene].median()
    median_z.insert(0, "axis", [next(axis for axis, genes in axes.items() if gene in genes) for gene in frozen_genes])
    median_z.to_csv(OUT / "subtype_gene_median_z_matrix.csv", index_label="gene")

    fig, (ax_heat, ax_axis) = plt.subplots(1, 2, figsize=(13.8, 8.5), gridspec_kw={"width_ratios": [1.5, 1.0]})
    heat_values = median_z[groups].to_numpy(float)
    image = ax_heat.imshow(heat_values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax_heat.set_xticks(range(3), ["Type 1\n(n=30)", "Type 2\n(n=25)", "SCLC-like\n(n=11)"])
    ax_heat.set_yticks(range(22), frozen_genes, fontsize=8.5)
    boundaries = np.cumsum([len(genes) for genes in axes.values()])[:-1] - 0.5
    for boundary in boundaries:
        ax_heat.axhline(boundary, color="black", linewidth=1.2)
    ax_heat.set_title("Median standardized expression of the frozen 22-gene panel", fontweight="bold")
    cbar = fig.colorbar(image, ax=ax_heat, shrink=0.82)
    cbar.set_label("Median gene-wise z score")

    ordered = axis_result.set_index("axis").loc[list(axes)]
    y = np.arange(len(ordered))
    colors = ["#1764ab" if passed else "#9aa0a6" for passed in ordered["fixed_axis_gate"]]
    ax_axis.barh(y, ordered["multivariate_r2"], color=colors, alpha=0.9)
    ax_axis.set_yticks(y, ["Quinolinate / de novo NAD", "ADP-ribose turnover", "Ascorbate / redox"])
    ax_axis.invert_yaxis()
    ax_axis.set_xlabel("Subtype-associated multivariate R2")
    ax_axis.set_title("Frozen-axis subtype effects", fontweight="bold")
    for i, row in enumerate(ordered.itertuples()):
        ax_axis.text(
            row.multivariate_r2 + 0.006,
            i,
            f"q={row.primary_bh_q_3:.3g}\nstage p={row.stage_stratified_permutation_p:.3g}",
            va="center",
            fontsize=9,
        )
    ax_axis.set_xlim(0, max(0.25, float(ordered["multivariate_r2"].max()) + 0.12))
    ax_axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Independent 66-tumor LCNEC transcriptomic subtype audit", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "Author-frozen subtypes; no matched normal tissue. Transcript heterogeneity is context, not metabolite replication or flux.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"frozen_axis_subtype_audit.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    passing_axes = axis_result.loc[axis_result["fixed_axis_gate"], "axis"].tolist()
    report = {
        "status": "lcnec_george2018_frozen_axis_subtype_audit_complete",
        "formal": True,
        "cohort": {
            "lcnec_tumors": 66,
            "type_1": 30,
            "type_2": 25,
            "sclc_like": 11,
            "known_stage": int(stage.notna().sum()),
            "matched_normal": 0,
        },
        "frozen_genes": 22,
        "axes_tested": 3,
        "axes_passing_fixed_gate": passing_axes,
        "genes_passing_secondary_bh22": gene_result.loc[
            gene_result["fixed_gene_secondary_gate"], ["axis", "gene", "bh_q_22"]
        ].to_dict("records"),
        "pairwise_axis_tests_passing_bh9": pair_result.loc[
            pair_result["bh_q_9"] < 0.05,
            ["axis", "left_subtype", "right_subtype", "multivariate_r2", "bh_q_9"],
        ].to_dict("records"),
        "gates": {
            "sample_overlap_66": len(sample_ids) == 66,
            "all_22_frozen_genes_measured": len(panel) == 22,
            "at_least_one_axis_passes_fixed_gate": bool(passing_axes),
        },
        "provenance": {
            "expression_sha256": sha256(EXPRESSION),
            "annotation_sha256": sha256(ANNOTATION),
            "preregistration_sha256": sha256(PREREG),
            "script_sha256": sha256(Path(__file__)),
            "article_doi": "10.1038/s41467-018-03099-x",
        },
        "claim_limit": prereg["claim_limit"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
