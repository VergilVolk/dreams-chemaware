"""Leave-one-gene-out robustness audit for external LCNEC genomic-stratum axes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analyze_lcnec_george2018_frozen_axes import bh, permutation_p, stage_bin


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external/LCNEC_George2018_transcriptome"
EXPRESSION = BASE / "Supplementary_Data_11.clean.xlsx"
ANNOTATION = BASE / "Supplementary_Data_12.xlsx"
PREREG = BASE / "frozen_axis_genomic_leave_one_gene_preregistration_v1.json"
STRATA = BASE / "frozen_axis_genomic_audit_v1/clean_genomic_strata.csv"
OUT = BASE / "frozen_axis_genomic_leave_one_gene_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if not prereg["formal"] or prereg["leave_one_gene_out_outcomes_inspected_before_freeze"]:
        raise RuntimeError("invalid leave-one-gene preregistration")
    axes: dict[str, list[str]] = prereg["axes"]
    genes = [gene for members in axes.values() for gene in members]
    if len(genes) != 22 or len(set(genes)) != 22:
        raise RuntimeError("frozen 22-gene panel changed")

    expression = pd.read_excel(EXPRESSION, header=2).set_index("Gene").loc[genes]
    expression.columns = [str(column).removeprefix("LCNEC_") for column in expression.columns]
    strata = pd.read_csv(STRATA).set_index("sample_id")
    if len(strata) != 39 or not set(strata.index).issubset(expression.columns):
        raise RuntimeError("clean genomic strata changed")
    values = np.log2(expression[strata.index].astype(float).T + 1.0)
    z = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
    labels = strata["genomic_group"].to_numpy(str)
    stages = strata["tumor stage (LCNEC)"].map(stage_bin)
    known = stages.notna().to_numpy()
    groups = ["STK11/KEAP1-altered", "RB1-altered"]
    if strata["genomic_group"].value_counts().to_dict() != {groups[0]: 22, groups[1]: 17}:
        raise RuntimeError("clean genomic group counts changed")

    rng = np.random.default_rng(int(prereg["seed"]))
    rows: list[dict[str, object]] = []
    for axis, members in axes.items():
        for omitted in members:
            retained = [gene for gene in members if gene != omitted]
            matrix = z[retained].to_numpy(float)
            primary_p, r2 = permutation_p(
                matrix, labels, rng, int(prereg["primary_permutations"])
            )
            stage_p, stage_r2 = permutation_p(
                matrix[known], labels[known], rng,
                int(prereg["stage_stratified_permutations"]),
                stages.loc[known].to_numpy(str),
            )
            centroids = {group: matrix[labels == group].mean(axis=0) for group in groups}
            distances = np.asarray([
                np.linalg.norm(row - centroids[label])
                for row, label in zip(matrix, labels, strict=True)
            ])
            dispersion_p = float(mannwhitneyu(
                distances[labels == groups[0]], distances[labels == groups[1]], alternative="two-sided"
            ).pvalue)
            rows.append({
                "axis": axis,
                "omitted_gene": omitted,
                "retained_genes": ";".join(retained),
                "n_retained_genes": len(retained),
                "primary_permutation_p": primary_p,
                "multivariate_r2": r2,
                "stage_stratified_permutation_p": stage_p,
                "stage_restricted_multivariate_r2": stage_r2,
                "dispersion_mannwhitney_p": dispersion_p,
            })

    result = pd.DataFrame(rows)
    if len(result) != 22:
        raise RuntimeError(f"expected 22 omissions, observed {len(result)}")
    result["primary_bh_q_22"] = bh(result["primary_permutation_p"].tolist())
    result["dispersion_bh_q_22"] = bh(result["dispersion_mannwhitney_p"].tolist())
    result["per_omission_gate"] = (
        (result["primary_bh_q_22"] < 0.05)
        & (result["multivariate_r2"] >= 0.08)
        & (result["stage_stratified_permutation_p"] < 0.05)
        & (result["dispersion_bh_q_22"] >= 0.05)
    )
    result.to_csv(OUT / "leave_one_gene_results.csv", index=False)

    summary = []
    for axis, block in result.groupby("axis", sort=False):
        summary.append({
            "axis": axis,
            "omissions": len(block),
            "omissions_passing": int(block["per_omission_gate"].sum()),
            "all_omissions_pass": bool(block["per_omission_gate"].all()),
            "minimum_r2": float(block["multivariate_r2"].min()),
            "maximum_primary_bh_q": float(block["primary_bh_q_22"].max()),
            "maximum_stage_p": float(block["stage_stratified_permutation_p"].max()),
            "minimum_dispersion_bh_q": float(block["dispersion_bh_q_22"].min()),
        })
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUT / "axis_leave_one_gene_summary.csv", index=False)
    report = {
        "status": "lcnec_george2018_genomic_leave_one_gene_complete",
        "formal": True,
        "omissions": len(result),
        "axes_all_omissions_passing": summary_frame.loc[
            summary_frame["all_omissions_pass"], "axis"
        ].tolist(),
        "axis_summary": summary,
        "provenance": {
            "expression_sha256": sha256(EXPRESSION),
            "annotation_sha256": sha256(ANNOTATION),
            "strata_sha256": sha256(STRATA),
            "preregistration_sha256": sha256(PREREG),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": prereg["claim_limit"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
