"""Fail-closed validation of the frozen George et al. LCNEC axis audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external/LCNEC_George2018_transcriptome"
SUBTYPE = BASE / "frozen_axis_subtype_audit_v1"
GENOMIC = BASE / "frozen_axis_genomic_audit_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(value: float, expected: float, tolerance: float = 1e-10) -> None:
    if abs(float(value) - expected) > tolerance:
        raise RuntimeError(f"numeric drift: observed={value}, expected={expected}")


def main() -> None:
    subtype_report = json.loads((SUBTYPE / "report.json").read_text(encoding="utf-8"))
    genomic_report = json.loads((GENOMIC / "report.json").read_text(encoding="utf-8"))
    loo_report = json.loads((BASE / "frozen_axis_genomic_leave_one_gene_v1/report.json").read_text(encoding="utf-8"))
    subtype_axes = pd.read_csv(SUBTYPE / "axis_subtype_results.csv").set_index("axis")
    genomic_axes = pd.read_csv(GENOMIC / "axis_genomic_results.csv").set_index("axis")
    genomic_genes = pd.read_csv(GENOMIC / "gene_genomic_results.csv").set_index("gene")
    strata = pd.read_csv(GENOMIC / "clean_genomic_strata.csv")

    expected_axes = {"quinolinate_de_novo_nad", "adp_ribose_turnover", "ascorbate_redox"}
    if set(subtype_axes.index) != expected_axes or set(genomic_axes.index) != expected_axes:
        raise RuntimeError("frozen axis set changed")
    if subtype_report["cohort"] != {
        "lcnec_tumors": 66,
        "type_1": 30,
        "type_2": 25,
        "sclc_like": 11,
        "known_stage": 63,
        "matched_normal": 0,
    }:
        raise RuntimeError("author subtype counts changed")
    if set(subtype_report["axes_passing_fixed_gate"]) != {"quinolinate_de_novo_nad"}:
        raise RuntimeError("expression-derived subtype gate result changed")

    if genomic_report["clean_genomic_group_counts"] != {
        "STK11/KEAP1-altered": 22,
        "RB1-altered": 17,
    }:
        raise RuntimeError("clean genomic group counts changed")
    if len(strata) != 39 or strata["sample_id"].nunique() != 39:
        raise RuntimeError("clean genomic strata must contain 39 unique tumors")
    if set(genomic_report["axes_passing_fixed_gate"]) != expected_axes:
        raise RuntimeError("all three frozen axes must pass the preregistered genomic gate")
    if loo_report["axes_all_omissions_passing"] != ["ascorbate_redox"]:
        raise RuntimeError("leave-one-gene robustness result changed")
    loo_summary = {row["axis"]: row for row in loo_report["axis_summary"]}
    if {axis: row["omissions_passing"] for axis, row in loo_summary.items()} != {
        "quinolinate_de_novo_nad": 8,
        "adp_ribose_turnover": 4,
        "ascorbate_redox": 8,
    }:
        raise RuntimeError("leave-one-gene passing counts changed")
    if not genomic_axes["fixed_axis_gate"].all():
        raise RuntimeError("genomic fixed gate column changed")
    if not ((genomic_axes["primary_bh_q_3"] < 0.05)
            & (genomic_axes["multivariate_r2"] >= 0.10)
            & (genomic_axes["stage_stratified_permutation_p"] < 0.05)
            & (genomic_axes["dispersion_bh_q_3"] >= 0.05)).all():
        raise RuntimeError("one or more genomic axis gate components failed")

    close(genomic_axes.loc["quinolinate_de_novo_nad", "multivariate_r2"], 0.11100650032794812)
    close(genomic_axes.loc["adp_ribose_turnover", "multivariate_r2"], 0.10409746834890381)
    close(genomic_axes.loc["ascorbate_redox", "multivariate_r2"], 0.1373471414466457)
    expected_gene_differences = {
        "NMNAT1": -0.5726285969956093,
        "NMNAT3": -1.6193730088954457,
        "PARP1": -0.8545484219704136,
        "TKT": 1.4233143969461413,
    }
    observed_genes = set(genomic_genes.index[genomic_genes["secondary_gene_gate"]])
    if observed_genes != set(expected_gene_differences):
        raise RuntimeError(f"secondary genomic gene set changed: {observed_genes}")
    for gene, expected in expected_gene_differences.items():
        close(genomic_genes.loc[gene, "median_difference_stk11_keap1_minus_rb1"], expected)

    expected_hashes = {
        "expression_sha256": "f028ab56602fae9339dd948f414cac02b834d7b3e10506ef0847827b42227556",
        "annotation_sha256": "d9eaa0a2194440eaa67df03fb0c48a1e34fec0403b26e140d90624f80584ad7f",
    }
    for key, expected in expected_hashes.items():
        if genomic_report["provenance"][key] != expected:
            raise RuntimeError(f"reported source hash changed: {key}")
    if sha256(BASE / "Supplementary_Data_11.clean.xlsx") != expected_hashes["expression_sha256"]:
        raise RuntimeError("expression workbook hash mismatch")
    if sha256(BASE / "Supplementary_Data_12.xlsx") != expected_hashes["annotation_sha256"]:
        raise RuntimeError("annotation workbook hash mismatch")

    for directory in (SUBTYPE, GENOMIC):
        for suffix in ("png", "pdf"):
            figures = list(directory.glob(f"*.{suffix}"))
            if len(figures) != 1 or figures[0].stat().st_size < 10_000:
                raise RuntimeError(f"missing or undersized {suffix} figure in {directory}")

    report = {
        "status": "lcnec_george2018_external_axis_validation_passed",
        "formal": True,
        "external_lcnec_tumors": 66,
        "clean_genomic_tumors": 39,
        "subtype_axis_gates_passing": 1,
        "genomic_axis_gates_passing": 3,
        "secondary_genes_passing": sorted(expected_gene_differences),
        "axes_fully_leave_one_gene_robust": ["ascorbate_redox"],
        "claim_limit": genomic_report["claim_limit"],
    }
    (BASE / "external_axis_validation_v1.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
