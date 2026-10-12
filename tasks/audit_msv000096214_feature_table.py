#!/usr/bin/env python
"""Truth-blind feature-table audit for MSV000096214 (contract 3.5, v1.2).

Checks the frozen gates against the consensus matrix WITHOUT reading any
genotype column: G1 consensus feature count, G2 per-file raw detection
floor, G3 pooled-QC pairwise log2 median differences (descriptive),
G4 per-sample missingness after gap filling. Emits a pass/fail verdict that
gates the effect stage. Tube numbers are used only as sample identifiers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/external/msv000096214"))
    parser.add_argument("--consensus-dir", type=Path, default=Path("data/external/msv000096214/consensus"))
    parser.add_argument("--features-dir", type=Path, default=Path("data/external/msv000096214/ms1_features"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/msv000096214_table_audit"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    extraction = pd.read_csv(args.features_dir / "extraction_manifest.csv")
    report: dict[str, object] = {
        "status": "MSV000096214_TABLE_AUDIT",
        "data_role": "development/consumed",
        "contract": "docs/GLM_MSV000096214_STRUCTURE_HIDDEN_CONTRACT_20261012.md",
        "contract_version": "v1.2",
        "genotype_column_read": False,
        "polarity": {},
    }
    gates_all: dict[str, bool] = {}

    for polarity in ("positive", "negative"):
        cons_dir = args.consensus_dir / polarity
        matrix = pd.read_csv(cons_dir / "matrix_filled.csv.gz", index_col="group_id")
        groups = pd.read_csv(cons_dir / "groups.csv")
        samples = [c for c in matrix.columns]

        # G1: consensus feature count
        g1 = len(matrix) >= 2000

        # G2: study files with raw detection below floor
        pole_ext = extraction.loc[(extraction["polarity"] == polarity)
                                  & (extraction["role"] == "study")]
        low_files = pole_ext.loc[pole_ext["n_features"] < 1000, "sample"].tolist()
        g2 = len(low_files) <= 3

        # G3 (descriptive): pooled-QC pairwise median |log2 difference|
        qc_cols = [c for c in samples if c.startswith("SJ-Mutant-Flavo-CQ")]
        qc_diffs = {}
        for i in range(len(qc_cols)):
            for j in range(i + 1, len(qc_cols)):
                a = matrix[qc_cols[i]]
                b = matrix[qc_cols[j]]
                both = a.notna() & b.notna()
                if both.sum() > 0:
                    qc_diffs[f"{qc_cols[i]}|{qc_cols[j]}"] = float(
                        np.log2(a[both] / b[both]).abs().median())
        g3 = all(v <= 1.0 for v in qc_diffs.values()) if qc_diffs else False

        # G4 (v1.2): per-study-sample missingness after gap filling
        study_cols = [c for c in samples if not c.startswith("SJ-Mutant-Flavo-CQ")]
        missingness = (matrix[study_cols].isna().mean() * 100).to_dict()
        over = [c for c, v in missingness.items() if v > 50.0]
        g4 = len(over) <= 2

        # detection coverage context (descriptive)
        coverage = (matrix[study_cols].notna().mean() * 100).describe().to_dict()
        report["polarity"][polarity] = {
            "n_consensus_features": int(len(matrix)),
            "n_samples": len(samples),
            "n_study_samples": len(study_cols),
            "G1_features_ge_2000": bool(g1),
            "G2_low_detection_files": low_files,
            "G2_le_3_low_files": bool(g2),
            "G3_qc_pairwise_median_abs_log2diff": qc_diffs,
            "G3_all_le_1": bool(g3),
            "G4_sample_missingness_pct": {k: round(v, 2) for k, v in missingness.items()},
            "G4_over_50pct_samples": over,
            "G4_le_2_over": bool(g4),
            "coverage_summary_pct": {k: round(float(v), 2) for k, v in coverage.items()},
            "qc_reference_file": json.loads((cons_dir / "report.json").read_text(encoding="utf-8"))["reference_file"],
        }
        gates_all[f"{polarity}_G1"] = bool(g1)
        gates_all[f"{polarity}_G2"] = bool(g2)
        gates_all[f"{polarity}_G4"] = bool(g4)

    report["gates"] = gates_all
    hard_pass = all(gates_all.values())
    report["pass_to_effect_stage"] = bool(hard_pass)
    report["claim_limit"] = ("Table-construction gates only; no genotype labels were "
                             "joined and no effect or annotation results were opened.")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("gates", "pass_to_effect_stage")}, indent=2))


if __name__ == "__main__":
    main()
