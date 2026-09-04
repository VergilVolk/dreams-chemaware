"""Validate the frozen independent LCNEC proteogenomic fixed-panel output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(
            "data/external/LCNEC_proteogenomic_2026/fixed_panel_patient_audit_v1"
        ),
    )
    args = parser.parse_args()
    report = json.loads((args.result_dir / "report.json").read_text(encoding="utf-8"))
    proteins = pd.read_csv(args.result_dir / "protein_results.csv")
    axes = pd.read_csv(args.result_dir / "axis_results.csv")
    pairs = pd.read_csv(args.result_dir / "pure_lcnec_patient_pair_differences.csv")

    assert report["formal"] is True
    assert report["cohort"]["protein_pairs"] == 103
    assert report["cohort"]["pure_lcnec_protein_pairs"] >= 75
    assert report["cohort"]["combined_lcnec_protein_pairs"] >= 20
    assert len(proteins) == 22
    assert proteins["gene"].nunique() == 22
    assert len(axes) == 3
    assert set(report["panel"]["missing_proteins"]) == set(
        proteins.loc[~proteins["measured"], "gene"]
    )
    assert (proteins.loc[~proteins["measured"], "primary_wilcoxon_p"] == 1.0).all()
    assert (proteins.loc[~proteins["measured"], "primary_bh_q_22"] == 1.0).all()
    expected_gate = (
        proteins["measured"]
        & (proteins["primary_pairs"] >= 20)
        & proteins["direction_stable"]
        & (proteins["primary_bh_q_22"] < 0.10)
    )
    assert (expected_gate == proteins["primary_protein_gate"]).all()
    assert pairs["patient_id"].nunique() == report["cohort"]["pure_lcnec_protein_pairs"]
    assert "not evaluated" in report["keap1_status"]
    print(
        "[validate_lcnec_independent_proteogenomic_fixed_panel] PASS "
        f"proteins={len(proteins)} axes={len(axes)} pairs={pairs['patient_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
