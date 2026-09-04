"""Validate the exploratory independent protein-axis covariation audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external/LCNEC_proteogenomic_2026/protein_axis_covariation_exploratory_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(OUT / "all_within_axis_pairwise_covariation.csv")
    assert report["formal"] is False
    assert report["patients"] == 80
    assert report["axes"] == 3
    assert report["pairwise_tests"] == 46 == len(pairs)
    assert pairs["patients"].eq(80).all()
    assert pairs["bh_q_46"].between(0, 1).all()
    print(f"[validate_lcnec_independent_protein_axis_covariation] PASS pairs=46 passing={report['pairs_passing_exploratory_gate']}")


if __name__ == "__main__":
    main()

