"""Validate the secondary-module patient-level sensitivity package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/secondary_module_independence_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    matrix = pd.read_csv(OUT / "patient_module_effects_and_technical_factors.csv")
    correlations = pd.read_csv(OUT / "module_correlation_audit.csv")
    assert report["formal"] is True
    assert len(matrix) == 10
    assert len(correlations.loc[correlations.comparison_type.eq("biology_vs_biology")]) == 6
    assert len(correlations.loc[correlations.comparison_type.eq("biology_vs_technical")]) == 12
    assert all(report["gates"].values())
    assert (OUT / "secondary_module_independence.png").stat().st_size > 10_000
    print("[validate_mtbls13729_secondary_module_independence] PASS")


if __name__ == "__main__":
    main()
