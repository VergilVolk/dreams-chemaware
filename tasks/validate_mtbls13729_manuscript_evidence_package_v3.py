"""Fail-closed validation for manuscript evidence package v3."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/manuscript_evidence_package_v3"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    results = pd.read_csv(OUT / "manuscript_result_evidence_table.csv")
    figures = pd.read_csv(OUT / "figure_manifest.csv")
    text = (OUT / "RESULTS.md").read_text(encoding="utf-8")
    assert report["formal"] is True
    assert report["biology_package_A_ready"] is True
    assert report["new_exact_metabolite_claims"] == 0
    assert len(results) == results.result_id.nunique() == 9
    assert len(figures) == 7
    for phrase in (
        "candidate-coverage numbers, not structure-accuracy estimates",
        "Full 13,155-target exact FDR10: zero hits",
        "threeway_application_v1/*__threeway_features.csv.gz",
        "No new exact metabolite claim survived",
        "missing-as-zero artifact in P28",
        "three secondary modules, not five new metabolites",
        "strengthens context and narrows, rather than inflates, novelty",
    ):
        assert phrase in text
    print("[validate_mtbls13729_manuscript_evidence_package_v3] PASS")


if __name__ == "__main__":
    main()
