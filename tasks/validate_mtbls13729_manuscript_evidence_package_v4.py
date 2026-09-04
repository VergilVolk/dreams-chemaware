"""Fail-closed validation for manuscript evidence package v4."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/manuscript_evidence_package_v4"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    results = pd.read_csv(OUT / "manuscript_result_evidence_table.csv")
    figures = pd.read_csv(OUT / "figure_manifest.csv")
    text = (OUT / "RESULTS.md").read_text(encoding="utf-8")
    assert report["formal"] is True
    assert report["biology_package_A_ready"] is True
    assert report["new_exact_metabolite_claims"] == 0
    assert report["bioaware_external_gain_confirmed"] is False
    assert len(results) == results.result_id.nunique() == 10
    assert len(figures) == 8
    assert (OUT / "bioaware_algorithm_biology_bridge.png").is_file()
    assert (OUT / "bioaware_algorithm_biology_bridge.pdf").is_file()
    assert "R10" in set(results.result_id)
    for phrase in (
        "BioAware is retained as a conservative context expert",
        "zero exact identity promotions",
        "Version-specific external evidence was mixed",
        "3,599 (21.23%)",
        "These are coverage estimates, not accuracy gains",
    ):
        assert phrase in text
    print("[validate_mtbls13729_manuscript_evidence_package_v4] PASS")


if __name__ == "__main__":
    main()
