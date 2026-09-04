"""Fail-closed validation for the LCNEC annotation benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_annotation_benchmark_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(OUT / "annotation_benchmark_ledger.csv")
    funnel = pd.read_csv(OUT / "discovery_funnel.csv")
    text = (OUT / "REPORT.md").read_text(encoding="utf-8")
    assert report["formal"] is True
    assert report["source_paper"]["annotation_rate_available"] is False
    assert report["source_paper"]["declared_metabolites_all_platforms"] == 1052
    assert report["source_paper"]["hsst3n_main_rows"] == 97
    assert report["reconstructed_qualified_universe"]["families"] == 263
    assert report["reconstructed_qualified_universe"]["source_hsst3n_overlap"] == 42
    assert report["frozen_dark_universe"]["modules"] == 81
    assert report["frozen_dark_universe"]["official_dreams_candidates"] == 51
    assert report["frozen_dark_universe"]["dreams_p2b_agreement"] == 45
    assert report["frozen_dark_universe"]["high_or_moderate_consistency_features"] == 22
    assert report["frozen_dark_universe"]["priority_hypotheses"] == 4
    assert len(ledger) == 10
    assert len(funnel) == 11
    for phrase in (
        "author annotation rate cannot be calculated",
        "source-table-absent analytical headroom",
        "candidate coverage",
        "not a ladder of accuracy estimates",
    ):
        assert phrase in text
    print("[validate_lcnec_annotation_biology_benchmark] PASS")


if __name__ == "__main__":
    main()
