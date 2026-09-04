"""Validate the frozen LCNEC biology readiness scorecard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_biology_readiness_scorecard_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    table = pd.read_csv(OUT / "biology_readiness_scorecard.csv")
    assert report["formal"] is True
    assert report["components"] == 13 == len(table)
    assert report["public_data_level2_application_ready"] is True
    assert report["new_exact_metabolite_claim_ready"] is False
    assert report["causal_mechanism_claim_ready"] is False
    assert table.loc[table["component"] == "patient_level_shared_module", "status"].item() == "not_supported"
    assert table.loc[table["component"] == "exact_identity_level1", "status"].item() == "missing"
    assert table.loc[table["component"] == "external_genomic_stratum_context", "status"].item() == "ready_context"
    print("[validate_lcnec_biology_readiness_scorecard] PASS components=13")


if __name__ == "__main__":
    main()
