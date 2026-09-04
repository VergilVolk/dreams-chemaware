"""Fail-closed validation for the MTBLS13729 identity-claim defense."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/identity_claim_defense_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(OUT / "candidate_identity_claim_defense.csv")
    objections = pd.read_csv(OUT / "reviewer_objection_response_matrix.csv")
    text = (OUT / "REPORT.md").read_text(encoding="utf-8")
    assert report["formal"] is True
    assert report["headline_source_msi"] == "Level 1"
    assert report["headline_is_new_identity_claim"] is False
    assert report["source_absent_signals"] == 5
    assert report["source_absent_modules"] == 3
    assert report["new_exact_metabolite_claims"] == 0
    assert report["standards_required_for_current_claim_set"] == 0
    assert len(candidates) == 18
    assert not candidates.new_exact_identity_claimed.astype(bool).any()
    assert len(objections) == 5
    for phrase in (
        "not a new identity invented by DreaMS",
        "collapsed into 3 family modules",
        "false dilemma",
        "existing Level-1 identity",
    ):
        assert phrase in text
    print("[validate_mtbls13729_identity_claim_defense] PASS")


if __name__ == "__main__":
    main()
