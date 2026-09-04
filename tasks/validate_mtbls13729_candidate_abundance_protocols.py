"""Fail-closed validation for the candidate abundance protocol reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/mtbls13729"


def main() -> None:
    out = BASE / "candidate_abundance_protocol_audit_v1"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(out / "candidate_abundance_protocol_summary.csv")
    detail = pd.read_csv(out / "candidate_abundance_patient_protocols.csv")
    corrected = pd.read_csv(BASE / "integrated_biology_ledger_v3/integrated_candidate_ledger_v3.csv")
    assert report["formal"] is True
    assert report["material_protocol_difference_features"] == [1717]
    assert len(summary) == 5 and len(detail) == 50
    row = summary.loc[summary.feature_id.eq(1717)].iloc[0]
    assert int(row.complete_detection_pairs) == 9
    assert int(row.complete_detection_positive_pairs) == 9
    assert abs(float(row.complete_detection_mean_log2fc) - 3.0094211614) < 1e-4
    c = corrected.loc[corrected.feature_id.eq(1717)].iloc[0]
    assert int(c.pairs) == 9 and int(c.positive_pairs) == 9
    assert "P28 excluded" in c.abundance_protocol_v3
    print("[validate_mtbls13729_candidate_abundance_protocols] PASS")


if __name__ == "__main__":
    main()
