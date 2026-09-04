"""Validate the frozen LCNEC priority patient-covariation audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_priority_patient_covariation_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    matrix = pd.read_csv(OUT / "patient_effect_matrix.csv", index_col=0)
    pairs = pd.read_csv(OUT / "pairwise_covariation.csv")
    assert report["formal"] is True
    assert matrix.shape == (34, 4) and not matrix.isna().any().any()
    assert len(pairs) == 6
    assert pairs["left"].str.cat(pairs["right"], sep="|").nunique() == 6
    assert (pairs["bh_q_6"].between(0, 1)).all()
    for name in ["priority_patient_covariation.png", "priority_patient_covariation.pdf"]:
        assert (OUT / name).stat().st_size > 100
    print(f"[validate_lcnec_priority_patient_covariation] PASS pairs=6 passing={report['pairs_passing_fixed_gate']}")


if __name__ == "__main__":
    main()
