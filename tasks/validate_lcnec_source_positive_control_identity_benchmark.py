"""Validate source-positive-control identity benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_source_positive_control_identity_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(OUT / "positive_control_identity_ledger.csv")
    assert report["formal"] is True
    assert report["source_matched_families"] == 42
    assert report["structure_resolvable_unique"] >= 15
    assert len(ledger) == 42
    evaluable = ledger.loc[ledger["resolution_status"] == "unique"]
    assert len(evaluable) == report["structure_resolvable_unique"]
    incorrect_retained = int(evaluable["full_tool_discordant_retained"].astype(bool).sum())
    assert incorrect_retained == report["metrics"]["full_tool_incorrect_retained"]
    same_formula_errors = int(evaluable["same_formula_isomer_error"].astype(bool).sum())
    assert same_formula_errors == report["metrics"]["full_tool_incorrect_retained_same_formula_isomers"]
    print(f"[validate_lcnec_source_positive_control_identity_benchmark] PASS evaluable={len(evaluable)}")


if __name__ == "__main__":
    main()
