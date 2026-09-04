"""Validate the LCNEC same-formula rival audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_priority_formula_rivals_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    table = pd.read_csv(OUT / "priority_formula_rival_summary.csv")
    assert report["formal"] is True
    assert report["priorities"] == 4 == len(table)
    assert report["priorities_with_same_formula_top5_rivals"] == 3
    assert report["new_exact_metabolite_claims"] == 0
    assert table.loc[table["priority_name"] == "quinolinate", "nearest_same_formula_rival"].item() == "3-Nitrobenzoic acid"
    assert table.loc[table["priority_name"] == "ascorbate", "dreams_margin_to_same_formula_rival"].item() > 0.25
    assert table["exact_identity_allowed"].astype(bool).sum() == 0
    print("[validate_lcnec_priority_formula_rivals] PASS priorities=4")


if __name__ == "__main__":
    main()

