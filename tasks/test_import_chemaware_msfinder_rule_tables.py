"""CPU contracts for the pinned MS-FINDER empirical rule tables."""
from __future__ import annotations

import json
from pathlib import Path

from import_chemaware_msfinder_rule_tables import PINNED_COMMIT, formula_mass


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    assert abs(formula_mass("H2O") - 18.01056468403) < 1e-9
    assert abs(formula_mass("C6H12O6") - 180.06338810418) < 1e-9
    try:
        formula_mass("C6H12O6+")
    except ValueError:
        pass
    else:
        raise AssertionError("malformed formula was accepted")
    output = ROOT / "data/validation/chemaware_msfinder_rule_tables_v1_20260928"
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    rules = json.loads((output / "msfinder_rules.json").read_text(encoding="utf-8"))
    assert report["source_commit"] == PINNED_COMMIT
    assert report["rules"] == 4_638
    assert report["neutral_loss_rules"] == 1_643
    assert report["product_ion_rules"] == 2_995
    assert all(report["gates"].values())
    assert len(rules["rules"]) == 4_638
    assert all(row["training_policy"].startswith("disabled_until") for row in rules["rules"])
    print("PASS: ChemAware MS-FINDER rule-table contracts", flush=True)


if __name__ == "__main__":
    main()
