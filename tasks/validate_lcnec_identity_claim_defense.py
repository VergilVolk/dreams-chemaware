"""Validate the frozen LCNEC identity-claim defense artifact."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/lcnec_hsst3n_identity_claim_defense_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((OUT / "priority_identity_claim_ledger.csv").open(encoding="utf-8-sig", newline="")))
    assert report["formal"] is True
    assert report["priority_hypotheses"] == 4 == len(rows)
    assert report["new_exact_metabolite_claims"] == 0
    assert report["standards_required_for_current_claim_set"] == 0
    assert all(row["exact_identity_allowed"] == "False" for row in rows)
    assert all(row["standard_required_for_level1_upgrade"] == "True" for row in rows)
    assert sum(row["bioaware_specific_anchor"] == "True" for row in rows) == 3
    assert sum(row["is_currency"] == "True" for row in rows) == 1
    assert (OUT / "REVIEWER_DEFENSE.md").stat().st_size > 1000
    print(f"[validate_lcnec_identity_claim_defense] PASS rows={len(rows)}")


if __name__ == "__main__":
    main()
