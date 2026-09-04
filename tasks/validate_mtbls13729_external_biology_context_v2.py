"""Validate the claim-aware external-context ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mtbls13729/external_biology_context_v2"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(OUT / "external_biology_context_ledger.csv")
    text = (OUT / "REPORT.md").read_text(encoding="utf-8")
    assert report["formal"] is False
    assert len(ledger) == 9 and ledger.axis.nunique() == 3
    assert ledger.url.str.startswith("https://").all()
    assert "does not validate local identities" in text
    assert "pool-to-destination decoupling" in text
    assert "not a previously unknown cancer metabolite" in text
    print("[validate_mtbls13729_external_biology_context_v2] PASS")


if __name__ == "__main__":
    main()
