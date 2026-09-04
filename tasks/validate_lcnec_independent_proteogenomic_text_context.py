from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external/LCNEC_proteogenomic_2026/text_context_audit_v1"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(OUT / "fixed_gene_text_mentions.csv")
    if report["formal"]:
        raise RuntimeError("text context cannot be formal outcome validation")
    if not report["fixed_panel_frozen_before_text_outcome_audit"]:
        raise RuntimeError("pre-outcome freeze was not preserved")
    if len(ledger) != 22 or ledger["gene"].duplicated().any():
        raise RuntimeError(f"unexpected fixed panel: rows={len(ledger)}")
    required_mentions = {"IDO1", "G6PD", "PGD", "TKT", "TALDO1"}
    observed_mentions = set(ledger.loc[ledger["mentioned_in_article_text"], "gene"])
    if not required_mentions.issubset(observed_mentions):
        raise RuntimeError(f"article-text extraction missed fixed genes: {sorted(required_mentions - observed_mentions)}")
    print(f"[validate_lcnec_independent_proteogenomic_text_context] PASS genes={len(ledger)}")


if __name__ == "__main__":
    main()
