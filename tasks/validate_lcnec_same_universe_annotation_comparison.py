"""Validate the LCNEC same-universe annotation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("data/validation/lcnec_hsst3n_same_universe_comparison_v1"),
    )
    args = parser.parse_args()
    report = json.loads((args.result_dir / "report.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(args.result_dir / "per_family_comparison.csv")
    comparison = pd.read_csv(args.result_dir / "same_universe_comparison.csv")
    assert report["formal"] is True
    assert report["source_annotation_rate_available"] is False
    assert len(ledger) == ledger["family_id"].nunique() == 263
    assert int(ledger["author_matched"].sum()) == 42
    assert int(ledger["official_dreams_candidate"].sum()) == 158
    assert int(ledger["multi_evidence_retained"].sum()) == 66
    assert report["source_table_absent_headroom"]["families"] == 221
    assert report["source_table_absent_headroom"]["multi_evidence_retained"] == 35
    assert report["source_table_absent_headroom"]["priority_author_unreported"] == 4
    assert comparison["boundary"].str.contains("not|accuracy|truth|absent", case=False).all()
    for name in ("same_universe_annotation_comparison.png", "same_universe_annotation_comparison.pdf"):
        assert (args.result_dir / name).stat().st_size > 10_000
    print("[validate_lcnec_same_universe_annotation_comparison] PASS families=263 retained=66")


if __name__ == "__main__":
    main()
