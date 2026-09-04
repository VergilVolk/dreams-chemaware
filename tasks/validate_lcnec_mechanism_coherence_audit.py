from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/validation/lcnec_hsst3n_mechanism_coherence_v1"),
    )
    args = parser.parse_args()
    report = json.loads((args.input_dir / "report.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(args.input_dir / "axis_evidence_ledger.csv")
    axes = pd.read_csv(args.input_dir / "axis_summary.csv")
    if report["formal"] or not report["descriptive_post_selection"]:
        raise RuntimeError("mechanism coherence must remain descriptive and post-selection")
    if len(axes) != 4 or len(ledger) != 14:
        raise RuntimeError(f"unexpected fixed-panel size: axes={len(axes)} members={len(ledger)}")
    if not ledger["direction_matches_fixed_axis"].all():
        raise RuntimeError("one or more fixed-axis directions do not match")
    if int(ledger["evidence_class"].str.startswith("R_").sum()) != 9:
        raise RuntimeError("expected nine source-atlas reproductions in the fixed axes")
    if int(ledger["evidence_class"].str.startswith("N_").sum()) != 4:
        raise RuntimeError("expected four author-unreported priority hypotheses in the fixed axes")
    if set(ledger["identity_claim"]) != {"MSI_Level_2_or_connectivity_family_only"}:
        raise RuntimeError("identity boundary drift")
    print("[validate_lcnec_mechanism_coherence_audit] PASS axes=4 members=14")


if __name__ == "__main__":
    main()
