"""Prepare test localization rows for frozen-panel-only confounder occlusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--localization", type=Path, default=Path("data/validation/large_test_failure_peak_localization/test_peak_evidence.csv"))
    parser.add_argument("--panel-hits", type=Path, default=Path("data/validation/large_test_frozen_peak_panel/test_frozen_panel_hits.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/large_test_frozen_panel_occlusion_input"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    localization = pd.read_csv(args.localization)
    hits = pd.read_csv(args.panel_hits)
    hits = hits.loc[hits["evidence"] == "confounder"].drop_duplicates(["query_index", "mz"])
    grouped = hits.groupby("query_index")["mz"].apply(lambda values: ";".join(f"{value:.5f}" for value in sorted(values))).rename("panel_mz")
    output = localization.merge(grouped, on="query_index", how="left", validate="one_to_one")
    output["fragment_all_confounder_support_mz"] = output["fragment_confounder_support_mz"]
    output["fragment_confounder_support_mz"] = output["panel_mz"]
    output["fragment_confounder_support_count"] = output["panel_mz"].fillna("").map(
        lambda value: len([item for item in value.split(";") if item])
    )
    output = output.loc[output["fragment_confounder_support_count"] > 0].drop(columns="panel_mz")
    output.to_csv(args.output_dir / "test_peak_evidence.csv", index=False)
    report = {
        "status": "frozen_panel_occlusion_input", "queries": len(output),
        "molecules": int(output["ik14"].nunique()), "formulas": int(output["formula"].nunique()),
        "target": "union of frozen-panel hits among confounder-supporting peaks",
        "random_exclusion": "all identity- and all confounder-supporting peaks excluded from random controls",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
