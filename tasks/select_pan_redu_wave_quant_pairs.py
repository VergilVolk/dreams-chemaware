#!/usr/bin/env python3
"""Select compact GNPS quantitative/MS2 candidates for schema qualification."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


EXCLUDE = re.compile(r"sirius|metadata|edge|annotation|full", re.I)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.inventory, sep="\t", dtype=str, keep_default_na=False)
    frame["bytes"] = pd.to_numeric(frame["bytes"], errors="raise").astype("int64")
    basename = frame["filepath"].map(lambda value: Path(value).name)
    mgf = (
        frame["asset_kind"].eq("ms2_export")
        & ~basename.str.contains(EXCLUDE)
        & (frame["bytes"] <= 100_000_000)
    )
    quant = (
        frame["asset_kind"].eq("quantitative_candidate")
        & (
            basename.str.contains("quant", case=False, regex=False)
            | basename.str.contains("feature", case=False, regex=False)
            | basename.str.contains("mzmineexport", case=False, regex=False)
            | basename.str.contains("ms2", case=False, regex=False)
        )
        & ~basename.str.contains(EXCLUDE)
        & (frame["bytes"] <= 200_000_000)
    )
    selected = frame.loc[mgf | quant].copy()
    kinds = selected.groupby(["dataset_accession", "asset_kind"]).size().unstack(fill_value=0)
    eligible = set(kinds.index[(kinds.get("ms2_export", 0) > 0) & (kinds.get("quantitative_candidate", 0) > 0)])
    selected = selected.loc[selected["dataset_accession"].isin(eligible)].copy()
    selected["selection_role"] = "processed_pair_schema_qualification"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "selected_assets.tsv", sep="\t", index=False)
    report = {
        "schema": "pan_redu_wave_quant_pair_selection_v1",
        "denominators": {
            "accessions_with_selected_pair_candidates": len(eligible),
            "selected_assets": len(selected),
            "selected_bytes": int(selected["bytes"].sum()),
            "selected_mgf": int(selected["asset_kind"].eq("ms2_export").sum()),
            "selected_quant_tables": int(selected["asset_kind"].eq("quantitative_candidate").sum()),
        },
        "leakage_status": "path_and_size_selection_only_contents_unopened",
        "claim_boundary": "Selection does not establish a valid feature-to-spectrum or sample-to-design join.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
