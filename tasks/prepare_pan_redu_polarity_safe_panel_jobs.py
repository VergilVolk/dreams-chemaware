#!/usr/bin/env python3
"""Freeze a polarity-safe Pan-ReDU panel-job table before positive-mode contracts run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from prepare_pan_redu_partial_transport_pairs import canonical_polarity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-jobs", required=True, type=Path)
    parser.add_argument("--panel-metadata", required=True, type=Path)
    parser.add_argument("--required-polarity", choices=("positive", "negative"), default="positive")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    jobs = pd.read_csv(args.panel_jobs, sep="\t", dtype=str)
    if jobs.panel_id.duplicated().any():
        raise RuntimeError("panel-job table contains duplicate panel IDs")
    metadata = pd.read_csv(args.panel_metadata, sep="\t", dtype=str, usecols=["panel_id", "polarity"])
    metadata["canonical_polarity"] = metadata.polarity.map(canonical_polarity)
    polarity = metadata[["panel_id", "canonical_polarity"]].drop_duplicates()
    if polarity.groupby("panel_id").canonical_polarity.nunique().max() != 1:
        raise RuntimeError("a panel maps to multiple polarities")
    merged = jobs.merge(polarity, on="panel_id", how="left", validate="one_to_one")
    if merged.canonical_polarity.isna().any():
        raise RuntimeError("panel-job table contains panels without polarity metadata")
    selected = merged.loc[merged.canonical_polarity.eq(args.required_polarity)].copy()
    selected = selected.sort_values("job_index", key=lambda value: value.astype(int), kind="stable")
    selected["job_index"] = range(len(selected))
    selected = selected[list(jobs.columns)]
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "panel_jobs.tsv"
    selected.to_csv(path, sep="\t", index=False)
    excluded = merged.loc[~merged.canonical_polarity.eq(args.required_polarity), [
        "panel_id", "canonical_polarity",
    ]].to_dict("records")
    report = {
        "schema": "pan_redu_polarity_safe_panel_jobs_v1",
        "status": "POLARITY_SAFE_PANEL_JOBS_FROZEN",
        "required_polarity": args.required_polarity,
        "denominators": {
            "input_panels": len(jobs), "eligible_panels": len(selected),
            "excluded_panels": len(excluded),
        },
        "excluded_panels": excluded,
        "provenance": {
            "input_panel_jobs_sha256": sha256(args.panel_jobs),
            "panel_metadata_sha256": sha256(args.panel_metadata),
            "output_panel_jobs_sha256": sha256(path),
        },
        "claim_boundary": (
            "Positive-library identity and positive-adduct ion-family contracts are undefined for "
            "excluded polarities; excluded panels are not unresolved chemical entities."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
