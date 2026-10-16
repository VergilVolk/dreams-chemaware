#!/usr/bin/env python3
"""Collapse byte-expensive accession mirrors without collapsing modalities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def normalized_stem(value: str) -> str:
    return Path(value.replace("\\", "/").rsplit("/", 1)[-1]).stem.casefold()


def stable_id(values: tuple[str, ...]) -> str:
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest, sep="\t", dtype=str, keep_default_na=False)
    frame["deposited_bytes"] = pd.to_numeric(frame["deposited_bytes"], errors="raise").astype("int64")
    frame["normalized_file_stem"] = frame["filename"].map(normalized_stem)
    key_columns = [
        "preliminary_source_family", "normalized_file_stem", "taxonomy", "body_part",
        "instrument", "polarity", "chromatography",
    ]
    frame["mirror_group_id"] = [
        stable_id(tuple(row)) for row in frame[key_columns].itertuples(index=False, name=None)
    ]
    frame["acquisition_keep"] = True
    frame["acquisition_decision"] = ""
    # MassIVE can expose the same bytes in both ``peak`` and ``ccms_peak``
    # collections inside one accession.  The former implementation only
    # collapsed mirrors across accessions and therefore downloaded/analysed
    # exact within-accession mirrors twice.  Require matching byte size and
    # analytical metadata before collapsing these collection aliases.
    within_accession_key = [
        "preliminary_source_family", "dataset_accession", "normalized_file_stem",
        "taxonomy", "body_part", "instrument", "polarity", "chromatography",
        "deposited_bytes",
    ]
    within_accession_dropped = 0
    for _, block in frame.groupby(within_accession_key, sort=False):
        if len(block) == 1:
            continue
        ranked = block.assign(
            canonical_collection=block.get(
                "inventory_collections", pd.Series("", index=block.index)
            ).str.casefold().eq("ccms_peak").astype(int),
        ).sort_values(["canonical_collection", "filename"], ascending=[False, True])
        keep_index = ranked.index[0]
        drop_indices = block.index.difference([keep_index])
        frame.loc[keep_index, "acquisition_decision"] = "within_accession_collection_representative"
        frame.loc[drop_indices, "acquisition_keep"] = False
        frame.loc[drop_indices, "acquisition_decision"] = "within_accession_collection_mirror"
        within_accession_dropped += len(drop_indices)

    stage_one = frame.loc[frame["acquisition_keep"]]
    for _, block in stage_one.groupby("mirror_group_id", sort=False):
        if block["dataset_accession"].nunique() == 1:
            undecided = block.index[frame.loc[block.index, "acquisition_decision"].eq("")]
            frame.loc[undecided, "acquisition_decision"] = "unique_accession_measurement"
            continue
        # Prefer open mzML over mzXML, then the higher accession (normally the
        # updated deposition). No abundance or outcome is used.
        rank = block.assign(
            mzml=block["filename"].str.lower().str.endswith(".mzml").astype(int),
            accession_number=pd.to_numeric(
                block["dataset_accession"].str.extract(r"(\d+)$", expand=False), errors="coerce"
            ).fillna(0),
        ).sort_values(["mzml", "accession_number", "filename"], ascending=[False, False, True])
        keep_accession = rank.iloc[0]["dataset_accession"]
        keep_indices = block.index[block["dataset_accession"].eq(keep_accession)]
        frame.loc[block.index, "acquisition_keep"] = False
        frame.loc[block.index, "acquisition_decision"] = "accession_mirror_collapsed"
        frame.loc[keep_indices, "acquisition_keep"] = True
        frame.loc[keep_indices, "acquisition_decision"] = "accession_mirror_representative"

    kept = frame.loc[frame["acquisition_keep"]].copy()
    dropped = frame.loc[~frame["acquisition_keep"]].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "acquisition_adjudication.tsv", sep="\t", index=False)
    kept.to_csv(args.output_dir / "acquisition_manifest.tsv", sep="\t", index=False)
    dropped.to_csv(args.output_dir / "collapsed_mirrors.tsv", sep="\t", index=False)
    report = {
        "schema": "pan_redu_systematic_wave_acquisition_v2",
        "denominators": {
            "input_files": len(frame),
            "acquisition_files": len(kept),
            "collapsed_mirror_files": len(dropped),
            "input_bytes": int(frame["deposited_bytes"].sum()),
            "acquisition_bytes": int(kept["deposited_bytes"].sum()),
            "acquisition_gib": float(kept["deposited_bytes"].sum() / 2**30),
            "collapsed_mirror_bytes": int(dropped["deposited_bytes"].sum()),
            "within_accession_collection_mirror_files": within_accession_dropped,
        },
        "leakage_status": "metadata_and_file_inventory_only",
        "claim_boundary": (
            "Mirror collapse saves acquisition bytes; it does not turn remaining accessions into independent cohorts."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
