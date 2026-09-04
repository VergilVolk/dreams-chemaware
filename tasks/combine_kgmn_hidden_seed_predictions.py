#!/usr/bin/env python3
"""Combine the fixed 10-repeat x 2-polarity KGMN prediction shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "repeat",
    "truth_inchikey1",
    "candidate_inchikey1",
    "candidate_score",
    "propagation_depth",
    "polarity",
    "peak_name",
    "adduct",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise RuntimeError("refusing to overwrite combined hidden-seed output")

    expected = {
        (repeat, polarity): args.shard_dir / f"repeat_{repeat:02d}_{polarity}.csv"
        for repeat in range(args.repeats)
        for polarity in ("positive", "negative")
    }
    missing = [str(path) for path in expected.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing prediction shards: {missing[:5]}")
    frames: list[pd.DataFrame] = []
    provenance: dict[str, str] = {}
    for (repeat, polarity), path in expected.items():
        frame = pd.read_csv(path)
        absent = REQUIRED.difference(frame.columns)
        if absent:
            raise RuntimeError(f"{path} misses columns: {sorted(absent)}")
        if not frame["repeat"].eq(repeat).all() or not frame["polarity"].eq(polarity).all():
            raise RuntimeError(f"shard labels disagree with path: {path}")
        frames.append(frame)
        provenance[path.name] = sha256(path)
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise RuntimeError("combined prediction table is empty")
    for column in ("truth_inchikey1", "candidate_inchikey1"):
        combined[column] = combined[column].fillna("").astype(str).str.strip().str.slice(0, 14)
        if combined[column].eq("").any():
            raise RuntimeError(f"combined table has empty {column}")
    combined["candidate_score"] = pd.to_numeric(combined["candidate_score"], errors="raise")
    combined["propagation_depth"] = pd.to_numeric(
        combined["propagation_depth"], errors="raise"
    ).astype(int)
    combined = combined.sort_values(
        ["repeat", "polarity", "truth_inchikey1", "candidate_score", "propagation_depth"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, compression="gzip")
    report = {
        "status": "kgmn_hidden_seed_predictions_combined",
        "formal": True,
        "repeats": args.repeats,
        "polarities": 2,
        "shards": len(expected),
        "rows": len(combined),
        "truth_identities": int(combined["truth_inchikey1"].nunique()),
        "candidate_identities": int(combined["candidate_inchikey1"].nunique()),
        "provenance": {
            "shards_sha256": provenance,
            "combined_sha256": sha256(args.output),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "Prediction assembly only; no performance claim.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
