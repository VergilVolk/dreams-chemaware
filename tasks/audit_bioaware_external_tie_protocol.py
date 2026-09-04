#!/usr/bin/env python
"""Audit strict-tie baseline reproduction across every available external unit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from develop_bioaware_rank_consensus_fusion import add_family_features, score_queries


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/validation/bioaware_metdna3_external_v3_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/bioaware_external_strict_tie_audit_20260830.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"fail-closed: output exists: {args.output}")
    rows, hashes = {}, {}
    for unit_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        ledger_path = unit_dir / "ledger" / "candidate_evidence.csv.gz"
        baseline_path = unit_dir / "baseline" / "raw_transitions.csv.gz"
        if not ledger_path.exists() or not baseline_path.exists():
            continue
        ledger = add_family_features(pd.read_csv(ledger_path))
        # Weights do not affect the baseline columns.
        prediction = score_queries(ledger, pd.Series([0.0] * 6).to_numpy())
        frozen = pd.read_csv(baseline_path).groupby("query_id", as_index=False).first()
        merged = prediction[["query_id", "baseline_candidate_id", "baseline_correct"]].merge(
            frozen[["query_id", "baseline_top_candidate", "baseline_correct"]],
            on="query_id", suffixes=("_replay", "_frozen"), validate="one_to_one",
        )
        id_mismatch = merged.baseline_candidate_id.astype(str).ne(merged.baseline_top_candidate.astype(str))
        correctness_mismatch = merged.baseline_correct_replay.astype(bool).ne(merged.baseline_correct_frozen.astype(bool))
        if id_mismatch.any() or correctness_mismatch.any():
            raise RuntimeError(f"{unit_dir.name}: strict baseline replay mismatch")
        display_correct = merged.baseline_candidate_id.astype(str).eq(
            ledger.groupby("query_id", sort=False).truth_candidate_id.first().reindex(merged.query_id).to_numpy().astype(str)
        )
        rows[unit_dir.name] = {
            "queries": int(len(merged)),
            "strict_recall1": float(merged.baseline_correct_replay.mean()),
            "display_id_recall1": float(display_correct.mean()),
            "ties_displaying_truth": int((display_correct & ~merged.baseline_correct_replay).sum()),
        }
        hashes[unit_dir.name] = {"ledger": sha256(ledger_path), "baseline": sha256(baseline_path)}
    report = {
        "status": "bioaware_external_strict_tie_protocol_audit_complete", "formal": True,
        "units": rows, "total_ties_displaying_truth": sum(row["ties_displaying_truth"] for row in rows.values()),
        "contract": "rank = 1 + number of negative candidates with score >= positive score; ties count against positive",
        "provenance": hashes,
        "claim_limit": "Protocol audit only; no BioAware performance result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
