#!/usr/bin/env python
"""Fail-closed integrity checks for the final noise E0 evidence ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    directory = args.output_dir.resolve()
    required = [
        "e0_manifest.json",
        "unified_query_action_ledger.csv.gz",
        "historical_cell_summary.csv",
        "unified_action_effect_matrix.csv",
        "changed_outcome_explanations.csv.gz",
        "e0_risk_weighted_action_matrix.png",
        "e0_correction_risk_tradeoff.png",
        "e0_a4_dose_response.png",
    ]
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise RuntimeError(f"E0 is incomplete: {missing}")

    manifest = json.loads((directory / "e0_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "noise_final_e0_unified_matrix_complete":
        raise RuntimeError("unexpected E0 status")
    if not args.allow_partial and (not manifest.get("formal") or not manifest.get("historical_sources_complete")):
        raise RuntimeError("formal E0 must include every locked historical stage")

    ledger_path = directory / "unified_query_action_ledger.csv.gz"
    if sha256(ledger_path) != manifest["outputs"]["ledger_sha256"]:
        raise RuntimeError("ledger hash mismatch")
    header = pd.read_csv(ledger_path, nrows=0).columns.tolist()
    forbidden = [name for name in header if any(token in name.lower() for token in ("p2b", "reranker", "rank_fusion", "raw_v1"))]
    if forbidden:
        raise RuntimeError(f"downstream fields found in noise-only ledger: {forbidden}")

    rows = 0
    corrected = 0
    introduced = 0
    for chunk in pd.read_csv(ledger_path, chunksize=100_000):
        if (chunk["corrected"].astype(bool) & chunk["introduced"].astype(bool)).any():
            raise RuntimeError("a row cannot be both corrected and introduced")
        rows += len(chunk)
        corrected += int(chunk["corrected"].astype(bool).sum())
        introduced += int(chunk["introduced"].astype(bool).sum())
    if rows != int(manifest["ledger_rows"]):
        raise RuntimeError(f"ledger row mismatch: {rows} != {manifest['ledger_rows']}")
    if corrected <= 0 or introduced <= 0:
        raise RuntimeError("E0 must preserve both corrections and collateral errors")

    cells = pd.read_csv(directory / "historical_cell_summary.csv")
    effects = pd.read_csv(directory / "unified_action_effect_matrix.csv")
    if len(cells) != int(manifest["cell_rows"]) or len(effects) != int(manifest["effect_matrix_rows"]):
        raise RuntimeError("summary row count mismatch")
    if not (cells["risk_net_lambda2"] < 0).any() or not (cells["risk_net_lambda2"] > 0).any():
        raise RuntimeError("historical matrix must preserve both beneficial and harmful cells")

    print(
        json.dumps(
            {
                "status": "noise_final_e0_validation_passed",
                "formal": bool(manifest["formal"]),
                "ledger_rows": rows,
                "corrected_action_rows": corrected,
                "introduced_action_rows": introduced,
                "stages": sorted(effects["stage"].unique().tolist()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
