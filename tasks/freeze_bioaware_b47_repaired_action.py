#!/usr/bin/env python
"""Freeze repaired B47 exact-event actions onto a unary candidate ranking.

This is the reconstructed, source-controlled successor to the lost U4 source.
It consumes no annotation truth.  A repaired event winner is promoted only
when it differs from the unique unary winner or resolves a unary tie; every
unmapped query and every already-winning candidate remains unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-dir", type=Path, required=True)
    parser.add_argument("--unary-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if "".join(path.suffixes).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    args = arguments()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".json").exists():
        raise FileExistsError(args.output)
    report_path = args.repair_dir / "report.json"
    action_path = args.repair_dir / "repaired_action_ledger.csv.gz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b47_gate_remediation_complete":
        raise RuntimeError("B47 repair artifact is not complete")
    if report.get("authorization_after_repair") is not True:
        raise RuntimeError("B47 repair did not authorize a confirmatory action freeze")
    scores = read_table(args.unary_scores)
    required_scores = {"query_id", "candidate_id", "score"}
    if not required_scores.issubset(scores.columns):
        raise RuntimeError(f"unary scores miss {sorted(required_scores - set(scores.columns))}")
    scores = scores.loc[:, ["query_id", "candidate_id", "score"]].copy()
    scores["query_id"] = scores["query_id"].astype(str)
    scores["candidate_id"] = scores["candidate_id"].astype(str)
    scores["score"] = pd.to_numeric(scores["score"], errors="raise").astype(float)
    if scores.duplicated(["query_id", "candidate_id"]).any() or not np.isfinite(scores["score"]).all():
        raise RuntimeError("unary score keys are duplicated or non-finite")
    if not scores.groupby("query_id", sort=False).size().ge(2).all():
        raise RuntimeError("every unary query must contain at least two candidates")

    actions = pd.read_csv(action_path)
    required_actions = {"query_id", "event_top_candidate"}
    if not required_actions.issubset(actions.columns):
        raise RuntimeError(f"action ledger misses {sorted(required_actions - set(actions.columns))}")
    actions = actions.copy()
    actions["query_id"] = actions["query_id"].astype(str)
    actions["event_top_candidate"] = actions["event_top_candidate"].astype(str)
    if actions["query_id"].duplicated().any():
        raise RuntimeError("repaired action ledger has more than one action per query")
    candidate_keys = pd.MultiIndex.from_frame(scores[["query_id", "candidate_id"]])
    action_keys = pd.MultiIndex.from_arrays(
        [actions["query_id"], actions["event_top_candidate"]],
        names=["query_id", "candidate_id"],
    )
    missing = action_keys.difference(candidate_keys)
    if len(missing):
        raise RuntimeError(f"{len(missing)} repaired actions do not exist in unary candidate keys")

    result = scores.copy()
    promoted = 0
    already_unique = 0
    tie_resolved = 0
    for action in actions.itertuples(index=False):
        query_id = str(action.query_id)
        candidate_id = str(action.event_top_candidate)
        mask = result["query_id"].eq(query_id)
        block = result.loc[mask, "score"].to_numpy(float)
        maximum = float(np.max(block))
        winners = int(np.sum(block == maximum))
        selected_mask = mask & result["candidate_id"].eq(candidate_id)
        selected_value = float(result.loc[selected_mask, "score"].iloc[0])
        if winners == 1 and selected_value == maximum:
            already_unique += 1
            continue
        result.loc[selected_mask, "score"] = np.nextafter(maximum, np.inf)
        promoted += 1
        tie_resolved += int(selected_value == maximum)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=args.output.parent, prefix=f".{args.output.name}.", suffix=".csv.gz",
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        result.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    frozen_report = {
        "status": "BIOAWARE_B47_REPAIRED_ACTION_FROZEN",
        "schema": "bioaware_b47_repaired_exact_event_action_v1",
        "truth_fields_used": [],
        "queries": int(result["query_id"].nunique()),
        "candidate_rows": int(len(result)),
        "authorized_actions": int(len(actions)),
        "promoted_actions": promoted,
        "tie_resolutions": tie_resolved,
        "already_unique_unary_winner": already_unique,
        "unmapped_queries_return_exact_unary_order": True,
        "provenance": {
            "repair_report_sha256": sha256(report_path),
            "repaired_action_ledger_sha256": sha256(action_path),
            "unary_scores_sha256": sha256(args.unary_scores),
            "frozen_scores_sha256": sha256(args.output),
        },
        "claim_limit": "Truth-blind score freeze only; no accuracy claim before one-time evaluation.",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(frozen_report, indent=2), encoding="utf-8",
    )
    print(json.dumps(frozen_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
