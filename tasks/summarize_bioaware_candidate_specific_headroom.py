#!/usr/bin/env python
"""Consolidate old network and new candidate-specific BioAware headroom."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def reaches_requirement(old_unique: int, new_unique: int, required: int) -> bool:
    if min(old_unique, new_unique, required) < 0:
        raise ValueError("headroom counts must be nonnegative")
    return old_unique + new_unique >= required


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-headroom", type=Path, default=Path(
        "data/validation/bioaware_10pp_headroom_v1/report.json"))
    parser.add_argument("--unresolved", type=Path, default=Path(
        "data/validation/bioaware_10pp_headroom_v1/unresolved_error_queries.csv.gz"))
    parser.add_argument("--decoder", type=Path, default=Path(
        "data/validation/bioaware_candidate_fragment_decoder_v1/query_headroom.csv.gz"))
    parser.add_argument("--rules", type=Path, default=Path(
        "data/validation/bioaware_candidate_rule_likelihood_v1/query_headroom.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/validation/bioaware_candidate_specific_headroom_v1"))
    args = parser.parse_args()
    for path in (args.old_headroom, args.unresolved, args.decoder, args.rules):
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    old = json.loads(args.old_headroom.read_text(encoding="utf-8"))
    unresolved = pd.read_csv(args.unresolved)
    decoder = pd.read_csv(args.decoder).set_index("query_id")
    rules = pd.read_csv(args.rules).set_index("query_id")
    rows: list[dict] = []
    for base in unresolved.itertuples(index=False):
        query_id = str(base.query_id)
        decoder_hit = bool(decoder.loc[query_id, "decoder_correct"])
        fixed_rule_hits = {
            "rule_overlap_idf": bool(rules.loc[query_id, "rule_overlap_idf_correct"]),
            "rule_jaccard_idf": bool(rules.loc[query_id, "rule_jaccard_idf_correct"]),
            "sparse_rule_overlap": bool(rules.loc[query_id, "sparse_rule_overlap_correct"]),
        }
        rows.append({
            "query_id": query_id,
            "truth_candidate_id": str(base.truth_candidate_id),
            "truth_formula": str(base.truth_formula),
            "truth_name": str(base.truth_name),
            "dreams_margin": float(base.dreams_margin),
            "decoder_headroom": decoder_hit,
            **{f"{name}_headroom": value for name, value in fixed_rule_hits.items()},
            "candidate_specific_headroom": bool(decoder_hit or any(fixed_rule_hits.values())),
            "evidence_arm_count": int(decoder_hit + sum(fixed_rule_hits.values())),
        })
    table = pd.DataFrame(rows)
    new_union = int(table["candidate_specific_headroom"].sum())
    old_actual = int(old["actual_union"]["unique_errors_corrected_by_at_least_one_current_rule"])
    combined_actual = old_actual + new_union
    required = int(old["protocol"]["required_net_corrections_for_10pp"])
    table_path = output / "unresolved_candidate_specific_headroom.csv.gz"
    table.to_csv(table_path, index=False)
    payload = {
        "status": "bioaware_candidate_specific_headroom_consolidated",
        "formal": True,
        "protocol_queries": int(old["protocol"]["queries"]),
        "required_net_corrections_for_10pp": required,
        "old_actual_unique_headroom": old_actual,
        "new_candidate_specific_unique_headroom": new_union,
        "new_candidate_specific_identities": int(
            table.loc[table["candidate_specific_headroom"], "truth_candidate_id"].nunique()
        ),
        "new_headroom_with_two_or_more_fixed_arms": int((table["evidence_arm_count"] >= 2).sum()),
        "combined_actual_unique_headroom": combined_actual,
        "combined_headroom_reaches_10pp_requirement": reaches_requirement(
            old_actual, new_union, required
        ),
        "remaining_without_candidate_specific_headroom": int((~table["candidate_specific_headroom"]).sum()),
        "decision": (
            "The +10 pp target is now mathematically reachable in consumed development, "
            "but direct decoder/rule overrides are unsafe. Build a risk-controlled global "
            "assignment with abstention; do not open RP until corrected-2*introduced, "
            "formula-cluster CI and decoy gates pass."
        ),
        "provenance": {
            "old_headroom_sha256": sha256(args.old_headroom),
            "unresolved_sha256": sha256(args.unresolved),
            "decoder_sha256": sha256(args.decoder),
            "rules_sha256": sha256(args.rules),
            "table_sha256": sha256(table_path),
        },
        "claim_limit": (
            "Union headroom uses truth only after each evidence arm was frozen. It is not "
            "a deployable fusion accuracy and cannot be called SOTA."
        ),
    }
    atomic_json(output / "report.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
