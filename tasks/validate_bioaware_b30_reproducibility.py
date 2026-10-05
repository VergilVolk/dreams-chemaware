#!/usr/bin/env python3
"""Fail-closed deterministic replay check for the BioAware B30 action."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


FILES = (
    "nested_sink_veto_transitions.csv.gz",
    "cross_source_candidate_history.csv.gz",
)

REPORT_FIELDS = (
    "status",
    "protocol",
    "strictly_better_action_than_B17",
    "frozen_B17_comparator",
    "nested_row_oof",
    "nested_physical_oof",
    "folds",
    "gates",
    "contracts",
)


def decompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_report(directory: Path) -> dict:
    path = directory / "report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b30_cross_source_sink_veto_complete":
        raise RuntimeError(f"wrong B30 status in {path}")
    if report.get("strictly_better_action_than_B17") is not True:
        raise RuntimeError(f"B30 is not strictly better in {path}")
    if not all(report.get("gates", {}).values()):
        raise RuntimeError(f"B30 gates are not all true in {path}")
    return report


def compare(first: Path, second: Path) -> dict:
    first_report = load_report(first)
    second_report = load_report(second)
    file_hashes: dict[str, str] = {}
    for name in FILES:
        first_path = first / name
        second_path = second / name
        if not first_path.is_file() or not second_path.is_file():
            raise FileNotFoundError(f"missing replay file: {name}")
        first_hash = decompressed_sha256(first_path)
        second_hash = decompressed_sha256(second_path)
        if first_hash != second_hash:
            raise RuntimeError(
                f"B30 replay mismatch for {name}: {first_hash} != {second_hash}"
            )
        file_hashes[name] = first_hash
    for field in REPORT_FIELDS:
        if first_report.get(field) != second_report.get(field):
            raise RuntimeError(f"B30 replay report mismatch: {field}")
    return {
        "status": "bioaware_b30_deterministic_replay_passed",
        "first": str(first),
        "second": str(second),
        "decompressed_sha256": file_hashes,
        "delta_recall1": first_report["nested_row_oof"]["delta_recall1"],
        "corrected": first_report["nested_row_oof"]["corrected"],
        "introduced": first_report["nested_row_oof"]["introduced"],
        "risk_net_lambda2": first_report["nested_row_oof"]["risk_net_lambda2"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.first, args.second)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
