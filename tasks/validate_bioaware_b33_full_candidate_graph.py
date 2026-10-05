#!/usr/bin/env python3
"""Fail-closed validation for a BioAware B33 full-reference candidate graph."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report_path = args.directory / "report.json"
    graph_path = args.directory / "full_candidate_graph.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "bioaware_b33_full_candidate_graph_complete"
        or report.get("pass_to_full_graph_bridge") is not True
        or not all(report.get("gates", {}).values())
        or report.get("provenance", {}).get("graph_sha256") != sha256(graph_path)
    ):
        raise RuntimeError("B33 full-candidate report gate failed")
    with np.load(graph_path, allow_pickle=False) as handle:
        body = {name: handle[name] for name in handle.files}
    query_ptr = body["candidate_ptr"].astype(np.int64)
    reference_ptr = body["candidate_all_reference_ptr"].astype(np.int64)
    positions = body["candidate_all_reference_position"].astype(np.int64)
    if (
        len(body["query_id"]) != 860
        or len(query_ptr) != 861
        or int(query_ptr[-1]) != len(body["candidate_id"])
        or len(reference_ptr) != len(body["candidate_id"]) + 1
        or int(reference_ptr[-1]) != len(positions)
        or np.any(np.diff(query_ptr) < 2)
        or np.any(np.diff(reference_ptr) < 1)
        or np.any(positions < 0)
        or np.any(positions >= len(body["candidate_reference_tensor"]))
    ):
        raise RuntimeError("B33 graph array contract failed")
    if (
        int(np.sum(body["direct_corrective"].astype(bool))) != 63
        or int(np.sum(body["direct_safety"].astype(bool))) != 14
        or not np.array_equal(
            body["baseline_rank_full_graph"].astype(np.int64) == 1,
            body["baseline_correct"].astype(bool),
        )
    ):
        raise RuntimeError("B33 frozen action/baseline contract failed")
    groups: dict[tuple[str, str], list[int]] = {}
    for position, key in enumerate(zip(
        body["source"].astype(str), body["physical_query_id"].astype(str), strict=True,
    )):
        groups.setdefault(key, []).append(position)
    first = []
    for key, positions_in_group in groups.items():
        for role in ("baseline_correct", "direct_corrective", "direct_safety"):
            values = body[role][positions_in_group].astype(bool)
            if not np.all(values == values[0]):
                raise RuntimeError(f"physical repeats disagree on {role}: {key}")
        first.append(positions_in_group[0])
    first = np.asarray(first, dtype=np.int64)
    observed_physical = {
        "queries": len(first),
        "official_correct": int(np.sum(body["baseline_correct"][first].astype(bool))),
        "official_wrong": int(np.sum(~body["baseline_correct"][first].astype(bool))),
        "corrective": int(np.sum(body["direct_corrective"][first].astype(bool))),
        "known_safety": int(np.sum(body["direct_safety"][first].astype(bool))),
    }
    expected_physical = {
        "queries": 753, "official_correct": 485, "official_wrong": 268,
        "corrective": 59, "known_safety": 14,
    }
    if observed_physical != expected_physical:
        raise RuntimeError(
            f"B33 physical training universe changed: {observed_physical} != {expected_physical}"
        )
    print(
        "[validate_bioaware_b33_full_candidate_graph] PASS "
        f"queries=860 candidates={len(body['candidate_id']):,} "
        f"reference-links={len(positions):,} spectra={len(body['candidate_reference_tensor']):,} "
        "physical=753 broad-safety-queries=485 corrective=59 known-safety=14",
        flush=True,
    )


if __name__ == "__main__":
    main()
