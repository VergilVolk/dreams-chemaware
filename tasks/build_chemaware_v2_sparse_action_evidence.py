#!/usr/bin/env python
"""Convert frozen ChemAware V2 actions to candidate-keyed sparse evidence.

An acted query exposes one categorical observation: the selected candidate is
supported relative to every other candidate.  Abstained queries expose no
chemical observation.  No truth array is read by this adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--action-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    # Deliberately load only candidate-axis arrays, never molecule_label.
    with np.load(args.panel, allow_pickle=False) as panel:
        query_ptr = np.asarray(panel["query_ptr"], dtype=np.int64)
        query_ids = np.asarray(panel["query_ik14"]).astype("U")
        candidate_ids = np.asarray(panel["molecule_ik14"]).astype("U")
    with np.load(args.action_ledger, allow_pickle=False) as ledger:
        query_index = np.asarray(ledger["query_index"], dtype=np.int64)
        abstained = np.asarray(ledger["abstained"], dtype=bool)
        selected = np.asarray(ledger["selected_candidate"], dtype=np.int64)

    queries = len(query_ptr) - 1
    if not (
        np.array_equal(query_index, np.arange(queries))
        and len(abstained) == queries
        and len(selected) == queries
        and int(query_ptr[-1]) == len(candidate_ids)
    ):
        raise RuntimeError("ChemAware action ledger does not align to panel candidate axis")

    scores = np.zeros(len(candidate_ids), dtype=np.float32)
    availability = np.zeros(len(candidate_ids), dtype=np.float32)
    acted = 0
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left, right = int(left), int(right)
        if abstained[query]:
            if selected[query] != -1:
                raise RuntimeError(f"abstained query {query} has a selected candidate")
            continue
        choice = int(selected[query])
        if not 0 <= choice < right - left:
            raise RuntimeError(f"ChemAware selected candidate is out of range at query {query}")
        # Availability covers the complete candidate comparison whenever the
        # selector acts: selected=1 is support and all non-selected=0 are the
        # explicitly contrasted alternatives.  Abstention remains unavailable.
        availability[left:right] = 1.0
        scores[left + choice] = 1.0
        acted += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        scores=scores,
        availability=availability,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
    )
    report = {
        "status": "CHEMAWARE_V2_SPARSE_ACTION_EVIDENCE_COMPLETE",
        "labels_opened": False,
        "queries": queries,
        "acted_queries": acted,
        "abstained_queries": int(abstained.sum()),
        "candidate_rows": int(len(candidate_ids)),
        "available_candidate_rows": int(availability.sum()),
        "semantics": (
            "acted query: selected candidate=1 and contrasted alternatives=0; "
            "abstained query: availability=0 for every candidate"
        ),
        "panel_sha256": sha256(args.panel),
        "action_ledger_sha256": sha256(args.action_ledger),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
