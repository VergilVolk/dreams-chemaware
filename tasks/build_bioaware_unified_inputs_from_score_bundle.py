#!/usr/bin/env python
"""Convert the formal GNPS article score bundle into unified-benchmark inputs.

The bundle ``method_scores.npz`` from run 2349091 stores per-reference-spectrum
scores aligned to each frozen panel's ``candidate_row`` ledger.  This converter
applies the deployment molecule-max aggregation per candidate molecule and
writes the canonical manifest plus one prediction table per method.  No score
is recomputed and no spectrum is reread; the frozen numbers pass through
unchanged.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-file", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--panel-label", required=True,
                        choices=("identity_disjoint", "formula_disjoint"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--expected-queries", type=int)
    parser.add_argument("--expected-candidate-rows", type=int)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite bundle conversion output: {output}")
    output.mkdir(parents=True)

    panel = np.load(args.panel_file.resolve(), allow_pickle=False)
    with np.load(args.scores.resolve(), allow_pickle=False) as body:
        method_names = [str(name) for name in body["method_names"]]
        scores = np.asarray(body[f"scores_{args.panel_label}"], dtype=np.float64)
    if scores.shape[0] != len(method_names):
        raise RuntimeError("method/score axis mismatch")
    if scores.shape[1] != len(panel["candidate_row"]):
        raise RuntimeError(
            "score rows do not align to the panel candidate ledger: "
            f"{scores.shape[1]} vs {len(panel['candidate_row'])}"
        )

    molecule_ptr = np.asarray(panel["molecule_ptr"], dtype=np.int64)
    query_ptr = np.asarray(panel["query_ptr"], dtype=np.int64)
    n_queries = len(panel["query_row"])
    query_slot = np.repeat(np.arange(n_queries, dtype=np.int64), np.diff(query_ptr))
    molecule_slots = np.arange(len(panel["molecule_ik14"]))
    nonempty = np.diff(molecule_ptr) > 0
    slots = molecule_slots[nonempty]

    # Molecule-max aggregation across each candidate's reference rows,
    # reduced along the pair axis for every method simultaneously.
    molecule_scores = np.maximum.reduceat(scores, molecule_ptr[:-1], axis=1)
    if molecule_scores.shape != (len(method_names), len(panel["molecule_ik14"])):
        raise RuntimeError("reduceat changed the molecule axis length")

    manifest = pd.DataFrame({
        "query_id": query_slot[slots].astype(str),
        "candidate_id": np.asarray(panel["molecule_ik14"], dtype=str)[slots],
        "is_truth": np.asarray(panel["molecule_label"], dtype=bool)[slots].astype(int),
        "formula_cluster": np.asarray(panel["query_formula"], dtype=str)[query_slot[slots]],
        "source": args.source_label,
        "polarity": "positive",
        "near_query": np.asarray(panel["near_query"], dtype=bool)[query_slot[slots]],
    })
    if manifest.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("panel exposes duplicate query/identity candidate slots")

    manifest.to_csv(output / "candidate_manifest.csv.gz", index=False, compression="gzip")
    for index, method in enumerate(method_names):
        frame = pd.DataFrame({
            "query_id": manifest["query_id"],
            "candidate_id": manifest["candidate_id"],
            "score": molecule_scores[index, slots],
        })
        frame.to_csv(
            output / f"predictions_{method}.csv.gz", index=False, compression="gzip"
        )

    n_queries_observed = int(manifest["query_id"].nunique())
    n_rows = int(len(manifest))
    if args.expected_queries is not None and n_queries_observed != args.expected_queries:
        raise RuntimeError(
            f"query mismatch: expected {args.expected_queries}, observed {n_queries_observed}"
        )
    if args.expected_candidate_rows is not None and n_rows != args.expected_candidate_rows:
        raise RuntimeError(
            f"candidate-row mismatch: expected {args.expected_candidate_rows}, observed {n_rows}"
        )
    payload = {
        "status": "bioaware_unified_inputs_from_formal_bundle",
        "scores": str(args.scores.resolve()),
        "panel": str(args.panel_file.resolve()),
        "panel_label": args.panel_label,
        "methods": method_names,
        "queries": n_queries_observed,
        "candidate_rows": n_rows,
        "aggregation": "molecule_max",
        "claim_limit": (
            "Scores are the frozen run-2349091 numbers passed through unchanged; "
            "this conversion adds no method and opens no truth."
        ),
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    temporary.replace(output / "report.json")
    summary = {k: payload[k] for k in ("status", "panel_label", "queries", "candidate_rows")}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
