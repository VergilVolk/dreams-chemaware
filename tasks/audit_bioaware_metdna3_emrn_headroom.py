#!/usr/bin/env python
"""Step-stratified one-hop headroom audit for the public MetDNA2 eMRN."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, default=Path(
        "data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz"))
    parser.add_argument("--scores", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--splits", type=Path, default=Path(
        "data/validation/bioaware_metdna3_development_v1/identity_splits.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/validation/bioaware_metdna3_emrn_headroom_v1.json"))
    args = parser.parse_args()
    for path in (args.edges, args.scores, args.splits):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise RuntimeError(f"fail-closed: {args.output}")
    edges = pd.read_csv(args.edges, usecols=["ik14_a", "ik14_b", "minimum_step"])
    scores = pd.read_csv(args.scores)
    level1 = set(pd.read_csv(args.splits)["ik14"].astype(str))

    query_rows: list[dict[str, object]] = []
    for query_id, group in scores.groupby("query_id", sort=False):
        truth = str(group["truth_candidate_id"].iloc[0])
        maximum = float(group["spectral_score"].max())
        tied = group[np.isclose(group["spectral_score"], maximum, rtol=0, atol=1e-12)]
        top = str(tied.sort_values("candidate_id").iloc[0].candidate_id)
        query_rows.append({
            "query_id": str(query_id), "truth": truth, "top": top,
            "correct": bool(len(tied) == 1 and top == truth),
        })
    queries = pd.DataFrame(query_rows)
    steps: list[dict[str, object]] = []
    prior_error_truth = None
    for step in range(9):
        selected = edges[edges["minimum_step"] <= step]
        adjacency: dict[str, set[str]] = {}
        for edge in selected.itertuples(index=False):
            adjacency.setdefault(str(edge.ik14_a), set()).add(str(edge.ik14_b))
            adjacency.setdefault(str(edge.ik14_b), set()).add(str(edge.ik14_a))
        truth_supported = queries["truth"].map(lambda value: bool(adjacency.get(value, set()) & level1))
        wrong_supported = queries["top"].map(lambda value: bool(adjacency.get(value, set()) & level1))
        error = ~queries["correct"]
        error_truth = truth_supported & error
        error_wrong = wrong_supported & error
        record = {
            "step": step, "cumulative_edges": int(len(selected)),
            "all_truth_supported": int(truth_supported.sum()),
            "error_truth_supported": int(error_truth.sum()),
            "error_wrong_top_supported": int(error_wrong.sum()),
            "error_truth_only": int((error_truth & ~error_wrong).sum()),
            "error_both": int((error_truth & error_wrong).sum()),
            "error_wrong_only": int((~truth_supported & error_wrong & error).sum()),
            "error_neither": int((~truth_supported & ~wrong_supported & error).sum()),
            "new_error_truth_support_vs_previous_step": (
                None if prior_error_truth is None else int(error_truth.sum()) - prior_error_truth
            ),
        }
        prior_error_truth = int(error_truth.sum())
        steps.append(record)
    report = {
        "status": "bioaware_metdna3_emrn_one_hop_headroom_complete", "formal": True,
        "steps": steps,
        "gates": {
            "predicted_steps_add_error_truth_coverage": any(
                row["new_error_truth_support_vs_previous_step"] not in (None, 0) for row in steps[1:]),
            "pass_to_one_hop_scoring": False,
        },
        "decision": (
            "Predicted eMRN steps do not connect additional held-out truths directly to the "
            "available Level-1 seed pool. One-hop seed propagation cannot exploit this topology; "
            "MS1 feature pre-mapping and recursive propagation are required."
        ),
        "contracts": {
            "outcome_used_only_for_coverage_audit": True, "threshold_selected": False,
            "P2b": "forbidden", "RP_opened": False,
        },
        "provenance": {
            "edges_sha256": sha256(args.edges), "scores_sha256": sha256(args.scores),
            "splits_sha256": sha256(args.splits),
        },
        "claim_limit": "Consumed development headroom audit; no recursive annotation gain is claimed.",
    }
    report["gates"]["pass_to_one_hop_scoring"] = bool(
        report["gates"]["predicted_steps_add_error_truth_coverage"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
