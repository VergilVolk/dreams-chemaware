#!/usr/bin/env python
"""Score development-query recall of a candidate-graph embedding cache.

Replays the exact clean-query loop of the V1-current residual atlas: one
candidate block per non-outer query, molecule-max aggregation, strict ties
counting against the positive.  This is the train-side safety gate of the V5
minimal round; it never touches outer-held queries and makes no claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_noise_v1_current_residual_atlas import embedding_lookup, load_npz
from noise_corrected_fullgraph_evaluation import score_embeddings
from noise_final_core import CandidateGraph, sha256_file, stable_fold

EXPECTED_STATUS = "noise_corrected_graph_checkpoint_encoding_complete"


def recall_summary(
    ranks: list[int], near_mask: list[bool], margins: list[float],
) -> dict[str, object]:
    if not ranks:
        raise RuntimeError("development replay produced no queries")
    rank_array = np.asarray(ranks, dtype=np.int64)
    margin_array = np.asarray(margins, dtype=np.float64)
    near_array = np.asarray(near_mask, dtype=bool)
    result: dict[str, object] = {
        "queries": int(len(rank_array)),
        "recall_at_1": float(np.mean(rank_array == 1)),
        "errors": int(np.sum(rank_array > 1)),
        "near_queries": int(np.sum(near_array)),
        "mean_margin": float(np.mean(margin_array)),
    }
    if np.any(near_array):
        result["near_recall_at_1"] = float(np.mean(rank_array[near_array] == 1))
    for label, quantile in (("p10", 0.10), ("median", 0.50), ("p90", 0.90)):
        result[f"margin_{label}"] = float(np.quantile(margin_array, quantile))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--fold-seed", type=int, default=20260825)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fold_seed != 20260825 or args.outer_fold != 0:
        raise RuntimeError("the atlas fold contract is fold seed 20260825, outer 0")
    if args.output.exists():
        raise FileExistsError(args.output)
    report_path = args.embedding_cache.with_suffix(".json")
    cache_report = json.loads(report_path.read_text(encoding="utf-8"))
    if cache_report.get("status") != EXPECTED_STATUS:
        raise RuntimeError("the embedding cache report is not a completed encoding")
    if cache_report.get("graph_sha256") != sha256_file(args.graph):
        raise RuntimeError("the embedding cache names a different candidate graph")

    graph = CandidateGraph(args.graph)
    cache = load_npz(args.embedding_cache)
    rows, embeddings, _ = embedding_lookup(cache, "candidate")
    scores = score_embeddings(graph, rows, embeddings)
    folds = np.asarray([
        stable_fold(formula, 5, args.fold_seed) for formula in graph.query_formula
    ], dtype=np.int8)
    development_queries = np.flatnonzero(folds != args.outer_fold)

    ranks: list[int] = []
    near_mask: list[bool] = []
    margins: list[float] = []
    single_molecule_queries = 0
    for ordinal, query_value in enumerate(development_queries, start=1):
        query = int(query_value)
        molecule_left = int(graph.query_ptr[query])
        molecule_right = int(graph.query_ptr[query + 1])
        _, candidate_rows, local_ptr, _ = graph.query_block(query)
        local_scores = scores.molecule[molecule_left:molecule_right]
        rank = 1 + int(np.sum(local_scores[1:] >= local_scores[0]))
        ranks.append(rank)
        near_mask.append(bool(graph.query_has_near[query]))
        if len(local_scores) > 1:
            margins.append(float(local_scores[0] - np.max(local_scores[1:])))
        else:
            single_molecule_queries += 1
        if ordinal % 10_000 == 0 or ordinal == len(development_queries):
            print(
                f"[dev-recall] {ordinal:,}/{len(development_queries):,}", flush=True,
            )
    summary = recall_summary(ranks, near_mask, margins)
    summary["single_molecule_queries"] = single_molecule_queries
    report = {
        "status": "noise_dev_graph_recall_scored",
        "scientific_scope": (
            "train-side development safety replay; outer-held queries excluded; "
            "no performance claim"
        ),
        **summary,
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "embedding_cache_sha256": sha256_file(args.embedding_cache),
            "embedding_cache_report_sha256": sha256_file(report_path),
        },
        "claim_limit": (
            "Development queries are inside the training pool lineage; this "
            "number is a damage detector, not evidence of retrieval gain."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_name(f".{args.output.name}.tmp")
    staging.write_text(json.dumps(report, indent=2), encoding="utf-8")
    staging.replace(args.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
