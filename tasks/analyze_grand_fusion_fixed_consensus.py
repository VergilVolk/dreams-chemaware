#!/usr/bin/env python
"""Post-hoc fixed-consensus screen on the already-opened GNPS score bundle.

This is a development diagnostic, never an external confirmation.  It tests
fully label-blind, zero-parameter rank aggregations and writes every attempted
arm so the best row cannot be reported without the multiplicity context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel


ROOT = Path(__file__).resolve().parents[1]
PANELS = ("identity_disjoint", "formula_disjoint")
METHOD_SETS = {
    "v1_wse": ("noise_v1", "weighted_spectral_entropy"),
    "v1_p2b": ("noise_v1", "p2b_noise_v1_frozen"),
    "wse_p2b": ("weighted_spectral_entropy", "p2b_noise_v1_frozen"),
    "v1_wse_p2b": ("noise_v1", "weighted_spectral_entropy", "p2b_noise_v1_frozen"),
    "all_spectral": (
        "noise_v1", "cosine_greedy", "modified_cosine", "weighted_spectral_entropy",
        "p2b_sqrt_cosine", "p2b_unweighted_entropy", "neutral_loss_sqrt_cosine",
        "p2b_noise_v1_frozen",
    ),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--identity-panel", type=Path, required=True)
    parser.add_argument("--formula-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def average_tie_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = np.mean(ranks[order[start:stop]])
        start = stop
    return ranks


def aggregate(blocks: list[np.ndarray], mode: str) -> np.ndarray:
    ranks = np.stack([average_tie_ranks(block) for block in blocks])
    if mode == "borda":
        # Normalize because queries differ in candidate count; highest is best.
        return -np.mean((ranks - 1.0) / max(ranks.shape[1] - 1, 1), axis=0)
    if mode == "rrf60":
        return np.sum(1.0 / (60.0 + ranks), axis=0)
    if mode == "minimax":
        return -np.max(ranks, axis=0)
    raise ValueError(mode)


def strict_correct(scores: np.ndarray, labels: np.ndarray) -> tuple[bool, bool]:
    winners = np.flatnonzero(scores == np.max(scores))
    tie = len(winners) != 1
    return bool(not tie and labels[int(winners[0])] == 1), tie


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.score_bundle, allow_pickle=False) as body:
        method_names = list(map(str, body["method_names"]))
        pair_scores = {panel: np.asarray(body[f"scores_{panel}"], dtype=np.float32) for panel in PANELS}
    required = set(sum((list(value) for value in METHOD_SETS.values()), [])) | {"official_dreams"}
    missing = sorted(required - set(method_names))
    if missing:
        raise RuntimeError(f"bundle misses methods: {missing}")
    panel_paths = {"identity_disjoint": args.identity_panel, "formula_disjoint": args.formula_panel}
    report = {
        "status": "GRAND_FUSION_FIXED_CONSENSUS_POSTHOC_COMPLETE",
        "method_sets": {name: list(methods) for name, methods in METHOD_SETS.items()},
        "attempted_aggregation_rules": ["borda", "rrf60", "minimax"],
        "panels": {},
    }
    for panel in PANELS:
        graph = graph_from_panel(panel_paths[panel])
        matrix = pair_scores[panel]
        molecule = {
            method: np.maximum.reduceat(
                matrix[method_names.index(method)], np.asarray(graph.molecule_ptr, dtype=np.int64)[:-1],
            )
            for method in required
        }
        arms: dict[str, dict] = {}
        for method in sorted(required):
            correct = ties = 0
            for left, right in zip(graph.query_ptr[:-1], graph.query_ptr[1:]):
                ok, tie = strict_correct(molecule[method][int(left):int(right)], graph.molecule_label[int(left):int(right)])
                correct += ok
                ties += tie
            arms[f"single::{method}"] = {
                "correct": correct, "recall_at_1": correct / graph.n_queries, "ties": ties,
            }
        for set_name, methods in METHOD_SETS.items():
            for mode in report["attempted_aggregation_rules"]:
                correct = ties = 0
                for left, right in zip(graph.query_ptr[:-1], graph.query_ptr[1:]):
                    left, right = int(left), int(right)
                    score = aggregate([molecule[method][left:right] for method in methods], mode)
                    ok, tie = strict_correct(score, graph.molecule_label[left:right])
                    correct += ok
                    ties += tie
                arms[f"{mode}::{set_name}"] = {
                    "correct": correct, "recall_at_1": correct / graph.n_queries, "ties": ties,
                }
        best_single = max(
            ((name, row) for name, row in arms.items() if name.startswith("single::")),
            key=lambda item: item[1]["recall_at_1"],
        )
        best_consensus = max(
            ((name, row) for name, row in arms.items() if not name.startswith("single::")),
            key=lambda item: item[1]["recall_at_1"],
        )
        report["panels"][panel] = {
            "queries": graph.n_queries,
            "arms": arms,
            "best_single": {"name": best_single[0], **best_single[1]},
            "best_consensus_posthoc": {
                "name": best_consensus[0], **best_consensus[1],
                "delta_vs_best_single_pp": 100.0 * (
                    best_consensus[1]["recall_at_1"] - best_single[1]["recall_at_1"]
                ),
            },
        }
        print(json.dumps({panel: report["panels"][panel]["best_consensus_posthoc"]}), flush=True)
    report["claim_limit"] = (
        "All arms were inspected on already-opened GNPS outcomes. This is a post-hoc development screen, "
        "not an external claim. Every attempted arm is retained; a selected rule requires new confirmation."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
