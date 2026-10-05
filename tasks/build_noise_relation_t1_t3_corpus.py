#!/usr/bin/env python
"""Build query-level T1/T3 relations from the corrected candidate graph.

No synthetic negative is created.  Each query keeps diverse measured spectra
of its true molecule, the live Stage-1 hardest same-formula rivals, and the
highest-scoring remaining boundary molecules.
Every representable Stage-1 targeted action belonging to an eligible query is
attached as another anchor inside that query, not as another training dose.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from noise_corrected_fullgraph_evaluation import score_embeddings
from noise_final_core import CandidateGraph, sha256_file, stable_fold


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        return {name: np.asarray(body[name]) for name in body.files}


def diverse_rows(rows: np.ndarray, scores: np.ndarray, maximum: int) -> np.ndarray:
    """Select hard-to-easy measured references rather than hardest-only."""
    order = np.argsort(scores, kind="stable")
    if len(order) <= maximum:
        return rows[order]
    positions = np.unique(np.rint(np.linspace(0, len(order) - 1, maximum)).astype(int))
    return rows[order[positions]]


def build_split(
    graph: CandidateGraph,
    pair_scores: np.ndarray,
    molecule_scores: np.ndarray,
    queries: np.ndarray,
    action_by_query: dict[int, list[int]],
    *,
    max_positive_refs: int,
    max_negative_molecules: int,
    additional_boundary_molecules: int,
    max_negative_refs: int,
) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    query_index: list[int] = []
    query_row: list[int] = []
    query_formula: list[str] = []
    action_ptr = [0]
    action_index: list[int] = []
    molecule_ptr = [0]
    molecule_index: list[int] = []
    reference_ptr = [0]
    reference_row: list[int] = []
    negative_grade: list[int] = []
    skipped_no_positive = 0
    queries_with_same_formula = 0
    complete_candidate_queries = 0

    for query_value in np.asarray(queries, dtype=np.int64):
        query = int(query_value)
        left, right = map(int, graph.query_ptr[query:query + 2])
        positive_molecule = left
        pleft, pright = map(int, graph.molecule_ptr[positive_molecule:positive_molecule + 2])
        positive_rows = graph.pair_candidate_row[pleft:pright]
        positive_pair_scores = pair_scores[pleft:pright]
        keep = positive_rows != int(graph.query_row[query])
        positive_rows = positive_rows[keep]
        positive_pair_scores = positive_pair_scores[keep]
        if not len(positive_rows):
            skipped_no_positive += 1
            continue
        candidate = np.arange(left + 1, right, dtype=np.int64)
        same_formula = candidate[
            graph.molecule_formula[candidate].astype(str) == str(graph.query_formula[query])
        ]
        if len(same_formula):
            queries_with_same_formula += 1
        same_order = np.argsort(-molecule_scores[same_formula], kind="stable")
        selected_same = list(map(
            int, same_formula[same_order[:max_negative_molecules]],
        ))
        selected_set = set(selected_same)
        remaining = np.asarray([
            int(value) for value in candidate if int(value) not in selected_set
        ], dtype=np.int64)
        remaining_order = np.argsort(-molecule_scores[remaining], kind="stable")
        selected_other = list(map(
            int, remaining[remaining_order[:additional_boundary_molecules]],
        ))
        negatives = np.asarray(selected_same + selected_other, dtype=np.int64)
        if not len(negatives):
            raise RuntimeError(f"query {query} unexpectedly has no negative molecule")
        complete_candidate_queries += int(len(negatives) == len(candidate))

        query_index.append(query)
        query_row.append(int(graph.query_row[query]))
        query_formula.append(str(graph.query_formula[query]))
        actions = sorted(set(action_by_query.get(query, ())))
        action_index.extend(actions)
        action_ptr.append(len(action_index))

        molecule_index.append(positive_molecule)
        selected_positive = diverse_rows(
            positive_rows, positive_pair_scores, max_positive_refs,
        )
        reference_row.extend(map(int, selected_positive))
        reference_ptr.append(len(reference_row))
        negative_grade.append(-1)
        for molecule in negatives:
            mleft, mright = map(int, graph.molecule_ptr[int(molecule):int(molecule) + 2])
            rows = graph.pair_candidate_row[mleft:mright]
            scores = pair_scores[mleft:mright]
            hard = np.argsort(-scores, kind="stable")[:max_negative_refs]
            molecule_index.append(int(molecule))
            reference_row.extend(map(int, rows[hard]))
            reference_ptr.append(len(reference_row))
            negative_grade.append(int(graph.molecule_mces_grade[int(molecule)]))
        molecule_ptr.append(len(molecule_index))

    arrays = {
        "query_index": np.asarray(query_index, dtype=np.int64),
        "query_row": np.asarray(query_row, dtype=np.int64),
        "query_formula": np.asarray(query_formula, dtype="U64"),
        "action_ptr": np.asarray(action_ptr, dtype=np.int64),
        "action_index": np.asarray(action_index, dtype=np.int64),
        "molecule_ptr": np.asarray(molecule_ptr, dtype=np.int64),
        "molecule_index": np.asarray(molecule_index, dtype=np.int64),
        "reference_ptr": np.asarray(reference_ptr, dtype=np.int64),
        "reference_row": np.asarray(reference_row, dtype=np.int64),
        "molecule_mces_grade": np.asarray(negative_grade, dtype=np.int8),
    }
    negative_counts = np.diff(arrays["molecule_ptr"]) - 1
    positive_counts = np.asarray([
        arrays["reference_ptr"][arrays["molecule_ptr"][i] + 1]
        - arrays["reference_ptr"][arrays["molecule_ptr"][i]]
        for i in range(len(arrays["query_index"]))
    ], dtype=np.int64)
    report = {
        "input_queries": int(len(queries)),
        "eligible_queries": int(len(query_index)),
        "skipped_no_distinct_positive": skipped_no_positive,
        "queries_with_same_formula_rival": queries_with_same_formula,
        "same_formula_relation_fraction": (
            float(queries_with_same_formula / len(query_index)) if query_index else 0.0
        ),
        "complete_candidate_queries": complete_candidate_queries,
        "complete_candidate_fraction": (
            float(complete_candidate_queries / len(query_index)) if query_index else 0.0
        ),
        "actions_attached": int(len(action_index)),
        "queries_with_actions": int(np.sum(np.diff(arrays["action_ptr"]) > 0)),
        "candidate_molecules": int(len(molecule_index)),
        "measured_reference_rows": int(len(reference_row)),
        "positive_references_median": float(np.median(positive_counts)) if len(positive_counts) else 0.0,
        "negative_molecules_median": float(np.median(negative_counts)) if len(negative_counts) else 0.0,
    }
    return arrays, report


def cap_action_anchors(
    action_by_query: dict[int, list[int]], maximum: int,
) -> tuple[dict[int, list[int]], int]:
    """Deduplicate every query's actions and even-spaced cap outliers.

    An uncapped query could exceed any per-step spectrum budget and abort
    training outright.  The even-spaced deterministic subsample keeps the
    packing budget unreachable and preserves the clean-0.5/actions-0.5
    anchor-share semantics.
    """
    if maximum < 1:
        raise ValueError("action anchor cap must be positive")
    capped: dict[int, list[int]] = {}
    capped_queries = 0
    for query, actions in action_by_query.items():
        unique = sorted(set(actions))
        if len(unique) > maximum:
            positions = np.unique(np.rint(np.linspace(
                0, len(unique) - 1, maximum,
            )).astype(int))
            capped[query] = [unique[int(i)] for i in positions]
            capped_queries += 1
        else:
            capped[query] = unique
    return capped, capped_queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--stage1-embeddings", type=Path, required=True)
    parser.add_argument("--stage1-triplets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--validation-fold", type=int, default=1)
    parser.add_argument("--fold-seed", type=int, default=20260825)
    parser.add_argument("--max-positive-refs", type=int, default=4)
    parser.add_argument("--max-negative-molecules", type=int, default=4)
    parser.add_argument("--additional-boundary-molecules", type=int, default=2)
    parser.add_argument("--max-negative-refs", type=int, default=3)
    parser.add_argument(
        "--max-actions-per-query", type=int, default=40,
        help=(
            "deterministic even-spaced cap on action anchors; 40 keeps every "
            "query inside a 64-spectrum step budget (1 anchor + 40 actions "
            "+ at most 22 references)"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (
        args.graph, args.stage1_embeddings,
        args.stage1_triplets / "train_pool.npz",
        args.stage1_triplets / "action_spectra.npz",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.outer_fold == args.validation_fold or not (0 <= args.outer_fold < 5 and 0 <= args.validation_fold < 5):
        raise ValueError("outer and validation folds must be distinct members of five folds")
    graph = CandidateGraph(args.graph)
    with np.load(args.stage1_embeddings, allow_pickle=False) as body:
        rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    scores = score_embeddings(graph, rows, embeddings)
    pool = load_npz(args.stage1_triplets / "train_pool.npz")
    bank = load_npz(args.stage1_triplets / "action_spectra.npz")
    event_query = np.asarray(pool["event_query"], dtype=np.int64)
    event_action = np.asarray(pool["event_action_index"], dtype=np.int64)
    representable = np.asarray(bank["native_action_view_representable"], dtype=bool)
    action_by_query: dict[int, list[int]] = defaultdict(list)
    for query, action in zip(event_query, event_action, strict=True):
        if action >= 0 and action < len(representable) and representable[action]:
            action_by_query[int(query)].append(int(action))
    # An uncapped query could exceed the per-step spectrum budget; cap the
    # anchors deterministically before any split consumes them.
    action_by_query, queries_with_capped_actions = cap_action_anchors(
        action_by_query, args.max_actions_per_query,
    )

    folds = np.asarray([
        stable_fold(formula, 5, args.fold_seed) for formula in graph.query_formula
    ], dtype=np.int8)
    train_queries = np.flatnonzero(
        (folds != args.outer_fold) & (folds != args.validation_fold)
    )
    validation_queries = np.flatnonzero(folds == args.validation_fold)
    train, train_report = build_split(
        graph, scores.pair, scores.molecule, train_queries, action_by_query,
        max_positive_refs=args.max_positive_refs,
        max_negative_molecules=args.max_negative_molecules,
        additional_boundary_molecules=args.additional_boundary_molecules,
        max_negative_refs=args.max_negative_refs,
    )
    validation, validation_report = build_split(
        graph, scores.pair, scores.molecule, validation_queries, {},
        max_positive_refs=args.max_positive_refs,
        max_negative_molecules=args.max_negative_molecules,
        additional_boundary_molecules=args.additional_boundary_molecules,
        max_negative_refs=args.max_negative_refs,
    )
    if not len(train["query_index"]) or not len(validation["query_index"]):
        raise RuntimeError("T1/T3 corpus has an empty train or validation split")
    if set(train["query_formula"]) & set(validation["query_formula"]):
        raise RuntimeError("T1/T3 train and validation formulas overlap")
    held_formulas = set(graph.query_formula[folds == args.outer_fold].astype(str))
    if held_formulas & (set(train["query_formula"]) | set(validation["query_formula"])):
        raise RuntimeError("outer-held formula entered the T1/T3 corpus")
    used_actions = set(map(int, train["action_index"]))
    expected_actions = {
        action for query in set(map(int, train["query_index"]))
        for action in action_by_query.get(query, ())
    }
    if used_actions != expected_actions:
        raise RuntimeError("eligible Stage-1 actions were lost from T1/T3 query anchors")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        np.savez(staging / "train.npz", **train)
        np.savez(staging / "validation.npz", **validation)
        report = {
            "status": "noise_relation_t1_t3_corpus_complete",
            "algorithm": {
                "T1": "all selected positive-reference x negative-reference hinge relations",
                "T3": "live exact molecule-max selected-boundary listwise",
                "query_dose": "one objective per query; anchors and references average within query",
                "negative_source": (
                    "measured Stage-1 top same-formula rivals plus top remaining boundary molecules"
                ),
                "positive_sampling": "Stage-1 hard-to-easy measured reference coverage",
                "actions": "all representable Stage-1 targeted actions for each eligible query",
            },
            "split": {
                "folds": 5, "seed": args.fold_seed,
                "outer_held_fold": args.outer_fold,
                "validation_fold": args.validation_fold,
                "training_folds": sorted(set(range(5)) - {args.outer_fold, args.validation_fold}),
                "validation_split_role": (
                    "diagnostic only; the validation fold lies inside Stage-1's "
                    "outer-train distribution, is never used for model "
                    "selection, and no performance claim may cite it"
                ),
            },
            "actions": {
                "max_per_query": args.max_actions_per_query,
                "queries_with_capped_actions": queries_with_capped_actions,
                "selection": "deterministic even-spaced subsample of sorted unique action indices",
            },
            "caps": {
                "positive_references": args.max_positive_refs,
                "negative_molecules": args.max_negative_molecules,
                "additional_boundary_molecules": args.additional_boundary_molecules,
                "references_per_negative_molecule": args.max_negative_refs,
            },
            "train": train_report,
            "validation": validation_report,
            "provenance": {
                "graph_sha256": sha256_file(args.graph),
                "stage1_embeddings_sha256": sha256_file(args.stage1_embeddings),
                "stage1_train_pool_sha256": sha256_file(args.stage1_triplets / "train_pool.npz"),
                "stage1_action_bank_sha256": sha256_file(args.stage1_triplets / "action_spectra.npz"),
            },
            "claim_limit": (
                "Outer held fold is absent; corpus construction makes no "
                "performance claim.  Queries with more representable actions "
                "than the cap keep an even-spaced deterministic subset, "
                "preserving the clean-0.5/actions-0.5 anchor dosage."
            ),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
