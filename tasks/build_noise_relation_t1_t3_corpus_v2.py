"""Build the v2 T1/T3 corpus: clean relations plus exact action events.

v1 attached every representable Stage-1 action as an extra anchor inside its
query's shared loss.  Two things were lost:

1. Each Stage-1 action event carries an exact hard-positive bridge
   (action spectrum -> one measured positive row, one mined hard-negative
   row); v1 regenerated a generic candidate pool from the clean-query
   geometry and let those exact rows re-enter only by luck.
2. All of a query's actions averaged into one anchor group, so the action
   signal collapsed to ~2.8% of the training dose and actions with opposing
   gradients cancelled inside a single step.

v2 keeps the clean T1/T3 relation corpus unchanged (without action anchors)
and writes every exact Stage-1 action bridge into its own event table for
one-action-per-step rotation training.

Author: GLM-5.3 via DeepSeek Harness, taking over the GPT-authored pipeline.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from build_noise_relation_t1_t3_corpus import load_npz
from noise_corrected_fullgraph_evaluation import score_embeddings
from noise_final_core import CandidateGraph, sha256_file, stable_fold

ACTION_EVENT_KIND = 2
REGISTRY_HDF5 = 0
REGISTRY_ACTION = 1


def extract_action_events(
    pool: dict[str, np.ndarray],
    query_formula: np.ndarray,
    folds: np.ndarray,
    representable: np.ndarray,
    outer_fold: int,
    validation_fold: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Recover every exact action bridge from the Stage-1 native pool.

    Returns the event table plus audit counters.  Events are labelled by the
    formula fold of their query and split into training folds (neither outer
    nor validation) and the diagnostic validation fold; a fold-0 event would
    mean Stage-1 trained on the outer held fold and aborts the build.
    """
    kinds = np.asarray(pool["event_kind"], dtype=np.int8)
    queries = np.asarray(pool["event_query"], dtype=np.int64)
    actions = np.asarray(pool["event_action_index"], dtype=np.int64)
    anchors = np.asarray(pool["anchor_idx"], dtype=np.int64)
    registry_kind = np.asarray(pool["registry_kind"], dtype=np.int8)
    registry_source = np.asarray(pool["registry_source_index"], dtype=np.int64)
    positive_ptr = np.asarray(pool["positive_ptr"], dtype=np.int64)
    positive_idx = np.asarray(pool["positive_idx"], dtype=np.int64)
    negative_ptr = np.asarray(pool["negative_ptr"], dtype=np.int64)
    negative_idx = np.asarray(pool["negative_idx"], dtype=np.int64)

    kept_event: list[int] = []
    kept_query: list[int] = []
    kept_action: list[int] = []
    kept_positive: list[int] = []
    kept_negative: list[int] = []
    kept_fold: list[int] = []
    audit = {
        "action_events_in_pool": 0,
        "rejected_bad_registry": 0,
        "rejected_ragged_bridge": 0,
        "rejected_unrepresentable": 0,
        "outer_fold_events": 0,
        "validation_fold_events": 0,
        "training_fold_events": 0,
    }
    for event in np.flatnonzero(kinds == ACTION_EVENT_KIND):
        audit["action_events_in_pool"] += 1
        anchor = int(anchors[int(event)])
        if int(registry_kind[anchor]) != REGISTRY_ACTION:
            audit["rejected_bad_registry"] += 1
            continue
        action = int(registry_source[anchor])
        if action != int(actions[int(event)]):
            audit["rejected_bad_registry"] += 1
            continue
        p0, p1 = int(positive_ptr[int(event)]), int(positive_ptr[int(event) + 1])
        n0, n1 = int(negative_ptr[int(event)]), int(negative_ptr[int(event) + 1])
        positives = positive_idx[p0:p1]
        negatives = negative_idx[n0:n1]
        if len(positives) != 1 or len(negatives) != 1:
            audit["rejected_ragged_bridge"] += 1
            continue
        positive = positives[0]
        negative = negatives[0]
        if (
            int(registry_kind[positive]) != REGISTRY_HDF5
            or int(registry_kind[negative]) != REGISTRY_HDF5
        ):
            audit["rejected_bad_registry"] += 1
            continue
        if not bool(representable[int(action)]):
            audit["rejected_unrepresentable"] += 1
            continue
        query = int(queries[int(event)])
        fold = int(folds[query])
        if fold == outer_fold:
            audit["outer_fold_events"] += 1
            continue
        kept_event.append(int(event))
        kept_query.append(query)
        kept_action.append(action)
        kept_positive.append(int(registry_source[positive]))
        kept_negative.append(int(registry_source[negative]))
        kept_fold.append(fold)
        if fold == validation_fold:
            audit["validation_fold_events"] += 1
        else:
            audit["training_fold_events"] += 1
    if audit["outer_fold_events"]:
        raise RuntimeError("Stage-1 pool leaked an outer-held-fold action event")

    arrays = {
        "pool_event": np.asarray(kept_event, dtype=np.int64),
        "query": np.asarray(kept_query, dtype=np.int64),
        "action_index": np.asarray(kept_action, dtype=np.int64),
        "positive_row": np.asarray(kept_positive, dtype=np.int64),
        "negative_row": np.asarray(kept_negative, dtype=np.int64),
        "fold": np.asarray(kept_fold, dtype=np.int8),
        "query_formula": np.asarray(
            [str(query_formula[q]) for q in kept_query], dtype="U64",
        ),
    }
    return arrays, audit


def audit_against_clean_corpus(
    events: dict[str, np.ndarray], clean: dict[str, np.ndarray],
) -> dict[str, float | int]:
    """Measure how often v1's generic pool contained each exact bridge row.

    This quantifies the supervision v1 kept only by luck: the fraction of
    exact positive rows that fell inside the clean corpus's selected
    positive references, and exact negative rows inside its selected
    negative-molecule references.
    """
    query_to_block: dict[int, tuple[int, int]] = {
        int(query): position for position, query in enumerate(clean["query_index"])
    }
    positive_hits = 0
    negative_hits = 0
    both_hits = 0
    matched = 0
    for position in range(len(events["query"])):
        query = int(events["query"][position])
        block = query_to_block.get(query)
        if block is None:
            continue
        left = int(clean["molecule_ptr"][block])
        right = int(clean["molecule_ptr"][block + 1])
        m0 = int(clean["reference_ptr"][left])
        m1 = int(clean["reference_ptr"][left + 1])
        positive_rows = set(map(int, clean["reference_row"][m0:m1]))
        negative_rows = set(map(int, clean["reference_row"][
            int(clean["reference_ptr"][left + 1]):int(clean["reference_ptr"][right])
        ]))
        exact_positive = int(events["positive_row"][position])
        exact_negative = int(events["negative_row"][position])
        p_hit = exact_positive in positive_rows
        n_hit = exact_negative in negative_rows
        positive_hits += int(p_hit)
        negative_hits += int(n_hit)
        both_hits += int(p_hit and n_hit)
        matched += 1
    return {
        "events_with_clean_counterpart": matched,
        "exact_positive_in_clean_pool_fraction": (
            positive_hits / matched if matched else None
        ),
        "exact_negative_in_clean_pool_fraction": (
            negative_hits / matched if matched else None
        ),
        "exact_bridge_fully_in_clean_pool_fraction": (
            both_hits / matched if matched else None
        ),
    }


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
    if args.outer_fold == args.validation_fold or not (
        0 <= args.outer_fold < 5 and 0 <= args.validation_fold < 5
    ):
        raise ValueError("outer and validation folds must be distinct members of five folds")

    from build_noise_relation_t1_t3_corpus import build_split

    graph = CandidateGraph(args.graph)
    with np.load(args.stage1_embeddings, allow_pickle=False) as body:
        rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    scores = score_embeddings(graph, rows, embeddings)
    bank = load_npz(args.stage1_triplets / "action_spectra.npz")
    representable = np.asarray(bank["native_action_view_representable"], dtype=bool)

    folds = np.asarray([
        stable_fold(formula, 5, args.fold_seed) for formula in graph.query_formula
    ], dtype=np.int8)
    train_queries = np.flatnonzero(
        (folds != args.outer_fold) & (folds != args.validation_fold)
    )
    validation_queries = np.flatnonzero(folds == args.validation_fold)
    # v2 clean relations carry NO action anchors: actions train in their own
    # rotation stream against their exact mined bridges.
    clean_action_by_query: dict[int, list[int]] = {}
    train, train_report = build_split(
        graph, scores.pair, scores.molecule, train_queries, clean_action_by_query,
        max_positive_refs=args.max_positive_refs,
        max_negative_molecules=args.max_negative_molecules,
        additional_boundary_molecules=args.additional_boundary_molecules,
        max_negative_refs=args.max_negative_refs,
    )
    validation, validation_report = build_split(
        graph, scores.pair, scores.molecule, validation_queries, clean_action_by_query,
        max_positive_refs=args.max_positive_refs,
        max_negative_molecules=args.max_negative_molecules,
        additional_boundary_molecules=args.additional_boundary_molecules,
        max_negative_refs=args.max_negative_refs,
    )
    if not len(train["query_index"]) or not len(validation["query_index"]):
        raise RuntimeError("T1/T3 v2 corpus has an empty train or validation split")
    if set(train["query_formula"]) & set(validation["query_formula"]):
        raise RuntimeError("T1/T3 v2 train and validation formulas overlap")
    held_formulas = set(graph.query_formula[folds == args.outer_fold].astype(str))
    if held_formulas & (
        set(train["query_formula"]) | set(validation["query_formula"])
    ):
        raise RuntimeError("outer-held formula entered the T1/T3 v2 corpus")

    pool = load_npz(args.stage1_triplets / "train_pool.npz")
    events, event_audit = extract_action_events(
        pool, graph.query_formula, folds, representable,
        args.outer_fold, args.validation_fold,
    )
    train_mask = events["fold"] != args.validation_fold
    training_events = {key: value[train_mask] for key, value in events.items()}
    validation_events = {key: value[~train_mask] for key, value in events.items()}
    clean_audit = audit_against_clean_corpus(training_events, train)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        np.savez(staging / "train.npz", **train)
        np.savez(staging / "validation.npz", **validation)
        np.savez(
            staging / "action_events.npz",
            **{f"train_{key}": value for key, value in training_events.items()},
            **{f"validation_{key}": value for key, value in validation_events.items()},
        )
        report = {
            "status": "noise_relation_t1_t3_corpus_v2_complete",
            "design": (
                "clean T1/T3 relations without action anchors; every exact "
                "Stage-1 hard-positive bridge trains in a dedicated "
                "one-action-per-step rotation stream"
            ),
            "split": {
                "folds": 5, "seed": args.fold_seed,
                "outer_held_fold": args.outer_fold,
                "validation_fold": args.validation_fold,
                "training_folds": sorted(set(range(5)) - {args.outer_fold, args.validation_fold}),
            },
            "caps": {
                "positive_references": args.max_positive_refs,
                "negative_molecules": args.max_negative_molecules,
                "additional_boundary_molecules": args.additional_boundary_molecules,
                "references_per_negative_molecule": args.max_negative_refs,
            },
            "train": train_report,
            "validation": validation_report,
            "action_events": event_audit,
            "v1_supervision_loss_audit": {
                "note": (
                    "fractions of exact bridges that v1's generic pool kept "
                    "purely by chance; 1 - these values is the supervision v1 "
                    "silently discarded"
                ),
                **clean_audit,
            },
            "provenance": {
                "graph_sha256": sha256_file(args.graph),
                "stage1_embeddings_sha256": sha256_file(args.stage1_embeddings),
                "stage1_train_pool_sha256": sha256_file(args.stage1_triplets / "train_pool.npz"),
                "stage1_action_bank_sha256": sha256_file(args.stage1_triplets / "action_spectra.npz"),
            },
            "claim_limit": (
                "Outer held fold is absent; corpus construction makes no "
                "performance claim.  Fold-1 material is diagnostic only."
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
