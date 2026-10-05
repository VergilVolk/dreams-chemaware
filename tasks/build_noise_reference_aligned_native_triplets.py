"""Build reference-aligned Noise triplets for the unmodified DreaMS runtime.

The previous adapter stacked a clean base, every exact boundary, and several
expanded boundaries for the same action relation.  This builder instead keeps
two native identity events for each affected query:

* one official-hard base for every query that receives a Noise action;
* one dynamic action pool containing every action-positive row, every
  action-selected false-reference row, and bounded hard references from every
  selected false molecule.

The native dataset draws one positive and one negative from that pool on every
visit. Raw duplicate actions remain aliases and cannot increase query dose. No
action tensor, teacher target, or custom loss enters training.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import pandas as pd

from build_noise_dreams_native_triplets import (
    REGISTERED_SOURCES,
    _report_allows_training,
    audit_formal_inputs,
    embedding_positions,
    load_npz,
    query_molecules,
    select_actions,
    sha256_file,
    stable_fold,
)


VERSION = "noise_reference_aligned_native_v3"
BASE_OFFICIAL_HARD = 1
ACTION_EXACT = 2
ACTION_REFERENCE_EXPANDED = 4
SAFE_SENTINEL = 8


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def audit_identity_edges(
    pool: Mapping[str, np.ndarray], data: Path,
) -> dict[str, int | bool]:
    """Verify native positive/negative identity semantics without shared code."""
    with h5py.File(data, "r") as handle:
        ik14 = np.asarray([value[:14] for value in _decode(handle["INCHIKEY"][:])])
    positive_edges = negative_edges = 0
    for event, anchor in enumerate(pool["anchor_idx"]):
        p0, p1 = map(int, pool["positive_ptr"][event:event + 2])
        n0, n1 = map(int, pool["negative_ptr"][event:event + 2])
        positive = pool["positive_idx"][p0:p1]
        negative = pool["negative_idx"][n0:n1]
        if np.any(positive == anchor) or not np.all(ik14[positive] == ik14[int(anchor)]):
            raise RuntimeError("native DreaMS positive edge violates identity contract")
        if not np.all(ik14[negative] != ik14[int(anchor)]):
            raise RuntimeError("native DreaMS negative edge violates identity contract")
        positive_edges += len(positive)
        negative_edges += len(negative)
    return {
        "identity_contract_passed": True,
        "positive_edges_checked": int(positive_edges),
        "negative_edges_checked": int(negative_edges),
    }


def locate_row(
    graph: Mapping[str, np.ndarray], query: int, row: int,
) -> tuple[int, bool, np.ndarray]:
    matches = [
        body for body in query_molecules(graph, int(query))
        if int(row) in set(map(int, body[2]))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"query {query} candidate row {row} maps to {len(matches)} molecules"
        )
    molecule, label, rows = matches[0]
    return int(molecule), bool(label), np.unique(np.asarray(rows, dtype=np.int64))


def action_row_audit(
    actions: pd.DataFrame, graph: Mapping[str, np.ndarray],
) -> None:
    query_formula = np.asarray(graph["query_formula"]).astype(str)
    query_ik14 = np.asarray(graph["query_ik14"]).astype(str)
    for row in actions.itertuples(index=False):
        query = int(row.query_index)
        if (
            int(row.query_row) != int(graph["query_row"][query])
            or str(row.query_formula) != query_formula[query]
            or str(row.query_ik14) != query_ik14[query]
        ):
            raise RuntimeError("Noise action disagrees with the frozen candidate graph")
        _, positive_label, _ = locate_row(
            graph, query, int(row.action_positive_row),
        )
        _, negative_label, _ = locate_row(
            graph, query, int(row.action_hard_negative_row),
        )
        if not positive_label or negative_label:
            raise RuntimeError("Noise action boundary has invalid identity labels")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--validation-folds", type=int, default=10)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--validation-seed", type=int, default=20260920)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--expected-actions", type=int, default=32114)
    parser.add_argument("--base-queries-per-formula", type=int, default=1)
    parser.add_argument("--negative-references-per-action-candidate", type=int, default=2)
    parser.add_argument("--positive-references-per-negative", type=int, default=2)
    parser.add_argument("--minimum-train-formulas", type=int, default=800)
    parser.add_argument("--minimum-triplet-events", type=int, default=6500)
    parser.add_argument("--minimum-active-events", type=int, default=4000)
    parser.add_argument("--minimum-exact-action-events", type=int, default=1000)
    parser.add_argument("--minimum-active-action-fraction", type=float, default=0.50)
    return parser.parse_args()


def _embeddings(
    cache: Mapping[str, np.ndarray], positions: np.ndarray, rows: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= len(positions)) or np.any(positions[rows] < 0):
        raise RuntimeError("triplet row is absent from the frozen embedding cache")
    return np.asarray(cache["embeddings"][positions[rows]], dtype=np.float32)


def _query_geometry(
    graph: Mapping[str, np.ndarray], cache: Mapping[str, np.ndarray],
    positions: np.ndarray, query: int,
) -> dict[str, object]:
    anchor = int(graph["query_row"][query])
    anchor_z = _embeddings(cache, positions, np.asarray([anchor]))[0]
    positive_blocks = [rows for _, label, rows in query_molecules(graph, query) if label]
    if not positive_blocks:
        raise RuntimeError(f"query {query} has no positive molecule")
    positive = np.unique(np.concatenate(positive_blocks).astype(np.int64))
    positive = positive[positive != anchor]
    if not len(positive):
        raise RuntimeError(f"query {query} has no distinct positive reference")
    positive_scores = _embeddings(cache, positions, positive) @ anchor_z
    false: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for molecule, label, rows in query_molecules(graph, query):
        if label:
            continue
        rows = np.asarray(rows, dtype=np.int64)
        false[int(molecule)] = (rows, _embeddings(cache, positions, rows) @ anchor_z)
    if not false:
        raise RuntimeError(f"query {query} has no false candidate molecule")
    return {
        "anchor": anchor,
        "positive_rows": positive,
        "positive_scores": np.asarray(positive_scores, dtype=np.float32),
        "false": false,
    }


def _split_queries(
    graph: Mapping[str, np.ndarray], actions: pd.DataFrame, *, outer_fold: int,
    formula_fold_seed: int, validation_folds: int, validation_fold: int,
    validation_seed: int, base_queries_per_formula: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    formulas = np.asarray(graph["query_formula"]).astype(str)
    outer_train = np.asarray([
        query for query, formula in enumerate(formulas)
        if stable_fold(formula, 5, formula_fold_seed) != outer_fold
    ], dtype=np.int64)
    action_queries = set(map(int, actions["query_index"]))
    action_formulas = set(actions["query_formula"].astype(str))
    clean_formulas = set(formulas[outer_train]) - action_formulas
    validation_formulas = {
        formula for formula in clean_formulas
        if stable_fold(formula, validation_folds, validation_seed) == validation_fold
    }
    validation_queries = np.asarray([
        query for query in outer_train if formulas[query] in validation_formulas
    ], dtype=np.int64)
    eligible_train = np.asarray([
        query for query in outer_train if formulas[query] not in validation_formulas
    ], dtype=np.int64)
    by_formula: dict[str, list[int]] = defaultdict(list)
    for query in eligible_train:
        by_formula[str(formulas[int(query)])].append(int(query))
    selected = set(action_queries)
    for formula in sorted(by_formula):
        selected.update(sorted(by_formula[formula])[:base_queries_per_formula])
    train_queries = np.asarray(sorted(selected), dtype=np.int64)
    if action_queries - set(map(int, train_queries)):
        raise RuntimeError("an action query was omitted from the training base")
    train_formulas = set(formulas[train_queries])
    if train_formulas & validation_formulas:
        raise RuntimeError("train and validation formulas overlap")
    expected_train_formulas = set(formulas[eligible_train])
    if train_formulas != expected_train_formulas:
        raise RuntimeError("formula-diverse base failed to cover every eligible train formula")
    outer_excluded = all(
        stable_fold(str(formulas[int(query)]), 5, formula_fold_seed) != outer_fold
        for query in np.concatenate((train_queries, validation_queries))
    )
    return train_queries, validation_queries, {
        "outer_train_queries": int(len(outer_train)),
        "eligible_train_queries": int(len(eligible_train)),
        "selected_train_queries": int(len(train_queries)),
        "selected_train_formulas": int(len(train_formulas)),
        "eligible_train_formulas": int(len(expected_train_formulas)),
        "validation_queries": int(len(validation_queries)),
        "validation_formulas": int(len(validation_formulas)),
        "action_queries": int(len(action_queries)),
        "action_formulas": int(len(action_formulas)),
        "formula_coverage_fraction": float(len(train_formulas) / max(1, len(expected_train_formulas))),
        "train_validation_formula_disjoint": True,
        "outer_held_formulas_excluded": bool(outer_excluded),
    }


class PoolBuilder:
    def __init__(self) -> None:
        self.anchor: list[int] = []
        self.positive: list[int] = []
        self.negative: list[int] = []
        self.positive_ptr = [0]
        self.negative_ptr = [0]
        self.query: list[int] = []
        self.negative_molecule: list[int] = []
        self.positive_reference_row: list[int] = []
        self.negative_reference_row: list[int] = []
        self.tag: list[int] = []
        self.activation: list[float] = []
        self.hinge: list[float] = []
        self.keys: dict[tuple[object, ...], int] = {}

    def add(
        self, *, key: tuple[object, ...], anchor: int, positives: np.ndarray,
        negatives: np.ndarray, query: int, negative_molecule: int, tag: int,
        positive_reference_row: int, negative_reference_row: int,
        activation: float, hinge: float,
    ) -> int:
        if key in self.keys:
            index = self.keys[key]
            self.tag[index] |= int(tag)
            return index
        positives = np.unique(np.asarray(positives, dtype=np.int64))
        negatives = np.unique(np.asarray(negatives, dtype=np.int64))
        if not len(positives) or not len(negatives):
            raise RuntimeError("native triplet event has an empty role")
        if anchor in positives or anchor in negatives or np.intersect1d(positives, negatives).size:
            raise RuntimeError("native triplet event has overlapping semantic roles")
        index = len(self.anchor)
        self.keys[key] = index
        self.anchor.append(int(anchor))
        self.positive.extend(map(int, positives))
        self.negative.extend(map(int, negatives))
        self.positive_ptr.append(len(self.positive))
        self.negative_ptr.append(len(self.negative))
        self.query.append(int(query))
        self.negative_molecule.append(int(negative_molecule))
        self.positive_reference_row.append(int(positive_reference_row))
        self.negative_reference_row.append(int(negative_reference_row))
        self.tag.append(int(tag))
        self.activation.append(float(activation))
        self.hinge.append(float(hinge))
        return index

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "anchor_idx": np.asarray(self.anchor, dtype=np.int64),
            "positive_ptr": np.asarray(self.positive_ptr, dtype=np.int64),
            "positive_idx": np.asarray(self.positive, dtype=np.int64),
            "negative_ptr": np.asarray(self.negative_ptr, dtype=np.int64),
            "negative_idx": np.asarray(self.negative, dtype=np.int64),
            "source_query": np.asarray(self.query, dtype=np.int64),
            "negative_molecule": np.asarray(self.negative_molecule, dtype=np.int64),
            "positive_reference_row": np.asarray(self.positive_reference_row, dtype=np.int64),
            "negative_reference_row": np.asarray(self.negative_reference_row, dtype=np.int64),
            "source_tag": np.asarray(self.tag, dtype=np.int8),
            "activation_probability": np.asarray(self.activation, dtype=np.float32),
            "mean_hinge_at_mining": np.asarray(self.hinge, dtype=np.float32),
        }


def build_pools(
    graph: Mapping[str, np.ndarray], cache: Mapping[str, np.ndarray],
    actions: pd.DataFrame, train_queries: np.ndarray, validation_queries: np.ndarray,
    *, margin: float, negative_reference_cap: int, positive_reference_cap: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, dict[str, object]]:
    positions = embedding_positions(cache)
    actions = actions.reset_index(drop=True)
    actions_by_query: dict[int, list[int]] = defaultdict(list)
    for index, query in enumerate(actions["query_index"].to_numpy(np.int64)):
        actions_by_query[int(query)].append(int(index))
    train = PoolBuilder()
    validation = PoolBuilder()
    action_event = np.full(len(actions), -1, dtype=np.int64)
    action_active = np.zeros(len(actions), dtype=bool)
    action_positive_represented = np.zeros(len(actions), dtype=bool)
    action_negative_represented = np.zeros(len(actions), dtype=bool)
    geometry_cache: dict[int, dict[str, object]] = {}

    def geometry(query: int) -> dict[str, object]:
        if query not in geometry_cache:
            geometry_cache[query] = _query_geometry(graph, cache, positions, query)
        return geometry_cache[query]

    def add_base(builder: PoolBuilder, query: int) -> None:
        body = geometry(query)
        false = body["false"]
        molecule = max(
            false, key=lambda value: (float(np.max(false[value][1])), -int(value)),
        )
        rows, scores = false[molecule]
        positives = np.asarray(body["positive_rows"], dtype=np.int64)
        positive_scores = np.asarray(body["positive_scores"], dtype=np.float32)
        hinge = np.maximum(
            margin + np.asarray(scores)[:, None] - positive_scores[None, :], 0.0,
        )
        builder.add(
            key=("base", query), anchor=int(body["anchor"]), positives=positives,
            negatives=np.asarray(rows, dtype=np.int64), query=query,
            negative_molecule=int(molecule), tag=BASE_OFFICIAL_HARD,
            positive_reference_row=-1, negative_reference_row=-1,
            activation=float(np.mean(hinge > 0.0)), hinge=float(np.mean(hinge)),
        )

    relations: dict[tuple[int, int], list[int]] = defaultdict(list)
    exact: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(actions.itertuples(index=False)):
        query = int(row.query_index)
        molecule, label, _ = locate_row(graph, query, int(row.action_hard_negative_row))
        if label:
            raise RuntimeError("Noise action selected a true molecule as negative")
        relations[(query, molecule)].append(index)
        exact[(query, int(row.action_positive_row), int(row.action_hard_negative_row))].append(index)

    query_formulas = np.asarray(graph["query_formula"]).astype(str)
    action_queries = {query for query, _ in relations}
    for query in sorted(action_queries):
        add_base(train, query)
    for query in map(int, validation_queries):
        add_base(validation, query)

    expanded_action_query_events = 0
    sentinel_action_query_events = 0
    for query in sorted(action_queries):
        indices = actions_by_query[query]
        body = geometry(query)
        exact_positive_rows = np.unique(
            actions.iloc[indices]["action_positive_row"].to_numpy(np.int64)
        )
        exact_negative_rows = np.unique(
            actions.iloc[indices]["action_hard_negative_row"].to_numpy(np.int64)
        )

        # Retain every action-proposed positive. If that pool is very small,
        # add hard same-identity references so native DreaMS can still resample.
        positive_rows = list(map(int, exact_positive_rows))
        positive_scores = np.asarray(body["positive_scores"], dtype=np.float32)
        for offset in np.argsort(positive_scores, kind="stable"):
            row = int(np.asarray(body["positive_rows"])[int(offset)])
            if row not in positive_rows:
                positive_rows.append(row)
            if len(positive_rows) >= positive_reference_cap:
                break
        positive_rows_array = np.asarray(sorted(set(positive_rows)), dtype=np.int64)

        # Every action-selected false row enters the pool. Add the hardest
        # measured references from every selected false molecule for diversity.
        negative_rows = set(map(int, exact_negative_rows))
        selected_molecules = sorted({
            molecule for (relation_query, molecule) in relations
            if relation_query == query
        })
        for molecule in selected_molecules:
            rows, scores = body["false"][molecule]
            order = np.argsort(-np.asarray(scores), kind="stable")[:negative_reference_cap]
            negative_rows.update(int(rows[int(offset)]) for offset in order)
        negative_rows_array = np.asarray(sorted(negative_rows), dtype=np.int64)

        anchor_z = _embeddings(
            cache, positions, np.asarray([int(body["anchor"])], dtype=np.int64),
        )[0]
        positive_pool_scores = _embeddings(cache, positions, positive_rows_array) @ anchor_z
        negative_pool_scores = _embeddings(cache, positions, negative_rows_array) @ anchor_z
        hinge = np.maximum(
            margin + negative_pool_scores[:, None] - positive_pool_scores[None, :], 0.0,
        )
        expanded = bool(set(map(int, negative_rows_array)) - set(map(int, exact_negative_rows)))
        active = bool(np.any(hinge > 0.0))
        tag = ACTION_EXACT | (ACTION_REFERENCE_EXPANDED if expanded else 0)
        if not active:
            tag |= SAFE_SENTINEL
            sentinel_action_query_events += 1
        if expanded:
            expanded_action_query_events += 1
        event = train.add(
            key=("action-query-pool", query), anchor=int(body["anchor"]),
            positives=positive_rows_array, negatives=negative_rows_array,
            query=query, negative_molecule=-1, tag=tag,
            positive_reference_row=-1, negative_reference_row=-1,
            activation=float(np.mean(hinge > 0.0)), hinge=float(np.mean(hinge)),
        )
        action_event[indices] = event
        action_active[indices] = active
        for index in indices:
            action_positive_represented[index] = (
                int(actions.iloc[index]["action_positive_row"])
                in set(map(int, positive_rows_array))
            )
            action_negative_represented[index] = (
                int(actions.iloc[index]["action_hard_negative_row"])
                in set(map(int, negative_rows_array))
            )

    if np.any(action_event < 0):
        raise RuntimeError("one or more qualified actions were not mapped to a native event")
    train_pool = train.arrays()
    val_pool = validation.arrays()
    tags = np.asarray(train_pool["source_tag"], dtype=np.int64)
    activation = np.asarray(train_pool["activation_probability"], dtype=np.float64)
    formulas = np.asarray(graph["query_formula"])[train_pool["source_query"]].astype(str)
    alias = pd.DataFrame({
        "qualified_action_index": np.arange(len(actions), dtype=np.int64),
        "qualified_action_id": actions["action_id"].astype(str),
        "query_index": actions["query_index"].to_numpy(np.int64),
        "source": actions["source"].astype(str),
        "optimization_event_index": action_event,
        "exact_positive_row_in_dynamic_pool": action_positive_represented,
        "exact_negative_row_in_dynamic_pool": action_negative_represented,
        "mapped_event_active_at_mining": action_active,
    })
    active_by_source = {
        source: int(np.sum(
            actions["source"].astype(str).eq(source).to_numpy() & action_active
        ))
        for source in sorted(REGISTERED_SOURCES)
    }
    query_event_counts = Counter(map(int, train_pool["source_query"]))
    maximum_events_per_query = max(query_event_counts.values(), default=0)
    base_query_counts = Counter(
        int(query) for query, tag in zip(
            train_pool["source_query"], train_pool["source_tag"],
        )
        if int(tag) & BASE_OFFICIAL_HARD
    )
    action_query_events = int(
        np.sum((tags & (ACTION_EXACT | ACTION_REFERENCE_EXPANDED)) > 0)
    )
    audit = {
        "triplet_events": int(len(train.anchor)),
        "base_official_hard_events": int(np.sum((tags & BASE_OFFICIAL_HARD) > 0)),
        "action_exact_events": int(np.sum((tags & ACTION_EXACT) > 0)),
        "action_reference_expanded_events": int(np.sum((tags & ACTION_REFERENCE_EXPANDED) > 0)),
        "safe_sentinel_events": int(np.sum((tags & SAFE_SENTINEL) > 0)),
        "active_events": int(np.sum(activation > 0.0)),
        "active_event_fraction": float(np.mean(activation > 0.0)),
        "mean_activation_probability": float(np.mean(activation)),
        "mean_native_hinge_at_mining": float(np.mean(train_pool["mean_hinge_at_mining"])),
        "unique_anchor_queries": int(len(np.unique(train_pool["source_query"]))),
        "unique_formulas": int(len(np.unique(formulas))),
        "positive_reference_edges": int(len(train_pool["positive_idx"])),
        "negative_reference_edges": int(len(train_pool["negative_idx"])),
        "qualified_action_aliases": int(len(alias)),
        "unique_exact_action_boundaries": int(len(exact)),
        "unique_action_query_negative_relations": int(len(relations)),
        "action_query_events": action_query_events,
        "expanded_action_query_events": int(expanded_action_query_events),
        "sentinel_action_query_events": int(sentinel_action_query_events),
        "actions_with_exact_positive_in_dynamic_pool": int(
            np.sum(action_positive_represented)
        ),
        "actions_with_exact_negative_in_dynamic_pool": int(
            np.sum(action_negative_represented)
        ),
        "actions_mapped_to_active_event": int(np.sum(action_active)),
        "active_mapped_action_fraction": float(np.mean(action_active)),
        "active_mapped_actions_by_source": active_by_source,
        "maximum_events_per_query": int(maximum_events_per_query),
        "action_query_base_events": int(sum(base_query_counts.values())),
        "every_action_query_has_one_base_event": bool(
            set(base_query_counts) == action_queries
            and all(base_query_counts[query] == 1 for query in action_queries)
        ),
        "non_action_queries_receive_no_base_dose": bool(
            not (set(base_query_counts) - action_queries)
        ),
        "all_action_aliases_participate_in_dynamic_pool": bool(
            np.all(action_positive_represented)
            and np.all(action_negative_represented)
        ),
        "one_dynamic_action_pool_per_query": bool(
            action_query_events == len(action_queries)
        ),
        "every_action_query_has_exactly_two_native_events": bool(
            set(query_event_counts) == action_queries
            and all(query_event_counts[query] == 2 for query in action_queries)
        ),
        "all_actions_mapped": bool(np.all(action_event >= 0)),
        "action_pool_sampling": "native_dynamic_one_positive_one_negative",
        "raw_duplicate_actions_are_not_optimizer_dose": True,
        "action_tensor_enters_model_input": False,
    }
    return train_pool, val_pool, alias, audit


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.base_queries_per_formula < 1:
        raise ValueError("--base-queries-per-formula must be positive")
    if min(args.negative_references_per_action_candidate, args.positive_references_per_negative) < 1:
        raise ValueError("reference expansion caps must be positive")
    if not 0.0 <= args.minimum_active_action_fraction <= 1.0:
        raise ValueError("--minimum-active-action-fraction must be in [0, 1]")
    report_path = args.ledger_dir / "report.json"
    actions_path = args.ledger_dir / "training_actions.csv.gz"
    spectra_path = args.ledger_dir / "action_spectra.npz"
    for path in (report_path, actions_path, spectra_path, args.graph, args.embedding_cache, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    ledger_report = json.loads(report_path.read_text(encoding="utf-8"))
    if not _report_allows_training(ledger_report):
        raise RuntimeError("Noise action ledger is not formal/outer-safe")
    if int(ledger_report.get("outer_formula_fold", -1)) != args.outer_fold:
        raise RuntimeError("Noise action ledger outer fold drifted")
    provenance = audit_formal_inputs(
        ledger_report, report_path=report_path, actions_path=actions_path,
        spectra_path=spectra_path, graph_path=args.graph,
        embedding_cache_path=args.embedding_cache,
    )
    actions = select_actions(
        pd.read_csv(actions_path, low_memory=False), margin_floor=5e-6,
        outer_fold=args.outer_fold, formula_fold_seed=args.formula_fold_seed,
        expected_actions=args.expected_actions,
    ).reset_index(drop=True)
    graph = load_npz(args.graph)
    cache = load_npz(args.embedding_cache)
    action_row_audit(actions, graph)
    train_queries, validation_queries, split = _split_queries(
        graph, actions, outer_fold=args.outer_fold,
        formula_fold_seed=args.formula_fold_seed,
        validation_folds=args.validation_folds,
        validation_fold=args.validation_fold,
        validation_seed=args.validation_seed,
        base_queries_per_formula=args.base_queries_per_formula,
    )
    train, validation, aliases, curriculum = build_pools(
        graph, cache, actions, train_queries, validation_queries,
        margin=args.margin,
        negative_reference_cap=args.negative_references_per_action_candidate,
        positive_reference_cap=args.positive_references_per_negative,
    )
    identity = {
        "train": audit_identity_edges(train, args.data),
        "validation": audit_identity_edges(validation, args.data),
    }
    training_formulas = set(
        np.asarray(graph["query_formula"])[train["source_query"]].astype(str)
    )
    validation_formulas = set(
        np.asarray(graph["query_formula"])[validation["source_query"]].astype(str)
    )
    gates = {
        "all_32114_actions_retained_in_alias_ledger": (
            len(actions) == len(aliases) == args.expected_actions
            and aliases["qualified_action_id"].nunique() == args.expected_actions
        ),
        "registered_source_closure": set(actions["source"].astype(str)) == REGISTERED_SOURCES,
        "all_actions_mapped_to_native_events": bool(curriculum["all_actions_mapped"]),
        "one_dynamic_action_pool_per_query": bool(
            curriculum["one_dynamic_action_pool_per_query"]
        ),
        "every_action_query_has_exactly_two_native_events": bool(
            curriculum["every_action_query_has_exactly_two_native_events"]
        ),
        "every_action_query_has_one_base_event": bool(
            curriculum["every_action_query_has_one_base_event"]
        ),
        "non_action_queries_receive_no_base_dose": bool(
            curriculum["non_action_queries_receive_no_base_dose"]
        ),
        "all_action_aliases_participate_in_dynamic_pool": bool(
            curriculum["all_action_aliases_participate_in_dynamic_pool"]
        ),
        "action_formula_coverage_is_complete": (
            curriculum["unique_formulas"] == split["action_formulas"]
        ),
        "formula_diversity_matches_successful_native_scale": (
            curriculum["unique_formulas"] >= args.minimum_train_formulas
        ),
        "triplet_count_matches_successful_native_scale": (
            curriculum["triplet_events"] >= args.minimum_triplet_events
        ),
        "active_event_count_is_large": (
            curriculum["active_events"] >= args.minimum_active_events
        ),
        "exact_action_event_count_is_large": (
            curriculum["action_exact_events"] >= args.minimum_exact_action_events
        ),
        "most_actions_map_to_an_active_event": (
            curriculum["active_mapped_action_fraction"]
            >= args.minimum_active_action_fraction
        ),
        "train_validation_formula_disjoint": not bool(training_formulas & validation_formulas),
        "outer_held_formulas_excluded": split["outer_held_formulas_excluded"] is True,
        "exact_action_boundaries_retained_before_dynamic_sampling": (
            curriculum["unique_exact_action_boundaries"] > 0
        ),
        "native_dynamic_pools_produce_active_events": curriculum["active_events"] > 0,
        "every_registered_source_reaches_an_active_native_event": all(
            count > 0
            for count in curriculum["active_mapped_actions_by_source"].values()
        ),
        "all_identity_edges_verified": all(body["identity_contract_passed"] for body in identity.values()),
        "action_tensor_excluded_from_model_input": curriculum["action_tensor_enters_model_input"] is False,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Noise reference-aligned triplet gates failed: {gates}")
    report = {
        "status": "NOISE_REFERENCE_ALIGNED_NATIVE_TRIPLETS_COMPLETE",
        "version": VERSION,
        "method": "one preservation base plus one all-action dynamic native pool per Noise action query",
        "training_runtime": "tasks/train_noise_reference_native.py (Noise-owned native DreaMS runtime)",
        "triplet_semantics": "measured clean query / measured same-identity positive / measured false-candidate negative",
        "qualified_actions": int(len(actions)),
        "qualified_source_counts": dict(sorted(Counter(actions["source"].astype(str)).items())),
        "split": split,
        "targeted_curriculum": curriculum,
        "validation": {
            "triplet_events": int(len(validation["anchor_idx"])),
            "unique_queries": int(len(np.unique(validation["source_query"]))),
            "unique_formulas": int(len(validation_formulas)),
        },
        "identity_audit": identity,
        "gates": gates,
        "provenance": provenance,
        "claim_limit": "Triplet construction only; retrieval gain requires frozen evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="noise_reference_aligned_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool_targeted.npz", **train)
        np.savez_compressed(temporary / "val_pool.npz", **validation)
        actions.to_csv(temporary / "selected_actions.csv.gz", index=False)
        aliases.to_csv(temporary / "action_aliases.csv.gz", index=False)
        report["output_artifacts"] = {
            name: sha256_file(temporary / name)
            for name in (
                "train_pool_targeted.npz", "val_pool.npz",
                "selected_actions.csv.gz", "action_aliases.csv.gz",
            )
        }
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
