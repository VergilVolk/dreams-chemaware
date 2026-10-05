"""Convert the frozen Noise action ledger into DreaMS-native hard triplets.

This is the Noise adapter for the training interface that produced the sealed
ChemAware +1.8144 pp result.  A Noise action may only choose a false candidate
molecule.  Every optimizer event remains an identity-valid native triplet:

    measured clean query / measured same-identity references /
    measured false-candidate references.

Action tensors, ranks and margins never enter the encoder or the loss.  Raw
action multiplicity is retained in an alias ledger but optimizer dose is
deduplicated by the successful ChemAware unit: one unique
``(query, negative molecule)`` relation per native event.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from build_chemaware_dreams_native_triplets import audit_identity_edges
from build_noise_dreams_native_triplets import (
    REGISTERED_FORMAL_COUNTS,
    REGISTERED_SOURCES,
    _report_allows_training,
    audit_formal_inputs,
    embedding_positions,
    hard_negative_rows,
    load_npz,
    query_molecules,
    select_actions,
    sha256_file,
    stable_fold,
)


NOISE_ACTION_HARD_NATIVE_VERSION = "noise_action_hard_native_v2"
OFFICIAL_HARD = 1
NOISE_ACTION_HARD = 2


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
    parser.add_argument("--margin-floor", type=float, default=5e-6)
    parser.add_argument("--expected-actions", type=int, default=32114)
    parser.add_argument("--minimum-train-queries", type=int, default=3000)
    parser.add_argument("--minimum-train-events", type=int, default=3000)
    parser.add_argument("--minimum-validation-events", type=int, default=1000)
    return parser.parse_args()


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


def positive_rows(
    graph: Mapping[str, np.ndarray], query: int,
) -> np.ndarray:
    anchor = int(graph["query_row"][query])
    blocks = [
        np.asarray(rows, dtype=np.int64)
        for _, label, rows in query_molecules(graph, query) if label
    ]
    if not blocks:
        raise RuntimeError(f"query {query} has no true candidate molecule")
    rows = np.unique(np.concatenate(blocks))
    rows = rows[rows != anchor]
    if not len(rows):
        raise RuntimeError(f"query {query} has no distinct positive reference")
    return rows


def action_row_audit(
    actions: pd.DataFrame,
    graph: Mapping[str, np.ndarray],
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


def build_arm_pool(
    actions: pd.DataFrame,
    graph: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    *,
    negative_column: str,
) -> tuple[dict[str, np.ndarray], dict[str, object], np.ndarray]:
    """Build one ChemAware-compatible pool and action-to-event mapping."""
    row_position = embedding_positions(cache)
    event_tags: dict[tuple[int, int], int] = {}
    event_rows: dict[tuple[int, int], np.ndarray] = {}
    action_pairs: list[tuple[int, int]] = []
    actions_by_query: dict[int, list[int]] = defaultdict(list)
    for index, query in enumerate(actions["query_index"].to_numpy(np.int64)):
        actions_by_query[int(query)].append(int(index))

    # The stable base is one official hard-negative molecule for every action
    # query.  This is the same coverage principle used by the successful
    # ChemAware pool and is identical in both causal arms.
    for query in sorted(actions_by_query):
        official_rows = hard_negative_rows(
            graph, query, cache, row_position, maximum_molecules=1,
        )
        molecule, label, rows = locate_row(graph, query, int(official_rows[0]))
        if label:
            raise RuntimeError("official hard-negative selector returned truth")
        key = (query, molecule)
        event_rows[key] = rows
        event_tags[key] = event_tags.get(key, 0) | OFFICIAL_HARD

    for row in actions.itertuples(index=False):
        query = int(row.query_index)
        molecule, label, rows = locate_row(
            graph, query, int(getattr(row, negative_column)),
        )
        if label:
            raise RuntimeError("Noise-selected hard negative is a true molecule")
        key = (query, molecule)
        event_rows[key] = rows
        event_tags[key] = event_tags.get(key, 0) | NOISE_ACTION_HARD
        action_pairs.append(key)

    ordered = sorted(event_rows)
    event_index = {key: index for index, key in enumerate(ordered)}
    aliases = np.asarray([event_index[key] for key in action_pairs], dtype=np.int64)
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    source_query: list[int] = []
    negative_molecule: list[int] = []
    source_tag: list[int] = []
    for query, molecule in ordered:
        pos = positive_rows(graph, query)
        neg = event_rows[(query, molecule)]
        anchors.append(int(graph["query_row"][query]))
        positives.extend(map(int, pos))
        negatives.extend(map(int, neg))
        positive_ptr.append(len(positives))
        negative_ptr.append(len(negatives))
        source_query.append(query)
        negative_molecule.append(molecule)
        source_tag.append(event_tags[(query, molecule)])
    pool = {
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positives, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negatives, dtype=np.int64),
        "source_query": np.asarray(source_query, dtype=np.int64),
        "negative_molecule": np.asarray(negative_molecule, dtype=np.int64),
        "source_tag": np.asarray(source_tag, dtype=np.int8),
    }
    query_array = np.asarray(source_query, dtype=np.int64)
    formulas = np.asarray(graph["query_formula"])[query_array].astype(str)
    tags = np.asarray(source_tag, dtype=np.int8)
    audit = {
        "triplet_events": int(len(anchors)),
        "unique_query_negative_pairs": int(len(ordered)),
        "unique_anchor_queries": int(len(np.unique(query_array))),
        "unique_formulas": int(len(np.unique(formulas))),
        "official_hard_events": int(np.sum((tags & OFFICIAL_HARD) > 0)),
        "noise_action_hard_events": int(np.sum((tags & NOISE_ACTION_HARD) > 0)),
        "multi_source_events": int(np.sum(tags == (OFFICIAL_HARD | NOISE_ACTION_HARD))),
        "positive_reference_edges": int(len(positives)),
        "negative_reference_edges": int(len(negatives)),
        "qualified_action_aliases": int(len(aliases)),
        "actions_collapsed_by_query_negative_deduplication": int(
            len(actions) - len(np.unique(aliases))
        ),
        "raw_action_multiplicity_is_optimizer_dose": False,
    }
    return pool, audit, aliases


def build_paired_pools(
    actions: pd.DataFrame,
    graph: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray],
    dict[str, object], np.ndarray,
]:
    """Build one native event per unique targeted query-negative relation.

    This is the successful ChemAware dose unit.  Repeated discoveries of the
    same targeted relation become aliases, not extra gradient dose.  Each
    targeted relation receives one deterministic matched-control molecule,
    chosen as the modal control paired with that relation (stable molecule-ID
    tie break).  The two arms therefore share event/query/positive ledgers and
    differ only in the selected false molecule.
    """
    actions = actions.reset_index(drop=True)
    row_position = embedding_positions(cache)
    units: dict[tuple[int, int], dict[str, object]] = {}
    official_by_query: dict[int, int] = {}
    for query in sorted(set(map(int, actions["query_index"]))):
        official_rows = hard_negative_rows(
            graph, query, cache, row_position, maximum_molecules=1,
        )
        official_molecule, label, official_rows = locate_row(
            graph, query, int(official_rows[0]),
        )
        if label:
            raise RuntimeError("official hard-negative selector returned truth")
        official_by_query[query] = official_molecule
        units[(query, official_molecule)] = {
            "targeted_rows": official_rows,
            "control_rows": official_rows,
            "control_molecule": official_molecule,
            "control_options": set(),
            "control_rows_by_molecule": {},
            "tag": OFFICIAL_HARD,
        }

    action_keys: list[tuple[int, int]] = []
    for row in actions.itertuples(index=False):
        query = int(row.query_index)
        target_molecule, target_label, target_rows = locate_row(
            graph, query, int(row.action_hard_negative_row),
        )
        control_molecule, control_label, control_rows = locate_row(
            graph, query, int(row.control_hard_negative_row),
        )
        if target_label or control_label:
            raise RuntimeError("paired Noise relation contains a true negative role")
        key = (query, target_molecule)
        if key not in units:
            units[key] = {
                "targeted_rows": target_rows,
                "control_rows": None,
                "control_molecule": None,
                "control_options": set(),
                "control_rows_by_molecule": {},
                "tag": NOISE_ACTION_HARD,
            }
        body = units[key]
        if not np.array_equal(body["targeted_rows"], target_rows):
            raise RuntimeError("targeted query-negative relation changed membership")
        body["tag"] = int(body["tag"]) | NOISE_ACTION_HARD
        if target_molecule != official_by_query[query]:
            options = body["control_options"]
            rows_by_molecule = body["control_rows_by_molecule"]
            options.add(control_molecule)
            rows_by_molecule[control_molecule] = control_rows
        action_keys.append(key)

    fallback_control_assignments = 0
    for query in sorted(official_by_query):
        target_keys = sorted(
            key for key in units
            if key[0] == query and key[1] != official_by_query[query]
        )
        false_rows = {
            int(molecule): np.asarray(rows, dtype=np.int64)
            for molecule, label, rows in query_molecules(graph, query)
            if not label
        }
        available = sorted(
            molecule for molecule in false_rows
            if molecule != official_by_query[query]
        )
        if len(available) < len(target_keys):
            raise RuntimeError(
                f"query {query} lacks enough unique false molecules for a matched control"
            )
        preferences: dict[tuple[int, int], list[int]] = {}
        for key in target_keys:
            target_molecule = key[1]
            associated = sorted(
                molecule for molecule in units[key]["control_options"]
                if molecule in available and molecule != target_molecule
            )
            remainder = [
                molecule for molecule in available
                if molecule not in associated and molecule != target_molecule
            ]
            preferences[key] = associated + remainder
            if not preferences[key]:
                raise RuntimeError(
                    f"query {query} target {target_molecule} has no distinct control"
                )

        owner: dict[int, tuple[int, int]] = {}
        assignment: dict[tuple[int, int], int] = {}

        def assign(key: tuple[int, int], seen: set[int]) -> bool:
            for molecule in preferences[key]:
                if molecule in seen:
                    continue
                seen.add(molecule)
                previous = owner.get(molecule)
                if previous is None or assign(previous, seen):
                    owner[molecule] = key
                    assignment[key] = molecule
                    return True
            return False

        for key in sorted(target_keys, key=lambda value: (len(preferences[value]), value)):
            if not assign(key, set()):
                raise RuntimeError(
                    f"query {query} has no one-to-one matched-control assignment"
                )
        if len(set(assignment.values())) != len(target_keys):
            raise RuntimeError("matched-control assignment is not relation-unique")
        for key, selected_control in assignment.items():
            body = units[key]
            if selected_control not in body["control_options"]:
                fallback_control_assignments += 1
            body["control_molecule"] = int(selected_control)
            body["control_rows"] = false_rows[selected_control]

    original_targeted_negative_edges = 0
    original_control_negative_edges = 0
    for (query, target_molecule), body in units.items():
        target_rows = np.asarray(body["targeted_rows"], dtype=np.int64)
        control_rows = np.asarray(body["control_rows"], dtype=np.int64)
        original_targeted_negative_edges += len(target_rows)
        original_control_negative_edges += len(control_rows)
        if target_molecule == official_by_query[query]:
            continue
        keep = min(len(target_rows), len(control_rows))
        salt = f"{query}|{target_molecule}|{body['control_molecule']}"
        def stable_subset(rows: np.ndarray) -> np.ndarray:
            return np.asarray(sorted(
                map(int, rows),
                key=lambda value: hashlib.sha256(
                    f"{salt}|{value}".encode("utf-8")
                ).digest(),
            )[:keep], dtype=np.int64)
        body["targeted_rows"] = stable_subset(target_rows)
        body["control_rows"] = stable_subset(control_rows)

    ordered = sorted(units)
    event_index = {key: index for index, key in enumerate(ordered)}
    aliases = np.asarray([event_index[key] for key in action_keys], dtype=np.int64)
    common_anchor: list[int] = []
    common_positive: list[int] = []
    common_positive_ptr = [0]
    targeted_negative: list[int] = []
    control_negative: list[int] = []
    targeted_negative_ptr = [0]
    control_negative_ptr = [0]
    source_query: list[int] = []
    targeted_molecules: list[int] = []
    control_molecules: list[int] = []
    source_tag: list[int] = []
    for query, target_molecule in ordered:
        body = units[(query, target_molecule)]
        positive = positive_rows(graph, query)
        target_rows = np.asarray(body["targeted_rows"], dtype=np.int64)
        control_rows = np.asarray(body["control_rows"], dtype=np.int64)
        if not len(target_rows) or not len(control_rows):
            raise RuntimeError("native relation has an empty negative reference set")
        common_anchor.append(int(graph["query_row"][query]))
        common_positive.extend(map(int, positive))
        common_positive_ptr.append(len(common_positive))
        targeted_negative.extend(map(int, target_rows))
        control_negative.extend(map(int, control_rows))
        targeted_negative_ptr.append(len(targeted_negative))
        control_negative_ptr.append(len(control_negative))
        source_query.append(query)
        targeted_molecules.append(target_molecule)
        control_molecules.append(int(body["control_molecule"]))
        source_tag.append(int(body["tag"]))

    common = {
        "anchor_idx": np.asarray(common_anchor, dtype=np.int64),
        "positive_ptr": np.asarray(common_positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(common_positive, dtype=np.int64),
        "source_query": np.asarray(source_query, dtype=np.int64),
        "source_tag": np.asarray(source_tag, dtype=np.int8),
    }
    targeted = {
        **common,
        "negative_ptr": np.asarray(targeted_negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(targeted_negative, dtype=np.int64),
        "negative_molecule": np.asarray(targeted_molecules, dtype=np.int64),
    }
    control = {
        **common,
        "negative_ptr": np.asarray(control_negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(control_negative, dtype=np.int64),
        "negative_molecule": np.asarray(control_molecules, dtype=np.int64),
    }
    query_array = np.asarray(source_query, dtype=np.int64)
    formulas = np.asarray(graph["query_formula"])[query_array].astype(str)
    tags = np.asarray(source_tag, dtype=np.int8)
    targeted_queries, targeted_counts = np.unique(
        targeted["source_query"], return_counts=True,
    )
    control_queries, control_counts = np.unique(
        control["source_query"], return_counts=True,
    )
    batches_per_complete_pass = (len(ordered) + 3) // 4
    planned_steps = (3 * batches_per_complete_pass + 1) // 2
    audit = {
        "paired_triplet_events_per_arm": int(len(ordered)),
        "unique_anchor_queries_per_arm": int(len(np.unique(query_array))),
        "unique_formulas_per_arm": int(len(np.unique(formulas))),
        "official_hard_events_per_arm": int(np.sum((tags & OFFICIAL_HARD) > 0)),
        "noise_action_hard_events_per_arm": int(np.sum((tags & NOISE_ACTION_HARD) > 0)),
        "targeted_positive_reference_edges": int(len(common_positive)),
        "control_positive_reference_edges": int(len(common_positive)),
        "targeted_negative_reference_edges": int(len(targeted_negative)),
        "control_negative_reference_edges": int(len(control_negative)),
        "original_targeted_negative_reference_edges": int(original_targeted_negative_edges),
        "original_control_negative_reference_edges": int(original_control_negative_edges),
        "matched_reference_edges_retained_per_arm": int(len(targeted_negative)),
        "equal_negative_reference_count_for_every_paired_event": bool(np.array_equal(
            np.diff(targeted["negative_ptr"]), np.diff(control["negative_ptr"])
        )),
        "relation_unique_control_pairs": int(len(set(zip(
            map(int, control["source_query"]), map(int, control["negative_molecule"])
        )))) == len(ordered),
        "fallback_control_assignments_outside_associated_ledger_controls": int(
            fallback_control_assignments
        ),
        "qualified_action_aliases": int(len(aliases)),
        "unique_targeted_query_negative_relations": int(len(ordered)),
        "actions_collapsed_by_target_relation_deduplication": int(
            len(actions) - len(np.unique(aliases))
        ),
        "all_targeted_action_relations_are_scheduled_native_events": bool(
            set(action_keys).issubset(set(ordered))
            and len(np.unique(aliases)) == len(set(action_keys))
        ),
        "action_rows_aliasing_shared_official_relation": int(sum(
            key[1] == official_by_query[key[0]] for key in action_keys
        )),
        "unique_action_relations_aliasing_shared_official_relation": int(len({
            key for key in action_keys
            if key[1] == official_by_query[key[0]]
        })),
        "minimum_events_per_query": int(targeted_counts.min()),
        "median_events_per_query": float(np.median(targeted_counts)),
        "maximum_events_per_query": int(targeted_counts.max()),
        "query_dose_is_relation_count_not_raw_action_count": True,
        "batch_size": 4,
        "batches_per_complete_pass_keep_partial": int(batches_per_complete_pass),
        "planned_fixed_steps": int(planned_steps),
        "planned_complete_passes": float(planned_steps / batches_per_complete_pass),
        "every_relation_scheduled_in_first_pass": True,
        "same_event_count": len(targeted["anchor_idx"]) == len(control["anchor_idx"]),
        "same_anchor_sequence": bool(np.array_equal(
            targeted["anchor_idx"], control["anchor_idx"]
        )),
        "same_positive_memberships": bool(
            np.array_equal(targeted["positive_ptr"], control["positive_ptr"])
            and np.array_equal(targeted["positive_idx"], control["positive_idx"])
        ),
        "same_source_query_sequence": bool(np.array_equal(
            targeted["source_query"], control["source_query"]
        )),
        "same_source_tag_sequence": bool(np.array_equal(
            targeted["source_tag"], control["source_tag"]
        )),
        "same_query_multiplicity": bool(np.array_equal(
            targeted_queries, control_queries,
        ) and np.array_equal(targeted_counts, control_counts)),
        "raw_action_multiplicity_is_optimizer_dose": False,
    }
    return targeted, control, audit, aliases


def build_validation_pool(
    graph: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    actions: pd.DataFrame,
    *,
    outer_fold: int,
    formula_fold_seed: int,
    validation_folds: int,
    validation_fold: int,
    validation_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    formulas = np.asarray(graph["query_formula"]).astype(str)
    action_formulas = set(actions["query_formula"].astype(str))
    queries = np.asarray([
        query for query, formula in enumerate(formulas)
        if stable_fold(formula, 5, formula_fold_seed) != outer_fold
        and formula not in action_formulas
        and stable_fold(formula, validation_folds, validation_seed) == validation_fold
    ], dtype=np.int64)
    if not len(queries):
        raise RuntimeError("formula-disjoint native validation pool is empty")
    synthetic = pd.DataFrame({"query_index": queries})
    # Reuse the same builder by supplying no action-specific event: construct
    # one official-hard row per query directly.
    row_position = embedding_positions(cache)
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    negative_molecules: list[int] = []
    for query in queries:
        query = int(query)
        hard = hard_negative_rows(
            graph, query, cache, row_position, maximum_molecules=1,
        )
        molecule, label, neg = locate_row(graph, query, int(hard[0]))
        if label:
            raise RuntimeError("validation hard negative is true")
        pos = positive_rows(graph, query)
        anchors.append(int(graph["query_row"][query]))
        positives.extend(map(int, pos))
        negatives.extend(map(int, neg))
        positive_ptr.append(len(positives))
        negative_ptr.append(len(negatives))
        negative_molecules.append(molecule)
    del synthetic
    pool = {
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positives, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negatives, dtype=np.int64),
        "source_query": queries,
        "negative_molecule": np.asarray(negative_molecules, dtype=np.int64),
        "source_tag": np.full(len(queries), OFFICIAL_HARD, dtype=np.int8),
    }
    return pool, {
        "triplet_events": int(len(queries)),
        "unique_queries": int(len(queries)),
        "unique_formulas": int(len(np.unique(formulas[queries]))),
        "train_formula_overlap": int(len(set(formulas[queries]) & action_formulas)),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report_path = args.ledger_dir / "report.json"
    actions_path = args.ledger_dir / "training_actions.csv.gz"
    spectra_path = args.ledger_dir / "action_spectra.npz"
    for path in (
        report_path, actions_path, spectra_path, args.graph,
        args.embedding_cache, args.data,
    ):
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
    complete = pd.read_csv(actions_path, low_memory=False)
    actions = select_actions(
        complete, margin_floor=args.margin_floor, outer_fold=args.outer_fold,
        formula_fold_seed=args.formula_fold_seed,
        expected_actions=args.expected_actions,
    ).reset_index(drop=True)
    graph = load_npz(args.graph)
    cache = load_npz(args.embedding_cache)
    if len(graph.get("query_row", [])) != REGISTERED_FORMAL_COUNTS["graph_queries"]:
        raise RuntimeError("registered corrected candidate graph drifted")
    action_row_audit(actions, graph)
    targeted, targeted_audit, targeted_alias = build_arm_pool(
        actions, graph, cache, negative_column="action_hard_negative_row",
    )
    train_events = int(targeted_audit["triplet_events"])
    batches_per_complete_pass = (train_events + 3) // 4
    planned_steps = (3 * batches_per_complete_pass + 1) // 2
    _, query_counts = np.unique(targeted["source_query"], return_counts=True)
    targeted_audit.update({
        "batch_size": 4,
        "batches_per_complete_pass_keep_partial": int(batches_per_complete_pass),
        "planned_fixed_steps": int(planned_steps),
        "planned_complete_passes": float(planned_steps / batches_per_complete_pass),
        "every_relation_scheduled_in_first_pass": True,
        "minimum_events_per_query": int(query_counts.min()),
        "median_events_per_query": float(np.median(query_counts)),
        "maximum_events_per_query": int(query_counts.max()),
        "active_hinge_fraction_requires_runtime_measurement": True,
    })
    validation, validation_audit = build_validation_pool(
        graph, cache, actions, outer_fold=args.outer_fold,
        formula_fold_seed=args.formula_fold_seed,
        validation_folds=args.validation_folds,
        validation_fold=args.validation_fold,
        validation_seed=args.validation_seed,
    )
    identity_audit = {
        "targeted": audit_identity_edges(targeted, args.data),
        "validation": audit_identity_edges(validation, args.data),
    }
    alias = pd.DataFrame({
        "qualified_action_index": np.arange(len(actions), dtype=np.int64),
        "qualified_action_id": actions["action_id"].astype(str),
        "query_index": actions["query_index"].to_numpy(np.int64),
        "source": actions["source"].astype(str),
        "targeted_event_index": targeted_alias,
    })
    training_query_formulas = np.asarray(graph["query_formula"]).astype(str)[
        np.asarray(targeted["source_query"], dtype=np.int64)
    ]
    training_formula_folds = np.asarray([
        stable_fold(value, 5, args.formula_fold_seed)
        for value in training_query_formulas
    ], dtype=np.int64)
    gates = {
        "formal_outer_safe_ledger": True,
        "all_32114_actions_retained_in_alias_ledger": (
            len(actions) == len(alias) == args.expected_actions
            and alias["qualified_action_id"].nunique() == args.expected_actions
        ),
        "registered_source_closure": set(actions["source"].astype(str)) == REGISTERED_SOURCES,
        "targeted_train_query_scale": targeted_audit["unique_anchor_queries"] >= args.minimum_train_queries,
        "targeted_train_event_scale": targeted_audit["triplet_events"] >= args.minimum_train_events,
        "validation_event_scale": validation_audit["triplet_events"] >= args.minimum_validation_events,
        "validation_formula_disjoint": validation_audit["train_formula_overlap"] == 0,
        "every_event_unique_by_query_negative_molecule": targeted_audit["triplet_events"] == targeted_audit["unique_query_negative_pairs"],
        "all_actions_mapped_to_targeted_relations": len(targeted_alias) == len(actions),
        "action_tensor_never_enters_model_input": True,
        "all_hdf5_identity_edges_verified": all(
            body["identity_contract_passed"] for body in identity_audit.values()
        ),
        "outer_held_formulas_excluded_from_training": bool(
            np.all(training_formula_folds != int(args.outer_fold))
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Noise action-hard native triplet gates failed: {gates}")
    report = {
        "status": "NOISE_ACTION_HARD_NATIVE_TRIPLETS_COMPLETE",
        "version": NOISE_ACTION_HARD_NATIVE_VERSION,
        "training_runtime": "tasks/train_noise_reference_native.py (Noise-owned copy of the native DreaMS interface)",
        "custom_component": "Noise-selected hard-negative curriculum only",
        "triplet_semantics": "measured clean query / measured same-identity positive / measured false-candidate negative",
        "dose_unit": "one native event per unique targeted query-negative molecule relation",
        "replication_design": "same frozen curriculum, two preregistered independent seeds",
        "qualified_actions": int(len(actions)),
        "qualified_source_counts": dict(sorted(Counter(actions["source"].astype(str)).items())),
        "targeted_curriculum": targeted_audit,
        "validation": validation_audit,
        "identity_audit": identity_audit,
        "gates": gates,
        "provenance": provenance,
        "claim_limit": "Triplet construction only; no shared-embedding gain is claimed before frozen evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="noise_action_hard_native_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool_targeted.npz", **targeted)
        np.savez_compressed(temporary / "val_pool.npz", **validation)
        actions.to_csv(temporary / "selected_actions.csv.gz", index=False)
        alias.to_csv(temporary / "action_aliases.csv.gz", index=False)
        report["output_artifacts"] = {
            name: sha256_file(temporary / name)
            for name in (
                "train_pool_targeted.npz", "val_pool.npz",
                "selected_actions.csv.gz", "action_aliases.csv.gz",
            )
        }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
