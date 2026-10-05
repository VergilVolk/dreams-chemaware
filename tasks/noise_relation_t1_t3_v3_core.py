"""Query-balanced exact-bridge helpers for Noise relation T1/T3 V3.

V3 keeps the clean relation-complete objective, but an action-bearing query
receives exactly one registered action bridge in the same query objective.
Action multiplicity therefore changes coverage across rotations, never the
number of optimizer updates assigned to a query.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F


def _query_span(
    corpus: dict[str, np.ndarray], position: int,
) -> tuple[np.ndarray, np.ndarray]:
    a0, a1 = map(int, corpus["action_ptr"][position:position + 2])
    m0, m1 = map(int, corpus["molecule_ptr"][position:position + 2])
    r0 = int(corpus["reference_ptr"][m0])
    r1 = int(corpus["reference_ptr"][m1])
    return (
        np.asarray(corpus["action_index"][a0:a1], dtype=np.int64),
        np.asarray(corpus["reference_row"][r0:r1], dtype=np.int64),
    )


def query_balanced_packed_batches(
    corpus: dict[str, np.ndarray],
    events: dict[str, np.ndarray],
    selected_by_query: dict[int, int],
    *,
    seed: int,
    maximum_spectra: int,
    maximum_queries: int,
) -> tuple[list[list[int]], dict[str, int | bool]]:
    """Pack each query once while accounting for its selected exact bridge."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(corpus["query_index"]))
    batches: list[list[int]] = []
    current: list[int] = []
    current_rows: set[int] = set()
    current_actions: set[int] = set()

    def resources(position: int) -> tuple[set[int], set[int]]:
        _unused, references = _query_span(corpus, position)
        rows = {int(corpus["query_row"][position]), *map(int, references)}
        actions: set[int] = set()
        query = int(corpus["query_index"][position])
        event = selected_by_query.get(query)
        if event is not None:
            rows.add(int(events["positive_row"][event]))
            rows.add(int(events["negative_row"][event]))
            actions.add(int(events["action_index"][event]))
        return rows, actions

    for value in order:
        position = int(value)
        rows, actions = resources(position)
        if len(rows) + len(actions) > maximum_spectra:
            raise RuntimeError(f"V3 query {position} exceeds spectrum budget")
        merged_rows = current_rows | rows
        merged_actions = current_actions | actions
        if current and (
            len(current) >= maximum_queries
            or len(merged_rows) + len(merged_actions) > maximum_spectra
        ):
            batches.append(current)
            current = []
            current_rows = set()
            current_actions = set()
            merged_rows, merged_actions = rows, actions
        current.append(position)
        current_rows = merged_rows
        current_actions = merged_actions
    if current:
        batches.append(current)

    flattened = [position for batch in batches for position in batch]
    if sorted(flattened) != list(range(len(corpus["query_index"]))):
        raise RuntimeError("V3 packed schedule lost or duplicated a query")
    action_query_count = sum(
        int(int(corpus["query_index"][position]) in selected_by_query)
        for position in flattened
    )
    report: dict[str, int | bool] = {
        "queries": int(len(flattened)),
        "optimizer_steps": int(len(batches)),
        "action_queries": int(action_query_count),
        "every_query_exactly_once": True,
        "one_selected_action_per_action_query": bool(
            action_query_count == len(selected_by_query)
        ),
        "action_multiplicity_changes_optimizer_dose": False,
    }
    if not report["one_selected_action_per_action_query"]:
        raise RuntimeError(f"V3 schedule lost an action query: {report}")
    return batches, report


def _query_seed(seed: int, query: int) -> int:
    payload = f"noise-t1t3-v3|{int(seed)}|{int(query)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def query_balanced_event_rotation(
    event_queries: np.ndarray,
    event_actions: np.ndarray,
    eligible_queries: np.ndarray,
    *,
    seed: int,
    rotation: int = 0,
) -> tuple[dict[int, int], dict[str, int | float | bool | str]]:
    """Select one exact action event per action query, coverage-first.

    The permutation inside a query is stable for ``seed``.  Consecutive
    rotations visit every event for that query before recycling.  Queries
    without actions remain ordinary clean T1/T3 queries.
    """
    queries = np.asarray(event_queries, dtype=np.int64)
    actions = np.asarray(event_actions, dtype=np.int64)
    eligible = np.asarray(eligible_queries, dtype=np.int64)
    if queries.ndim != 1 or actions.shape != queries.shape:
        raise RuntimeError("V3 action event ledgers are not aligned vectors")
    if eligible.ndim != 1 or len(np.unique(eligible)) != len(eligible):
        raise RuntimeError("V3 eligible query ledger is not unique")
    if rotation < 0:
        raise ValueError("V3 rotation must be nonnegative")
    eligible_set = set(map(int, eligible))
    by_query: dict[int, list[int]] = {}
    for event, query in enumerate(queries):
        value = int(query)
        if value not in eligible_set:
            raise RuntimeError(f"V3 action query {value} is absent from clean corpus")
        by_query.setdefault(value, []).append(int(event))
    if not by_query:
        raise RuntimeError("V3 received no exact action events")

    selected: dict[int, int] = {}
    for query in sorted(by_query):
        indices = by_query[query]
        # Sort semantically before seeded permutation so archive row order is
        # not a hidden source of selection.
        indices.sort(key=lambda index: (int(actions[index]), index))
        rng = np.random.default_rng(_query_seed(seed, query))
        order = np.asarray(indices, dtype=np.int64)
        rng.shuffle(order)
        selected[query] = int(order[rotation % len(order)])

    counts = np.asarray([len(value) for value in by_query.values()], dtype=np.int64)
    selected_events = list(selected.values())
    selected_ledger = np.asarray(
        [(query, selected[query]) for query in sorted(selected)], dtype=np.int64,
    )
    report: dict[str, int | float | bool | str] = {
        "event_count": int(len(queries)),
        "action_query_count": int(len(by_query)),
        "selected_event_count": int(len(selected_events)),
        "one_event_per_action_query": bool(len(selected_events) == len(by_query)),
        "selected_events_unique": bool(len(set(selected_events)) == len(selected_events)),
        "minimum_events_per_query": int(counts.min()),
        "maximum_events_per_query": int(counts.max()),
        "mean_events_per_query": float(counts.mean()),
        "rotation": int(rotation),
        "selected_query_event_sha256": hashlib.sha256(
            np.ascontiguousarray(selected_ledger).tobytes()
        ).hexdigest(),
        "query_dose_is_independent_of_action_multiplicity": True,
    }
    if not report["one_event_per_action_query"] or not report["selected_events_unique"]:
        raise RuntimeError(f"V3 query-balanced selection failed: {report}")
    return selected, report


def exact_cosine_triplet_hinge(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    """Native cosine triplet hinge, returned per event."""
    if anchor.ndim != 2 or positive.shape != anchor.shape or negative.shape != anchor.shape:
        raise RuntimeError("V3 exact triplet tensors must be aligned matrices")
    if margin <= 0:
        raise ValueError("V3 triplet margin must be positive")
    a = F.normalize(anchor.float(), dim=1)
    p = F.normalize(positive.float(), dim=1)
    n = F.normalize(negative.float(), dim=1)
    return torch.relu(float(margin) + torch.sum(a * n, dim=1) - torch.sum(a * p, dim=1))


def combine_query_losses(
    clean_losses: list[torch.Tensor],
    action_losses: Mapping[int, torch.Tensor],
    *,
    action_fraction: float = 0.5,
) -> torch.Tensor:
    """Average once over queries; an action query remains one dose unit."""
    if not clean_losses:
        raise RuntimeError("V3 cannot combine an empty clean query batch")
    if not 0.0 < action_fraction < 1.0:
        raise ValueError("V3 action fraction must lie strictly inside (0, 1)")
    invalid = set(action_losses) - set(range(len(clean_losses)))
    if invalid:
        raise RuntimeError(f"V3 action losses reference invalid query positions: {invalid}")
    combined = []
    for position, clean in enumerate(clean_losses):
        if clean.ndim != 0:
            raise RuntimeError("V3 clean query loss must be scalar")
        action = action_losses.get(position)
        if action is None:
            combined.append(clean)
        else:
            if action.ndim != 0:
                raise RuntimeError("V3 action query loss must be scalar")
            combined.append((1.0 - action_fraction) * clean + action_fraction * action)
    return torch.stack(combined).mean()
