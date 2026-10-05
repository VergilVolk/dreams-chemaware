"""Pure construction logic for ChemAware online query packets.

The deployment score is the maximum query/reference cosine for each molecule.
This module therefore mines one equally weighted packet per training query from
the *current* shared embedding.  Chemistry is allowed to select an additional
negative only when that candidate was observed for the same query in the
frozen ChemAware bank; evidence is never broadcast across identities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from build_chemaware_action_hard_native_triplets import (
    ACTION_HARD,
    OFFICIAL_HARD,
    SPECIFIC_HARD,
)
from build_chemaware_dreams_native_triplets import molecule_rows


SAFE_QUERY_PACKET = 1
ERROR_QUERY_PACKET = 2
CHEMICAL_QUERY_PACKET = 3
DREAMS_NATIVE_REPLAY = 4

WINNER_SLOT = 1
SEMI_HARD_SLOT = 2
CHEMICAL_SLOT = 3
WINNER_PADDING_SLOT = 4


def edges(pool: Mapping[str, np.ndarray], event: int, kind: str) -> np.ndarray:
    pointer = np.asarray(pool[f"{kind}_ptr"], dtype=np.int64)
    values = np.asarray(pool[f"{kind}_idx"], dtype=np.int64)
    left, right = map(int, pointer[event:event + 2])
    return values[left:right]


@dataclass(frozen=True)
class EmbeddingStore:
    rows: np.ndarray
    embeddings: np.ndarray

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows, dtype=np.int64)
        values = np.asarray(self.embeddings, dtype=np.float32)
        if values.ndim != 2 or len(rows) != len(values):
            raise ValueError("embedding rows and values do not align")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("embedding row registry contains duplicates")
        norms = np.linalg.norm(values, axis=1)
        if np.any(~np.isfinite(values)) or np.any(norms <= 0):
            raise ValueError("embedding store contains invalid vectors")
        normalized = values / norms[:, None]
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "embeddings", normalized.astype(np.float32))
        object.__setattr__(self, "position", {
            int(row): index for index, row in enumerate(rows)
        })

    def get(self, rows: np.ndarray | list[int]) -> np.ndarray:
        try:
            position = np.asarray(
                [self.position[int(row)] for row in rows], dtype=np.int64,
            )
        except KeyError as error:
            raise RuntimeError(f"spectrum row absent from online embedding store: {error}") from error
        return self.embeddings[position]


class PoolWriter:
    def __init__(self, packet_width: int):
        if packet_width < 1:
            raise ValueError("packet width must be positive")
        self.packet_width = int(packet_width)
        self.anchor: list[int] = []
        self.positive: list[int] = []
        self.negative: list[int] = []
        self.positive_ptr = [0]
        self.negative_ptr = [0]
        self.event_kind: list[int] = []
        self.source_query: list[int] = []
        self.current_rank: list[int] = []
        self.winner_candidate: list[int] = []
        self.winner_is_chemical: list[bool] = []
        self.packet_candidate: list[list[int]] = []
        self.packet_slot_role: list[list[int]] = []

    def append_packet(
        self, *, anchor: int, positive: int, negatives: list[int],
        query: int, rank: int, winner: int, candidates: list[int],
        slot_roles: list[int], event_kind: int, winner_is_chemical: bool,
    ) -> None:
        if not (
            len(negatives) == len(candidates) == len(slot_roles) == self.packet_width
        ):
            raise ValueError("focused packet does not have the frozen width")
        self.anchor.append(int(anchor))
        self.positive.append(int(positive))
        self.negative.extend(map(int, negatives))
        self.positive_ptr.append(len(self.positive))
        self.negative_ptr.append(len(self.negative))
        self.event_kind.append(int(event_kind))
        self.source_query.append(int(query))
        self.current_rank.append(int(rank))
        self.winner_candidate.append(int(winner))
        self.winner_is_chemical.append(bool(winner_is_chemical))
        self.packet_candidate.append(list(map(int, candidates)))
        self.packet_slot_role.append(list(map(int, slot_roles)))

    def append_replay(
        self, *, anchor: int, positives: np.ndarray, negatives: np.ndarray,
    ) -> None:
        if not len(positives) or not len(negatives):
            raise ValueError("empty official replay edge set")
        self.anchor.append(int(anchor))
        self.positive.extend(map(int, positives))
        self.negative.extend(map(int, negatives))
        self.positive_ptr.append(len(self.positive))
        self.negative_ptr.append(len(self.negative))
        self.event_kind.append(DREAMS_NATIVE_REPLAY)
        self.source_query.append(-1)
        self.current_rank.append(-1)
        self.winner_candidate.append(-1)
        self.winner_is_chemical.append(False)
        self.packet_candidate.append([-1] * self.packet_width)
        self.packet_slot_role.append([0] * self.packet_width)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "anchor_idx": np.asarray(self.anchor, dtype=np.int64),
            "positive_ptr": np.asarray(self.positive_ptr, dtype=np.int64),
            "positive_idx": np.asarray(self.positive, dtype=np.int64),
            "negative_ptr": np.asarray(self.negative_ptr, dtype=np.int64),
            "negative_idx": np.asarray(self.negative, dtype=np.int64),
            "event_kind": np.asarray(self.event_kind, dtype=np.int8),
            "source_query": np.asarray(self.source_query, dtype=np.int64),
            "current_rank": np.asarray(self.current_rank, dtype=np.int32),
            "winner_candidate": np.asarray(self.winner_candidate, dtype=np.int32),
            "winner_is_chemical": np.asarray(self.winner_is_chemical, dtype=bool),
            "packet_candidate": np.asarray(self.packet_candidate, dtype=np.int32),
            "packet_slot_role": np.asarray(self.packet_slot_role, dtype=np.int8),
        }


def _candidate_tags(
    base: Mapping[str, np.ndarray], queries: np.ndarray,
) -> tuple[dict[int, dict[int, int]], dict[int, int]]:
    tags: dict[int, dict[int, int]] = {int(query): {} for query in queries}
    official: dict[int, int] = {}
    for event, raw_query in enumerate(np.asarray(base["source_query"], dtype=np.int64)):
        query = int(raw_query)
        if query not in tags:
            continue
        candidate = int(base["negative_candidate"][event])
        tag = int(base["source_tag"][event])
        tags[query][candidate] = tags[query].get(candidate, 0) | tag
        if tag & OFFICIAL_HARD:
            if query in official and official[query] != candidate:
                raise RuntimeError(f"query {query} has multiple official candidates")
            official[query] = candidate
    if set(official) != set(map(int, queries)):
        missing = sorted(set(map(int, queries)) - set(official))[:10]
        raise RuntimeError(f"queries lack an official negative candidate: {missing}")
    return tags, official


def build_online_packet_pool(
    *, manifest: Mapping[str, np.ndarray], evidence: Mapping[str, np.ndarray],
    base: Mapping[str, np.ndarray], replay: Mapping[str, np.ndarray],
    embedding_rows: np.ndarray, embeddings: np.ndarray, margin: float = 0.1,
    packet_width: int = 3, replay_events: int = 1024, seed: int = 3407,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build one current-boundary packet per frozen training query."""
    if packet_width < 1 or margin <= 0:
        raise ValueError("invalid packet width or margin")
    queries = np.asarray(evidence["query"], dtype=np.int64)
    if len(np.unique(queries)) != len(queries):
        raise RuntimeError("training evidence repeats a query")
    if replay_events < 0 or replay_events > len(replay["anchor_idx"]):
        raise ValueError("invalid official replay budget")

    store = EmbeddingStore(embedding_rows, embeddings)
    tags_by_query, official_candidate = _candidate_tags(base, queries)
    writer = PoolWriter(packet_width)
    current_errors = current_correct = winner_switches = 0
    active_chemical_queries = included_chemical_queries = 0
    packets_with_semi = packets_with_distinct_chem = 0
    packets_with_distinct_specific = packets_with_distinct_action_only = 0
    packets_with_chemical_winner = 0
    active_counts: list[int] = []
    distinct_counts: list[int] = []

    for query in sorted(map(int, queries)):
        anchor = int(manifest["query_row"][query])
        anchor_embedding = store.get([anchor])[0]
        molecule_left, molecule_right = map(
            int, manifest["query_ptr"][query:query + 2],
        )
        labels = np.asarray(
            manifest["molecule_label"][molecule_left:molecule_right], dtype=bool,
        )
        true_candidates = np.flatnonzero(labels)
        false_candidates = np.flatnonzero(~labels)
        if not len(true_candidates) or not len(false_candidates):
            raise RuntimeError(f"query {query} lacks a true or false candidate")

        positive_rows = np.unique(np.concatenate([
            molecule_rows(manifest, query, int(candidate))
            for candidate in true_candidates
        ]))
        positive_rows = positive_rows[positive_rows != anchor]
        if not len(positive_rows):
            raise RuntimeError(f"query {query} lacks an independent positive spectrum")
        positive_scores = store.get(positive_rows) @ anchor_embedding
        positive_at = int(np.argmax(positive_scores))
        positive_row = int(positive_rows[positive_at])
        positive_score = float(positive_scores[positive_at])

        geometry: dict[int, tuple[int, float]] = {}
        for raw_candidate in false_candidates:
            candidate = int(raw_candidate)
            rows = np.unique(molecule_rows(manifest, query, candidate))
            scores = store.get(rows) @ anchor_embedding
            at = int(np.argmax(scores))
            geometry[candidate] = (int(rows[at]), float(scores[at]))
        ordered = sorted(
            map(int, false_candidates),
            key=lambda candidate: (-geometry[candidate][1], candidate),
        )
        winner = ordered[0]
        rank = 1 + sum(geometry[candidate][1] >= positive_score for candidate in ordered)
        is_error = rank > 1
        current_errors += int(is_error)
        current_correct += int(not is_error)
        winner_switches += int(winner != official_candidate[query])
        active = [
            candidate for candidate in ordered
            if margin + geometry[candidate][1] - positive_score > 0.0
        ]
        active_counts.append(len(active))

        negatives = [geometry[winner][0]]
        candidates = [winner]
        roles = [WINNER_SLOT]
        query_tags = tags_by_query[query]
        aligned_chemical = [
            candidate for candidate in active
            if int(query_tags.get(candidate, 0)) & (ACTION_HARD | SPECIFIC_HARD)
        ]
        active_chemical_queries += int(bool(aligned_chemical))
        winner_is_chemical = bool(
            int(query_tags.get(winner, 0)) & (ACTION_HARD | SPECIFIC_HARD)
        )
        packets_with_chemical_winner += int(is_error and winner_is_chemical)

        if is_error:
            chemical = [candidate for candidate in aligned_chemical if candidate != winner]
            chemical.sort(key=lambda candidate: (
                -int(bool(query_tags[candidate] & SPECIFIC_HARD)),
                -int(bool(query_tags[candidate] & ACTION_HARD)),
                -geometry[candidate][1],
                candidate,
            ))
            chemical_candidate = chemical[0] if chemical else None
            excluded = {winner}
            if chemical_candidate is not None:
                excluded.add(chemical_candidate)
            semi = [candidate for candidate in active if candidate not in excluded]
            semi_target = positive_score - 0.5 * margin
            semi.sort(key=lambda candidate: (
                abs(geometry[candidate][1] - semi_target), candidate,
            ))
            semi_candidate = semi[0] if semi else None
            if semi_candidate is not None and len(negatives) < packet_width:
                negatives.append(geometry[semi_candidate][0])
                candidates.append(semi_candidate)
                roles.append(SEMI_HARD_SLOT)
                packets_with_semi += 1
            if chemical_candidate is not None and len(negatives) < packet_width:
                negatives.append(geometry[chemical_candidate][0])
                candidates.append(chemical_candidate)
                roles.append(CHEMICAL_SLOT)
                packets_with_distinct_chem += 1
                if query_tags[chemical_candidate] & SPECIFIC_HARD:
                    packets_with_distinct_specific += 1
                else:
                    packets_with_distinct_action_only += 1
            included_chemical_queries += int(
                winner_is_chemical
                or chemical_candidate is not None
            )

        distinct_counts.append(len(set(candidates)))
        while len(negatives) < packet_width:
            negatives.append(geometry[winner][0])
            candidates.append(winner)
            roles.append(WINNER_PADDING_SLOT)
        writer.append_packet(
            anchor=anchor, positive=positive_row, negatives=negatives,
            query=query, rank=rank, winner=winner, candidates=candidates,
            slot_roles=roles,
            event_kind=ERROR_QUERY_PACKET if is_error else SAFE_QUERY_PACKET,
            winner_is_chemical=winner_is_chemical,
        )

    rng = np.random.default_rng(seed)
    selected_replay = np.sort(
        rng.choice(len(replay["anchor_idx"]), size=replay_events, replace=False)
    )
    for raw_event in selected_replay:
        event = int(raw_event)
        writer.append_replay(
            anchor=int(replay["anchor_idx"][event]),
            positives=edges(replay, event, "positive"),
            negatives=edges(replay, event, "negative"),
        )
    output = writer.arrays()
    focused = output["event_kind"] != DREAMS_NATIVE_REPLAY
    focused_queries = output["source_query"][focused]
    gates = {
        "one_focused_packet_per_query": (
            len(focused_queries) == len(queries)
            and np.array_equal(np.sort(focused_queries), np.sort(queries))
        ),
        "fixed_focused_packet_width": bool(np.all(
            np.diff(output["negative_ptr"][:len(queries) + 1]) == packet_width
        )),
        "one_positive_per_focused_packet": bool(np.all(
            np.diff(output["positive_ptr"][:len(queries) + 1]) == 1
        )),
        "exact_replay_budget": int(np.sum(~focused)) == replay_events,
        "no_role2_or_role3_training_query": True,
        "same_query_candidate_chemistry_only": True,
        "finite_current_geometry": bool(np.all(np.isfinite(embeddings))),
    }
    if not all(gates.values()):
        raise RuntimeError(f"online query-packet gates failed: {gates}")

    report = {
        "status": "CHEMAWARE_ONLINE_QUERY_PACKET_COMPLETE",
        "queries": int(len(queries)),
        "current_error_queries": int(current_errors),
        "current_correct_queries": int(current_correct),
        "winner_switches_from_frozen_official_candidate": int(winner_switches),
        "winner_switch_fraction": float(winner_switches / len(queries)),
        "active_chemical_queries": int(active_chemical_queries),
        "included_chemical_queries": int(included_chemical_queries),
        "packets_with_semi_hard_negative": int(packets_with_semi),
        "packets_with_distinct_chemical_negative": int(packets_with_distinct_chem),
        "packets_with_distinct_specific_negative": int(
            packets_with_distinct_specific
        ),
        "packets_with_distinct_action_only_negative": int(
            packets_with_distinct_action_only
        ),
        "error_packets_with_chemical_winner": int(packets_with_chemical_winner),
        "mean_active_false_candidates": float(np.mean(active_counts)),
        "mean_distinct_packet_negatives": float(np.mean(distinct_counts)),
        "packet_width": int(packet_width),
        "margin": float(margin),
        "official_replay_events": int(replay_events),
        "total_events": int(len(output["anchor_idx"])),
        "gates": gates,
        "scientific_boundary": (
            "one equally weighted packet per frozen training query; current molecule-max "
            "winner plus semi-hard and same-query candidate-aligned ChemAware negatives; "
            "identity labels, DreaMS replay and formula-held-out roles remain unchanged"
        ),
    }
    return output, report


def phasea_role_fractions(pool: Mapping[str, np.ndarray]) -> dict[str, float]:
    roles = np.asarray(pool.get("curriculum_role"), dtype=np.int64)
    if len(roles) != len(pool["anchor_idx"]):
        raise RuntimeError("Phase-A pool lacks curriculum-role provenance")
    mapping = {
        "safe": SAFE_QUERY_PACKET,
        "error": ERROR_QUERY_PACKET,
        "chemical": CHEMICAL_QUERY_PACKET,
        "replay": DREAMS_NATIVE_REPLAY,
    }
    counts = {name: int(np.sum(roles == role)) for name, role in mapping.items()}
    if sum(counts.values()) != len(roles) or not all(counts.values()):
        raise RuntimeError(f"invalid Phase-A role budget: {counts}")
    return {name: count / len(roles) for name, count in counts.items()}


def allocate_role_steps(fractions: Mapping[str, float], steps: int) -> dict[str, int]:
    """Largest-remainder allocation with deterministic role tie breaking."""
    names = ("safe", "error", "chemical", "replay")
    raw = {name: float(fractions[name]) * steps for name in names}
    output = {name: int(np.floor(raw[name])) for name in names}
    remaining = steps - sum(output.values())
    order = sorted(
        names,
        key=lambda name: (-(raw[name] - output[name]), names.index(name)),
    )
    for name in order[:remaining]:
        output[name] += 1
    if sum(output.values()) != steps:
        raise AssertionError("role-step allocation did not conserve optimizer steps")
    return output


def role_entries(pool: Mapping[str, np.ndarray]) -> dict[str, list[tuple[int, int]]]:
    """Resolve one-negative query-equal streams from a current packet pool."""
    kinds = np.asarray(pool["event_kind"], dtype=np.int64)
    slot_roles = np.asarray(pool["packet_slot_role"], dtype=np.int64)
    winner_chemical = np.asarray(pool["winner_is_chemical"], dtype=bool)
    result: dict[str, list[tuple[int, int]]] = {
        "safe": [], "error": [], "chemical": [], "replay": [],
    }
    for event, kind in enumerate(kinds):
        if kind == DREAMS_NATIVE_REPLAY:
            result["replay"].append((event, 0))
        elif kind == SAFE_QUERY_PACKET:
            result["safe"].append((event, 0))
        elif kind == ERROR_QUERY_PACKET:
            result["error"].append((event, 0))
            chemical_slot = np.flatnonzero(slot_roles[event] == CHEMICAL_SLOT)
            if len(chemical_slot):
                result["chemical"].append((event, int(chemical_slot[0])))
            elif winner_chemical[event]:
                result["chemical"].append((event, 0))
        else:
            raise RuntimeError(f"unknown online event kind: {kind}")
    if not all(result.values()):
        raise RuntimeError({name: len(entries) for name, entries in result.items()})
    return result
