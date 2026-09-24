"""Build reference-aligned ChemAware triplets for native DreaMS training.

The deployed retrieval score of a candidate molecule is the maximum cosine
over all of its reference spectra.  The stock DreaMS dataset, however, samples
one reference uniformly from every event.  A molecule-level hard negative can
therefore yield an easy spectrum pair and no useful gradient.  This builder
keeps the native DreaMS dataset, model and loss, but makes the curriculum agree
with deployed retrieval:

* two arm-independent checkpoint-hard candidate molecules per query;
* one distinct chemistry-selected candidate (or a hard fallback);
* up to K highest-scoring reference spectra expanded into explicit events;
* all same-identity positive references retained for native random sampling.

No query-negative-reference triplet is duplicated.  Formula role 2 freezes the
chemical recipe; roles 0--1 train, role 3 evaluates, and role 4 is never read.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_action_hard_native_triplets import load_npz, true_locals
from build_chemaware_dreams_native_triplets import audit_identity_edges, molecule_rows
from build_chemaware_residual_native_triplets import (
    cache_arrays,
    metric_cache,
    positions,
    strict_negative_candidates,
)
from chemaware_native_triplet_mining_core import (
    directional_masks,
    recipe_dict,
    select_recipe,
    validate_evidence,
)


ROOT = Path(__file__).resolve().parents[1]

PRIMARY_HARD = 1
SECONDARY_HARD = 2
CHEMICAL_HARD = 4
STRICT_SPECIFIC = 8
HARD_FALLBACK = 16
RETRIEVAL_MARGIN_VIOLATION = 32
SAFE_SENTINEL = 64


@dataclass(frozen=True)
class QueryGeometry:
    query_row: int
    positive_rows: np.ndarray
    positive_scores: np.ndarray
    molecule_scores: np.ndarray
    negative_rows: tuple[np.ndarray, ...]
    negative_scores: tuple[np.ndarray, ...]
    activation_probability: np.ndarray
    mean_hinge: np.ndarray


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--candidates-per-query", type=int, default=3)
    parser.add_argument("--chemical-candidates-per-query", type=int, default=1)
    parser.add_argument("--negative-references-per-candidate", type=int, default=2)
    parser.add_argument(
        "--explicit-positive-references-per-negative", type=int, default=0,
        help=(
            "Zero preserves native random positive sampling; two emits one boundary "
            "positive and one hard positive for every active negative reference."
        ),
    )
    parser.add_argument("--max-active-spectrum-events-per-query", type=int, default=4)
    parser.add_argument("--chemical-hardness-window", type=float, default=0.50)
    parser.add_argument(
        "--min-chemical-activation-probability", type=float, default=0.05,
        help="Minimum fraction of positive-reference pairs with nonzero native hinge loss.",
    )
    parser.add_argument("--min-train-queries", type=int, default=3500)
    parser.add_argument("--min-candidate-events", type=int, default=4500)
    parser.add_argument("--min-spectrum-events", type=int, default=8000)
    parser.add_argument("--min-train-formulas", type=int, default=2000)
    parser.add_argument("--min-chemical-candidates", type=int, default=300)
    parser.add_argument("--min-chemical-spectrum-events", type=int, default=1000)
    return parser.parse_args()


def _query_geometry(
    manifest: Mapping[str, np.ndarray], query: int, cache_rows: np.ndarray,
    embedding: np.ndarray, margin: float, negative_reference_cap: int,
) -> QueryGeometry:
    mleft, mright = map(int, manifest["query_ptr"][query:query + 2])
    pleft = int(manifest["molecule_ptr"][mleft])
    pright = int(manifest["molecule_ptr"][mright])
    all_rows = np.asarray(manifest["pair_candidate_row"][pleft:pright], dtype=np.int64)
    query_row = int(manifest["query_row"][query])
    query_z = np.asarray(embedding[positions(cache_rows, np.asarray([query_row]))[0]])
    all_z = np.asarray(embedding[positions(cache_rows, all_rows)])
    all_scores = all_z @ query_z
    local_ptr = np.asarray(manifest["molecule_ptr"][mleft:mright + 1], dtype=np.int64) - pleft
    molecule_scores = np.maximum.reduceat(all_scores, local_ptr[:-1]).astype(np.float32)

    truth = true_locals(manifest, query)
    positive_rows = np.unique(np.concatenate([
        molecule_rows(manifest, query, int(candidate)) for candidate in truth
    ]))
    positive_rows = positive_rows[positive_rows != query_row]
    if not len(positive_rows):
        raise RuntimeError(f"query {query} has no distinct positive reference")
    positive_z = np.asarray(embedding[positions(cache_rows, positive_rows)])
    positive_scores = (positive_z @ query_z).astype(np.float32)

    negative_rows: list[np.ndarray] = []
    negative_scores: list[np.ndarray] = []
    activation_probability = np.zeros(mright - mleft, dtype=np.float32)
    mean_hinge = np.zeros(mright - mleft, dtype=np.float32)
    for candidate in range(mright - mleft):
        rows = all_rows[local_ptr[candidate]:local_ptr[candidate + 1]]
        scores = all_scores[local_ptr[candidate]:local_ptr[candidate + 1]]
        order = np.argsort(-scores, kind="stable")[:negative_reference_cap]
        chosen_rows = np.asarray(rows[order], dtype=np.int64)
        chosen_scores = np.asarray(scores[order], dtype=np.float32)
        negative_rows.append(chosen_rows)
        negative_scores.append(chosen_scores)
        hinge = np.maximum(
            float(margin) + chosen_scores[:, None] - positive_scores[None, :], 0.0,
        )
        activation_probability[candidate] = float(np.mean(hinge > 0.0))
        mean_hinge[candidate] = float(np.mean(hinge))
    return QueryGeometry(
        query_row=query_row,
        positive_rows=positive_rows,
        positive_scores=positive_scores,
        molecule_scores=molecule_scores,
        negative_rows=tuple(negative_rows),
        negative_scores=tuple(negative_scores),
        activation_probability=activation_probability,
        mean_hinge=mean_hinge,
    )


def _base_order(false: np.ndarray, geometry: QueryGeometry) -> np.ndarray:
    return np.asarray(sorted(
        map(int, false),
        key=lambda candidate: (
            float(geometry.molecule_scores[candidate]),
            float(geometry.mean_hinge[candidate]),
            float(geometry.activation_probability[candidate]),
            -candidate,
        ),
        reverse=True,
    ), dtype=np.int64)


def chemical_rejection_order_key(
    slot: int, row: int, center: int, metrics: Mapping[str, np.ndarray],
    current_score: float, strict: bool,
) -> tuple[float, ...]:
    """Rank false candidates by *directed* chemical rejection.

    A false candidate is useful as a ChemAware negative when the center arm
    rejects it more strongly than the median of the other three arms.  Using
    an absolute center-vs-null contrast is invalid here: it also rewards a
    center arm that promotes the false candidate and makes the correct and
    content-permuted arms select the same molecules.  Strict directional
    correction/protection events retain first priority.
    """
    arms = metrics["action_top_fraction"].shape[0]
    others = np.asarray([index for index in range(arms) if index != center])

    def rejection(name: str) -> float:
        values = metrics[name]
        return float(np.median(values[others, row, slot]) - values[center, row, slot])

    contrasts = (
        rejection("action_top_fraction"),
        rejection("action_largest_region_fraction"),
        rejection("action_best_advantage_over_baseline"),
        rejection("candidate_rule_max"),
        rejection("delta_rule_max"),
    )
    positive = tuple(max(0.0, value) for value in contrasts)
    rank = metrics["candidate_rule_rank_fraction"]
    return (
        float(strict),
        float(sum(value > 0.0 for value in contrasts)),
        *positive,
        -float(rank[center, row, slot]),
        -float(metrics["candidate_rule_max"][center, row, slot]),
        -float(metrics["candidate_rule_top2_mean"][center, row, slot]),
        -float(metrics["delta_rule_max"][center, row, slot]),
        -float(metrics["delta_rule_top2_mean"][center, row, slot]),
        -float(metrics["action_top_fraction"][center, row, slot]),
        -float(metrics["action_largest_region_fraction"][center, row, slot]),
        -float(metrics["action_same_neighbor_fraction"][center, row, slot]),
        float(current_score),
    )


def select_candidate_slots(
    evidence: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    metrics: Mapping[str, np.ndarray], row: int, center: int,
    correction: np.ndarray, protection: np.ndarray, geometry: QueryGeometry,
    candidates_per_query: int, chemical_hardness_window: float,
    min_chemical_activation_probability: float, chemical_candidates_per_query: int,
) -> list[tuple[int, int]]:
    """Choose fixed base slots and at most one arm-specific chemical slot."""
    query = int(np.asarray(evidence["query"])[row])
    truth = set(map(int, true_locals(manifest, query)))
    false = np.asarray([
        candidate for candidate in range(len(geometry.molecule_scores))
        if candidate not in truth
    ], dtype=np.int64)
    if not len(false):
        raise RuntimeError(f"query {query} has no false molecule candidate")
    ordered = _base_order(false, geometry)
    budget = min(int(candidates_per_query), len(ordered))
    base_count = min(2, budget)
    selected: dict[int, int] = {}
    if base_count:
        selected[int(ordered[0])] = PRIMARY_HARD
    if base_count > 1:
        selected[int(ordered[1])] = SECONDARY_HARD

    if len(selected) < budget:
        strict = strict_negative_candidates(evidence, row, center, correction, protection)
        valid = np.asarray(evidence["valid"], dtype=bool)[row]
        proposed = np.asarray(evidence["proposed_candidate"], dtype=np.int64)[row]
        candidate_slot = {
            int(proposed[slot]): int(slot)
            for slot in np.flatnonzero(valid)
            if int(proposed[slot]) not in truth
        }
        hardest_score = float(geometry.molecule_scores[int(ordered[0])])
        eligible = [
            candidate for candidate in candidate_slot
            if float(geometry.molecule_scores[candidate])
            >= hardest_score - float(chemical_hardness_window)
            and float(geometry.activation_probability[candidate])
            >= float(min_chemical_activation_probability)
        ]
        baseline = int(np.asarray(evidence["baseline_candidate"])[row])
        if (
            baseline in strict and baseline not in truth and baseline not in selected
            and float(geometry.molecule_scores[baseline])
            >= hardest_score - float(chemical_hardness_window)
            and float(geometry.activation_probability[baseline])
            >= float(min_chemical_activation_probability)
        ):
            eligible.append(baseline)
        eligible = sorted(set(eligible))
        if eligible:
            def key(candidate: int) -> tuple[float, ...]:
                if candidate in candidate_slot:
                    chemistry = chemical_rejection_order_key(
                        candidate_slot[candidate], row, center, metrics,
                        float(geometry.molecule_scores[candidate]), candidate in strict,
                    )
                else:
                    chemistry = (float(candidate in strict),) + (0.0,) * 15
                return (
                    *chemistry,
                    float(geometry.mean_hinge[candidate]),
                    float(geometry.activation_probability[candidate]),
                )
            chemical_budget = min(
                int(chemical_candidates_per_query), budget,
            )
            for chemical in sorted(eligible, key=key, reverse=True)[:chemical_budget]:
                tag = selected.get(int(chemical), 0) | CHEMICAL_HARD
                if chemical in strict:
                    tag |= STRICT_SPECIFIC
                selected[int(chemical)] = tag

    for candidate in ordered:
        if len(selected) >= budget:
            break
        candidate = int(candidate)
        if candidate not in selected:
            selected[candidate] = HARD_FALLBACK
    return sorted(
        selected.items(),
        key=lambda item: (-float(geometry.molecule_scores[item[0]]), item[0]),
    )


def retain_query_events(
    query_events: list[dict[str, float | int]], max_active_events: int,
) -> tuple[list[dict[str, float | int]], bool]:
    """Keep diverse active constraints or one nearest-negative safety sentinel."""
    if not query_events:
        raise RuntimeError("query produced no reference-level triplet candidates")
    active = [event for event in query_events if float(event["activation"]) > 0.0]
    if not active:
        sentinel = max(
            query_events,
            key=lambda event: (
                bool(int(event["tag"]) & PRIMARY_HARD),
                float(event["negative_score"]),
            ),
        )
        return [{**sentinel, "tag": int(sentinel["tag"]) | SAFE_SENTINEL}], True

    def priority(event: Mapping[str, float | int]) -> tuple[bool, bool, bool, bool, float, float]:
        tag = int(event["tag"])
        return (
            bool(tag & CHEMICAL_HARD), bool(tag & STRICT_SPECIFIC),
            bool(tag & PRIMARY_HARD), bool(tag & SECONDARY_HARD),
            float(event["hinge"]), float(event["negative_score"]),
        )

    best_by_candidate: dict[int, dict[str, float | int]] = {}
    for event in active:
        candidate = int(event["candidate"])
        if candidate not in best_by_candidate or priority(event) > priority(best_by_candidate[candidate]):
            best_by_candidate[candidate] = event
    retained = sorted(best_by_candidate.values(), key=priority, reverse=True)
    retained_ids = {id(event) for event in retained}
    remaining = [event for event in active if id(event) not in retained_ids]
    retained.extend(sorted(remaining, key=priority, reverse=True))
    return retained[:max_active_events], False


def explicit_positive_indices(
    positive_scores: np.ndarray, hinge: np.ndarray, cap: int,
) -> list[int]:
    """Select a boundary positive and a hard positive without duplication."""
    active = np.flatnonzero(np.asarray(hinge) > 0.0)
    if not len(active):
        return [int(np.argmax(positive_scores))]
    boundary = int(active[np.argmax(np.asarray(positive_scores)[active])])
    hard = int(active[np.argmin(np.asarray(positive_scores)[active])])
    chosen = [boundary]
    if hard != boundary:
        chosen.append(hard)
    remaining = sorted(
        set(map(int, active)) - set(chosen),
        key=lambda index: float(hinge[index]), reverse=True,
    )
    chosen.extend(remaining)
    return chosen[:cap]


def build_pool(
    evidence: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    geometry_by_query: Mapping[int, QueryGeometry], center: int, recipe,
    candidates_per_query: int, chemical_hardness_window: float,
    min_chemical_activation_probability: float, chemical_candidates_per_query: int,
    margin: float,
    max_active_spectrum_events_per_query: int,
    explicit_positive_references_per_negative: int,
) -> tuple[
    dict[str, np.ndarray], dict[str, object], set[tuple[int, int]],
    set[tuple[int, int, int, int]],
]:
    validate_evidence(evidence)
    correction, protection, _ = directional_masks(evidence, recipe, center)
    metrics = metric_cache(evidence)
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    source_query: list[int] = []
    negative_candidate: list[int] = []
    positive_reference_row: list[int] = []
    negative_reference_row: list[int] = []
    source_tag: list[int] = []
    activation_probability: list[float] = []
    mean_hinge: list[float] = []
    candidate_pairs: set[tuple[int, int]] = set()
    spectrum_triplets: set[tuple[int, int, int, int]] = set()
    selected_reference_edges = 0
    available_reference_edges = 0
    safe_sentinel_events = 0
    for row, query_value in enumerate(np.asarray(evidence["query"], dtype=np.int64)):
        query = int(query_value)
        geometry = geometry_by_query[query]
        chosen = select_candidate_slots(
            evidence, manifest, metrics, row, center, correction, protection,
            geometry, candidates_per_query, chemical_hardness_window,
            min_chemical_activation_probability, chemical_candidates_per_query,
        )
        query_events = []
        for candidate, tag in chosen:
            chosen_rows = geometry.negative_rows[candidate]
            chosen_scores = geometry.negative_scores[candidate]
            top_positive = float(np.max(geometry.positive_scores))
            for negative_row, negative_score in zip(chosen_rows, chosen_scores, strict=True):
                event_tag = int(tag)
                if float(margin) + float(negative_score) - top_positive > 0.0:
                    event_tag |= RETRIEVAL_MARGIN_VIOLATION
                hinge = np.maximum(
                    float(margin) + float(negative_score) - geometry.positive_scores, 0.0,
                )
                if explicit_positive_references_per_negative > 0:
                    # One boundary positive gives the smallest still-active
                    # perturbation; one hard positive transfers replicate
                    # invariance. A safe negative supplies only a sentinel
                    # possibility, later pruned unless the whole query is safe.
                    chosen_positive = explicit_positive_indices(
                        geometry.positive_scores, hinge,
                        explicit_positive_references_per_negative,
                    )
                    for positive_index in chosen_positive:
                        query_events.append({
                            "candidate": int(candidate),
                            "positive_row": int(geometry.positive_rows[positive_index]),
                            "negative_row": int(negative_row),
                            "negative_score": float(negative_score),
                            "tag": int(event_tag),
                            "activation": float(hinge[positive_index] > 0.0),
                            "hinge": float(hinge[positive_index]),
                        })
                else:
                    query_events.append({
                        "candidate": int(candidate),
                        "positive_row": -1,
                        "negative_row": int(negative_row),
                        "negative_score": float(negative_score),
                        "tag": int(event_tag),
                        "activation": float(np.mean(hinge > 0.0)),
                        "hinge": float(np.mean(hinge)),
                    })
        retained, used_sentinel = retain_query_events(
            query_events, max_active_spectrum_events_per_query,
        )
        if used_sentinel:
            safe_sentinel_events += 1
        for event in retained:
            candidate = int(event["candidate"])
            positive_row = int(event["positive_row"])
            negative_row = int(event["negative_row"])
            triplet = (query, positive_row, candidate, negative_row)
            if triplet in spectrum_triplets:
                raise RuntimeError("query-positive-negative-reference triplet was duplicated")
            spectrum_triplets.add(triplet)
            candidate_pairs.add((query, candidate))
            anchors.append(geometry.query_row)
            event_positive_rows = (
                geometry.positive_rows if positive_row < 0
                else np.asarray([positive_row], dtype=np.int64)
            )
            positives.extend(map(int, event_positive_rows))
            negatives.append(negative_row)
            positive_ptr.append(len(positives))
            negative_ptr.append(len(negatives))
            source_query.append(query)
            negative_candidate.append(candidate)
            positive_reference_row.append(positive_row)
            negative_reference_row.append(negative_row)
            source_tag.append(int(event["tag"]))
            activation_probability.append(float(event["activation"]))
            mean_hinge.append(float(event["hinge"]))
        retained_candidates = {int(event["candidate"]) for event in retained}
        selected_reference_edges += int(len(retained))
        available_reference_edges += sum(
            len(np.unique(molecule_rows(manifest, query, candidate)))
            for candidate in retained_candidates
        )
    pool = {
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positives, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negatives, dtype=np.int64),
        "source_query": np.asarray(source_query, dtype=np.int64),
        "negative_candidate": np.asarray(negative_candidate, dtype=np.int16),
        "positive_reference_row": np.asarray(positive_reference_row, dtype=np.int64),
        "negative_reference_row": np.asarray(negative_reference_row, dtype=np.int64),
        "source_tag": np.asarray(source_tag, dtype=np.int8),
        "activation_probability": np.asarray(activation_probability, dtype=np.float32),
        "mean_hinge_at_mining": np.asarray(mean_hinge, dtype=np.float32),
    }
    query_array = np.asarray(source_query, dtype=np.int64)
    formulas = np.asarray(manifest["query_formula"])[query_array].astype(str)
    tags = np.asarray(source_tag, dtype=np.int64)
    activation = np.asarray(activation_probability, dtype=np.float64)
    hinges = np.asarray(mean_hinge, dtype=np.float64)
    chemical_mask = (tags & CHEMICAL_HARD) > 0
    chemical_candidate_pairs = set(zip(
        query_array[chemical_mask].tolist(),
        np.asarray(negative_candidate, dtype=np.int64)[chemical_mask].tolist(),
    ))
    audit = {
        "spectrum_triplet_events": int(len(anchors)),
        "candidate_events": int(len(candidate_pairs)),
        "unique_query_positive_negative_reference_triplets": int(len(spectrum_triplets)),
        "anchor_queries": int(len(np.unique(query_array))),
        "unique_formulas": int(len(np.unique(formulas))),
        "primary_hard_spectrum_events": int(np.sum((tags & PRIMARY_HARD) > 0)),
        "secondary_hard_spectrum_events": int(np.sum((tags & SECONDARY_HARD) > 0)),
        "chemical_spectrum_events": int(np.sum((tags & CHEMICAL_HARD) > 0)),
        "chemical_candidate_events": int(len(chemical_candidate_pairs)),
        "strict_specific_spectrum_events": int(np.sum((tags & STRICT_SPECIFIC) > 0)),
        "fallback_spectrum_events": int(np.sum((tags & HARD_FALLBACK) > 0)),
        "retrieval_margin_violating_spectrum_events": int(
            np.sum((tags & RETRIEVAL_MARGIN_VIOLATION) > 0)
        ),
        "positive_reference_edges": int(len(positives)),
        "selected_negative_reference_edges": int(selected_reference_edges),
        "available_negative_reference_edges_for_selected_candidates": int(available_reference_edges),
        "mean_pair_activation_probability": float(np.mean(activation)),
        "median_pair_activation_probability": float(np.median(activation)),
        "zero_pair_activation_spectrum_events": int(np.sum(activation == 0.0)),
        "safe_sentinel_events": int(safe_sentinel_events),
        "active_spectrum_events": int(np.sum(activation > 0.0)),
        "mean_native_hinge_at_mining": float(np.mean(hinges)),
    }
    return pool, audit, candidate_pairs, spectrum_triplets


def _tagged_candidate_pairs(pool: Mapping[str, np.ndarray], tag: int) -> set[tuple[int, int]]:
    query = np.asarray(pool["source_query"], dtype=np.int64)
    candidate = np.asarray(pool["negative_candidate"], dtype=np.int64)
    tags = np.asarray(pool["source_tag"], dtype=np.int64)
    return {
        (int(q), int(c))
        for q, c in zip(query[(tags & tag) > 0], candidate[(tags & tag) > 0], strict=True)
    }


def structural_gates(
    train: Mapping[str, object], *, min_train_queries: int,
    min_candidate_events: int, min_spectrum_events: int,
    min_train_formulas: int, min_chemical_candidates: int,
    min_chemical_spectrum_events: int, role2_specific_surplus: int,
) -> dict[str, bool]:
    """Checkpoint-portable floors; relative activity is gated separately."""
    return {
        "train_queries": int(train["anchor_queries"]) >= min_train_queries,
        "train_candidate_events": int(train["candidate_events"]) >= min_candidate_events,
        "train_spectrum_events": int(train["spectrum_triplet_events"]) >= min_spectrum_events,
        "train_formulas": int(train["unique_formulas"]) >= min_train_formulas,
        "train_chemical_candidates": (
            int(train["chemical_candidate_events"]) >= min_chemical_candidates
        ),
        "train_chemical_spectrum_events": (
            int(train["chemical_spectrum_events"]) >= min_chemical_spectrum_events
        ),
        "unique_spectrum_triplets": (
            int(train["spectrum_triplet_events"])
            == int(train["unique_query_positive_negative_reference_triplets"])
        ),
        "role2_recipe_specific": int(role2_specific_surplus) > 0,
        "formula_roles_disjoint": True,
        "outer_role_4_untouched": True,
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.candidates_per_query < 1:
        raise ValueError("--candidates-per-query must be positive")
    if args.chemical_candidates_per_query < 0:
        raise ValueError("--chemical-candidates-per-query must be nonnegative")
    if args.negative_references_per_candidate < 1:
        raise ValueError("--negative-references-per-candidate must be positive")
    if args.explicit_positive_references_per_negative < 0:
        raise ValueError("--explicit-positive-references-per-negative must be nonnegative")
    if args.max_active_spectrum_events_per_query < 1:
        raise ValueError("--max-active-spectrum-events-per-query must be positive")
    if not 0.0 <= args.min_chemical_activation_probability <= 1.0:
        raise ValueError("--min-chemical-activation-probability must be in [0, 1]")
    evidence = {
        role: load_npz(args.evidence_dir / f"{file_role}_triplet_evidence.npz")
        for role, file_role in (
            ("train", "train"), ("selection", "selection"),
            ("confirmation", "confirmation"),
        )
    }
    manifest = load_npz(args.manifest)
    role_formulas = {role: set(body["formula"].astype(str)) for role, body in evidence.items()}
    if any(
        role_formulas[left] & role_formulas[right]
        for left, right in (("train", "selection"), ("train", "confirmation"), ("selection", "confirmation"))
    ):
        raise RuntimeError("formula roles overlap in reference-aligned triplet evidence")
    cache_rows, embedding, embedding_path = cache_arrays(args.embedding_cache)
    recipe, selection, grid = select_recipe(evidence["selection"])
    arm_names = tuple(map(str, evidence["train"]["arm_names"].tolist()))
    geometry = {
        role: {
            int(query): _query_geometry(
                manifest, int(query), cache_rows, embedding, args.margin,
                args.negative_references_per_candidate,
            )
            for query in np.asarray(body["query"], dtype=np.int64)
        }
        for role, body in evidence.items()
    }
    pools: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    audits: dict[str, dict[str, object]] = {}
    candidate_sets: dict[tuple[str, str], set[tuple[int, int]]] = {}
    spectrum_sets: dict[tuple[str, str], set[tuple[int, int, int, int]]] = {}
    for role, body in evidence.items():
        for center, arm in enumerate(arm_names):
            pool, audit, candidates, spectra = build_pool(
                body, manifest, geometry[role], center, recipe,
                args.candidates_per_query, args.chemical_hardness_window,
                args.min_chemical_activation_probability,
                args.chemical_candidates_per_query, args.margin,
                args.max_active_spectrum_events_per_query,
                args.explicit_positive_references_per_negative,
            )
            pools[(role, arm)] = pool
            audits[f"{role}:{arm}"] = audit
            candidate_sets[(role, arm)] = candidates
            spectrum_sets[(role, arm)] = spectra
    train = audits["train:correct"]
    gates = structural_gates(
        train,
        min_train_queries=args.min_train_queries,
        min_candidate_events=args.min_candidate_events,
        min_spectrum_events=args.min_spectrum_events,
        min_train_formulas=args.min_train_formulas,
        min_chemical_candidates=args.min_chemical_candidates,
        min_chemical_spectrum_events=args.min_chemical_spectrum_events,
        role2_specific_surplus=int(selection["specific_candidate_surplus"]),
    )
    if not all(gates.values()):
        raise RuntimeError(f"reference-aligned triplet gates failed: {gates}; train={train}")
    identity_audit = {
        role: audit_identity_edges(pools[(role, "correct")], args.data)
        for role in ("train", "selection", "confirmation")
    }
    overlaps = {}
    correct_candidates = candidate_sets[("selection", "correct")]
    correct_chemical = _tagged_candidate_pairs(pools[("selection", "correct")], CHEMICAL_HARD)
    for arm in arm_names[1:]:
        null_candidates = candidate_sets[("selection", arm)]
        null_chemical = _tagged_candidate_pairs(pools[("selection", arm)], CHEMICAL_HARD)
        overlaps[arm] = {
            "candidate_jaccard": float(
                len(correct_candidates & null_candidates)
                / max(1, len(correct_candidates | null_candidates))
            ),
            "correct_only_candidates": int(len(correct_candidates - null_candidates)),
            "null_only_candidates": int(len(null_candidates - correct_candidates)),
            "chemical_jaccard": float(
                len(correct_chemical & null_chemical)
                / max(1, len(correct_chemical | null_chemical))
            ),
            "correct_only_chemical_candidates": int(len(correct_chemical - null_chemical)),
            "null_only_chemical_candidates": int(len(null_chemical - correct_chemical)),
        }
    report = {
        "status": "CHEMAWARE_REFERENCE_ALIGNED_NATIVE_TRIPLETS_COMPLETE",
        "method": "checkpoint-hard candidate slots with explicit positive-negative reference expansion",
        "training_runtime": "unmodified DreaMS ContrastiveSpectraDataset plus ContrastiveHead",
        "initialization_cache": str(args.embedding_cache.resolve()),
        "embedding_cache_file": str(embedding_path.resolve()),
        "margin": float(args.margin),
        "candidates_per_query": int(args.candidates_per_query),
        "chemical_candidates_per_query": int(args.chemical_candidates_per_query),
        "negative_references_per_candidate": int(args.negative_references_per_candidate),
        "explicit_positive_references_per_negative": int(
            args.explicit_positive_references_per_negative
        ),
        "max_active_spectrum_events_per_query": int(args.max_active_spectrum_events_per_query),
        "chemical_hardness_window": float(args.chemical_hardness_window),
        "min_chemical_activation_probability": float(args.min_chemical_activation_probability),
        "role2_frozen_specific_recipe": recipe_dict(recipe),
        "role2_recipe_selection": selection,
        "role2_recipe_grid_rows": int(len(grid)),
        "arms": list(arm_names),
        "roles": {
            "optimization": "formula roles 0-1",
            "checkpoint_selection": "formula role 2",
            "development_evaluation": "formula role 3",
            "outer": "formula role 4 untouched and not loaded",
        },
        "audits": audits,
        "role2_correct_vs_null_overlap": overlaps,
        "identity_audit": identity_audit,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_ref_aligned_", dir=args.output.parent))
    try:
        for role in ("train", "selection", "confirmation"):
            np.savez_compressed(temporary / f"{role}_pool.npz", **pools[(role, "correct")])
            for arm in arm_names[1:]:
                safe = arm.removeprefix("rule_response_")
                np.savez_compressed(
                    temporary / f"{role}_pool_{safe}.npz", **pools[(role, arm)],
                )
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
