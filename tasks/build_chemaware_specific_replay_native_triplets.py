"""Build a safety-replayed, counterfactual-specific native DreaMS curriculum.

The pair-expanded pilot increased role-2 Recall@1 but introduced too many new
top-1 errors.  Its pool gave most optimizer mass to generic hard/fallback
events, while chemically selected events were a minority.  This builder keeps
the official DreaMS dataset and triplet loss and changes only the examples:

* exactly one stage-1 nearest-boundary safety event for every query;
* up to K active correct-arm chemical events per query;
* correct chemical candidates are prioritized by disagreement with all three
  frozen content-permuted arms;
* null pools use the identical query/event schedule and the identical safety
  background, replacing only the chemical event with a matched null event.

Formula role 2 is used only for checkpoint selection, role 3 only after that
selection, and role 4 is not loaded.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_dreams_native_triplets import audit_identity_edges
from build_chemaware_reference_aligned_native_triplets import (
    CHEMICAL_HARD,
    PRIMARY_HARD,
)


ROOT = Path(__file__).resolve().parents[1]
NULL_NAMES = ("content_permuted", "content_permuted_b", "content_permuted_c")
SAFETY_REPLAY = 1
SPECIFIC_CHEMISTRY = 2
MATCHED_NULL = 3


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-bank", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chemical-events-per-query", type=int, default=3)
    parser.add_argument(
        "--maximum-null-candidate-agreement", type=int, default=2,
        help=(
            "Upper bound for an automatic strictest-admissible scan from zero to this "
            "many agreeing nulls; three would admit fully nonspecific candidates."
        ),
    )
    parser.add_argument("--min-train-queries", type=int, default=3500)
    parser.add_argument("--min-specific-chemical-events", type=int, default=300)
    parser.add_argument("--min-specific-chemical-queries", type=int, default=150)
    parser.add_argument("--min-chemical-fraction", type=float, default=0.06)
    parser.add_argument("--min-active-chemical-fraction", type=float, default=0.15)
    parser.add_argument("--max-null-mean-hinge-gap", type=float, default=0.05)
    parser.add_argument("--max-null-mean-activation-gap", type=float, default=0.10)
    parser.add_argument("--max-null-event-hinge-gap", type=float, default=0.25)
    parser.add_argument("--max-null-event-activation-gap", type=float, default=0.25)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def event_signature(pool: Mapping[str, np.ndarray], index: int) -> tuple[object, ...]:
    p0, p1 = map(int, pool["positive_ptr"][index:index + 2])
    n0, n1 = map(int, pool["negative_ptr"][index:index + 2])
    return (
        int(pool["anchor_idx"][index]),
        tuple(map(int, pool["positive_idx"][p0:p1])),
        tuple(map(int, pool["negative_idx"][n0:n1])),
    )


def indices_by_query(pool: Mapping[str, np.ndarray]) -> dict[int, np.ndarray]:
    query = np.asarray(pool["source_query"], dtype=np.int64)
    return {
        int(value): np.flatnonzero(query == value).astype(np.int64)
        for value in np.unique(query)
    }


def validate_source_pool(pool: Mapping[str, np.ndarray], name: str) -> None:
    required = {
        "anchor_idx", "positive_ptr", "positive_idx", "negative_ptr", "negative_idx",
        "source_query", "negative_candidate", "positive_reference_row",
        "negative_reference_row", "source_tag", "activation_probability",
        "mean_hinge_at_mining",
    }
    if missing := sorted(required - set(pool)):
        raise RuntimeError(f"{name} lacks source-pool fields: {missing}")
    events = len(pool["anchor_idx"])
    aligned = (
        "source_query", "negative_candidate", "positive_reference_row",
        "negative_reference_row", "source_tag", "activation_probability",
        "mean_hinge_at_mining",
    )
    if any(len(pool[key]) != events for key in aligned):
        raise RuntimeError(f"{name} event metadata does not align")
    if len(pool["positive_ptr"]) != events + 1 or len(pool["negative_ptr"]) != events + 1:
        raise RuntimeError(f"{name} pointer arrays do not align")
    if int(pool["positive_ptr"][0]) != 0 or int(pool["negative_ptr"][0]) != 0:
        raise RuntimeError(f"{name} pointers do not start at zero")
    if int(pool["positive_ptr"][-1]) != len(pool["positive_idx"]):
        raise RuntimeError(f"{name} positive pointer terminus is invalid")
    if int(pool["negative_ptr"][-1]) != len(pool["negative_idx"]):
        raise RuntimeError(f"{name} negative pointer terminus is invalid")
    if np.any(np.diff(pool["positive_ptr"]) < 1) or np.any(np.diff(pool["negative_ptr"]) < 1):
        raise RuntimeError(f"{name} contains an empty positive or negative event")


def event_priority(pool: Mapping[str, np.ndarray], index: int) -> tuple[float, ...]:
    tag = int(pool["source_tag"][index])
    return (
        float(bool(tag & CHEMICAL_HARD)),
        float(pool["mean_hinge_at_mining"][index]),
        float(pool["activation_probability"][index]),
        -float(pool["negative_candidate"][index]),
        -float(pool["negative_reference_row"][index]),
    )


def select_safety(pool: Mapping[str, np.ndarray]) -> dict[int, int]:
    selected: dict[int, int] = {}
    for query, indices in indices_by_query(pool).items():
        primary = [
            int(index) for index in indices
            if int(pool["source_tag"][index]) & PRIMARY_HARD
        ]
        if not primary:
            raise RuntimeError(f"query {query} lacks a stage-1 primary safety event")
        selected[query] = max(primary, key=lambda index: event_priority(pool, index))
    return selected


def chemical_candidates_by_query(
    pool: Mapping[str, np.ndarray],
) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for query, indices in indices_by_query(pool).items():
        result[query] = {
            int(pool["negative_candidate"][index])
            for index in indices
            if (int(pool["source_tag"][index]) & CHEMICAL_HARD)
            and float(pool["activation_probability"][index]) > 0.0
        }
    return result


def select_specific_schedule(
    correct: Mapping[str, np.ndarray], nulls: Mapping[str, Mapping[str, np.ndarray]],
    safety: Mapping[int, int], cap: int, maximum_null_agreement: int,
) -> dict[int, list[int]]:
    correct_by_query = indices_by_query(correct)
    null_candidates = {
        name: chemical_candidates_by_query(pool) for name, pool in nulls.items()
    }
    schedule: dict[int, list[int]] = {}
    for query, safety_index in safety.items():
        candidates = []
        for index in correct_by_query[query]:
            index = int(index)
            tag = int(correct["source_tag"][index])
            if not tag & CHEMICAL_HARD:
                continue
            if float(correct["activation_probability"][index]) <= 0.0:
                continue
            candidate = int(correct["negative_candidate"][index])
            null_agreement = sum(
                candidate in by_query.get(query, set())
                for by_query in null_candidates.values()
            )
            if null_agreement > maximum_null_agreement:
                continue
            if event_signature(correct, index) == event_signature(correct, safety_index):
                continue
            candidates.append((index, null_agreement))
        candidates.sort(
            key=lambda item: (-item[1], *event_priority(correct, item[0])),
            reverse=True,
        )
        chosen: list[int] = []
        seen = {event_signature(correct, safety_index)}
        for index, _agreement in candidates:
            signature = event_signature(correct, index)
            if signature in seen:
                continue
            seen.add(signature)
            chosen.append(index)
            if len(chosen) >= cap:
                break
        schedule[query] = chosen
    return schedule


def matched_null_schedule(
    pool: Mapping[str, np.ndarray], target: Mapping[int, list[int]],
    safety_pool: Mapping[str, np.ndarray], safety: Mapping[int, int],
) -> dict[int, list[int]]:
    by_query = indices_by_query(pool)
    result: dict[int, list[int]] = {}
    for query, correct_indices in target.items():
        needed = len(correct_indices)
        if needed == 0:
            result[query] = []
            continue
        safety_signature = event_signature(safety_pool, safety[query])
        forbidden_candidates = {
            int(safety_pool["negative_candidate"][index]) for index in correct_indices
        }
        forbidden_signatures = {
            event_signature(safety_pool, int(index)) for index in correct_indices
        }
        candidates = [
            int(index) for index in by_query[query]
            if float(pool["activation_probability"][index]) > 0.0
            and int(pool["source_tag"][index]) & CHEMICAL_HARD
            and event_signature(pool, int(index)) != safety_signature
            and int(pool["negative_candidate"][index]) not in forbidden_candidates
            and event_signature(pool, int(index)) not in forbidden_signatures
        ]
        chosen: list[int] = []
        seen = {safety_signature}
        remaining = list(candidates)
        # Greedily match each correct chemical event by native loss geometry,
        # removing the chosen null signature so event multiplicity is exact.
        for correct_index in correct_indices:
            target_hinge = float(safety_pool["mean_hinge_at_mining"][correct_index])
            target_activation = float(safety_pool["activation_probability"][correct_index])
            eligible = [
                index for index in remaining
                if event_signature(pool, index) not in seen
            ]
            if not eligible:
                break
            index = min(eligible, key=lambda value: (
                abs(float(pool["mean_hinge_at_mining"][value]) - target_hinge),
                abs(float(pool["activation_probability"][value]) - target_activation),
                -float(pool["mean_hinge_at_mining"][value]),
                int(pool["negative_candidate"][value]),
                int(pool["negative_reference_row"][value]),
            ))
            signature = event_signature(pool, index)
            seen.add(signature); chosen.append(index); remaining.remove(index)
        if len(chosen) != needed:
            raise RuntimeError(
                f"null arm lacks {needed} matched active events for query {query}; "
                f"available={len(chosen)} after excluding correct chemical candidates"
            )
        result[query] = chosen
    return result


def prune_to_matched_null_capacity(
    correct: Mapping[str, np.ndarray],
    nulls: Mapping[str, Mapping[str, np.ndarray]],
    safety: Mapping[int, int],
    schedule: Mapping[int, list[int]],
) -> tuple[dict[int, list[int]], int]:
    """Drop lowest-priority chemical events until every null can match them."""
    null_by_query = {name: indices_by_query(pool) for name, pool in nulls.items()}
    pruned: dict[int, list[int]] = {}
    removed = 0
    for query, initial in schedule.items():
        chosen = list(initial)
        safety_signature = event_signature(correct, safety[query])
        while chosen:
            forbidden_candidates = {
                int(correct["negative_candidate"][index]) for index in chosen
            }
            capacities = []
            for name, pool in nulls.items():
                signatures = {
                    event_signature(pool, int(index))
                    for index in null_by_query[name][query]
                    if float(pool["activation_probability"][index]) > 0.0
                    and int(pool["source_tag"][index]) & CHEMICAL_HARD
                    and event_signature(pool, int(index)) != safety_signature
                    and int(pool["negative_candidate"][index]) not in forbidden_candidates
                }
                capacities.append(len(signatures))
            if min(capacities) >= len(chosen):
                break
            chosen.pop()
            removed += 1
        pruned[query] = chosen
    return pruned, removed


def prune_to_matched_null_geometry(
    correct: Mapping[str, np.ndarray],
    nulls: Mapping[str, Mapping[str, np.ndarray]],
    safety: Mapping[int, int],
    schedule: Mapping[int, list[int]],
    max_mean_hinge_gap: float,
    max_mean_activation_gap: float,
    max_event_hinge_gap: float,
    max_event_activation_gap: float,
) -> tuple[dict[int, list[int]], int]:
    """Prune worst-matched events until every null has comparable geometry.

    Candidate specificity and optimizer geometry are separate constraints. A
    chemically specific event can still have a much larger native hinge than
    every available null event, which confounds a correct-vs-null comparison.
    The check therefore happens before a profile is admitted. Events from
    queries with multiple chemical constraints are removed first, preserving
    query diversity whenever possible.
    """
    current = {int(query): list(indices) for query, indices in schedule.items()}
    removed = 0
    while True:
        matches = {
            name: matched_null_schedule(pool, current, correct, safety)
            for name, pool in nulls.items()
        }
        if not any(current.values()):
            return current, removed

        hinge_by_arm: dict[str, list[float]] = {name: [] for name in nulls}
        activation_by_arm: dict[str, list[float]] = {name: [] for name in nulls}
        penalties: dict[tuple[int, int], float] = {}
        ordered_keys: list[tuple[int, int]] = []
        maximum_hinge: dict[tuple[int, int], float] = {}
        maximum_activation: dict[tuple[int, int], float] = {}
        for query, correct_indices in current.items():
            for position, correct_index in enumerate(correct_indices):
                event_key = (query, position)
                ordered_keys.append(event_key)
                hinge = float(correct["mean_hinge_at_mining"][correct_index])
                activation = float(correct["activation_probability"][correct_index])
                event_penalty = 0.0
                event_hinge = 0.0
                event_activation = 0.0
                for name, pool in nulls.items():
                    null_index = int(matches[name][query][position])
                    hinge_gap = abs(
                        float(pool["mean_hinge_at_mining"][null_index]) - hinge
                    )
                    activation_gap = abs(
                        float(pool["activation_probability"][null_index]) - activation
                    )
                    hinge_by_arm[name].append(hinge_gap)
                    activation_by_arm[name].append(activation_gap)
                    event_hinge = max(event_hinge, hinge_gap)
                    event_activation = max(event_activation, activation_gap)
                    event_penalty = max(
                        event_penalty,
                        hinge_gap / max(float(max_event_hinge_gap), 1e-12),
                        activation_gap / max(float(max_event_activation_gap), 1e-12),
                    )
                penalties[event_key] = event_penalty
                maximum_hinge[event_key] = event_hinge
                maximum_activation[event_key] = event_activation

        geometry_ok = all(
            float(np.mean(hinge_by_arm[name])) <= float(max_mean_hinge_gap)
            and float(np.mean(activation_by_arm[name])) <= float(max_mean_activation_gap)
            and float(np.max(hinge_by_arm[name])) <= float(max_event_hinge_gap)
            and float(np.max(activation_by_arm[name])) <= float(max_event_activation_gap)
            for name in nulls
        )
        if geometry_ok:
            return current, removed

        # Remove all event-level violations together. For a mean-only failure,
        # remove the smallest frozen set of worst gaps needed by each failing
        # arm under the current matching, then rematch. This avoids hundreds
        # of full 4,032-query scans while retaining the same deterministic gate.
        remove_keys = {
            key for key in ordered_keys
            if maximum_hinge[key] > float(max_event_hinge_gap)
            or maximum_activation[key] > float(max_event_activation_gap)
        }
        if not remove_keys:
            for values, threshold in (
                (hinge_by_arm, float(max_mean_hinge_gap)),
                (activation_by_arm, float(max_mean_activation_gap)),
            ):
                for name, gaps in values.items():
                    if float(np.mean(gaps)) <= threshold:
                        continue
                    total = float(np.sum(gaps)); remaining = len(gaps)
                    for index in sorted(
                        range(len(gaps)), key=lambda value: gaps[value], reverse=True,
                    ):
                        remove_keys.add(ordered_keys[index])
                        total -= float(gaps[index]); remaining -= 1
                        if remaining == 0 or total / remaining <= threshold:
                            break
        for query in sorted({key[0] for key in remove_keys}):
            positions = sorted(
                (position for q, position in remove_keys if q == query),
                reverse=True,
            )
            for position in positions:
                current[query].pop(position)
                removed += 1


def compose_pool(
    safety_pool: Mapping[str, np.ndarray], safety: Mapping[int, int],
    event_pool: Mapping[str, np.ndarray], schedule: Mapping[int, list[int]],
    chemical_role: int,
) -> dict[str, np.ndarray]:
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    metadata = {key: [] for key in (
        "source_query", "negative_candidate", "positive_reference_row",
        "negative_reference_row", "source_tag", "activation_probability",
        "mean_hinge_at_mining", "curriculum_role",
    )}

    def append(source: Mapping[str, np.ndarray], index: int, role: int) -> None:
        p0, p1 = map(int, source["positive_ptr"][index:index + 2])
        n0, n1 = map(int, source["negative_ptr"][index:index + 2])
        anchors.append(int(source["anchor_idx"][index]))
        positives.extend(map(int, source["positive_idx"][p0:p1]))
        negatives.extend(map(int, source["negative_idx"][n0:n1]))
        positive_ptr.append(len(positives)); negative_ptr.append(len(negatives))
        for key in metadata:
            if key == "curriculum_role":
                metadata[key].append(role)
            else:
                metadata[key].append(source[key][index])

    for query in sorted(safety):
        append(safety_pool, int(safety[query]), SAFETY_REPLAY)
        for index in schedule[query]:
            append(event_pool, int(index), chemical_role)
    return {
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positives, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negatives, dtype=np.int64),
        "source_query": np.asarray(metadata["source_query"], dtype=np.int64),
        "negative_candidate": np.asarray(metadata["negative_candidate"], dtype=np.int16),
        "positive_reference_row": np.asarray(metadata["positive_reference_row"], dtype=np.int64),
        "negative_reference_row": np.asarray(metadata["negative_reference_row"], dtype=np.int64),
        "source_tag": np.asarray(metadata["source_tag"], dtype=np.int8),
        "activation_probability": np.asarray(metadata["activation_probability"], dtype=np.float32),
        "mean_hinge_at_mining": np.asarray(metadata["mean_hinge_at_mining"], dtype=np.float32),
        "curriculum_role": np.asarray(metadata["curriculum_role"], dtype=np.int8),
    }


def audit_pool(pool: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray]) -> dict[str, object]:
    role = np.asarray(pool["curriculum_role"], dtype=np.int64)
    queries = np.asarray(pool["source_query"], dtype=np.int64)
    chemical = role != SAFETY_REPLAY
    active = np.asarray(pool["activation_probability"], dtype=np.float64) > 0.0
    counts = np.unique(queries, return_counts=True)[1]
    return {
        "events": int(len(queries)),
        "queries": int(len(np.unique(queries))),
        "formulas": int(len(np.unique(np.asarray(manifest["query_formula"])[queries].astype(str)))),
        "safety_events": int(np.sum(role == SAFETY_REPLAY)),
        "chemical_or_matched_events": int(np.sum(chemical)),
        "chemical_or_matched_fraction": float(np.mean(chemical)),
        "active_events": int(np.sum(active)),
        "active_safety_events": int(np.sum(active & ~chemical)),
        "active_chemical_or_matched_events": int(np.sum(active & chemical)),
        "active_chemical_or_matched_fraction": (
            float(np.sum(active & chemical) / np.sum(active)) if np.any(active) else 0.0
        ),
        "inactive_chemical_or_matched_events": int(np.sum(
            chemical & (np.asarray(pool["activation_probability"]) <= 0.0)
        )),
        "queries_with_chemical_or_matched_event": int(len(np.unique(queries[chemical]))),
        "minimum_events_per_query": int(np.min(counts)),
        "maximum_events_per_query": int(np.max(counts)),
        "mean_events_per_query": float(np.mean(counts)),
    }


def matched_geometry_audit(
    correct: Mapping[str, np.ndarray], null: Mapping[str, np.ndarray],
) -> dict[str, float | int]:
    correct_role = np.asarray(correct["curriculum_role"], dtype=np.int64)
    null_role = np.asarray(null["curriculum_role"], dtype=np.int64)
    chemical = correct_role == SPECIFIC_CHEMISTRY
    if not np.array_equal(chemical, null_role == MATCHED_NULL):
        raise RuntimeError("correct/null chemical event positions do not align")
    hinge_gap = np.abs(
        np.asarray(correct["mean_hinge_at_mining"], dtype=np.float64)[chemical]
        - np.asarray(null["mean_hinge_at_mining"], dtype=np.float64)[chemical]
    )
    activation_gap = np.abs(
        np.asarray(correct["activation_probability"], dtype=np.float64)[chemical]
        - np.asarray(null["activation_probability"], dtype=np.float64)[chemical]
    )
    return {
        "matched_chemical_events": int(np.sum(chemical)),
        "mean_absolute_hinge_gap": float(np.mean(hinge_gap)) if len(hinge_gap) else 0.0,
        "maximum_absolute_hinge_gap": float(np.max(hinge_gap)) if len(hinge_gap) else 0.0,
        "mean_absolute_activation_gap": (
            float(np.mean(activation_gap)) if len(activation_gap) else 0.0
        ),
        "maximum_absolute_activation_gap": (
            float(np.max(activation_gap)) if len(activation_gap) else 0.0
        ),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.chemical_events_per_query < 1:
        raise ValueError("chemical event cap must be positive")
    if not 0 <= args.maximum_null_candidate_agreement < len(NULL_NAMES):
        raise ValueError("invalid maximum null-candidate agreement")
    report_path = args.pair_bank / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "CHEMAWARE_REFERENCE_ALIGNED_NATIVE_TRIPLETS_COMPLETE":
        raise RuntimeError("pair-expanded source bank is incomplete")
    manifest = load_npz(args.manifest)
    # Select the strictest candidate-disagreement level that can support a
    # full-model native continuation.  This structural scan uses roles 0--1
    # only and never consults role-2 performance or role-3 outcomes.
    train_correct = load_npz(args.pair_bank / "train_pool.npz")
    train_nulls = {
        name: load_npz(args.pair_bank / f"train_pool_{name}.npz")
        for name in NULL_NAMES
    }
    validate_source_pool(train_correct, "train:correct")
    for name, null_pool in train_nulls.items():
        validate_source_pool(null_pool, f"train:{name}")
    train_safety = select_safety(train_correct)
    specificity_profiles: dict[str, dict[str, object]] = {}
    selected_null_agreement: int | None = None
    for agreement in range(args.maximum_null_candidate_agreement + 1):
        profile_schedule = select_specific_schedule(
            train_correct, train_nulls, train_safety,
            args.chemical_events_per_query, agreement,
        )
        profile_schedule, removed = prune_to_matched_null_capacity(
            train_correct, train_nulls, train_safety, profile_schedule,
        )
        profile_schedule, geometry_removed = prune_to_matched_null_geometry(
            train_correct, train_nulls, train_safety, profile_schedule,
            args.max_null_mean_hinge_gap, args.max_null_mean_activation_gap,
            args.max_null_event_hinge_gap, args.max_null_event_activation_gap,
        )
        profile_pool = compose_pool(
            train_correct, train_safety, train_correct, profile_schedule,
            SPECIFIC_CHEMISTRY,
        )
        profile = audit_pool(profile_pool, manifest)
        profile_gates = {
            "specific_chemical_events": (
                int(profile["chemical_or_matched_events"])
                >= args.min_specific_chemical_events
            ),
            "specific_chemical_queries": (
                int(profile["queries_with_chemical_or_matched_event"])
                >= args.min_specific_chemical_queries
            ),
            "chemical_fraction": (
                float(profile["chemical_or_matched_fraction"])
                >= args.min_chemical_fraction
            ),
            "active_chemical_fraction": (
                float(profile["active_chemical_or_matched_fraction"])
                >= args.min_active_chemical_fraction
            ),
        }
        specificity_profiles[str(agreement)] = {
            "audit": profile,
            "matched_capacity_pruned_events": int(removed),
            "matched_geometry_pruned_events": int(geometry_removed),
            "gates": profile_gates,
            "admissible": bool(all(profile_gates.values())),
        }
        if all(profile_gates.values()):
            selected_null_agreement = agreement
            break
    if selected_null_agreement is None:
        raise RuntimeError(
            "no counterfactual-specific replay profile reaches the frozen coverage "
            f"gates without admitting three-null consensus: {specificity_profiles}"
        )
    print(json.dumps({
        "status": "CHEMAWARE_SPECIFIC_REPLAY_PROFILE_SELECTED",
        "selected_maximum_null_candidate_agreement": selected_null_agreement,
        "profiles": specificity_profiles,
    }, indent=2), flush=True)

    pools: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    audits: dict[str, dict[str, object]] = {}
    capacity_pruned: dict[str, dict[str, int]] = {}
    matching_audits: dict[str, dict[str, float | int]] = {}
    role_formulas: dict[str, set[str]] = {}
    for role in ("train", "selection", "confirmation"):
        correct = load_npz(args.pair_bank / f"{role}_pool.npz")
        nulls = {
            name: load_npz(args.pair_bank / f"{role}_pool_{name}.npz")
            for name in NULL_NAMES
        }
        validate_source_pool(correct, f"{role}:correct")
        for name, null_pool in nulls.items():
            validate_source_pool(null_pool, f"{role}:{name}")
            if set(indices_by_query(null_pool)) != set(indices_by_query(correct)):
                raise RuntimeError(f"{role}:{name} query support differs from correct")
        role_formulas[role] = set(
            np.asarray(manifest["query_formula"])[
                np.unique(np.asarray(correct["source_query"], dtype=np.int64))
            ].astype(str)
        )
        safety = select_safety(correct)
        schedule = select_specific_schedule(
            correct, nulls, safety, args.chemical_events_per_query,
            selected_null_agreement,
        )
        schedule, removed = prune_to_matched_null_capacity(
            correct, nulls, safety, schedule,
        )
        schedule, geometry_removed = prune_to_matched_null_geometry(
            correct, nulls, safety, schedule,
            args.max_null_mean_hinge_gap, args.max_null_mean_activation_gap,
            args.max_null_event_hinge_gap, args.max_null_event_activation_gap,
        )
        capacity_pruned[role] = {
            "capacity": int(removed),
            "geometry": int(geometry_removed),
        }
        correct_output = compose_pool(
            correct, safety, correct, schedule, SPECIFIC_CHEMISTRY,
        )
        pools[(role, "correct")] = correct_output
        audits[f"{role}:correct"] = audit_pool(correct_output, manifest)
        for name, null_pool in nulls.items():
            null_schedule = matched_null_schedule(null_pool, schedule, correct, safety)
            output = compose_pool(
                correct, safety, null_pool, null_schedule, MATCHED_NULL,
            )
            pools[(role, name)] = output
            audits[f"{role}:{name}"] = audit_pool(output, manifest)
            matching_audits[f"{role}:{name}"] = matched_geometry_audit(
                correct_output, output,
            )
            if not np.array_equal(output["source_query"], correct_output["source_query"]):
                raise RuntimeError(f"{role}:{name} query/event schedule drifted")
            if not np.array_equal(
                output["anchor_idx"][output["curriculum_role"] == SAFETY_REPLAY],
                correct_output["anchor_idx"][correct_output["curriculum_role"] == SAFETY_REPLAY],
            ):
                raise RuntimeError(f"{role}:{name} safety replay drifted")

    if any(
        role_formulas[left] & role_formulas[right]
        for left, right in (
            ("train", "selection"), ("train", "confirmation"),
            ("selection", "confirmation"),
        )
    ):
        raise RuntimeError("formula roles overlap in the specific-replay source pools")
    train = audits["train:correct"]
    gates = {
        "train_query_coverage": int(train["queries"]) >= args.min_train_queries,
        "specific_chemical_events": (
            int(train["chemical_or_matched_events"]) >= args.min_specific_chemical_events
        ),
        "specific_chemical_queries": (
            int(train["queries_with_chemical_or_matched_event"])
            >= args.min_specific_chemical_queries
        ),
        "chemical_fraction": (
            float(train["chemical_or_matched_fraction"]) >= args.min_chemical_fraction
        ),
        "active_chemical_fraction": (
            float(train["active_chemical_or_matched_fraction"])
            >= args.min_active_chemical_fraction
        ),
        "one_safety_event_per_query": (
            int(train["safety_events"]) == int(train["queries"])
        ),
        "all_chemical_and_matched_events_active": all(
            int(audits[f"{role}:{arm}"]["inactive_chemical_or_matched_events"]) == 0
            for role in ("train", "selection", "confirmation")
            for arm in ("correct", *NULL_NAMES)
        ),
        "exact_null_event_budgets": all(
            audits[f"{role}:{name}"]["events"] == audits[f"{role}:correct"]["events"]
            for role in ("train", "selection", "confirmation") for name in NULL_NAMES
        ),
        "formula_roles_disjoint": True,
        "null_hinge_geometry_matched": all(
            float(value["mean_absolute_hinge_gap"]) <= args.max_null_mean_hinge_gap
            for value in matching_audits.values()
        ),
        "null_activation_geometry_matched": all(
            float(value["mean_absolute_activation_gap"])
            <= args.max_null_mean_activation_gap
            for value in matching_audits.values()
        ),
        "null_event_hinge_geometry_matched": all(
            float(value["maximum_absolute_hinge_gap"])
            <= args.max_null_event_hinge_gap
            for value in matching_audits.values()
        ),
        "null_event_activation_geometry_matched": all(
            float(value["maximum_absolute_activation_gap"])
            <= args.max_null_event_activation_gap
            for value in matching_audits.values()
        ),
        "outer_role_4_untouched": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"specific replay gates failed: {gates}; train={train}")
    identity = {
        role: audit_identity_edges(pools[(role, "correct")], args.data)
        for role in ("train", "selection", "confirmation")
    }
    output_report = {
        "status": "CHEMAWARE_SPECIFIC_REPLAY_NATIVE_TRIPLETS_COMPLETE",
        "method": "one stage-1 safety replay per query plus counterfactual-specific chemical events",
        "training_runtime": "unmodified DreaMS ContrastiveSpectraDataset plus ContrastiveHead",
        "source_pair_bank": str(args.pair_bank.resolve()),
        "chemical_events_per_query": int(args.chemical_events_per_query),
        "maximum_null_candidate_agreement_requested": int(
            args.maximum_null_candidate_agreement
        ),
        "maximum_null_candidate_agreement_selected": int(selected_null_agreement),
        "matched_null_geometry_limits": {
            "mean_hinge_gap": float(args.max_null_mean_hinge_gap),
            "mean_activation_gap": float(args.max_null_mean_activation_gap),
            "event_hinge_gap": float(args.max_null_event_hinge_gap),
            "event_activation_gap": float(args.max_null_event_activation_gap),
        },
        "specificity_profiles": specificity_profiles,
        "arms": ["correct", *NULL_NAMES],
        "roles": {
            "optimization": "formula roles 0-1",
            "checkpoint_selection": "formula role 2",
            "development_evaluation": "formula role 3",
            "outer": "formula role 4 untouched and not loaded",
        },
        "audits": audits,
        "matched_null_capacity_pruned_events": capacity_pruned,
        "matched_null_geometry": matching_audits,
        "identity_audit": identity,
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_specific_replay_", dir=args.output.parent))
    try:
        for role in ("train", "selection", "confirmation"):
            np.savez_compressed(temporary / f"{role}_pool.npz", **pools[(role, "correct")])
            for name in NULL_NAMES:
                np.savez_compressed(
                    temporary / f"{role}_pool_{name}.npz", **pools[(role, name)],
                )
        (temporary / "report.json").write_text(
            json.dumps(output_report, indent=2), encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(output_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
