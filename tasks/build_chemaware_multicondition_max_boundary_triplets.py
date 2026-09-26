"""Build identity-balanced, multi-condition max-boundary native triplets.

This is the performance successor to the frozen ChemAware Phase-A curriculum.
It preserves every mature DreaMS component and changes only which native
triplets are sampled:

* every frozen one-query-per-identity ChemAware anchor is retained;
* at most a small number of additional spectra per identity are preselected
  from distinct instrument/collision-energy conditions;
* an additional anchor is admitted only when it is currently wrong, lies near
  the retrieval boundary, or exposes a different hardest false identity;
* every focused event is the exact singleton max-positive/max-negative pair;
* ChemAware candidates discovered on the frozen base anchor may transfer to a
  second condition only when the same false identity is present and its exact
  max-reference hinge is active;
* official DreaMS replay is kept as a fixed sampling fraction; and
* stored sampling weights give every focused identity equal expected mass and
  every replay identity equal expected mass.

Formula roles 2--4 are never used by this builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from build_chemaware_action_hard_native_triplets import (
    ACTION_HARD,
    OFFICIAL_HARD,
    SPECIFIC_HARD,
)
from build_chemaware_dreams_native_triplets import audit_identity_edges, molecule_rows
from build_chemaware_max_boundary_native_triplets import (
    DREAMS_NATIVE_REPLAY,
    FrozenEmbeddings,
    PoolWriter,
    append_dreams_replay,
    build_focused_pool,
    load_npz,
)
from chemaware_numpy_sampling import stable_formula_folds


ROOT = Path(__file__).resolve().parents[1]
BASE_SAFE_BOUNDARY = 1
BASE_ERROR_BOUNDARY = 2
BASE_CHEMICAL_BOUNDARY = 3
EXTRA_SAFE_BOUNDARY = 5
EXTRA_ERROR_BOUNDARY = 6
EXTRA_CHEMICAL_BOUNDARY = 7


def pool_prefix_semantic_sha256(pool: Mapping[str, np.ndarray], events: int) -> str:
    """Hash one CSR triplet prefix independent of compact integer dtypes."""
    if events < 0 or events > len(pool["anchor_idx"]):
        raise ValueError("invalid triplet-prefix length")
    digest = hashlib.sha256()
    for key in (
        "anchor_idx", "source_query", "negative_candidate", "source_tag",
        "curriculum_role",
    ):
        values = np.asarray(pool[key][:events], dtype=np.int64)
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(values).tobytes())
    for pointer_key, index_key in (
        ("positive_ptr", "positive_idx"),
        ("negative_ptr", "negative_idx"),
    ):
        pointers = np.asarray(pool[pointer_key][:events + 1], dtype=np.int64)
        edge_count = int(pointers[-1]) if len(pointers) else 0
        indices = np.asarray(pool[index_key][:edge_count], dtype=np.int64)
        digest.update(pointer_key.encode("utf-8"))
        digest.update(np.ascontiguousarray(pointers).tobytes())
        digest.update(index_key.encode("utf-8"))
        digest.update(np.ascontiguousarray(indices).tobytes())
    return digest.hexdigest()


def pool_prefix_equal(
    expected: Mapping[str, np.ndarray], observed: Mapping[str, np.ndarray], events: int,
) -> bool:
    """Require exact semantic equality for every native-triplet prefix field."""
    for key in (
        "anchor_idx", "source_query", "negative_candidate", "source_tag",
        "curriculum_role",
    ):
        if not np.array_equal(expected[key][:events], observed[key][:events]):
            return False
    for pointer_key, index_key in (
        ("positive_ptr", "positive_idx"),
        ("negative_ptr", "negative_idx"),
    ):
        expected_ptr = np.asarray(expected[pointer_key][:events + 1], dtype=np.int64)
        observed_ptr = np.asarray(observed[pointer_key][:events + 1], dtype=np.int64)
        if not np.array_equal(expected_ptr, observed_ptr):
            return False
        expected_edges = int(expected_ptr[-1])
        observed_edges = int(observed_ptr[-1])
        if expected_edges != observed_edges or not np.array_equal(
            expected[index_key][:expected_edges], observed[index_key][:observed_edges],
        ):
            return False
    return True


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bank", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--embedding-rows", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy",
    )
    parser.add_argument(
        "--official-embeddings", type=Path,
        default=(ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1"
                 / "official_embeddings_f32.npy"),
    )
    parser.add_argument(
        "--dreams-replay-pool", type=Path,
        default=ROOT / "data/e1/e1_train_triplet_pool_10ppm.npz",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--training-roles", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--near-boundary-margin", type=float, default=0.1)
    parser.add_argument("--max-anchors-per-identity", type=int, default=3)
    parser.add_argument("--preselect-anchors-per-identity", type=int, default=6)
    parser.add_argument("--collision-energy-bin", type=float, default=10.0)
    parser.add_argument("--negative-references-per-error", type=int, default=2)
    parser.add_argument("--chemical-candidates-per-error", type=int, default=2)
    parser.add_argument("--transferred-chemical-candidates-per-error", type=int, default=1)
    parser.add_argument("--replay-fraction", type=float, default=0.20)
    parser.add_argument("--error-sampling-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def _text(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def condition_signature(instrument: object, energy: float, width: float) -> tuple[str, str]:
    if width <= 0:
        raise ValueError("collision-energy bin width must be positive")
    if np.isfinite(energy):
        energy_key = str(int(math.floor(float(energy) / width)))
    else:
        energy_key = "missing"
    return _text(instrument), energy_key


def action_identities_by_base_query(
    base: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
) -> dict[int, dict[str, int]]:
    output: dict[int, dict[str, int]] = defaultdict(dict)
    for event, query_value in enumerate(np.asarray(base["source_query"], dtype=np.int64)):
        tag = int(base["source_tag"][event])
        if not tag & (ACTION_HARD | SPECIFIC_HARD):
            continue
        query = int(query_value)
        candidate = int(base["negative_candidate"][event])
        left, right = map(int, manifest["query_ptr"][query:query + 2])
        if candidate < 0 or left + candidate >= right:
            raise RuntimeError("base ChemAware candidate is outside its query graph")
        identity = str(manifest["molecule_ik14"][left + candidate])
        output[query][identity] = output[query].get(identity, 0) | tag
    return dict(output)


def query_geometry(
    manifest: Mapping[str, np.ndarray], cache: FrozenEmbeddings,
    query: int, margin: float,
) -> dict[str, object]:
    anchor = int(manifest["query_row"][query])
    anchor_embedding = cache.get([anchor])[0]
    left, right = map(int, manifest["query_ptr"][query:query + 2])
    labels = np.asarray(manifest["molecule_label"][left:right], dtype=bool)
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
        raise RuntimeError(f"query {query} lacks a distinct positive reference")
    positive_scores = cache.get(positive_rows) @ anchor_embedding
    positive_row = int(positive_rows[int(np.argmax(positive_scores))])
    positive_score = float(np.max(positive_scores))

    candidates: dict[int, dict[str, object]] = {}
    for candidate_value in false_candidates:
        candidate = int(candidate_value)
        rows = np.unique(molecule_rows(manifest, query, candidate))
        scores = cache.get(rows) @ anchor_embedding
        order = np.argsort(-scores, kind="stable")
        rows = rows[order]
        scores = scores[order]
        active = (margin + scores - positive_score) > 0.0
        candidates[candidate] = {
            "identity": str(manifest["molecule_ik14"][left + candidate]),
            "rows": rows,
            "scores": scores,
            "active_rows": rows[active],
            "maximum_hinge": float(margin + float(scores[0]) - positive_score),
        }
    hardest = max(candidates, key=lambda candidate: float(candidates[candidate]["scores"][0]))
    hardest_score = float(candidates[hardest]["scores"][0])
    return {
        "query": int(query),
        "anchor": anchor,
        "positive_row": positive_row,
        "positive_score": positive_score,
        "hardest_candidate": int(hardest),
        "hardest_identity": str(candidates[hardest]["identity"]),
        "margin": float(positive_score - hardest_score),
        "error": bool(hardest_score >= positive_score),
        "candidates": candidates,
    }


def preselect_queries(
    queries: np.ndarray, base_query: int, manifest: Mapping[str, np.ndarray],
    cache: FrozenEmbeddings, signatures: Mapping[int, tuple[str, str]], limit: int,
) -> list[tuple[int, float]]:
    if limit < 1:
        raise ValueError("preselection limit must be positive")
    queries = np.asarray(queries, dtype=np.int64)
    others = queries[queries != int(base_query)]
    if not len(others):
        return [(int(base_query), 0.0)]
    base_embedding = cache.get([int(manifest["query_row"][base_query])])[0]
    other_embedding = cache.get(np.asarray(manifest["query_row"])[others])
    distances = 1.0 - other_embedding @ base_embedding
    ordered = sorted(
        zip(map(int, others), map(float, distances), strict=True),
        key=lambda row: (
            signatures[row[0]] != signatures[int(base_query)], row[1], -row[0]
        ),
        reverse=True,
    )
    selected: list[tuple[int, float]] = [(int(base_query), 0.0)]
    seen = {signatures[int(base_query)]}
    for query, distance in ordered:
        if len(selected) >= limit:
            break
        if signatures[query] in seen:
            continue
        selected.append((query, distance))
        seen.add(signatures[query])
    selected_ids = {query for query, _ in selected}
    for query, distance in ordered:
        if len(selected) >= limit:
            break
        if query not in selected_ids:
            selected.append((query, distance))
            selected_ids.add(query)
    return selected


def select_identity_anchors(
    candidates: list[dict[str, object]], base_query: int,
    signatures: Mapping[int, tuple[str, str]], max_anchors: int,
    near_margin: float,
) -> list[dict[str, object]]:
    if max_anchors < 1:
        raise ValueError("max anchors per identity must be positive")
    by_query = {int(row["query"]): row for row in candidates}
    if int(base_query) not in by_query:
        raise RuntimeError("base identity anchor is absent after preselection")
    selected = [by_query[int(base_query)]]
    seen_conditions = {signatures[int(base_query)]}
    seen_negatives = {str(by_query[int(base_query)]["hardest_identity"])}
    others = [row for row in candidates if int(row["query"]) != int(base_query)]
    others.sort(key=lambda row: (
        bool(row["error"]),
        float(row["margin"]) <= near_margin,
        signatures[int(row["query"])] != signatures[int(base_query)],
        str(row["hardest_identity"]) != str(by_query[int(base_query)]["hardest_identity"]),
        -float(row["margin"]),
        float(row.get("embedding_distance", 0.0)),
        -int(row["query"]),
    ), reverse=True)
    for row in others:
        if len(selected) >= max_anchors:
            break
        query = int(row["query"])
        condition_novel = signatures[query] not in seen_conditions
        negative_novel = str(row["hardest_identity"]) not in seen_negatives
        boundary_relevant = bool(row["error"]) or float(row["margin"]) <= near_margin
        if not boundary_relevant and not negative_novel:
            continue
        if not condition_novel and not negative_novel:
            continue
        selected.append(row)
        seen_conditions.add(signatures[query])
        seen_negatives.add(str(row["hardest_identity"]))
    return selected


def append_query_events(
    writer: PoolWriter, geometry: Mapping[str, object], action_identity: Mapping[str, int],
    *, is_extra: bool, negative_references: int, chemical_candidates: int,
) -> dict[str, int]:
    query = int(geometry["query"])
    anchor = int(geometry["anchor"])
    positive = int(geometry["positive_row"])
    hardest = int(geometry["hardest_candidate"])
    candidates = geometry["candidates"]
    if not isinstance(candidates, dict):
        raise TypeError("invalid candidate geometry")
    hardest_body = candidates[hardest]
    counts = {"safe": 0, "official_error": 0, "chemical_error": 0}
    if not bool(geometry["error"]):
        writer.append(
            anchor, [positive], [int(hardest_body["rows"][0])], query, hardest,
            OFFICIAL_HARD, EXTRA_SAFE_BOUNDARY if is_extra else BASE_SAFE_BOUNDARY,
        )
        counts["safe"] += 1
        return counts

    active = np.asarray(hardest_body["active_rows"], dtype=np.int64)
    if not len(active):
        raise RuntimeError(f"official-error query {query} has no active hardest boundary")
    for negative in active[:negative_references]:
        writer.append(
            anchor, [positive], [int(negative)], query, hardest, OFFICIAL_HARD,
            EXTRA_ERROR_BOUNDARY if is_extra else BASE_ERROR_BOUNDARY,
        )
        counts["official_error"] += 1

    transferable = []
    for candidate, body in candidates.items():
        identity = str(body["identity"])
        if int(candidate) == hardest or identity not in action_identity:
            continue
        candidate_active = np.asarray(body["active_rows"], dtype=np.int64)
        if not len(candidate_active):
            continue
        tag = int(action_identity[identity])
        transferable.append((
            bool(tag & SPECIFIC_HARD), float(body["maximum_hinge"]),
            -int(candidate), int(candidate), tag, candidate_active,
        ))
    transferable.sort(reverse=True)
    for _, _, _, candidate, tag, candidate_active in transferable[:chemical_candidates]:
        for negative in candidate_active[:negative_references]:
            writer.append(
                anchor, [positive], [int(negative)], query, candidate, tag,
                EXTRA_CHEMICAL_BOUNDARY if is_extra else BASE_CHEMICAL_BOUNDARY,
            )
            counts["chemical_error"] += 1
    return counts


def identity_equal_sampling_weights(
    output: Mapping[str, np.ndarray], data: Path, replay_fraction: float,
    error_fraction: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if not 0.0 < replay_fraction < 0.5 or not 0.0 < error_fraction < 0.5:
        raise ValueError("replay and error fractions must each lie in (0, 0.5)")
    if replay_fraction + error_fraction >= 1.0:
        raise ValueError("replay plus error sampling mass must be below one")
    anchors = np.asarray(output["anchor_idx"], dtype=np.int64)
    roles = np.asarray(output["curriculum_role"], dtype=np.int8)
    with h5py.File(data, "r") as handle:
        identity = np.asarray([_text(handle["INCHIKEY"][int(row)]) for row in anchors])
    replay = roles == DREAMS_NATIVE_REPLAY
    error = np.isin(roles, [BASE_ERROR_BOUNDARY, BASE_CHEMICAL_BOUNDARY,
                            EXTRA_ERROR_BOUNDARY, EXTRA_CHEMICAL_BOUNDARY])
    safety = np.isin(roles, [BASE_SAFE_BOUNDARY, EXTRA_SAFE_BOUNDARY])
    if not np.all(replay | error | safety) or np.any(
        (replay.astype(np.int8) + error.astype(np.int8) + safety.astype(np.int8)) != 1
    ):
        raise RuntimeError("curriculum roles do not form replay/error/safety strata")
    if not np.any(safety) or not np.any(error) or not np.any(replay):
        raise RuntimeError("weighted pool requires safety, error and replay events")
    weights = np.zeros(len(anchors), dtype=np.float64)
    audits: dict[str, object] = {}
    for name, mask, mass in (
        ("error_boundary", error, error_fraction),
        ("safety_boundary", safety, 1.0 - replay_fraction - error_fraction),
        ("replay", replay, replay_fraction),
    ):
        unique, inverse = np.unique(identity[mask], return_inverse=True)
        count = np.bincount(inverse)
        local = mass / len(unique) / count[inverse]
        weights[np.flatnonzero(mask)] = local
        per_identity = np.bincount(inverse, weights=local)
        audits[name] = {
            "events": int(np.sum(mask)),
            "identities": int(len(unique)),
            "total_sampling_mass": float(np.sum(local)),
            "minimum_identity_sampling_mass": float(np.min(per_identity)),
            "maximum_identity_sampling_mass": float(np.max(per_identity)),
        }
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("invalid identity-balanced sampling weights")
    weights /= weights.sum()
    audits["normalized_total_sampling_mass"] = float(weights.sum())
    return weights, audits


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.max_anchors_per_identity < 2:
        raise ValueError("multi-condition construction requires at least two anchors per identity")
    if args.preselect_anchors_per_identity < args.max_anchors_per_identity:
        raise ValueError("preselection must be at least the retained anchor cap")
    if args.negative_references_per_error < 1:
        raise ValueError("negative reference cap must be positive")
    if args.chemical_candidates_per_error < 0 or args.transferred_chemical_candidates_per_error < 0:
        raise ValueError("chemical candidate caps cannot be negative")
    if set(args.training_roles) != {0, 1}:
        raise ValueError("formal multi-condition builder is restricted to formula roles 0 and 1")

    base_report = json.loads((args.base_bank / "report.json").read_text(encoding="utf-8"))
    if base_report.get("status") != "CHEMAWARE_ACTION_HARD_NATIVE_TRIPLETS_COMPLETE":
        raise RuntimeError("base bank is not the frozen successful ChemAware curriculum")
    base = load_npz(args.base_bank / "train_pool.npz")
    evidence = load_npz(args.evidence)
    manifest = load_npz(args.manifest)
    replay_pool = load_npz(args.dreams_replay_pool)
    cache = FrozenEmbeddings(args.embedding_rows, args.official_embeddings)
    folds = stable_formula_folds(manifest["query_formula"], args.folds, args.fold_seed)
    allowed = np.isin(folds, np.asarray(args.training_roles, dtype=folds.dtype))
    training_queries = np.flatnonzero(allowed)
    groups: dict[str, list[int]] = defaultdict(list)
    for query in training_queries:
        groups[str(manifest["query_ik14"][query])].append(int(query))

    base_queries = np.asarray(evidence["query"], dtype=np.int64)
    base_identity = np.asarray(evidence["identity"]).astype(str)
    if len(np.unique(base_identity)) != len(base_identity):
        raise RuntimeError("base evidence is not one-query-per-identity")
    base_by_identity = dict(zip(map(str, base_identity), map(int, base_queries), strict=True))
    if set(base_by_identity) != set(groups):
        raise RuntimeError("base evidence does not cover every training identity exactly once")
    if np.any(~allowed[base_queries]):
        raise RuntimeError("base evidence reaches outside formula roles 0 and 1")
    expected_rank = dict(zip(map(int, base_queries), map(int, evidence["baseline_rank"]), strict=True))
    action_by_query = action_identities_by_base_query(base, manifest)

    with h5py.File(args.data, "r") as handle:
        instrument = np.asarray(handle["INSTRUMENT_TYPE"][:])
        collision_energy = np.asarray(handle["COLLISION_ENERGY"][:], dtype=np.float64)
    signatures = {
        int(query): condition_signature(
            instrument[int(manifest["query_row"][query])],
            collision_energy[int(manifest["query_row"][query])],
            args.collision_energy_bin,
        )
        for query in training_queries
    }

    writer = PoolWriter()
    # Preserve the exact successful Phase-A base curriculum.  In particular,
    # use its frozen baseline ranks and official-hard event rather than
    # recomputing the one-query-per-identity arm under a subtly different tie
    # convention.  Multi-condition events are an additive intervention.
    base_focused = build_focused_pool(
        base, evidence, cache, writer, args.margin,
        args.negative_references_per_error, args.chemical_candidates_per_error,
    )
    base_focused_events = len(writer.anchor)
    base_focused_snapshot = writer.arrays()
    base_focused_sha256 = pool_prefix_semantic_sha256(
        base_focused_snapshot, base_focused_events,
    )
    selected_queries: list[int] = []
    selected_by_identity: dict[str, int] = {}
    geometry_by_query: dict[int, dict[str, object]] = {}
    condition_diverse_extras = boundary_novel_extras = 0
    counts = defaultdict(int, {
        "base_safe": int(base_focused["safe_max_boundary_events"]),
        "base_official_error": int(base_focused["error_official_max_boundary_events"]),
        "base_chemical_error": int(base_focused["error_chemical_max_boundary_events"]),
    })
    base_error_queries = int(np.sum(np.asarray(evidence["baseline_rank"]) != 1))
    base_error_boundaries: set[tuple[str, str]] = set()
    selected_error_boundaries: set[tuple[str, str]] = set()

    for identity in sorted(groups):
        base_query = base_by_identity[identity]
        preselected = preselect_queries(
            np.asarray(groups[identity], dtype=np.int64), base_query, manifest,
            cache, signatures, args.preselect_anchors_per_identity,
        )
        geometries = []
        for query, distance in preselected:
            geometry = query_geometry(manifest, cache, query, args.margin)
            geometry["embedding_distance"] = float(distance)
            geometry_by_query[query] = geometry
            geometries.append(geometry)
        # The successful Phase-A ledger is authoritative for its base anchor.
        # Exact cosine ties can otherwise differ by rank implementation even
        # when every underlying score is identical.
        geometry_by_query[base_query]["error"] = expected_rank[base_query] != 1
        if expected_rank[base_query] != 1:
            base_error_boundaries.add((identity, str(geometry_by_query[base_query]["hardest_identity"])))
        selected = select_identity_anchors(
            geometries, base_query, signatures, args.max_anchors_per_identity,
            args.near_boundary_margin,
        )
        selected_by_identity[identity] = len(selected)
        base_condition = signatures[base_query]
        base_negative = str(geometry_by_query[base_query]["hardest_identity"])
        for geometry in selected:
            query = int(geometry["query"])
            is_extra = query != base_query
            selected_queries.append(query)
            if is_extra and signatures[query] != base_condition:
                condition_diverse_extras += 1
            if is_extra and str(geometry["hardest_identity"]) != base_negative:
                boundary_novel_extras += 1
            if is_extra:
                event_counts = append_query_events(
                    writer, geometry, action_by_query.get(base_query, {}),
                    is_extra=True,
                    negative_references=args.negative_references_per_error,
                    chemical_candidates=args.transferred_chemical_candidates_per_error,
                )
                for key, value in event_counts.items():
                    counts["extra_" + key] += value
            if bool(geometry["error"]):
                selected_error_boundaries.add((identity, str(geometry["hardest_identity"])))

    focused_events = len(writer.anchor)
    replay_events = int(round(focused_events * args.replay_fraction / (1.0 - args.replay_fraction)))
    append_dreams_replay(writer, replay_pool, replay_events, args.seed)
    output = writer.arrays()
    phase_a_prefix_immutable = pool_prefix_equal(
        base_focused_snapshot, output, base_focused_events,
    )
    observed_base_focused_sha256 = pool_prefix_semantic_sha256(
        output, base_focused_events,
    )
    weights, weight_audit = identity_equal_sampling_weights(
        output, args.data, args.replay_fraction, args.error_sampling_fraction,
    )
    output["sampling_weight"] = weights
    roles = np.asarray(output["curriculum_role"], dtype=np.int8)
    selected_queries_array = np.asarray(selected_queries, dtype=np.int64)
    base_query_set = set(map(int, base_queries))
    selected_error_queries = int(base_error_queries + sum(
        query not in base_query_set and bool(geometry_by_query[query]["error"])
        for query in selected_queries
    ))
    extra_queries = int(len(selected_queries) - len(base_queries))
    extra_error_queries = int(sum(
        query not in base_query_set and bool(geometry_by_query[query]["error"])
        for query in selected_queries
    ))
    focused_singleton = bool(np.all(
        np.diff(output["positive_ptr"][:focused_events + 1]) == 1
    ) and np.all(np.diff(output["negative_ptr"][:focused_events + 1]) == 1))
    selected_folds = folds[selected_queries_array]
    gates = {
        "all_frozen_base_queries_retained": set(map(int, base_queries)).issubset(selected_queries),
        "phase_a_base_event_prefix_immutable": (
            phase_a_prefix_immutable
            and observed_base_focused_sha256 == base_focused_sha256
        ),
        "only_training_formula_roles_used": set(map(int, np.unique(selected_folds))) == {0, 1},
        "outer_roles_2_3_4_untouched": True,
        "anchor_cap_respected": max(selected_by_identity.values()) <= args.max_anchors_per_identity,
        "additional_condition_anchors_exist": extra_queries > 0,
        "additional_official_error_anchors_exist": extra_error_queries > 0,
        "official_error_anchor_coverage_increased": selected_error_queries > base_error_queries,
        "distinct_error_boundaries_not_reduced": (
            len(selected_error_boundaries) >= len(base_error_boundaries)
        ),
        "condition_diverse_extras_exist": condition_diverse_extras > 0,
        "focused_events_are_singleton_boundaries": focused_singleton,
        "exact_proportional_replay_budget": int(np.sum(roles == DREAMS_NATIVE_REPLAY)) == replay_events,
        "sampling_weights_positive_and_normalized": bool(
            np.all(weights > 0) and abs(float(weights.sum()) - 1.0) <= 1e-12
        ),
        "unique_triplet_signatures": len(writer.signatures) == len(output["anchor_idx"]),
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"multi-condition max-boundary gates failed: {gates}; "
            f"base_errors={base_error_queries} selected_errors={selected_error_queries}"
        )
    identity_audit = audit_identity_edges(output, args.data)
    report = {
        "status": "CHEMAWARE_MULTICONDITION_MAX_BOUNDARY_TRIPLETS_COMPLETE",
        "method": (
            "identity-equal multi-condition exact max-reference triplets with "
            "active ChemAware identity transfer and proportional native DreaMS replay"
        ),
        "training_runtime": "native DreaMS ContrastiveSpectraDataset plus ContrastiveHead",
        "formula_roles": {"training": [0, 1], "selection": 2, "confirmation": 3, "outer": 4},
        "settings": {
            "margin": float(args.margin),
            "near_boundary_margin": float(args.near_boundary_margin),
            "max_anchors_per_identity": int(args.max_anchors_per_identity),
            "preselect_anchors_per_identity": int(args.preselect_anchors_per_identity),
            "collision_energy_bin": float(args.collision_energy_bin),
            "negative_references_per_error": int(args.negative_references_per_error),
            "chemical_candidates_per_base_error": int(args.chemical_candidates_per_error),
            "transferred_chemical_candidates_per_extra_error": int(
                args.transferred_chemical_candidates_per_error
            ),
            "replay_fraction": float(args.replay_fraction),
            "error_sampling_fraction": float(args.error_sampling_fraction),
        },
        "coverage": {
            "training_queries_available": int(len(training_queries)),
            "training_identities": int(len(groups)),
            "frozen_base_queries": int(len(base_queries)),
            "selected_anchor_queries": int(len(selected_queries)),
            "extra_anchor_queries": extra_queries,
            "base_official_error_queries": base_error_queries,
            "selected_official_error_queries": selected_error_queries,
            "extra_official_error_queries": extra_error_queries,
            "base_distinct_identity_negative_boundaries": int(len(base_error_boundaries)),
            "selected_distinct_identity_negative_boundaries": int(len(selected_error_boundaries)),
            "condition_diverse_extra_anchors": int(condition_diverse_extras),
            "new_hardest_identity_extra_anchors": int(boundary_novel_extras),
            "identities_with_one_anchor": int(sum(value == 1 for value in selected_by_identity.values())),
            "identities_with_multiple_anchors": int(sum(value > 1 for value in selected_by_identity.values())),
        },
        "phase_a_base_preservation": {
            "events": int(base_focused_events),
            "semantic_sha256_before_expansion": base_focused_sha256,
            "semantic_sha256_after_expansion": observed_base_focused_sha256,
            "exact_prefix_equal": bool(phase_a_prefix_immutable),
        },
        "events": {
            **{key: int(value) for key, value in sorted(counts.items())},
            "focused": int(focused_events),
            "replay": int(replay_events),
            "total": int(len(output["anchor_idx"])),
        },
        "sampling": weight_audit,
        "identity_audit": identity_audit,
        "gates": gates,
        "claim_boundary": (
            "construction and sampling audit only; retrieval improvement requires "
            "role-2 selection and role-3 confirmation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_multicondition_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **output)
        shutil.copy2(args.base_bank / "val_pool.npz", temporary / "val_pool.npz")
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
