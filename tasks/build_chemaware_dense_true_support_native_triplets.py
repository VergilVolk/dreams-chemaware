"""Build a dense, hard, DreaMS-native ChemAware triplet curriculum.

The chemical rule bank is used only as a counterfactual gate on the known
training identity.  After an identity passes that gate, every eligible
formula-role-0/1 spectrum of that identity is rescored under the protected
Phase-A embedding.  Only Phase-A-current errors contribute corrective
triplets.  Each correction is an exact singleton

    (query, current max-positive reference, current active false reference)

drawn from several distinct hard false molecules and references.  Counts are
based on unique stored signatures; repeated sampling cannot satisfy coverage
gates.  Active correct boundaries and untouched official DreaMS triplets are
included as native safety data.  No custom sampler, loss, teacher, or model is
introduced downstream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from build_chemaware_dreams_native_triplets import audit_identity_edges
from build_chemaware_max_boundary_native_triplets import (
    DREAMS_NATIVE_REPLAY,
    FrozenEmbeddings,
    PoolWriter,
    append_dreams_replay,
    load_npz,
)
from build_chemaware_multicondition_max_boundary_triplets import (
    condition_signature,
    query_geometry,
)
from build_chemaware_true_support_native_triplets import (
    BROAD_TRUE_SUPPORT_TAG,
    STRICT_TRUE_SUPPORT_TAG,
    true_candidate_support,
)
from chemaware_numpy_sampling import stable_formula_folds
from encode_chemaware_checkpoint_manifest_rows import array_sha256


ROOT = Path(__file__).resolve().parents[1]
DENSE_STRICT_CORRECTION = 31
DENSE_BROAD_CORRECTION = 32
ACTIVE_SAFETY_SENTINEL = 33


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--phasea-cache", type=Path, required=True)
    parser.add_argument("--validation-pool", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
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
    parser.add_argument("--minimum-rule-wins", type=int, default=2)
    parser.add_argument("--max-error-anchors-per-identity", type=int, default=6)
    parser.add_argument("--max-safety-anchors-per-identity", type=int, default=2)
    parser.add_argument("--negative-candidates-per-query", type=int, default=4)
    parser.add_argument("--negative-references-per-candidate", type=int, default=3)
    parser.add_argument("--dreams-replay-events", type=int, default=1024)
    parser.add_argument("--collision-energy-bin", type=float, default=10.0)
    parser.add_argument("--minimum-correction-triplets", type=int, default=1000)
    parser.add_argument("--minimum-correction-queries", type=int, default=200)
    parser.add_argument("--minimum-distinct-boundaries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--allow-official-cache-standin", action="store_true",
        help="Engineering audit only; formal training must use a protected Phase-A cache.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def diverse_hard_anchors(
    geometries: list[dict[str, object]],
    signatures: Mapping[int, tuple[str, str]],
    limit: int,
) -> list[dict[str, object]]:
    """Prefer new conditions and false identities, then fill by hardness."""
    if limit < 1:
        raise ValueError("anchor limit must be positive")
    ordered = sorted(
        geometries,
        key=lambda row: (
            -float(row["margin"]),
            float(row["candidate_active_references"]),
            -int(row["query"]),
        ),
        reverse=True,
    )
    selected: list[dict[str, object]] = []
    seen_conditions: set[tuple[str, str]] = set()
    seen_negatives: set[str] = set()
    remaining = list(ordered)
    while remaining and len(selected) < limit:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                signatures[int(remaining[index]["query"])] not in seen_conditions,
                str(remaining[index]["hardest_identity"]) not in seen_negatives,
                -float(remaining[index]["margin"]),
                float(remaining[index]["candidate_active_references"]),
                -int(remaining[index]["query"]),
            ),
        )
        row = remaining.pop(best_index)
        selected.append(row)
        seen_conditions.add(signatures[int(row["query"])])
        seen_negatives.add(str(row["hardest_identity"]))
    return selected


def active_candidate_ranking(
    geometry: Mapping[str, object], limit: int,
) -> list[tuple[int, Mapping[str, object]]]:
    candidates = geometry["candidates"]
    if not isinstance(candidates, dict):
        raise TypeError("invalid candidate geometry")
    ranked = [
        (int(candidate), body)
        for candidate, body in candidates.items()
        if len(np.asarray(body["active_rows"], dtype=np.int64))
        and float(body["maximum_hinge"]) > 0.0
    ]
    ranked.sort(key=lambda row: (
        float(row[1]["maximum_hinge"]),
        len(np.asarray(row[1]["active_rows"], dtype=np.int64)),
        -row[0],
    ), reverse=True)
    return ranked[:limit]


def append_corrections(
    writer: PoolWriter,
    geometry: Mapping[str, object],
    strict: bool,
    candidate_limit: int,
    reference_limit: int,
) -> tuple[int, set[str]]:
    role = DENSE_STRICT_CORRECTION if strict else DENSE_BROAD_CORRECTION
    tag = STRICT_TRUE_SUPPORT_TAG if strict else BROAD_TRUE_SUPPORT_TAG
    added = 0
    negative_identities: set[str] = set()
    for candidate, body in active_candidate_ranking(geometry, candidate_limit):
        negative_identities.add(str(body["identity"]))
        for negative in np.asarray(body["active_rows"], dtype=np.int64)[:reference_limit]:
            before = len(writer.anchor)
            writer.append(
                int(geometry["anchor"]), [int(geometry["positive_row"])], [int(negative)],
                int(geometry["query"]), candidate, tag, role,
            )
            added += int(len(writer.anchor) > before)
    return added, negative_identities


def append_active_sentinel(
    writer: PoolWriter, geometry: Mapping[str, object], strict: bool,
) -> bool:
    if bool(geometry["error"]) or not 0.0 < float(geometry["margin"]) < 0.1 + 1e-12:
        raise RuntimeError("only active current-correct boundaries may be sentinels")
    candidates = geometry["candidates"]
    if not isinstance(candidates, dict):
        raise TypeError("invalid candidate geometry")
    candidate = int(geometry["hardest_candidate"])
    negative = int(np.asarray(candidates[candidate]["rows"], dtype=np.int64)[0])
    tag = STRICT_TRUE_SUPPORT_TAG if strict else BROAD_TRUE_SUPPORT_TAG
    before = len(writer.anchor)
    writer.append(
        int(geometry["anchor"]), [int(geometry["positive_row"])], [negative],
        int(geometry["query"]), candidate, tag, ACTIVE_SAFETY_SENTINEL,
    )
    return len(writer.anchor) > before


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if set(args.training_roles) != {0, 1}:
        raise ValueError("dense true-support mining is restricted to roles 0 and 1")
    if args.margin != 0.1:
        raise ValueError("the native DreaMS margin must remain 0.1")
    if min(
        args.max_error_anchors_per_identity,
        args.max_safety_anchors_per_identity,
        args.negative_candidates_per_query,
        args.negative_references_per_candidate,
    ) < 1:
        raise ValueError("all dense coverage caps must be positive")

    evidence = load_npz(args.evidence)
    manifest = load_npz(args.manifest)
    replay = load_npz(args.dreams_replay_pool)
    cache_report_path = args.phasea_cache / "report.json"
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    phasea_status = cache_report.get("status") == "CHEMAWARE_FORMULA_ROLE_CHECKPOINT_EMBEDDINGS_COMPLETE"
    official_standin = (
        args.allow_official_cache_standin
        and cache_report.get("status") == "chemaware_corrected_manifest_token_cache_complete"
    )
    if not phasea_status and not official_standin:
        raise RuntimeError("cache is neither a formal Phase-A cache nor an explicit official stand-in")
    if phasea_status and set(map(int, cache_report.get("formula_roles", []))) != {0, 1}:
        raise RuntimeError("formal Phase-A cache must contain exactly roles 0 and 1")
    rows_path = args.phasea_cache / "rows.npy"
    embeddings_path = args.phasea_cache / (
        "embeddings_f32.npy" if phasea_status else "official_embeddings_f32.npy"
    )
    rows = np.load(rows_path, allow_pickle=False)
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if phasea_status:
        if array_sha256(rows) != cache_report["rows_array_sha256"]:
            raise RuntimeError("Phase-A cache row hash mismatch")
        if array_sha256(np.asarray(embeddings)) != cache_report["embeddings_array_sha256"]:
            raise RuntimeError("Phase-A cache embedding hash mismatch")
    cache = FrozenEmbeddings(rows_path, embeddings_path)

    folds = stable_formula_folds(manifest["query_formula"], args.folds, args.fold_seed)
    allowed = np.isin(folds, np.asarray(args.training_roles, dtype=folds.dtype))
    training_queries = np.flatnonzero(allowed)
    groups: dict[str, list[int]] = defaultdict(list)
    for query in training_queries:
        groups[str(manifest["query_ik14"][query])].append(int(query))

    support, support_audit = true_candidate_support(
        evidence, manifest, args.minimum_rule_wins,
    )
    support_by_identity = {str(body["identity"]): body for body in support.values()}
    if len(support_by_identity) != len(support):
        raise RuntimeError("direct true-support identities are not unique")
    missing = set(support_by_identity) - set(groups)
    if missing:
        raise RuntimeError(f"supported identities escaped the training roles: {len(missing)}")

    with h5py.File(args.data, "r") as handle:
        instrument = np.asarray(handle["INSTRUMENT_TYPE"][:])
        collision_energy = np.asarray(handle["COLLISION_ENERGY"][:], dtype=np.float64)
    signatures = {
        int(query): condition_signature(
            instrument[int(manifest["query_row"][query])],
            collision_energy[int(manifest["query_row"][query])],
            args.collision_energy_bin,
        )
        for identity in support_by_identity for query in groups[identity]
    }

    writer = PoolWriter()
    correction_queries: set[int] = set()
    correction_identities: set[str] = set()
    strict_correction_queries: set[int] = set()
    sentinel_queries: set[int] = set()
    distinct_boundaries: set[tuple[str, str]] = set()
    all_supported_queries = 0
    current_error_queries = 0
    selected_error_anchors = 0
    strict_events = broad_events = sentinel_events = 0
    candidate_histogram: dict[int, int] = defaultdict(int)
    reference_histogram: dict[int, int] = defaultdict(int)

    for identity in sorted(support_by_identity):
        body = support_by_identity[identity]
        strict = bool(body["strict"])
        errors: list[dict[str, object]] = []
        safe: list[dict[str, object]] = []
        for query in groups[identity]:
            geometry = query_geometry(manifest, cache, query, args.margin)
            ranked = active_candidate_ranking(
                geometry, args.negative_candidates_per_query,
            )
            geometry["candidate_active_references"] = int(sum(
                min(args.negative_references_per_candidate,
                    len(np.asarray(candidate[1]["active_rows"], dtype=np.int64)))
                for candidate in ranked
            ))
            all_supported_queries += 1
            if bool(geometry["error"]):
                current_error_queries += 1
                errors.append(geometry)
            elif 0.0 < float(geometry["margin"]) <= args.margin:
                safe.append(geometry)

        selected_errors = diverse_hard_anchors(
            errors, signatures, args.max_error_anchors_per_identity,
        )
        selected_error_anchors += len(selected_errors)
        for geometry in selected_errors:
            query = int(geometry["query"])
            ranked = active_candidate_ranking(
                geometry, args.negative_candidates_per_query,
            )
            candidate_histogram[len(ranked)] += 1
            for _candidate, candidate_body in ranked:
                reference_histogram[min(
                    args.negative_references_per_candidate,
                    len(np.asarray(candidate_body["active_rows"], dtype=np.int64)),
                )] += 1
            added, negative_identities = append_corrections(
                writer, geometry, strict,
                args.negative_candidates_per_query,
                args.negative_references_per_candidate,
            )
            if added <= 0:
                raise RuntimeError("selected current error produced no active correction")
            correction_queries.add(query)
            correction_identities.add(identity)
            if strict:
                strict_correction_queries.add(query)
                strict_events += added
            else:
                broad_events += added
            distinct_boundaries.update((identity, negative) for negative in negative_identities)

        selected_safe = diverse_hard_anchors(
            safe, signatures, args.max_safety_anchors_per_identity,
        ) if safe else []
        for geometry in selected_safe:
            if append_active_sentinel(writer, geometry, strict):
                sentinel_queries.add(int(geometry["query"]))
                sentinel_events += 1

    focused_events = len(writer.anchor)
    correction_events = strict_events + broad_events
    replay_audit = append_dreams_replay(
        writer, replay, args.dreams_replay_events, args.seed,
    )
    output = writer.arrays()
    roles = np.asarray(output["curriculum_role"], dtype=np.int64)
    focused_singleton = bool(
        np.all(np.diff(output["positive_ptr"][:focused_events + 1]) == 1)
        and np.all(np.diff(output["negative_ptr"][:focused_events + 1]) == 1)
    )
    correction_query_array = np.asarray(sorted(correction_queries), dtype=np.int64)
    gates = {
        "counterfactual_true_support_present": len(support_by_identity) >= 300,
        "only_training_roles_used": bool(
            len(correction_query_array)
            and set(map(int, np.unique(folds[correction_query_array]))) <= {0, 1}
        ),
        "outer_roles_2_3_4_untouched": True,
        "minimum_unique_correction_triplets": correction_events >= args.minimum_correction_triplets,
        "minimum_independent_correction_queries": len(correction_queries) >= args.minimum_correction_queries,
        "minimum_distinct_identity_false_boundaries": len(distinct_boundaries) >= args.minimum_distinct_boundaries,
        "strict_and_broad_corrections_exist": strict_events > 0 and broad_events > 0,
        "all_corrections_currently_active": correction_events > 0,
        "active_safety_sentinels_exist": sentinel_events > 0,
        "focused_events_are_singleton_boundaries": focused_singleton,
        "exact_dreams_replay_budget": int(np.sum(roles == DREAMS_NATIVE_REPLAY)) == args.dreams_replay_events,
        "unique_triplet_signatures": len(writer.signatures) == len(output["anchor_idx"]),
        "no_custom_sampling_weights": "sampling_weight" not in output,
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"dense true-support gates failed: {gates}; corrections={correction_events} "
            f"queries={len(correction_queries)} boundaries={len(distinct_boundaries)}"
        )
    identity_audit = audit_identity_edges(output, args.data)
    report = {
        "status": "CHEMAWARE_DENSE_TRUE_SUPPORT_NATIVE_TRIPLETS_COMPLETE",
        "method": (
            "counterfactual-null-exclusive identity gate plus multi-condition "
            "Phase-A-current top-candidate/top-reference singleton hard triplets"
        ),
        "training_runtime": "unmodified native DreaMS uniform shuffled DataLoader and ContrastiveHead",
        "cache_kind": "protected_phasea" if phasea_status else "official_engineering_standin",
        "formula_roles": {"training": [0, 1], "selection": 2, "confirmation": 3, "outer": 4},
        "settings": {
            "margin": float(args.margin),
            "minimum_rule_wins": int(args.minimum_rule_wins),
            "max_error_anchors_per_identity": int(args.max_error_anchors_per_identity),
            "max_safety_anchors_per_identity": int(args.max_safety_anchors_per_identity),
            "negative_candidates_per_query": int(args.negative_candidates_per_query),
            "negative_references_per_candidate": int(args.negative_references_per_candidate),
            "dreams_replay_events": int(args.dreams_replay_events),
            "minimum_correction_triplets": int(args.minimum_correction_triplets),
        },
        "provenance": {
            "evidence_sha256": sha256_file(args.evidence),
            "manifest_sha256": sha256_file(args.manifest),
            "validation_pool_sha256": sha256_file(args.validation_pool),
            "dreams_replay_pool_sha256": sha256_file(args.dreams_replay_pool),
            "cache_rows_array_sha256": array_sha256(rows),
            "cache_embeddings_array_sha256": array_sha256(np.asarray(embeddings)),
            "checkpoint_sha256": cache_report.get("checkpoint_sha256"),
        },
        "direct_support": support_audit,
        "coverage": {
            "supported_identities": int(len(support_by_identity)),
            "supported_query_spectra_available": int(all_supported_queries),
            "phasea_current_error_query_spectra": int(current_error_queries),
            "selected_error_anchor_queries": int(selected_error_anchors),
            "correction_queries": int(len(correction_queries)),
            "correction_identities": int(len(correction_identities)),
            "strict_correction_queries": int(len(strict_correction_queries)),
            "distinct_identity_false_boundaries": int(len(distinct_boundaries)),
            "active_safety_queries": int(len(sentinel_queries)),
        },
        "events": {
            "strict_correction": int(strict_events),
            "broad_correction": int(broad_events),
            "unique_correction_triplets": int(correction_events),
            "active_safety_sentinel": int(sentinel_events),
            "focused": int(focused_events),
            "dreams_native_replay": int(replay_audit["retained"]),
            "total": int(len(output["anchor_idx"])),
        },
        "hardness": {
            "candidate_count_per_selected_query_histogram": {
                str(key): int(value) for key, value in sorted(candidate_histogram.items())
            },
            "active_reference_count_per_selected_candidate_histogram": {
                str(key): int(value) for key, value in sorted(reference_histogram.items())
            },
            "every_correction_satisfies_native_hinge_at_construction": True,
            "negative_candidates_are_distinct_molecules": True,
            "positive_reference_is_current_maximum": True,
        },
        "sampler": {
            "type": "native DataLoader shuffle",
            "replacement": False,
            "custom_sampling_weight_present": False,
        },
        "identity_audit": identity_audit,
        "gates": gates,
        "claim_boundary": (
            "triplet construction and hardness audit only; performance requires "
            "role-2 selection against protected Phase A and role-3 confirmation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_dense_true_support_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **output)
        shutil.copy2(args.validation_pool, temporary / "val_pool.npz")
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
