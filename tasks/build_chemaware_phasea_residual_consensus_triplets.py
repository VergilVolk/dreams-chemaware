"""Build Phase-A-current, multi-condition residual native triplets.

The successful Phase-A pool is copied byte-semantically as the protection
prefix.  New events are mined under the protected Phase-A embedding, not under
official DreaMS.  Residual errors that repeat the same false identity across
experimental conditions receive the largest additional sampling mass; isolated
errors receive less mass; and correct but active near-boundary conditions serve
as overshoot sentinels.  Model, loss and optimizer remain native DreaMS.
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

import h5py
import numpy as np

from build_chemaware_dreams_native_triplets import audit_identity_edges
from build_chemaware_max_boundary_native_triplets import (
    FrozenEmbeddings,
    PoolWriter,
    load_npz,
)
from build_chemaware_multicondition_max_boundary_triplets import (
    action_identities_by_base_query,
    condition_signature,
    pool_prefix_equal,
    pool_prefix_semantic_sha256,
    preselect_queries,
    query_geometry,
)
from chemaware_numpy_sampling import stable_formula_folds
from encode_chemaware_checkpoint_manifest_rows import array_sha256


ROOT = Path(__file__).resolve().parents[1]
RESIDUAL_CONSENSUS = 20
RESIDUAL_CONSENSUS_CHEMICAL = 21
RESIDUAL_ISOLATED = 22
RESIDUAL_ISOLATED_CHEMICAL = 23
PROTECTION_SENTINEL = 24


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phasea-bank", type=Path, required=True)
    parser.add_argument("--base-action-bank", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--phasea-cache", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
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
    parser.add_argument("--protection-margin", type=float, default=0.1)
    parser.add_argument("--collision-energy-bin", type=float, default=10.0)
    parser.add_argument("--preselect-anchors-per-identity", type=int, default=8)
    parser.add_argument("--max-residual-anchors-per-identity", type=int, default=2)
    parser.add_argument("--chemical-candidates-per-error", type=int, default=1)
    parser.add_argument("--base-sampling-mass", type=float, default=0.70)
    parser.add_argument("--consensus-sampling-mass", type=float, default=0.18)
    parser.add_argument("--isolated-sampling-mass", type=float, default=0.07)
    parser.add_argument("--sentinel-sampling-mass", type=float, default=0.05)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def event_edges(pool: Mapping[str, np.ndarray], event: int, kind: str) -> np.ndarray:
    pointer = np.asarray(pool[f"{kind}_ptr"], dtype=np.int64)
    values = np.asarray(pool[f"{kind}_idx"], dtype=np.int64)
    left, right = map(int, pointer[event:event + 2])
    return values[left:right]


def copy_pool(writer: PoolWriter, pool: Mapping[str, np.ndarray]) -> None:
    events = len(pool["anchor_idx"])
    for event in range(events):
        writer.append(
            int(pool["anchor_idx"][event]),
            event_edges(pool, event, "positive"),
            event_edges(pool, event, "negative"),
            int(pool["source_query"][event]),
            int(pool["negative_candidate"][event]),
            int(pool["source_tag"][event]),
            int(pool["curriculum_role"][event]),
        )
    if len(writer.anchor) != events:
        raise RuntimeError("Phase-A pool contains duplicate triplet signatures")


def transferable_chemical_candidates(
    geometry: Mapping[str, object], action_identities: Mapping[str, int], limit: int,
) -> list[tuple[int, int, int]]:
    candidates = geometry["candidates"]
    if not isinstance(candidates, dict):
        raise TypeError("invalid candidate geometry")
    hardest = int(geometry["hardest_candidate"])
    ranked = []
    for candidate, body in candidates.items():
        identity = str(body["identity"])
        active = np.asarray(body["active_rows"], dtype=np.int64)
        if int(candidate) == hardest or identity not in action_identities or not len(active):
            continue
        tag = int(action_identities[identity])
        ranked.append((
            float(body["maximum_hinge"]), -int(candidate),
            int(candidate), tag, int(active[0]),
        ))
    ranked.sort(reverse=True)
    return [(candidate, tag, row) for _, _, candidate, tag, row in ranked[:limit]]


def append_residual_error(
    writer: PoolWriter, geometry: Mapping[str, object],
    action_identities: Mapping[str, int], consensus: bool,
    chemical_candidates: int,
) -> dict[str, int]:
    if not bool(geometry["error"]):
        raise RuntimeError("non-error geometry reached residual-error writer")
    candidates = geometry["candidates"]
    if not isinstance(candidates, dict):
        raise TypeError("invalid candidate geometry")
    hardest = int(geometry["hardest_candidate"])
    hardest_body = candidates[hardest]
    active = np.asarray(hardest_body["active_rows"], dtype=np.int64)
    if not len(active):
        raise RuntimeError("Phase-A current error has no active hardest reference")
    query = int(geometry["query"])
    anchor = int(geometry["anchor"])
    positive = int(geometry["positive_row"])
    official_role = RESIDUAL_CONSENSUS if consensus else RESIDUAL_ISOLATED
    chemical_role = (
        RESIDUAL_CONSENSUS_CHEMICAL if consensus
        else RESIDUAL_ISOLATED_CHEMICAL
    )
    before = len(writer.anchor)
    writer.append(
        anchor, [positive], [int(active[0])], query, hardest, 0, official_role,
    )
    official_added = int(len(writer.anchor) > before)
    chemical_added = 0
    for candidate, tag, negative in transferable_chemical_candidates(
        geometry, action_identities, chemical_candidates,
    ):
        before = len(writer.anchor)
        writer.append(
            anchor, [positive], [negative], query, candidate, tag, chemical_role,
        )
        chemical_added += int(len(writer.anchor) > before)
    return {"official": official_added, "chemical": chemical_added}


def append_protection_sentinel(
    writer: PoolWriter, geometry: Mapping[str, object],
) -> bool:
    if bool(geometry["error"]):
        raise RuntimeError("error geometry reached protection sentinel")
    candidates = geometry["candidates"]
    if not isinstance(candidates, dict):
        raise TypeError("invalid candidate geometry")
    hardest = int(geometry["hardest_candidate"])
    row = int(np.asarray(candidates[hardest]["rows"], dtype=np.int64)[0])
    before = len(writer.anchor)
    writer.append(
        int(geometry["anchor"]), [int(geometry["positive_row"])], [row],
        int(geometry["query"]), hardest, 0, PROTECTION_SENTINEL,
    )
    return len(writer.anchor) > before


def identity_equal_mass(
    weights: np.ndarray, mask: np.ndarray, anchor_identity: np.ndarray,
    mass: float,
) -> dict[str, object]:
    positions = np.flatnonzero(mask)
    if not len(positions):
        raise RuntimeError("a frozen residual sampling stratum is empty")
    identities = anchor_identity[positions]
    unique, inverse = np.unique(identities, return_inverse=True)
    counts = np.bincount(inverse)
    local = mass / (len(unique) * counts[inverse])
    weights[positions] = local
    per_identity = np.bincount(inverse, weights=local)
    return {
        "events": int(len(positions)),
        "identities": int(len(unique)),
        "sampling_mass": float(np.sum(local)),
        "minimum_identity_mass": float(np.min(per_identity)),
        "maximum_identity_mass": float(np.max(per_identity)),
    }


def residual_sampling_weights(
    output: Mapping[str, np.ndarray], base_events: int, data: Path,
    masses: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, object]]:
    if abs(sum(masses.values()) - 1.0) > 1e-12 or any(
        value <= 0 for value in masses.values()
    ):
        raise ValueError("residual sampling masses must be positive and sum to one")
    events = len(output["anchor_idx"])
    roles = np.asarray(output["curriculum_role"], dtype=np.int64)
    with h5py.File(data, "r") as handle:
        anchor_identity = np.asarray(handle["INCHIKEY"])[
            np.asarray(output["anchor_idx"], dtype=np.int64)
        ].astype(str)
    weights = np.zeros(events, dtype=np.float64)
    weights[:base_events] = masses["phasea_base"] / base_events
    audit: dict[str, object] = {
        "phasea_base": {
            "events": int(base_events),
            "sampling_mass": float(np.sum(weights[:base_events])),
            "within_stratum": "exact uniform Phase-A event replay",
        },
    }
    audit["residual_consensus"] = identity_equal_mass(
        weights,
        np.isin(roles, [RESIDUAL_CONSENSUS, RESIDUAL_CONSENSUS_CHEMICAL]),
        anchor_identity, masses["residual_consensus"],
    )
    audit["residual_isolated"] = identity_equal_mass(
        weights,
        np.isin(roles, [RESIDUAL_ISOLATED, RESIDUAL_ISOLATED_CHEMICAL]),
        anchor_identity, masses["residual_isolated"],
    )
    audit["protection_sentinel"] = identity_equal_mass(
        weights, roles == PROTECTION_SENTINEL,
        anchor_identity, masses["protection_sentinel"],
    )
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("residual sampling weights are invalid")
    weights /= weights.sum()
    audit["normalized_total_sampling_mass"] = float(weights.sum())
    return weights, audit


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if set(args.training_roles) != {0, 1}:
        raise ValueError("formal residual mining is restricted to roles 0 and 1")
    if not 0 < args.protection_margin <= args.margin:
        raise ValueError("protection margin must lie in (0, native triplet margin]")
    if args.max_residual_anchors_per_identity < 1:
        raise ValueError("at least one residual anchor per identity is required")
    phasea_report = json.loads(
        (args.phasea_bank / "report.json").read_text(encoding="utf-8")
    )
    if phasea_report.get("status") != "CHEMAWARE_MAX_BOUNDARY_NATIVE_TRIPLETS_COMPLETE":
        raise RuntimeError("Phase-A bank is not a max-boundary native pool")
    action_report = json.loads(
        (args.base_action_bank / "report.json").read_text(encoding="utf-8")
    )
    if action_report.get("status") != "CHEMAWARE_ACTION_HARD_NATIVE_TRIPLETS_COMPLETE":
        raise RuntimeError("base action bank is not the frozen successful curriculum")
    cache_report = json.loads(
        (args.phasea_cache / "report.json").read_text(encoding="utf-8")
    )
    if cache_report.get("status") != "CHEMAWARE_FORMULA_ROLE_CHECKPOINT_EMBEDDINGS_COMPLETE":
        raise RuntimeError("Phase-A cache has the wrong provenance status")
    if set(map(int, cache_report.get("formula_roles", []))) != {0, 1}:
        raise RuntimeError("Phase-A cache does not contain exactly training roles 0 and 1")

    phasea = load_npz(args.phasea_bank / "train_pool.npz")
    action_bank = load_npz(args.base_action_bank / "train_pool.npz")
    evidence = load_npz(args.evidence)
    manifest = load_npz(args.manifest)
    cache_rows = np.load(args.phasea_cache / "rows.npy", allow_pickle=False)
    cache_embeddings = np.load(
        args.phasea_cache / "embeddings_f32.npy", mmap_mode="r", allow_pickle=False,
    )
    if array_sha256(cache_rows) != cache_report["rows_array_sha256"]:
        raise RuntimeError("Phase-A cache row hash mismatch")
    if array_sha256(np.asarray(cache_embeddings)) != cache_report["embeddings_array_sha256"]:
        raise RuntimeError("Phase-A cache embedding hash mismatch")
    cache = FrozenEmbeddings(
        args.phasea_cache / "rows.npy", args.phasea_cache / "embeddings_f32.npy",
    )
    folds = stable_formula_folds(manifest["query_formula"], args.folds, args.fold_seed)
    allowed = np.isin(folds, np.asarray(args.training_roles, dtype=folds.dtype))
    training_queries = np.flatnonzero(allowed)
    groups: dict[str, list[int]] = defaultdict(list)
    for query in training_queries:
        groups[str(manifest["query_ik14"][query])].append(int(query))

    base_queries = np.asarray(evidence["query"], dtype=np.int64)
    base_identities = np.asarray(evidence["identity"]).astype(str)
    if len(np.unique(base_identities)) != len(base_identities):
        raise RuntimeError("Phase-A evidence is not one-query-per-identity")
    base_by_identity = dict(zip(base_identities, map(int, base_queries), strict=True))
    if set(base_by_identity) != set(groups):
        raise RuntimeError("Phase-A evidence does not cover every training identity")
    action_by_query = action_identities_by_base_query(action_bank, manifest)

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
    copy_pool(writer, phasea)
    base_events = len(writer.anchor)
    base_snapshot = writer.arrays()
    base_sha256 = pool_prefix_semantic_sha256(base_snapshot, base_events)
    residual_queries: set[int] = set()
    consensus_queries: set[int] = set()
    isolated_queries: set[int] = set()
    sentinel_queries: set[int] = set()
    condition_diverse_queries: set[int] = set()
    current_error_queries = current_error_identities = 0
    chemical_events = 0
    residual_official_events = 0
    identities_with_consensus = identities_with_mixed_outcomes = 0
    residual_boundary_counts: Counter[tuple[str, str]] = Counter()

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
            geometries.append(geometry)
        errors = [row for row in geometries if bool(row["error"])]
        correct = [row for row in geometries if not bool(row["error"])]
        current_error_queries += len(errors)
        current_error_identities += int(bool(errors))
        negative_conditions: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for row in errors:
            negative_conditions[str(row["hardest_identity"])].add(
                signatures[int(row["query"])]
            )
        if any(len(value) >= 2 for value in negative_conditions.values()):
            identities_with_consensus += 1
        actions = action_by_query.get(base_query, {})
        ranked_errors = []
        for row in errors:
            query = int(row["query"])
            negative = str(row["hardest_identity"])
            consensus = len(negative_conditions[negative]) >= 2
            candidates = row["candidates"]
            if not isinstance(candidates, dict):
                raise TypeError("invalid candidate geometry")
            chemical_supported = negative in actions or any(
                str(body["identity"]) in actions and len(body["active_rows"])
                for body in candidates.values()
            )
            ranked_errors.append((
                consensus, chemical_supported,
                signatures[query] != signatures[base_query],
                float(row["margin"]), float(row["embedding_distance"]),
                -query, row,
            ))
        ranked_errors.sort(reverse=True)
        selected = [row[-1] for row in ranked_errors[:args.max_residual_anchors_per_identity]]
        selected_conditions = {signatures[int(row["query"])] for row in selected}
        for row in selected:
            query = int(row["query"])
            negative = str(row["hardest_identity"])
            consensus = len(negative_conditions[negative]) >= 2
            counts = append_residual_error(
                writer, row, actions, consensus, args.chemical_candidates_per_error,
            )
            if not counts["official"]:
                continue
            residual_queries.add(query)
            residual_official_events += counts["official"]
            chemical_events += counts["chemical"]
            residual_boundary_counts[(identity, negative)] += counts["official"]
            (consensus_queries if consensus else isolated_queries).add(query)
            if signatures[query] != signatures[base_query]:
                condition_diverse_queries.add(query)

        eligible_sentinels = [
            row for row in correct
            if 0.0 < float(row["margin"]) <= args.protection_margin
            and int(row["query"]) not in residual_queries
        ]
        eligible_sentinels.sort(key=lambda row: (
            signatures[int(row["query"])] not in selected_conditions,
            -float(row["margin"]), float(row["embedding_distance"]),
            -int(row["query"]),
        ), reverse=True)
        if selected and eligible_sentinels:
            if append_protection_sentinel(writer, eligible_sentinels[0]):
                sentinel_queries.add(int(eligible_sentinels[0]["query"]))
                identities_with_mixed_outcomes += 1

    output = writer.arrays()
    base_immutable = pool_prefix_equal(base_snapshot, output, base_events)
    observed_base_sha256 = pool_prefix_semantic_sha256(output, base_events)
    masses = {
        "phasea_base": float(args.base_sampling_mass),
        "residual_consensus": float(args.consensus_sampling_mass),
        "residual_isolated": float(args.isolated_sampling_mass),
        "protection_sentinel": float(args.sentinel_sampling_mass),
    }
    weights, sampling_audit = residual_sampling_weights(
        output, base_events, args.data, masses,
    )
    output["sampling_weight"] = weights
    roles = np.asarray(output["curriculum_role"], dtype=np.int64)
    residual_mask = np.arange(len(roles)) >= base_events
    residual_singleton = bool(
        np.all(np.diff(output["positive_ptr"])[residual_mask] == 1)
        and np.all(np.diff(output["negative_ptr"])[residual_mask] == 1)
    )
    residual_query_array = np.asarray(sorted(residual_queries), dtype=np.int64)
    gates = {
        "phase_a_base_prefix_immutable": (
            base_immutable and base_sha256 == observed_base_sha256
        ),
        "only_training_roles_used_for_residuals": bool(
            len(residual_query_array)
            and set(map(int, np.unique(folds[residual_query_array]))) <= {0, 1}
        ),
        "outer_roles_2_3_4_untouched": True,
        "phase_a_current_errors_exist": current_error_queries > 0,
        "residual_error_events_exist": residual_official_events > 0,
        "cross_condition_consensus_exists": len(consensus_queries) > 0,
        "isolated_residual_control_exists": len(isolated_queries) > 0,
        "chemical_transfer_events_exist": chemical_events > 0,
        "protection_sentinels_exist": len(sentinel_queries) > 0,
        "residual_events_are_singleton_boundaries": residual_singleton,
        "sampling_weights_positive_and_normalized": bool(
            np.all(weights > 0) and abs(float(weights.sum()) - 1.0) <= 1e-12
        ),
        "unique_triplet_signatures": len(writer.signatures) == len(output["anchor_idx"]),
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"Phase-A residual-consensus triplet gates failed: {gates}; "
            f"current_errors={current_error_queries} residual_queries={len(residual_queries)}"
        )
    identity_audit = audit_identity_edges(output, args.data)
    report = {
        "status": "CHEMAWARE_PHASEA_RESIDUAL_CONSENSUS_TRIPLETS_COMPLETE",
        "method": (
            "immutable Phase-A protection prefix plus Phase-A-current cross-condition "
            "residual errors and active protection sentinels"
        ),
        "training_runtime": "native DreaMS ContrastiveSpectraDataset plus ContrastiveHead",
        "formula_roles": {"training": [0, 1], "selection": 2, "confirmation": 3, "outer": 4},
        "settings": {
            "margin": float(args.margin),
            "protection_margin": float(args.protection_margin),
            "preselect_anchors_per_identity": int(args.preselect_anchors_per_identity),
            "max_residual_anchors_per_identity": int(args.max_residual_anchors_per_identity),
            "chemical_candidates_per_error": int(args.chemical_candidates_per_error),
            "sampling_masses": masses,
        },
        "provenance": {
            "phasea_train_pool_sha256": sha256_file(args.phasea_bank / "train_pool.npz"),
            "base_action_train_pool_sha256": sha256_file(
                args.base_action_bank / "train_pool.npz"
            ),
            "evidence_sha256": sha256_file(args.evidence),
            "manifest_sha256": sha256_file(args.manifest),
            "phasea_checkpoint_sha256": cache_report["checkpoint_sha256"],
            "cache_rows_array_sha256": cache_report["rows_array_sha256"],
            "cache_embeddings_array_sha256": cache_report["embeddings_array_sha256"],
        },
        "phase_a_base_preservation": {
            "events": int(base_events),
            "semantic_sha256_before_residuals": base_sha256,
            "semantic_sha256_after_residuals": observed_base_sha256,
            "exact_prefix_equal": bool(base_immutable),
        },
        "coverage": {
            "training_queries_available": int(len(training_queries)),
            "training_identities": int(len(groups)),
            "phase_a_current_error_queries_preselected": int(current_error_queries),
            "phase_a_current_error_identities_preselected": int(current_error_identities),
            "selected_residual_queries": int(len(residual_queries)),
            "consensus_residual_queries": int(len(consensus_queries)),
            "isolated_residual_queries": int(len(isolated_queries)),
            "condition_diverse_residual_queries": int(len(condition_diverse_queries)),
            "protection_sentinel_queries": int(len(sentinel_queries)),
            "identities_with_repeated_false_identity": int(identities_with_consensus),
            "identities_with_error_and_safe_sentinel": int(identities_with_mixed_outcomes),
            "distinct_identity_to_false_identity_boundaries": int(
                len(residual_boundary_counts)
            ),
        },
        "events": {
            "phase_a_base": int(base_events),
            "residual_official": int(residual_official_events),
            "residual_chemical": int(chemical_events),
            "protection_sentinel": int(len(sentinel_queries)),
            "total": int(len(output["anchor_idx"])),
        },
        "sampling": sampling_audit,
        "identity_audit": identity_audit,
        "gates": gates,
        "claim_boundary": (
            "triplet construction only; performance requires role-2 selection "
            "against protected Phase A and independent role-3 confirmation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_phasea_residual_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **output)
        shutil.copy2(args.phasea_bank / "val_pool.npz", temporary / "val_pool.npz")
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
