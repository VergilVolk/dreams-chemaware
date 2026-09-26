"""Build Phase-A-current native triplets from direct true-candidate support.

This curriculum is deliberately narrower than the earlier action-hard and
multi-condition expansions.  A correction is admitted only when the known
training truth candidate receives rule evidence that is strictly stronger
than every one of three matched rule-content nulls.  The chemical evidence is
used only to select a native DreaMS triplet; it is never used by the model at
inference time and no teacher score is distilled.

For every admitted Phase-A-current error, the triplet uses the exact current
max-positive reference and current hardest false reference.  A correction is
retained only when at least one current-correct sentinel can be constructed
for either the true identity or the false counterparty identity.  This avoids
the unpaired, generic error pressure that caused the V6 continuation to churn
correct boundaries.  Model, loss and optimizer remain native DreaMS.
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
    FrozenEmbeddings,
    PoolWriter,
    load_npz,
)
from build_chemaware_multicondition_max_boundary_triplets import query_geometry
from chemaware_numpy_sampling import stable_formula_folds
from encode_chemaware_checkpoint_manifest_rows import array_sha256


ROOT = Path(__file__).resolve().parents[1]

STRICT_TRUE_SUPPORT_CORRECTION = 30
BROAD_TRUE_SUPPORT_CORRECTION = 31
CHEMISTRY_SUPPORTED_PRESERVATION = 32
PAIRED_ENDPOINT_SENTINEL = 33

STRICT_TRUE_SUPPORT_TAG = 8
BROAD_TRUE_SUPPORT_TAG = 16
ENDPOINT_SENTINEL_TAG = 32

SUPPORT_METRICS = (
    "action_top_fraction",
    "action_largest_region_fraction",
    "action_same_neighbor_fraction",
)
ADVANTAGE_METRICS = (
    "action_best_advantage_over_baseline",
    "global_action_advantage_over_baseline",
)
RULE_METRICS = (
    "candidate_rule_max",
    "candidate_rule_top2_mean",
    "delta_rule_max",
    "delta_rule_top2_mean",
)


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
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--training-roles", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--negative-references-per-error", type=int, default=2)
    parser.add_argument("--minimum-rule-wins", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def decoded(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(values).tolist()
    )


def metric_index(evidence: Mapping[str, np.ndarray]) -> dict[str, int]:
    names = decoded(evidence["metric_names"])
    required = set(SUPPORT_METRICS + ADVANTAGE_METRICS + RULE_METRICS)
    missing = sorted(required - set(names))
    if missing:
        raise RuntimeError(f"triplet evidence lacks required metrics: {missing}")
    return {name: names.index(name) for name in required}


def true_candidate_support(
    evidence: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    minimum_rule_wins: int,
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    """Return role-0/1 queries whose true candidate beats every matched null."""
    if not 1 <= minimum_rule_wins <= len(RULE_METRICS):
        raise ValueError("minimum rule wins is outside the available rule metrics")
    arms = decoded(evidence["arm_names"])
    if len(arms) != 4 or arms[0] != "correct":
        raise RuntimeError("true-support mining requires correct plus exactly three null arms")
    if len(set(arms)) != len(arms):
        raise RuntimeError("triplet evidence contains duplicate arm names")
    index = metric_index(evidence)
    arm_metric = np.asarray(evidence["arm_metric"], dtype=np.float64)
    if arm_metric.shape[0] != len(arms):
        raise RuntimeError("arm metric tensor does not align with arm names")
    center = arm_metric[0]
    null_maximum = np.max(arm_metric[1:], axis=0)
    valid = np.asarray(evidence["valid"], dtype=bool)
    benefit = np.asarray(evidence["benefit"], dtype=bool)
    proposed = np.asarray(evidence["proposed_candidate"], dtype=np.int64)
    queries = np.asarray(evidence["query"], dtype=np.int64)
    identities = np.asarray(evidence["identity"]).astype(str)
    if len(np.unique(queries)) != len(queries) or len(np.unique(identities)) != len(identities):
        raise RuntimeError("training evidence must contain one query per identity")

    high = {
        name: center[..., column] > null_maximum[..., column]
        for name, column in index.items()
    }
    support_all = np.logical_and.reduce([high[name] for name in SUPPORT_METRICS])
    advantage_all = np.logical_and.reduce([high[name] for name in ADVANTAGE_METRICS])
    rule_wins = np.sum(np.stack([high[name] for name in RULE_METRICS]), axis=0)

    selected: dict[int, dict[str, object]] = {}
    missing_benefit_slot = multiple_benefit_slots = 0
    for row, query_value in enumerate(queries):
        slots = np.flatnonzero(valid[row] & benefit[row])
        if not len(slots):
            missing_benefit_slot += int(np.asarray(evidence["baseline_rank"])[row] > 1)
            continue
        if len(slots) != 1:
            multiple_benefit_slots += 1
            continue
        slot = int(slots[0])
        if int(rule_wins[row, slot]) < minimum_rule_wins:
            continue
        query = int(query_value)
        candidate = int(proposed[row, slot])
        left, right = map(int, manifest["query_ptr"][query:query + 2])
        if candidate < 0 or left + candidate >= right:
            raise RuntimeError("direct true-support candidate is outside its query graph")
        if not bool(manifest["molecule_label"][left + candidate]):
            raise RuntimeError("benefit-labelled candidate is not the manifest truth")
        strict = bool(support_all[row, slot] and advantage_all[row, slot])
        selected[query] = {
            "identity": str(identities[row]),
            "candidate": candidate,
            "strict": strict,
            "rule_wins": int(rule_wins[row, slot]),
            "minimum_support_null_advantage": float(min(
                center[row, slot, index[name]] - null_maximum[row, slot, index[name]]
                for name in SUPPORT_METRICS
            )),
            "minimum_action_null_advantage": float(min(
                center[row, slot, index[name]] - null_maximum[row, slot, index[name]]
                for name in ADVANTAGE_METRICS
            )),
        }
    strict_queries = [query for query, body in selected.items() if bool(body["strict"])]
    formulas = np.asarray(evidence["formula"]).astype(str)
    row_by_query = {int(query): row for row, query in enumerate(queries)}
    audit = {
        "evidence_queries": int(len(queries)),
        "official_error_queries": int(np.sum(np.asarray(evidence["baseline_rank"]) > 1)),
        "broad_true_support_queries": int(len(selected)),
        "strict_true_support_queries": int(len(strict_queries)),
        "broad_true_support_identities": int(len({str(body["identity"]) for body in selected.values()})),
        "broad_true_support_formulas": int(len({formulas[row_by_query[q]] for q in selected})),
        "strict_true_support_formulas": int(len({formulas[row_by_query[q]] for q in strict_queries})),
        "official_errors_without_unique_benefit_slot": int(missing_benefit_slot),
        "queries_with_multiple_benefit_slots_rejected": int(multiple_benefit_slots),
        "definition": {
            "broad": f"truth candidate beats all three nulls on >= {minimum_rule_wins} of 4 rule metrics",
            "strict": "broad plus all 3 support and both advantage metrics beat every null",
        },
    }
    return selected, audit


def append_boundary(
    writer: PoolWriter, geometry: Mapping[str, object], role: int, tag: int,
    negative_limit: int,
) -> int:
    candidates = geometry["candidates"]
    if not isinstance(candidates, dict):
        raise TypeError("invalid current geometry")
    candidate = int(geometry["hardest_candidate"])
    body = candidates[candidate]
    if role in (STRICT_TRUE_SUPPORT_CORRECTION, BROAD_TRUE_SUPPORT_CORRECTION):
        rows = np.asarray(body["active_rows"], dtype=np.int64)
        if not len(rows):
            raise RuntimeError("current error lacks an active hardest reference")
        rows = rows[:negative_limit]
    else:
        rows = np.asarray(body["rows"], dtype=np.int64)[:1]
    added = 0
    for negative in rows:
        before = len(writer.anchor)
        writer.append(
            int(geometry["anchor"]), [int(geometry["positive_row"])], [int(negative)],
            int(geometry["query"]), candidate, int(tag), int(role),
        )
        added += int(len(writer.anchor) > before)
    return added


def select_endpoint_sentinel(
    identity: str,
    groups: Mapping[str, list[int]],
    geometry_for_query,
    margin: float,
    excluded_queries: set[int],
) -> dict[str, object] | None:
    """Choose the closest current-correct guard, preferring zero-loss guards."""
    eligible: list[dict[str, object]] = []
    for query in groups.get(str(identity), []):
        if int(query) in excluded_queries:
            continue
        geometry = geometry_for_query(int(query))
        if bool(geometry["error"]) or float(geometry["margin"]) <= 0.0:
            continue
        eligible.append(geometry)
    if not eligible:
        return None
    eligible.sort(key=lambda row: (
        float(row["margin"]) >= margin,
        -abs(float(row["margin"]) - margin),
        -int(row["query"]),
    ), reverse=True)
    return eligible[0]


def identity_equal_weights(
    pool: Mapping[str, np.ndarray], data: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    raw_mass = {
        STRICT_TRUE_SUPPORT_CORRECTION: 0.45,
        BROAD_TRUE_SUPPORT_CORRECTION: 0.20,
        CHEMISTRY_SUPPORTED_PRESERVATION: 0.20,
        PAIRED_ENDPOINT_SENTINEL: 0.15,
    }
    role_name = {
        STRICT_TRUE_SUPPORT_CORRECTION: "strict_true_support_correction",
        BROAD_TRUE_SUPPORT_CORRECTION: "broad_true_support_correction",
        CHEMISTRY_SUPPORTED_PRESERVATION: "chemistry_supported_preservation",
        PAIRED_ENDPOINT_SENTINEL: "paired_endpoint_sentinel",
    }
    roles = np.asarray(pool["curriculum_role"], dtype=np.int64)
    anchors = np.asarray(pool["anchor_idx"], dtype=np.int64)
    with h5py.File(data, "r") as handle:
        identities = np.asarray(handle["INCHIKEY"])[anchors].astype(str)
    present = {role: np.flatnonzero(roles == role) for role in raw_mass}
    available_total = sum(raw_mass[role] for role, pos in present.items() if len(pos))
    if available_total <= 0:
        raise RuntimeError("true-support curriculum has no sampling stratum")
    weights = np.zeros(len(roles), dtype=np.float64)
    audit: dict[str, object] = {}
    for role, positions in present.items():
        name = role_name[role]
        if not len(positions):
            audit[name] = {"events": 0, "sampling_mass": 0.0, "available": False}
            continue
        mass = raw_mass[role] / available_total
        unique, inverse = np.unique(identities[positions], return_inverse=True)
        counts = np.bincount(inverse)
        local = mass / (len(unique) * counts[inverse])
        weights[positions] = local
        per_identity = np.bincount(inverse, weights=local)
        audit[name] = {
            "events": int(len(positions)),
            "identities": int(len(unique)),
            "sampling_mass": float(np.sum(local)),
            "minimum_identity_mass": float(np.min(per_identity)),
            "maximum_identity_mass": float(np.max(per_identity)),
            "available": True,
        }
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("true-support sampling weights are invalid")
    weights /= weights.sum()
    audit["normalized_total_sampling_mass"] = float(weights.sum())
    audit["absent_strata_are_zero_then_fixed_masses_renormalized"] = True
    return weights, audit


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if set(args.training_roles) != {0, 1}:
        raise ValueError("formal true-support training is restricted to roles 0 and 1")
    if args.margin != 0.1:
        raise ValueError("true-support triplets must retain the native DreaMS margin 0.1")
    if args.negative_references_per_error not in (1, 2, 3):
        raise ValueError("negative reference cap must be in {1,2,3}")

    evidence = load_npz(args.evidence)
    manifest = load_npz(args.manifest)
    cache_report = json.loads((args.phasea_cache / "report.json").read_text(encoding="utf-8"))
    if cache_report.get("status") != "CHEMAWARE_FORMULA_ROLE_CHECKPOINT_EMBEDDINGS_COMPLETE":
        raise RuntimeError("Phase-A cache has the wrong provenance status")
    if set(map(int, cache_report.get("formula_roles", []))) != {0, 1}:
        raise RuntimeError("Phase-A cache must contain exactly training roles 0 and 1")
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
    training_queries = np.flatnonzero(
        np.isin(folds, np.asarray(args.training_roles, dtype=folds.dtype))
    )
    groups: dict[str, list[int]] = defaultdict(list)
    for query in training_queries:
        groups[str(manifest["query_ik14"][query])].append(int(query))
    support, support_audit = true_candidate_support(
        evidence, manifest, args.minimum_rule_wins,
    )
    if set(map(int, support)) - set(map(int, training_queries)):
        raise RuntimeError("true-support evidence escaped the training formula roles")

    geometry_cache: dict[int, dict[str, object]] = {}

    def geometry_for_query(query: int) -> dict[str, object]:
        if query not in geometry_cache:
            geometry_cache[query] = query_geometry(manifest, cache, query, args.margin)
        return geometry_cache[query]

    writer = PoolWriter()
    correction_queries: set[int] = set()
    paired_correction_queries: set[int] = set()
    strict_correction_queries: set[int] = set()
    broad_correction_queries: set[int] = set()
    preserve_queries: set[int] = set()
    endpoint_queries: set[int] = set()
    skipped_unpaired: set[int] = set()
    strict_events = broad_events = preserve_events = endpoint_events = 0
    inactive_endpoint_events = active_endpoint_events = 0
    pair_sources = {"self": 0, "counterparty": 0, "both": 0}

    # Correct true-support identities are protected before error corrections
    # add endpoint sentinels.  PoolWriter then deterministically deduplicates a
    # guard that serves both roles without multiplying its reported scale.
    for query in sorted(support):
        body = support[query]
        geometry = geometry_for_query(query)
        if bool(geometry["error"]):
            continue
        role_tag = STRICT_TRUE_SUPPORT_TAG if bool(body["strict"]) else BROAD_TRUE_SUPPORT_TAG
        added = append_boundary(
            writer, geometry, CHEMISTRY_SUPPORTED_PRESERVATION,
            role_tag, 1,
        )
        if added:
            preserve_queries.add(query)
            preserve_events += added

    for query in sorted(support):
        body = support[query]
        geometry = geometry_for_query(query)
        if not bool(geometry["error"]):
            continue
        true_identity = str(body["identity"])
        false_identity = str(geometry["hardest_identity"])
        self_guard = select_endpoint_sentinel(
            true_identity, groups, geometry_for_query, args.margin, {query},
        )
        counterparty_guard = select_endpoint_sentinel(
            false_identity, groups, geometry_for_query, args.margin, {query},
        )
        guards: list[tuple[str, dict[str, object]]] = []
        seen_guard_queries: set[int] = set()
        for source, guard in (("self", self_guard), ("counterparty", counterparty_guard)):
            if guard is None or int(guard["query"]) in seen_guard_queries:
                continue
            guards.append((source, guard))
            seen_guard_queries.add(int(guard["query"]))
        if not guards:
            skipped_unpaired.add(query)
            continue

        strict = bool(body["strict"])
        correction_role = (
            STRICT_TRUE_SUPPORT_CORRECTION if strict else BROAD_TRUE_SUPPORT_CORRECTION
        )
        correction_tag = STRICT_TRUE_SUPPORT_TAG if strict else BROAD_TRUE_SUPPORT_TAG
        added = append_boundary(
            writer, geometry, correction_role, correction_tag,
            args.negative_references_per_error,
        )
        if not added:
            continue
        correction_queries.add(query)
        (strict_correction_queries if strict else broad_correction_queries).add(query)
        if strict:
            strict_events += added
        else:
            broad_events += added
        logical_sources = {source for source, _guard in guards}
        paired_correction_queries.add(query)
        if logical_sources == {"self", "counterparty"}:
            pair_sources["both"] += 1
        else:
            pair_sources[next(iter(logical_sources))] += 1
        for source, guard in guards:
            guard_added = append_boundary(
                writer, guard, PAIRED_ENDPOINT_SENTINEL, ENDPOINT_SENTINEL_TAG, 1,
            )
            if not guard_added:
                continue
            endpoint_queries.add(int(guard["query"]))
            endpoint_events += guard_added
            if float(guard["margin"]) >= args.margin:
                inactive_endpoint_events += guard_added
            else:
                active_endpoint_events += guard_added
    output = writer.arrays()
    weights, sampling_audit = identity_equal_weights(output, args.data)
    output["sampling_weight"] = weights
    identity_audit = audit_identity_edges(output, args.data)
    correction_array = np.asarray(sorted(correction_queries), dtype=np.int64)
    roles = np.asarray(output["curriculum_role"], dtype=np.int64)
    singleton = bool(
        np.all(np.diff(output["positive_ptr"]) == 1)
        and np.all(np.diff(output["negative_ptr"]) == 1)
    )
    gates = {
        "four_matched_arms_present": len(decoded(evidence["arm_names"])) == 4,
        "broad_true_support_not_sparse": support_audit["broad_true_support_queries"] >= 300,
        "strict_true_support_not_sparse": support_audit["strict_true_support_queries"] >= 100,
        "only_training_roles_used": bool(
            len(correction_array)
            and set(map(int, np.unique(folds[correction_array]))) <= {0, 1}
        ),
        "outer_roles_2_3_4_untouched": True,
        "phasea_current_corrections_exist": len(correction_queries) > 0,
        "strict_and_broad_corrections_exist": bool(
            strict_correction_queries and broad_correction_queries
        ),
        "every_retained_correction_has_endpoint_guard": (
            paired_correction_queries == correction_queries
        ),
        "endpoint_sentinels_exist": endpoint_events > 0,
        "all_events_singleton_exact_boundaries": singleton,
        "sampling_weights_positive_and_normalized": bool(
            np.all(weights > 0) and abs(float(weights.sum()) - 1.0) <= 1e-12
        ),
        "only_declared_curriculum_roles": set(map(int, np.unique(roles))) <= {
            STRICT_TRUE_SUPPORT_CORRECTION,
            BROAD_TRUE_SUPPORT_CORRECTION,
            CHEMISTRY_SUPPORTED_PRESERVATION,
            PAIRED_ENDPOINT_SENTINEL,
        },
        "identity_edges_valid": bool(identity_audit["identity_contract_passed"]),
        "unique_triplet_signatures": len(writer.signatures) == len(output["anchor_idx"]),
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"true-support native-triplet gates failed: {gates}; "
            f"corrections={len(correction_queries)} skipped_unpaired={len(skipped_unpaired)}"
        )

    report = {
        "status": "CHEMAWARE_TRUE_SUPPORT_NATIVE_TRIPLETS_COMPLETE",
        "method": (
            "three-null-exclusive true-candidate support selects Phase-A-current "
            "native max-boundary corrections paired with current-correct endpoint sentinels"
        ),
        "training_runtime": "native DreaMS ContrastiveSpectraDataset plus ContrastiveHead",
        "formula_roles": {"training": [0, 1], "selection": 2, "confirmation": 3, "outer": 4},
        "settings": {
            "margin": float(args.margin),
            "minimum_rule_wins": int(args.minimum_rule_wins),
            "negative_references_per_error": int(args.negative_references_per_error),
            "support_metrics": list(SUPPORT_METRICS),
            "advantage_metrics": list(ADVANTAGE_METRICS),
            "rule_metrics": list(RULE_METRICS),
        },
        "provenance": {
            "evidence_sha256": sha256_file(args.evidence),
            "manifest_sha256": sha256_file(args.manifest),
            "validation_pool_sha256": sha256_file(args.validation_pool),
            "phasea_checkpoint_sha256": cache_report["checkpoint_sha256"],
            "cache_rows_array_sha256": cache_report["rows_array_sha256"],
            "cache_embeddings_array_sha256": cache_report["embeddings_array_sha256"],
        },
        "direct_support": support_audit,
        "phasea_current": {
            "correction_queries": int(len(correction_queries)),
            "strict_correction_queries": int(len(strict_correction_queries)),
            "broad_correction_queries": int(len(broad_correction_queries)),
            "chemistry_supported_preservation_queries": int(len(preserve_queries)),
            "endpoint_sentinel_queries": int(len(endpoint_queries)),
            "eligible_queries_skipped_without_guard": int(len(skipped_unpaired)),
            "pair_source_counts": pair_sources,
            "geometry_queries_computed": int(len(geometry_cache)),
        },
        "events": {
            "strict_true_support_correction": int(strict_events),
            "broad_true_support_correction": int(broad_events),
            "chemistry_supported_preservation": int(preserve_events),
            "paired_endpoint_sentinel": int(endpoint_events),
            "inactive_zero_loss_endpoint_at_construction": int(inactive_endpoint_events),
            "active_near_boundary_endpoint_at_construction": int(active_endpoint_events),
            "total": int(len(output["anchor_idx"])),
        },
        "sampling": sampling_audit,
        "identity_audit": identity_audit,
        "gates": gates,
        "claim_boundary": (
            "triplet construction only; performance requires role-2 selection against "
            "the protected Phase-A embedding and independent role-3 confirmation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_true_support_", dir=args.output.parent))
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
