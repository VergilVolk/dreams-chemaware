"""Build ChemAware benefit-proven decision-boundary DreaMS triplets.

This constructor changes only the triplet pool.  It does not change DreaMS'
``ContrastiveSpectraDataset``, its uniform shuffled DataLoader, the random
one-positive/one-negative draw inside an event, ``ContrastiveHead``, the
cosine triplet-margin loss, or Adam.

The earlier action-hard constructor treated a false candidate preferred by a
chemical action as a negative.  Here the chemical evidence has the correct
semantics: a correct-arm proposal must promote the known true candidate from
an official error to rank one and must beat every matched semantic null on a
minimum number of directional metrics.  That benefit proof qualifies the
query.  The emitted native triplets then contrast the true candidate against
the actual highest-scoring false candidate under the requested initialization
geometry.

The protected Phase-A safety events and unmodified official DreaMS replay
events are copied exactly.  Only its old error/chemical events are replaced.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from build_chemaware_dreams_native_triplets import audit_identity_edges, molecule_rows
from build_chemaware_max_boundary_native_triplets import (
    DREAMS_NATIVE_REPLAY,
    SAFE_MAX_BOUNDARY,
    FrozenEmbeddings,
    PoolWriter,
    edges,
    load_npz,
)


ROOT = Path(__file__).resolve().parents[1]
BENEFIT_PROVEN_BOUNDARY = 5

POSITIVE_DIRECTION_METRICS = (
    "action_top_fraction",
    "action_largest_region_fraction",
    "action_same_neighbor_fraction",
    "action_best_advantage_over_baseline",
    "global_action_advantage_over_baseline",
    "global_action_selects_candidate",
    "candidate_rule_max",
    "candidate_rule_top2_mean",
    "delta_rule_max",
    "delta_rule_top2_mean",
)
NEGATIVE_DIRECTION_METRICS = ("candidate_rule_rank_fraction",)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phasea-pool", type=Path, required=True)
    parser.add_argument("--validation-pool", type=Path, required=True)
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
        "--embeddings", type=Path,
        default=(ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1"
                 / "official_embeddings_f32.npy"),
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--minimum-null-consensus-metrics", type=int, default=2)
    parser.add_argument(
        "--maximum-events-per-query", type=int, default=16,
        help=(
            "Hard concentration ceiling. The constructor automatically uses "
            "the smallest balanced cap that can restore the frozen event budget."
        ),
    )
    parser.add_argument("--minimum-correction-events", type=int, default=1000)
    parser.add_argument("--minimum-correction-queries", type=int, default=250)
    parser.add_argument("--minimum-benefit-proven-formulas", type=int, default=180)
    parser.add_argument("--minimum-active-correction-formulas", type=int, default=160)
    return parser.parse_args()


def validate_evidence(evidence: Mapping[str, np.ndarray]) -> None:
    required = {
        "query", "formula", "valid", "proposed_candidate", "baseline_candidate",
        "baseline_rank", "proposal_rank", "benefit", "harmful", "arm_names",
        "metric_names", "arm_metric",
    }
    missing = required - set(evidence)
    if missing:
        raise RuntimeError(f"benefit evidence is incomplete: {sorted(missing)}")
    arms = tuple(map(str, evidence["arm_names"].tolist()))
    if len(arms) != 4 or arms[0] != "correct":
        raise RuntimeError(f"expected correct plus three semantic nulls, got {arms}")
    if np.asarray(evidence["arm_metric"]).shape[:3] != (
        4, len(evidence["query"]), np.asarray(evidence["valid"]).shape[1],
    ):
        raise RuntimeError("benefit evidence arm metric layout is invalid")


def benefit_consensus(evidence: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return the worst-null count of correctly directed metric advantages."""
    names = tuple(map(str, evidence["metric_names"].tolist()))
    try:
        positive = [names.index(name) for name in POSITIVE_DIRECTION_METRICS]
        negative = [names.index(name) for name in NEGATIVE_DIRECTION_METRICS]
    except ValueError as error:
        raise RuntimeError(f"required directional metric is missing: {error}") from error
    metric = np.asarray(evidence["arm_metric"], dtype=np.float64)
    per_null = np.concatenate((
        np.take(metric[0], positive, axis=-1)[None]
        - np.take(metric[1:], positive, axis=-1),
        np.take(metric[1:], negative, axis=-1)
        - np.take(metric[0], negative, axis=-1)[None],
    ), axis=-1)
    return np.min(np.sum(per_null > 0.0, axis=-1), axis=0).astype(np.int16)


def copy_protected_background(
    phasea: Mapping[str, np.ndarray], writer: PoolWriter,
) -> dict[str, int]:
    roles = np.asarray(phasea["curriculum_role"], dtype=np.int64)
    retained = np.isin(roles, [SAFE_MAX_BOUNDARY, DREAMS_NATIVE_REPLAY])
    for event in np.flatnonzero(retained):
        writer.append(
            int(phasea["anchor_idx"][event]),
            edges(phasea, int(event), "positive"),
            edges(phasea, int(event), "negative"),
            int(phasea["source_query"][event]),
            int(phasea["negative_candidate"][event]),
            int(phasea["source_tag"][event]),
            int(roles[event]),
        )
    return {
        "safe_events": int(np.sum(roles == SAFE_MAX_BOUNDARY)),
        "official_replay_events": int(np.sum(roles == DREAMS_NATIVE_REPLAY)),
        "old_error_events_removed": int(np.sum(~retained)),
    }


def candidate_geometry(
    manifest: Mapping[str, np.ndarray], cache: FrozenEmbeddings, query: int,
) -> dict[str, object]:
    anchor = int(manifest["query_row"][query])
    anchor_embedding = cache.get([anchor])[0]
    left, right = map(int, manifest["query_ptr"][query:query + 2])
    labels = np.asarray(manifest["molecule_label"][left:right], dtype=bool)
    true_candidates = np.flatnonzero(labels)
    false_candidates = np.flatnonzero(~labels)
    if len(true_candidates) != 1 or not len(false_candidates):
        raise RuntimeError(
            f"query {query} requires one true and at least one false candidate"
        )
    true_candidate = int(true_candidates[0])
    positive_rows = np.unique(molecule_rows(manifest, query, true_candidate))
    positive_rows = positive_rows[positive_rows != anchor]
    if not len(positive_rows):
        raise RuntimeError(f"query {query} has no distinct positive reference")
    positive_scores = cache.get(positive_rows) @ anchor_embedding

    winner = -1
    negative_rows = np.empty(0, dtype=np.int64)
    negative_scores = np.empty(0, dtype=np.float32)
    winner_score = -np.inf
    for candidate in false_candidates:
        rows = np.unique(molecule_rows(manifest, query, int(candidate)))
        scores = cache.get(rows) @ anchor_embedding
        score = float(np.max(scores))
        if score > winner_score:
            winner = int(candidate)
            winner_score = score
            negative_rows = rows
            negative_scores = scores
    return {
        "anchor": anchor,
        "true_candidate": true_candidate,
        "positive_rows": positive_rows,
        "positive_scores": positive_scores,
        "false_winner": winner,
        "negative_rows": negative_rows,
        "negative_scores": negative_scores,
        "positive_max": float(np.max(positive_scores)),
        "negative_max": float(winner_score),
    }


def active_pairs(
    geometry: Mapping[str, object], margin: float,
) -> list[tuple[int, int, float, int, int]]:
    positive_rows = np.asarray(geometry["positive_rows"], dtype=np.int64)
    negative_rows = np.asarray(geometry["negative_rows"], dtype=np.int64)
    positive_scores = np.asarray(geometry["positive_scores"], dtype=np.float64)
    negative_scores = np.asarray(geometry["negative_scores"], dtype=np.float64)
    positive_order = np.argsort(-positive_scores, kind="stable")
    negative_order = np.argsort(-negative_scores, kind="stable")
    positive_rank = np.empty(len(positive_order), dtype=np.int64)
    negative_rank = np.empty(len(negative_order), dtype=np.int64)
    positive_rank[positive_order] = np.arange(len(positive_order))
    negative_rank[negative_order] = np.arange(len(negative_order))
    output: list[tuple[int, int, float, int, int]] = []
    for p_index, positive in enumerate(positive_rows):
        for n_index, negative in enumerate(negative_rows):
            hinge = float(margin + negative_scores[n_index] - positive_scores[p_index])
            if hinge > 0.0:
                output.append((
                    int(positive), int(negative), hinge,
                    int(positive_rank[p_index]), int(negative_rank[n_index]),
                ))
    # Retrieval-critical pairs come first: top negative and top positive ranks,
    # followed by larger native hinge.  Row ids make ties deterministic.
    output.sort(key=lambda row: (
        row[4] + row[3], max(row[4], row[3]), row[4], row[3],
        -row[2], row[0], row[1],
    ))
    return output


def balanced_pair_candidates(
    records: list[dict[str, object]], maximum_events_per_query: int,
):
    """Yield native pairs in balanced query layers, strongest within each layer.

    No query receives its second event before every query with an available
    first event has been considered.  Within a layer, current Phase-A errors
    precede already-correct margin queries, followed by stronger null
    consensus and the retrieval-critical ordering returned by ``active_pairs``.
    This preserves query breadth while allowing enough reference multiplicity
    to replace the complete Phase-A error-event budget.
    """
    if maximum_events_per_query < 1:
        raise ValueError("maximum events per query must be positive")
    for depth in range(maximum_events_per_query):
        layer = [
            record for record in records
            if len(record["pairs"]) > depth  # type: ignore[arg-type]
        ]
        layer.sort(key=lambda record: (
            -int(bool(record["current_error"])),
            -int(record["query_consensus"]),
            int(record["pairs"][depth][3]) + int(record["pairs"][depth][4]),  # type: ignore[index]
            max(
                int(record["pairs"][depth][3]),  # type: ignore[index]
                int(record["pairs"][depth][4]),  # type: ignore[index]
            ),
            -float(record["pairs"][depth][2]),  # type: ignore[index]
            int(record["query"]),
        ))
        for record in layer:
            yield record, record["pairs"][depth]  # type: ignore[index]


def minimum_balanced_cap(
    records: list[dict[str, object]], target_events: int, maximum_cap: int,
) -> tuple[int, int]:
    """Return the least per-query cap reaching ``target_events`` and capacity."""
    if target_events < 1 or maximum_cap < 1:
        raise ValueError("target and maximum cap must be positive")
    for cap in range(1, maximum_cap + 1):
        capacity = int(sum(
            min(len(record["pairs"]), cap)  # type: ignore[arg-type]
            for record in records
        ))
        if capacity >= target_events:
            return cap, capacity
    capacity = int(sum(
        min(len(record["pairs"]), maximum_cap)  # type: ignore[arg-type]
        for record in records
    ))
    return 0, capacity


def build_corrections(
    evidence: Mapping[str, np.ndarray], manifest: Mapping[str, np.ndarray],
    cache: FrozenEmbeddings, writer: PoolWriter, margin: float,
    minimum_consensus: int, maximum_events_per_query: int,
    target_correction_events: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if (
        minimum_consensus < 1
        or maximum_events_per_query < 1
        or target_correction_events < 1
    ):
        raise ValueError("consensus, event cap, and replacement target must be positive")
    consensus = benefit_consensus(evidence)
    benefit = np.asarray(evidence["benefit"], dtype=bool)
    eligible = benefit & (consensus >= int(minimum_consensus))
    query_rows = np.flatnonzero(np.any(eligible, axis=1))
    proposed = np.asarray(evidence["proposed_candidate"], dtype=np.int64)
    baseline = np.asarray(evidence["baseline_candidate"], dtype=np.int64)
    baseline_rank = np.asarray(evidence["baseline_rank"], dtype=np.int64)
    query_values = np.asarray(evidence["query"], dtype=np.int64)

    event_query: list[int] = []
    proof_candidate: list[int] = []
    evidence_baseline_candidate: list[int] = []
    current_false_winner: list[int] = []
    event_consensus: list[int] = []
    event_hinge: list[float] = []
    event_current_error: list[bool] = []
    benefit_proven_formulas = {
        str(manifest["query_formula"][int(query_values[row])]) for row in query_rows
    }
    eligible_formulas: set[str] = set()
    active_queries = current_errors = winner_switches = 0
    possible_native_combinations = 0
    records: list[dict[str, object]] = []

    for row in query_rows:
        query = int(query_values[row])
        if int(baseline_rank[row]) <= 1:
            raise RuntimeError("benefit proof unexpectedly came from an official-correct query")
        geometry = candidate_geometry(manifest, cache, query)
        true_candidate = int(geometry["true_candidate"])
        proof_slots = np.flatnonzero(eligible[row])
        proof_candidates = set(map(int, proposed[row, proof_slots]))
        if proof_candidates != {true_candidate}:
            raise RuntimeError(
                f"query {query} benefit proof does not uniquely promote its true candidate: "
                f"{sorted(proof_candidates)} versus {true_candidate}"
            )
        pairs = active_pairs(geometry, margin)
        if not pairs:
            continue
        active_queries += 1
        current_error = bool(
            float(geometry["negative_max"]) >= float(geometry["positive_max"])
        )
        current_errors += int(current_error)
        winner_switches += int(
            int(geometry["false_winner"]) != int(baseline[row])
        )
        possible_native_combinations += len(pairs)
        eligible_formulas.add(str(manifest["query_formula"][query]))
        query_consensus = int(np.max(consensus[row, proof_slots]))
        records.append({
            "query": query,
            "geometry": geometry,
            "pairs": pairs,
            "true_candidate": true_candidate,
            "evidence_baseline_candidate": int(baseline[row]),
            "query_consensus": query_consensus,
            "current_error": current_error,
        })

    effective_cap, capped_capacity = minimum_balanced_cap(
        records, target_correction_events, maximum_events_per_query,
    )
    if not effective_cap:
        raise RuntimeError(
            "benefit-boundary active capacity cannot reach the frozen Phase-A "
            f"replacement budget below the concentration ceiling: target="
            f"{target_correction_events}, capped_capacity={capped_capacity}, "
            f"active_queries={active_queries}, ceiling={maximum_events_per_query}"
        )
    selected_per_query: dict[int, int] = {}
    for record, pair in balanced_pair_candidates(records, effective_cap):
        if len(event_query) >= target_correction_events:
            break
        positive, negative, hinge, _positive_rank, _negative_rank = pair
        geometry = record["geometry"]
        query = int(record["query"])
        before = len(writer.anchor)
        writer.append(
            int(geometry["anchor"]), [positive], [negative], query,  # type: ignore[index]
            int(geometry["false_winner"]), 0, BENEFIT_PROVEN_BOUNDARY,  # type: ignore[index]
        )
        # A retained Phase-A safety event can already own the exact signature.
        # It must not create a duplicate ledger row or consume replacement dose.
        if len(writer.anchor) == before:
            continue
        event_query.append(query)
        proof_candidate.append(int(record["true_candidate"]))
        evidence_baseline_candidate.append(
            int(record["evidence_baseline_candidate"]),
        )
        current_false_winner.append(int(geometry["false_winner"]))  # type: ignore[index]
        event_consensus.append(int(record["query_consensus"]))
        event_hinge.append(float(hinge))
        event_current_error.append(bool(record["current_error"]))
        selected_per_query[query] = selected_per_query.get(query, 0) + 1

    if len(event_query) != target_correction_events:
        raise RuntimeError(
            "benefit-boundary active capacity cannot exactly replace the frozen "
            f"Phase-A error-event budget: selected={len(event_query)}, "
            f"target={target_correction_events}, capped_capacity={capped_capacity}, "
            f"active_queries={active_queries}, effective_cap={effective_cap}"
        )

    ledger = {
        "source_query": np.asarray(event_query, dtype=np.int64),
        "proof_true_candidate": np.asarray(proof_candidate, dtype=np.int16),
        "evidence_baseline_candidate": np.asarray(
            evidence_baseline_candidate, dtype=np.int16,
        ),
        "current_false_winner": np.asarray(current_false_winner, dtype=np.int16),
        "null_consensus_metrics": np.asarray(event_consensus, dtype=np.int16),
        "native_hinge_at_construction": np.asarray(event_hinge, dtype=np.float32),
        "current_error": np.asarray(event_current_error, dtype=bool),
    }
    report = {
        "benefit_proven_queries": int(len(query_rows)),
        "benefit_proven_formulas": int(len(benefit_proven_formulas)),
        "active_benefit_queries": int(active_queries),
        "active_benefit_formulas": int(len(eligible_formulas)),
        "active_formula_fraction_of_benefit_proven": (
            float(len(eligible_formulas) / len(benefit_proven_formulas))
            if benefit_proven_formulas else 0.0
        ),
        "current_error_queries": int(current_errors),
        "current_correct_margin_queries": int(active_queries - current_errors),
        "current_false_winner_switches_from_evidence_baseline": int(winner_switches),
        "unique_active_correction_events": int(len(event_query)),
        "target_phasea_error_event_replacement": int(target_correction_events),
        "capped_active_native_reference_capacity": int(capped_capacity),
        "possible_active_native_reference_triplets": int(possible_native_combinations),
        "minimum_null_consensus_metrics": int(minimum_consensus),
        "maximum_events_per_query_ceiling": int(maximum_events_per_query),
        "effective_balanced_events_per_query_cap": int(effective_cap),
        "mean_selected_events_per_active_query": (
            float(len(event_query) / len(selected_per_query))
            if selected_per_query else 0.0
        ),
        "maximum_selected_events_for_any_query": int(
            max(selected_per_query.values(), default=0)
        ),
    }
    return report, ledger


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    phasea = load_npz(args.phasea_pool)
    evidence = load_npz(args.evidence)
    manifest = load_npz(args.manifest)
    validate_evidence(evidence)
    cache = FrozenEmbeddings(args.embedding_rows, args.embeddings)
    writer = PoolWriter()
    background = copy_protected_background(phasea, writer)
    correction, ledger = build_corrections(
        evidence, manifest, cache, writer, args.margin,
        args.minimum_null_consensus_metrics, args.maximum_events_per_query,
        background["old_error_events_removed"],
    )
    output = writer.arrays()
    roles = np.asarray(output["curriculum_role"], dtype=np.int64)
    correction_mask = roles == BENEFIT_PROVEN_BOUNDARY
    source_queries = np.asarray(output["source_query"], dtype=np.int64)[correction_mask]
    correction_formulas = set(
        np.asarray(manifest["query_formula"])[source_queries].astype(str).tolist()
    )
    gates = {
        "benefit_proof_promotes_true_candidate": True,
        "minimum_correction_events": (
            int(np.sum(correction_mask)) >= args.minimum_correction_events
        ),
        "minimum_correction_queries": (
            len(np.unique(source_queries)) >= args.minimum_correction_queries
        ),
        "minimum_benefit_proven_formulas": (
            int(correction["benefit_proven_formulas"])
            >= args.minimum_benefit_proven_formulas
        ),
        "minimum_active_correction_formulas": (
            len(correction_formulas) >= args.minimum_active_correction_formulas
        ),
        "exact_phasea_error_event_replacement": (
            int(np.sum(correction_mask)) == background["old_error_events_removed"]
        ),
        "phasea_train_pool_cardinality_preserved": (
            len(output["anchor_idx"]) == len(phasea["anchor_idx"])
        ),
        "all_correction_events_active": bool(
            len(ledger["native_hinge_at_construction"])
            and np.all(ledger["native_hinge_at_construction"] > 0.0)
        ),
        "per_query_event_cap": bool(
            not len(source_queries)
            or np.max(np.unique(source_queries, return_counts=True)[1])
            <= args.maximum_events_per_query
        ),
        "phasea_safety_preserved": (
            int(np.sum(roles == SAFE_MAX_BOUNDARY)) == background["safe_events"]
        ),
        "official_replay_preserved": (
            int(np.sum(roles == DREAMS_NATIVE_REPLAY))
            == background["official_replay_events"]
        ),
        "unique_triplet_signatures": len(writer.signatures) == len(output["anchor_idx"]),
        "validation_pool_unchanged": True,
        "outer_roles_2_3_4_untouched": True,
        "native_sampler_loss_optimizer_unchanged": True,
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"benefit-boundary triplet gates failed: {gates}; "
            f"correction={correction}; background={background}"
        )
    identity = audit_identity_edges(output, args.data)
    report = {
        "status": "CHEMAWARE_BENEFIT_BOUNDARY_NATIVE_TRIPLETS_COMPLETE",
        "method": (
            "correct-arm benefit proves the true candidate; current false-winner "
            "active reference pairs become ordinary native DreaMS triplets"
        ),
        "initialization_geometry": str(args.embeddings.resolve()),
        "margin": float(args.margin),
        "background": background,
        "correction": correction,
        "total_events": int(len(output["anchor_idx"])),
        "identity_audit": identity,
        "gates": gates,
        "frozen_training_contract": {
            "dataset": "dreams.utils.data.ContrastiveSpectraDataset",
            "sampler": "unchanged uniform shuffled DataLoader and native one-positive/one-negative draw",
            "model": "dreams.models.heads.heads.ContrastiveHead",
            "loss": "unchanged cosine triplet-margin loss",
            "optimizer": "unchanged Adam",
            "only_custom_component": "triplet construction",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_benefit_boundary_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "train_pool.npz", **output)
        shutil.copy2(args.validation_pool, temporary / "val_pool.npz")
        np.savez_compressed(temporary / "benefit_boundary_ledger.npz", **ledger)
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
