"""Route all 66 unique mature P actions under a fold-aligned E8 geometry.

This bounded audit reconstructs the original positive-guided intensity matrix
plus E10B/E11/E12B positive-reference actions and their wrong-identity direction
controls.  It never reads held action outcomes and is not itself a training or
performance result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
import time

import h5py
import numpy as np
import pandas as pd
import torch

from audit_noise_final_positive_guided_matrix import reference_profile
from audit_noise_final_positive_peak_transfer import recurrent_missing_peaks
from calibrate_noise_final_e1_empirical import clean_instrument, decode
from noise_corrected_action_routing import RoutingThresholds, route_action
from noise_corrected_action_routing_v3 import score_candidate_boundary
from noise_corrected_action_panel import (
    lossless_selector_frontier,
    select_initial_e8_error_boundary_queries,
)
from noise_corrected_full_action_registry import (
    P_MISSING_PEAK_FAMILIES,
    choose_reference_rows,
    materialize_p_action,
    registered_p_recipes,
)
from noise_final_core import CandidateGraph, sha256_file, stable_fold, strict_rank
from train_e1_identity import load_base_model, torch_load_compat
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, forward_embeddings


REGISTERED_FORMAL_P_ROUTE_CONFIGURATION: dict[str, object] = {
    "formula_fold_seed": 20260825,
    "query_scope": "initial_error_boundary",
    "max_queries": 0,
    "sample_seed": 20260906,
    "fragment_tolerance": 0.02,
    "paired_advantage_threshold": 0.01,
    "harm_margin_threshold": 0.01,
    "robustness_slack": 0.005,
    "boundary_correct_multiplier": 2.0,
    "minimum_boundary_correct": 4096,
    "maximum_boundary_correct": 8192,
    "maximum_corrective_frontier": 16,
    "maximum_harmful_frontier": 8,
    "maximum_robust_frontier": 8,
    "n_highest_peaks": 100,
    "amp": False,
}


def _validate_registered_formal_configuration(args: argparse.Namespace) -> None:
    if not args.formal:
        return
    mismatches = {}
    for name, expected in REGISTERED_FORMAL_P_ROUTE_CONFIGURATION.items():
        observed = getattr(args, name)
        equal = (
            bool(np.isclose(float(observed), float(expected), rtol=1e-12, atol=1e-12))
            if isinstance(expected, float) else
            observed == expected and type(observed) is type(expected)
        )
        if not equal:
            mismatches[name] = {"observed": observed, "expected": expected}
    if mismatches:
        raise RuntimeError(
            "formal P66 route configuration drifted: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _exact_tensor_deduplication_plan(
    spectra: list[torch.Tensor],
) -> tuple[list[torch.Tensor], np.ndarray]:
    """Return exact unique spectra and a lossless raw-to-unique inverse index."""
    unique: list[torch.Tensor] = []
    positions: dict[tuple[str, tuple[int, ...], bytes], int] = {}
    inverse = np.empty(len(spectra), dtype=np.int64)
    for raw_index, tensor in enumerate(spectra):
        array = tensor.detach().cpu().contiguous().numpy()
        key = (array.dtype.str, tuple(array.shape), array.tobytes())
        unique_index = positions.get(key)
        if unique_index is None:
            unique_index = len(unique)
            positions[key] = unique_index
            unique.append(tensor)
        inverse[raw_index] = unique_index
    if len(inverse) and (
        int(inverse.min()) < 0 or int(inverse.max()) >= len(unique)
    ):
        raise RuntimeError("exact spectrum deduplication produced an invalid inverse index")
    return unique, inverse


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-student-checkpoint", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument(
        "--query-scope",
        choices=("official_errors", "official_correct", "all", "initial_error_boundary"),
        default="official_errors",
    )
    parser.add_argument("--max-queries", type=int, default=64)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=20260906)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--paired-advantage-threshold", type=float, default=0.01)
    parser.add_argument("--harm-margin-threshold", type=float, default=0.01)
    parser.add_argument("--robustness-slack", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--boundary-correct-multiplier", type=float, default=2.0)
    parser.add_argument("--minimum-boundary-correct", type=int, default=4096)
    parser.add_argument("--maximum-boundary-correct", type=int, default=8192)
    parser.add_argument("--maximum-corrective-frontier", type=int, default=16)
    parser.add_argument("--maximum-harmful-frontier", type=int, default=8)
    parser.add_argument("--maximum-robust-frontier", type=int, default=8)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _select_queries(graph: CandidateGraph, folds: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.query_scope == "initial_error_boundary":
        raise ValueError("initial-E8 query selection requires the encoded initialization")
    eligible = np.flatnonzero(folds != args.outer_fold)
    ranks = np.asarray([strict_rank(graph.official_molecule_scores(int(q))) for q in eligible])
    if args.query_scope == "official_errors":
        eligible = eligible[ranks > 1]
    elif args.query_scope == "official_correct":
        eligible = eligible[ranks == 1]
    if args.max_queries < 0:
        raise ValueError("max-queries cannot be negative")
    if args.max_queries and args.max_queries < len(eligible):
        eligible = np.sort(np.random.default_rng(args.sample_seed).choice(
            eligible, args.max_queries, replace=False,
        ))
    return np.asarray(eligible, dtype=np.int64)


def _initial_geometry(
    graph: CandidateGraph,
    queries: np.ndarray,
    embeddings: np.ndarray,
    embedding_index: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.empty(len(queries), dtype=np.int16)
    margins = np.empty(len(queries), dtype=np.float32)
    for position, query_value in enumerate(queries):
        query = int(query_value)
        _, rows, ptr, _ = graph.query_block(query)
        qvector = embeddings[embedding_index[int(graph.query_row[query])]]
        candidate = embeddings[[embedding_index[int(row)] for row in rows]]
        molecule = np.maximum.reduceat(candidate @ qvector, ptr[:-1])
        ranks[position] = strict_rank(molecule)
        margins[position] = float(molecule[0] - np.max(molecule[1:]))
    return ranks, margins


def _route(
    clean_rank: int,
    clean_margin: float,
    action_rank: int,
    action_margin: float,
    control_margin: float,
    args: argparse.Namespace,
) -> str:
    """Route one P action through the shared production routing contract."""
    return route_action(
        clean_rank=clean_rank,
        clean_margin=clean_margin,
        action_rank=action_rank,
        action_margin=action_margin,
        control_margin=control_margin,
        thresholds=RoutingThresholds(
            paired_advantage=float(args.paired_advantage_threshold),
            harm_margin=float(args.harm_margin_threshold),
            robustness_slack=float(args.robustness_slack),
        ),
    )


def main() -> None:
    args = arguments()
    _validate_registered_formal_configuration(args)
    started = time.time()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.outer_fold not in range(5):
        raise ValueError("outer fold must be 0..4")
    if args.formal and args.max_queries != 0:
        raise ValueError("formal P routing must consume the complete selected scope")
    if (
        args.query_chunk_size < 1
        or min(
            args.maximum_corrective_frontier,
            args.maximum_harmful_frontier,
            args.maximum_robust_frontier,
        ) < 1
    ):
        raise ValueError("P routing chunk/frontier limits must be positive")
    graph_path = args.graph_dir / "candidate_graph.npz"
    graph_report_path = args.graph_dir / "report.json"
    required = (
        graph_path, graph_report_path, args.data, args.official_checkpoint,
        args.architecture_checkpoint, args.initial_student_checkpoint,
        args.initial_student_checkpoint.parent / "decision.json",
    )
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    if graph_report.get("formal_training_authorized") is not True:
        raise RuntimeError("corrected graph is not training-authorized")
    graph = CandidateGraph(graph_path)
    folds = np.asarray([
        stable_fold(str(value), 5, args.formula_fold_seed) for value in graph.query_formula
    ], dtype=np.int8)
    eligible = np.flatnonzero(folds != args.outer_fold).astype(np.int64)
    queries = (
        eligible if args.query_scope == "initial_error_boundary"
        else _select_queries(graph, folds, args)
    )

    needed: set[int] = set(map(int, graph.query_row[queries]))
    for query in queries:
        _, rows, _, _ = graph.query_block(int(query))
        needed.update(map(int, rows))
    reachable = np.asarray(sorted(needed), dtype=np.int64)
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    with h5py.File(args.data, "r") as handle:
        instrument_values = decode(handle["INSTRUMENT_TYPE"][store.rows])
        collision_values = np.asarray(handle["COLLISION_ENERGY"][store.rows], dtype=float)
    instruments = {
        int(row): clean_instrument(str(value)) for row, value in zip(store.rows, instrument_values)
    }
    collision_energy = {
        int(row): float(value) for row, value in zip(store.rows, collision_values)
    }

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, _ = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
    decision = json.loads((args.initial_student_checkpoint.parent / "decision.json").read_text(encoding="utf-8"))
    if (
        package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
        or package.get("inference_clean_only") is not True
        or package.get("P2b_used") is not False
        or int(package.get("outer_fold", -1)) != args.outer_fold
        or decision.get("formal") is not True
    ):
        raise RuntimeError("mature E4/E8 initialization contract failed")
    model.load_state_dict(package["model_state"], strict=True)
    model.eval()
    embeddings = encode_rows(
        model, store, reachable, device, args.batch_size, args.amp, "full-P-router-initial",
    )
    embedding_index = {int(row): index for index, row in enumerate(reachable)}
    panel_report: dict[str, object] = {
        "selection": args.query_scope,
        "selected_action_queries": int(len(queries)),
    }
    if args.query_scope == "initial_error_boundary":
        initial_ranks, initial_margins = _initial_geometry(
            graph, eligible, embeddings, embedding_index,
        )
        queries, panel_report = select_initial_e8_error_boundary_queries(
            eligible,
            initial_ranks,
            initial_margins,
            graph.query_ik14[eligible],
            boundary_multiplier=args.boundary_correct_multiplier,
            minimum_boundary_correct=args.minimum_boundary_correct,
            maximum_boundary_correct=args.maximum_boundary_correct,
        )
        panel_report["selection"] = "all_initial_E8_errors_plus_identity_diverse_boundary_correct"
    if not len(queries):
        raise RuntimeError("P router query panel is empty")
    recipes = registered_p_recipes()

    policies = sorted({recipe.reference_policy for recipe in recipes})
    missing_context_keys = {
        (
            recipe.reference_policy,
            float(recipe.minimum_reference_prevalence),
            int(recipe.maximum_transferred_peaks),
        )
        for recipe in recipes
        if recipe.family in P_MISSING_PEAK_FAMILIES
    }
    frontier_frames: list[pd.DataFrame] = []
    frontier_action_chunks: list[np.ndarray] = []
    frontier_control_chunks: list[np.ndarray] = []
    per_query_frames: list[pd.DataFrame] = []
    route_counts: Counter[str] = Counter()
    total_positive_actions = 0
    maximum_resident_spectra = 0
    maximum_resident_unique_spectra = 0
    maximum_resident_embedding_bytes = 0
    total_materialized_spectra = 0
    total_unique_encoded_spectra = 0
    reference_profile_builds = 0
    recurrent_missing_peak_builds = 0
    for chunk_left in range(0, len(queries), args.query_chunk_size):
        chunk = queries[chunk_left:chunk_left + args.query_chunk_size]
        contexts: dict[int, dict[str, object]] = {}
        spectra: list[torch.Tensor] = []
        metadata: list[dict[str, object]] = []
        for query_value in chunk:
            query = int(query_value)
            _, rows, ptr, _ = graph.query_block(query)
            qrow = int(graph.query_row[query])
            qvector = embeddings[embedding_index[qrow]]
            candidate_vectors = embeddings[[embedding_index[int(row)] for row in rows]]
            pair_scores = candidate_vectors @ qvector
            molecule_scores = np.maximum.reduceat(pair_scores, ptr[:-1])
            wrong = int(np.argmax(molecule_scores[1:])) + 1
            clean_rank = strict_rank(molecule_scores)
            clean_margin = float(molecule_scores[0] - molecule_scores[wrong])
            references: dict[tuple[str, str], np.ndarray] = {}
            for policy in policies:
                for direction, molecule in (("positive", 0), ("direction_control", wrong)):
                    left, right = map(int, ptr[molecule:molecule + 2])
                    local_rows = np.asarray(rows[left:right], dtype=np.int64)
                    references[(policy, direction)] = choose_reference_rows(
                        local_rows, pair_scores[left:right], candidate_vectors[left:right], policy,
                        query_instrument=instruments[qrow],
                        query_collision_energy=collision_energy[qrow],
                        instruments=instruments, collision_energy=collision_energy,
                    )
            contexts[query] = {
                "candidate_rows": np.asarray(rows, dtype=np.int64),
                "candidate_vectors": candidate_vectors,
                "ptr": np.asarray(ptr, dtype=np.int64),
                "clean_rank": clean_rank,
                "clean_margin": clean_margin,
            }
            clean = store.one(qrow)
            reference_spectra: dict[tuple[str, str], list[torch.Tensor]] = {}
            profiles: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
            missing_contexts: dict[tuple[str, str, float, int], np.ndarray] = {}
            for recipe in recipes:
                action_id = f"q{query}|{recipe.recipe_id}"
                for direction in ("positive", "direction_control"):
                    reference_key = (recipe.reference_policy, direction)
                    selected_reference = references[reference_key]
                    if reference_key not in reference_spectra:
                        reference_spectra[reference_key] = [
                            store.one(int(row)) for row in selected_reference
                        ]
                    if reference_key not in profiles:
                        profiles[reference_key] = reference_profile(
                            clean, reference_spectra[reference_key],
                            args.fragment_tolerance,
                        )
                        reference_profile_builds += 1
                    missing = None
                    if recipe.family in P_MISSING_PEAK_FAMILIES:
                        missing_key = (
                            recipe.reference_policy,
                            direction,
                            float(recipe.minimum_reference_prevalence),
                            int(recipe.maximum_transferred_peaks),
                        )
                        if missing_key not in missing_contexts:
                            missing_contexts[missing_key] = recurrent_missing_peaks(
                                clean,
                                reference_spectra[reference_key],
                                args.fragment_tolerance,
                                recipe.minimum_reference_prevalence,
                                recipe.maximum_transferred_peaks,
                            )
                            recurrent_missing_peak_builds += 1
                        missing = missing_contexts[missing_key]
                    spectra.append(materialize_p_action(
                        clean,
                        reference_spectra[reference_key],
                        recipe,
                        args.fragment_tolerance,
                        profile=profiles[reference_key],
                        missing=missing,
                    ))
                    metadata.append({
                        "action_id": action_id, "direction": direction,
                        "query_index": query, "query_row": qrow,
                        "query_ik14": str(graph.query_ik14[query]),
                        "query_formula": str(graph.query_formula[query]),
                        "near": bool(graph.query_has_near[query]),
                        "formula_fold": int(folds[query]),
                        "source": recipe.source,
                        "reference_policy": recipe.reference_policy,
                        "family": recipe.family, "dose": recipe.dose,
                        "auxiliary_dose": recipe.auxiliary_dose,
                        "minimum_reference_prevalence": recipe.minimum_reference_prevalence,
                        "maximum_transferred_peaks": recipe.maximum_transferred_peaks,
                        "support_weighted": recipe.support_weighted,
                        "recipe_id": recipe.recipe_id,
                        "reference_rows": ",".join(map(str, selected_reference)),
                    })

        maximum_resident_spectra = max(maximum_resident_spectra, len(spectra))
        unique_spectra, raw_to_unique = _exact_tensor_deduplication_plan(spectra)
        maximum_resident_unique_spectra = max(
            maximum_resident_unique_spectra, len(unique_spectra),
        )
        total_materialized_spectra += len(spectra)
        total_unique_encoded_spectra += len(unique_spectra)
        encoded = np.empty((len(unique_spectra), embeddings.shape[1]), dtype=np.float32)
        maximum_resident_embedding_bytes = max(
            maximum_resident_embedding_bytes, int(encoded.nbytes),
        )
        with torch.inference_mode():
            for left in range(0, len(unique_spectra), args.batch_size):
                right = min(left + args.batch_size, len(unique_spectra))
                encoded[left:right] = forward_embeddings(
                    model, torch.stack(unique_spectra[left:right]).to(device), args.amp,
                ).float().cpu().numpy()

        result = pd.DataFrame(metadata)
        scores = []
        for row, unique_index in zip(result.itertuples(index=False), raw_to_unique):
            context = contexts[int(row.query_index)]
            scores.append(score_candidate_boundary(
                np.asarray(context["candidate_rows"]),
                np.asarray(context["ptr"]),
                np.asarray(context["candidate_vectors"]),
                encoded[int(unique_index)],
            ))
        result["rank"] = np.asarray([value.rank for value in scores], dtype=np.int16)
        result["margin"] = np.asarray([value.margin for value in scores], dtype=np.float32)
        result["positive_row"] = np.asarray(
            [value.positive_row for value in scores], dtype=np.int64,
        )
        result["hard_negative_molecule_index"] = np.asarray(
            [value.hard_negative_molecule_index for value in scores], dtype=np.int32,
        )
        result["hard_negative_row"] = np.asarray(
            [value.hard_negative_row for value in scores], dtype=np.int64,
        )
        positive = result.loc[result.direction.eq("positive")].reset_index(drop=True)
        control = result.loc[result.direction.eq("direction_control")].reset_index(drop=True)
        if not np.array_equal(positive.action_id.astype(str), control.action_id.astype(str)):
            raise RuntimeError("positive/control P action pairs drifted")
        positive = positive.rename(columns={
            "rank": "action_rank", "margin": "action_margin",
            "positive_row": "action_positive_row",
            "hard_negative_molecule_index": "action_hard_negative_molecule_index",
            "hard_negative_row": "action_hard_negative_row",
        })
        positive["control_rank"] = control["rank"].to_numpy(np.int16)
        positive["control_margin"] = control["margin"].to_numpy(np.float32)
        positive["control_kind"] = "wrong_identity_direction"
        positive["control_semantic"] = "wrong_identity_direction"
        positive["control_positive_row"] = control["positive_row"].to_numpy(np.int64)
        positive["control_hard_negative_molecule_index"] = control[
            "hard_negative_molecule_index"
        ].to_numpy(np.int32)
        positive["control_hard_negative_row"] = control[
            "hard_negative_row"
        ].to_numpy(np.int64)
        positive["clean_rank"] = [
            int(contexts[int(query)]["clean_rank"]) for query in positive.query_index
        ]
        positive["clean_margin"] = [
            float(contexts[int(query)]["clean_margin"]) for query in positive.query_index
        ]
        positive["margin_change"] = positive.action_margin - positive.clean_margin
        positive["paired_advantage"] = positive.action_margin - positive.control_margin
        positive["route"] = [
            _route(
                int(row.clean_rank), float(row.clean_margin),
                int(row.action_rank), float(row.action_margin),
                float(row.control_margin), args,
            )
            for row in positive.itertuples(index=False)
        ]
        positive["corrective_weight"] = positive.route.eq("corrective").astype(np.float32)
        positive["action_tensor_index"] = np.arange(len(positive), dtype=np.int64)
        route_counts.update(map(str, positive.route))
        total_positive_actions += len(positive)
        per_query_frames.append(positive.groupby("query_index", sort=True).agg(
            clean_rank=("clean_rank", "first"),
            corrective_actions=("corrective_weight", "sum"),
            best_action_rank=("action_rank", "min"),
            best_margin_change=("margin_change", "max"),
            harmful_actions=("route", lambda value: int(np.sum(value == "harmful"))),
        ).reset_index())

        frontier = lossless_selector_frontier(
            positive,
            maximum_corrective_per_query=args.maximum_corrective_frontier,
            maximum_harmful_per_query=args.maximum_harmful_frontier,
            maximum_robust_per_query=args.maximum_robust_frontier,
        )
        source_indices = frontier.action_tensor_index.to_numpy(np.int64)
        if len(frontier):
            frontier_action_chunks.append(np.stack([
                spectra[2 * index].detach().cpu().numpy() for index in source_indices
            ]).astype(np.float32))
            frontier_control_chunks.append(np.stack([
                spectra[2 * index + 1].detach().cpu().numpy() for index in source_indices
            ]).astype(np.float32))
            frontier_frames.append(frontier)
        processed = chunk_left + len(chunk)
        print(
            f"[full-P-router chunk] {processed:,}/{len(queries):,}; "
            f"resident_spectra={len(spectra):,}; "
            f"unique_encoded={len(unique_spectra):,}; frontier={len(frontier):,}",
            flush=True,
        )

    if not frontier_frames:
        raise RuntimeError("P routing produced no corrective/harmful/robust selector frontier")
    positive = pd.concat(frontier_frames, ignore_index=True, sort=False)
    action_spectra = np.concatenate(frontier_action_chunks, axis=0)
    control_spectra = np.concatenate(frontier_control_chunks, axis=0)
    if len(positive) != len(action_spectra) or action_spectra.shape != control_spectra.shape:
        raise RuntimeError("P selector frontier tensor alignment failed")
    positive["action_tensor_index"] = np.arange(len(positive), dtype=np.int64)
    per_query = pd.concat(per_query_frames, ignore_index=True, sort=False)
    report = {
        "status": "noise_corrected_full_p_router_audit_complete",
        "formal_training_authorized": bool(args.formal),
        "outer_formula_fold": args.outer_fold,
        "query_scope": args.query_scope,
        "configuration": {
            name: getattr(args, name)
            for name in REGISTERED_FORMAL_P_ROUTE_CONFIGURATION
        },
        "queries": int(len(queries)), "recipes": len(recipes),
        "recipe_source_counts": {
            str(key): int(value) for key, value in Counter(
                recipe.source for recipe in recipes
            ).items()
        },
        "recipe_family_counts": {
            str(key): int(value) for key, value in Counter(
                recipe.family for recipe in recipes
            ).items()
        },
        "positive_actions": int(total_positive_actions),
        "direction_controls": int(total_positive_actions),
        "positive_actions_evaluated": int(total_positive_actions),
        "direction_controls_evaluated": int(total_positive_actions),
        "selector_frontier_actions_retained": int(len(positive)),
        "route_counts": {str(key): int(value) for key, value in route_counts.items()},
        "queries_with_corrective_action": int(per_query.corrective_actions.gt(0).sum()),
        "error_queries": int(per_query.clean_rank.gt(1).sum()),
        "error_queries_with_rank1_action": int((per_query.clean_rank.gt(1) & per_query.best_action_rank.eq(1)).sum()),
        "error_queries_with_corrective_action": int((per_query.clean_rank.gt(1) & per_query.corrective_actions.gt(0)).sum()),
        "action_panel": panel_report,
        "bounded_memory": {
            "query_chunk_size": int(args.query_chunk_size),
            "maximum_resident_spectra": int(maximum_resident_spectra),
            "maximum_resident_unique_encoded_spectra": int(
                maximum_resident_unique_spectra
            ),
            "maximum_resident_action_embedding_bytes": int(maximum_resident_embedding_bytes),
            "monolithic_all_action_embedding_allocation": False,
        },
        "exact_spectrum_encoding_deduplication": {
            "materialized_action_and_control_spectra": int(total_materialized_spectra),
            "unique_spectra_sent_to_encoder": int(total_unique_encoded_spectra),
            "duplicate_encoder_forwards_avoided": int(
                total_materialized_spectra - total_unique_encoded_spectra
            ),
            "unique_fraction": float(
                total_unique_encoded_spectra / total_materialized_spectra
            ),
        },
        "reference_context_cache": {
            "reference_profile_builds": int(reference_profile_builds),
            "expected_reference_profile_builds": int(
                len(queries) * len(policies) * 2
            ),
            "recurrent_missing_peak_builds": int(recurrent_missing_peak_builds),
            "expected_recurrent_missing_peak_builds": int(
                len(queries) * len(missing_context_keys) * 2
            ),
            "naive_uncached_materialization_calls": int(
                len(queries) * len(recipes) * 2
            ),
        },
        "selector_frontier": {
            "maximum_corrective_per_query": int(args.maximum_corrective_frontier),
            "maximum_harmful_per_query": int(args.maximum_harmful_frontier),
            "maximum_robust_per_query": int(args.maximum_robust_frontier),
        },
        "contracts": {
            "all_66_unique_mature_P_cells_materialized": len(recipes) == 66,
            "registered_formal_route_configuration_verified": bool(args.formal),
            "original_12_cell_P_intensity_matrix_complete_without_duplicates": bool(
                {
                    (recipe.family, float(recipe.dose))
                    for recipe in recipes
                    if recipe.reference_policy == "top3"
                    and recipe.family in {
                        "matched_intensity_transport",
                        "prevalence_attenuation",
                        "consensus_projection",
                    }
                }
                == {
                    (family, dose)
                    for family in (
                        "matched_intensity_transport",
                        "prevalence_attenuation",
                        "consensus_projection",
                    )
                    for dose in (0.25, 0.50, 0.75, 1.00)
                }
            ),
            "reference_profiles_cached_per_query_policy_direction": bool(
                reference_profile_builds == len(queries) * len(policies) * 2
            ),
            "missing_peaks_cached_per_query_policy_direction_parameters": bool(
                recurrent_missing_peak_builds
                == len(queries) * len(missing_context_keys) * 2
            ),
            "exact_action_tensor_deduplication_before_encoder_is_lossless": bool(
                total_materialized_spectra == total_positive_actions * 2
                and 0 < total_unique_encoded_spectra <= total_materialized_spectra
            ),
            "wrong_identity_direction_control_paired": True,
            "control_semantics_explicit": True,
            "route_mined_only_on_outer_train": True,
            "current_E8_error_queries_all_retained": bool(
                args.query_scope != "initial_error_boundary"
                or panel_report["initial_E8_error_queries"]
                == int(per_query.clean_rank.gt(1).sum())
            ),
            "selector_frontier_is_lossless_for_global_caps": True,
            "exact_action_control_candidate_switch_rows_recorded": True,
            "nonfrontier_action_metadata_aggregated_not_materialized": True,
            "noncorrective_weight_exact_zero": bool(
                positive.loc[~positive.route.eq("corrective"), "corrective_weight"].eq(0).all()
            ),
            "outer_held_formula_consumed": False,
            "teacher_embedding_target_used": False,
            "P3_consumed": False,
        },
        "model_provenance": {
            "initial_student_checkpoint_sha256": sha256_file(args.initial_student_checkpoint),
            "initial_decision_sha256": sha256_file(args.initial_student_checkpoint.parent / "decision.json"),
        },
        "provenance": {
            "candidate_graph_sha256": sha256_file(graph_path),
            "graph_report_sha256": sha256_file(graph_report_path),
            "script_sha256": sha256_file(Path(__file__)),
            "full_action_registry_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_full_action_registry.py")
            ),
            "e10_action_executor_sha256": sha256_file(
                Path(__file__).with_name(
                    "audit_noise_final_e10_positive_residual_matrix.py"
                )
            ),
            "e11_reference_selector_sha256": sha256_file(
                Path(__file__).with_name(
                    "audit_noise_final_e11_reference_diversity_matrix.py"
                )
            ),
            "positive_guided_profile_sha256": sha256_file(
                Path(__file__).with_name("audit_noise_final_positive_guided_matrix.py")
            ),
            "positive_peak_transfer_sha256": sha256_file(
                Path(__file__).with_name("audit_noise_final_positive_peak_transfer.py")
            ),
            "action_routing_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_action_routing.py")
            ),
            "action_panel_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_action_panel.py")
            ),
            "action_routing_v3_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_action_routing_v3.py")
            ),
        },
        "runtime_seconds": time.time() - started,
        "claim_limit": "Bounded outer-train action routing audit; not encoder performance.",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        positive.to_csv(staging / "routed_actions.csv.gz", index=False, compression="gzip")
        per_query.to_csv(staging / "per_query.csv.gz", index=False, compression="gzip")
        np.savez_compressed(
            staging / "action_spectra.npz",
            action_ids=np.asarray(positive.action_id.astype(str), dtype=str),
            action_spectra=action_spectra,
            control_spectra=control_spectra,
        )
        report["provenance"]["action_spectra_sha256"] = sha256_file(staging / "action_spectra.npz")
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
