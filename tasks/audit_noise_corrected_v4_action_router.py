"""Route the frozen v4 best raw-spectrum actions in current-E8 geometry.

The producer is intentionally bounded and outcome-free at panel selection.
Every outer-training E8 error plus an identity-diverse low-margin correct panel
is chosen before a v4 action is executed.  Target actions and same-role matched
controls are then scored on the complete molecule candidate block and routed
into corrective, robustness, harmful or uncertain semantics.  Only a lossless
per-query frontier is materialized for the downstream direct trainer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from audit_noise_corrected_action_expansion_v4 import (
    build_recipe_views,
    build_sequential_supported_boost_views,
)
from build_noise_corrected_full_action_bank import current_context, load_initial_model
from noise_corrected_action_panel import (
    lossless_selector_frontier,
    select_initial_e8_error_boundary_queries,
)
from noise_corrected_action_routing import RoutingThresholds, route_action
from noise_corrected_action_routing_v3 import score_candidate_boundary
from noise_corrected_best_action_v5 import (
    V4_SOURCE,
    action_lineage_contract,
    registered_v4_action_recipes,
    select_registered_v4_views,
)
from noise_final_core import CandidateGraph, sha256_file, stable_fold
from noise_v3_core import stable_seed
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, forward_embeddings


REGISTERED_FORMAL_V4_ROUTE_CONFIGURATION: dict[str, object] = {
    "formula_fold_seed": 20260825,
    "query_scope": "initial_error_boundary",
    "max_queries": 0,
    "sample_seed": 20260907,
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
    "top_k_negatives": 5,
    "softmax_temperature": 0.10,
    "sequential_boost_steps": 4,
    "sequential_boost_dose": 0.50,
    "n_highest_peaks": 100,
    "amp": False,
}


def _validate_registered_formal_configuration(args: argparse.Namespace) -> None:
    if not args.formal:
        return
    mismatches = {}
    for name, expected in REGISTERED_FORMAL_V4_ROUTE_CONFIGURATION.items():
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
            "formal v4 best-action route configuration drifted: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _exact_tensor_deduplication_plan(
    spectra: list[torch.Tensor],
) -> tuple[list[torch.Tensor], np.ndarray]:
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
    return unique, inverse


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
        score = score_candidate_boundary(
            rows,
            ptr,
            embeddings[[embedding_index[int(row)] for row in rows]],
            embeddings[embedding_index[int(graph.query_row[query])]],
        )
        ranks[position] = score.rank
        margins[position] = score.margin
    return ranks, margins


def _bounded_development_panel(
    queries: np.ndarray,
    ranks_by_query: dict[int, int],
    maximum: int,
    seed: int,
) -> np.ndarray:
    if maximum < 0:
        raise ValueError("max-queries cannot be negative")
    if not maximum or len(queries) <= maximum:
        return np.asarray(queries, dtype=np.int64)
    errors = np.asarray(
        [int(query) for query in queries if ranks_by_query[int(query)] > 1],
        dtype=np.int64,
    )
    correct = np.asarray(
        [int(query) for query in queries if ranks_by_query[int(query)] == 1],
        dtype=np.int64,
    )
    rng = np.random.default_rng(int(seed))
    error_count = min(len(errors), max(1, maximum // 2))
    correct_count = min(len(correct), maximum - error_count)
    if error_count + correct_count < maximum:
        error_count = min(len(errors), maximum - correct_count)
    chosen = []
    if error_count:
        chosen.extend(rng.choice(errors, error_count, replace=False).tolist())
    if correct_count:
        chosen.extend(rng.choice(correct, correct_count, replace=False).tolist())
    return np.asarray(sorted(map(int, chosen)), dtype=np.int64)


def _action_id(query_row: int, recipe: dict[str, object]) -> str:
    payload = json.dumps({
        "version": "best-v5",
        "query_row": int(query_row),
        "recipe_id": str(recipe["recipe_id"]),
        "target_down": list(map(int, recipe["target_down"])),
        "target_up": list(map(int, recipe["target_up"])),
        "control_down": list(map(int, recipe["control_down"])),
        "control_up": list(map(int, recipe["control_up"])),
    }, sort_keys=True, separators=(",", ":")).encode()
    return "V4-" + hashlib.sha256(payload).hexdigest()[:24]


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
        "--query-scope", choices=("initial_error_boundary",),
        default="initial_error_boundary",
    )
    parser.add_argument("--max-queries", type=int, default=32)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=20260907)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--paired-advantage-threshold", type=float, default=0.01)
    parser.add_argument("--harm-margin-threshold", type=float, default=0.01)
    parser.add_argument("--robustness-slack", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-chunk-size", type=int, default=32)
    parser.add_argument("--gradient-batch-size", type=int, default=2)
    parser.add_argument("--boundary-correct-multiplier", type=float, default=2.0)
    parser.add_argument("--minimum-boundary-correct", type=int, default=4096)
    parser.add_argument("--maximum-boundary-correct", type=int, default=8192)
    parser.add_argument("--maximum-corrective-frontier", type=int, default=16)
    parser.add_argument("--maximum-harmful-frontier", type=int, default=8)
    parser.add_argument("--maximum-robust-frontier", type=int, default=8)
    parser.add_argument("--top-k-negatives", type=int, default=5)
    parser.add_argument("--softmax-temperature", type=float, default=0.10)
    parser.add_argument("--sequential-boost-steps", type=int, default=4)
    parser.add_argument("--sequential-boost-dose", type=float, default=0.50)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    _validate_registered_formal_configuration(args)
    started = time.time()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.outer_fold not in range(5):
        raise ValueError("outer fold must be 0..4")
    if args.formal and args.max_queries != 0:
        raise ValueError("formal v4 routing must consume the complete selected scope")
    if min(
        args.batch_size,
        args.query_chunk_size,
        args.gradient_batch_size,
        args.maximum_corrective_frontier,
        args.maximum_harmful_frontier,
        args.maximum_robust_frontier,
    ) < 1:
        raise ValueError("v4 router batch/frontier limits must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    graph_path = args.graph_dir / "candidate_graph.npz"
    graph_report_path = args.graph_dir / "report.json"
    decision_path = args.initial_student_checkpoint.parent / "decision.json"
    required = (
        graph_path, graph_report_path, args.data, args.official_checkpoint,
        args.architecture_checkpoint, args.initial_student_checkpoint, decision_path,
    )
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    if graph_report.get("formal_training_authorized") is not True:
        raise RuntimeError("corrected candidate graph is not training-authorized")

    graph = CandidateGraph(graph_path)
    folds = np.asarray([
        stable_fold(str(value), 5, args.formula_fold_seed)
        for value in graph.query_formula
    ], dtype=np.int8)
    eligible = np.flatnonzero(folds != args.outer_fold).astype(np.int64)
    needed: set[int] = set(map(int, graph.query_row[eligible]))
    for query_value in eligible:
        _, rows, _, _ = graph.query_block(int(query_value))
        needed.update(map(int, rows))
    reachable = np.asarray(sorted(needed), dtype=np.int64)
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    tensors = {
        int(row): store.tensor[position] for position, row in enumerate(store.rows)
    }
    device = torch.device(args.device)
    model, model_provenance = load_initial_model(args, device)
    embeddings = encode_rows(
        model, store, reachable, device, args.batch_size, args.amp,
        "v4-best-router-initial",
    )
    embedding_index = {int(row): position for position, row in enumerate(reachable)}
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
    ranks_by_query = {
        int(query): int(rank) for query, rank in zip(eligible, initial_ranks)
    }
    margins_by_query = {
        int(query): float(margin)
        for query, margin in zip(eligible, initial_margins)
    }
    full_panel_queries = np.asarray(queries, dtype=np.int64)
    queries = _bounded_development_panel(
        full_panel_queries, ranks_by_query, args.max_queries, args.sample_seed,
    )
    panel_report["selection"] = (
        "all_initial_E8_errors_plus_identity_diverse_boundary_correct"
    )
    panel_report["development_query_truncation"] = int(
        len(full_panel_queries) - len(queries)
    )
    panel_report["full_selected_action_queries"] = int(len(full_panel_queries))
    panel_report["selected_action_queries"] = int(len(queries))
    panel_report["all_initial_E8_errors_retained_before_development_truncation"] = bool(
        panel_report["initial_E8_error_queries"]
        <= panel_report["full_selected_action_queries"]
    )
    panel_report["all_initial_E8_errors_retained_in_executed_panel"] = bool(
        args.max_queries == 0
        and panel_report["initial_E8_error_queries"]
        <= panel_report["selected_action_queries"]
    )
    if not len(queries):
        raise RuntimeError("v4 best-action query panel is empty")

    thresholds = RoutingThresholds(
        paired_advantage=args.paired_advantage_threshold,
        harm_margin=args.harm_margin_threshold,
        robustness_slack=args.robustness_slack,
    )
    frontier_frames: list[pd.DataFrame] = []
    frontier_action_chunks: list[np.ndarray] = []
    frontier_control_chunks: list[np.ndarray] = []
    per_query_frames: list[pd.DataFrame] = []
    route_counts: Counter[str] = Counter()
    materialized_recipe_counts: Counter[str] = Counter()
    incomplete_control_counts: Counter[str] = Counter()
    total_actions = 0
    total_raw_encoder_views = 0
    total_unique_encoder_views = 0
    maximum_resident_spectra = 0
    maximum_resident_embedding_bytes = 0

    for chunk_left in range(0, len(queries), args.query_chunk_size):
        chunk = queries[chunk_left:chunk_left + args.query_chunk_size]
        query_meta: list[dict[str, object]] = []
        gradients: dict[int, np.ndarray] = {}
        roles_by_query: dict[int, np.ndarray] = {}
        for batch_left in range(0, len(chunk), args.gradient_batch_size):
            local = chunk[batch_left:batch_left + args.gradient_batch_size]
            clean = torch.stack([
                tensors[int(graph.query_row[int(query)])] for query in local
            ]).to(device)
            clean.requires_grad_(True)
            current = forward_embeddings(model, clean, False)
            contexts = [
                current_context(
                    graph, int(query), vector, embeddings, embedding_index,
                    tensors, tensors[int(graph.query_row[int(query)])],
                    args.top_k_negatives, args.fragment_tolerance,
                )
                for query, vector in zip(
                    local, current.detach().float().cpu().numpy(),
                )
            ]
            positive = torch.as_tensor(np.stack([
                embeddings[embedding_index[int(context[0].positive_row)]]
                for context in contexts
            ]), device=device, dtype=current.dtype)
            maximum = max(len(context[0].negative_rows) for context in contexts)
            negative = torch.zeros(
                (len(local), maximum, embeddings.shape[1]),
                device=device, dtype=current.dtype,
            )
            valid = torch.zeros(
                (len(local), maximum), device=device, dtype=torch.bool,
            )
            for position, context in enumerate(contexts):
                rows = context[0].negative_rows
                negative[position, :len(rows)] = torch.as_tensor(np.stack([
                    embeddings[embedding_index[int(row)]] for row in rows
                ]), device=device, dtype=current.dtype)
                valid[position, :len(rows)] = True
            positive_score = torch.sum(current * positive, dim=1)
            negative_score = torch.einsum(
                "bd,bkd->bk", current, negative,
            ).masked_fill(~valid, -1e9)
            weight = torch.softmax(
                negative_score / args.softmax_temperature, dim=1,
            ).detach()
            objective = positive_score - torch.sum(weight * negative_score, dim=1)
            local_gradient = torch.autograd.grad(objective.sum(), clean)[0][:, :, 1]
            for position, query_value in enumerate(local):
                query = int(query_value)
                gradients[query] = local_gradient[
                    position
                ].detach().float().cpu().numpy()
                roles_by_query[query] = contexts[position][1]
                query_meta.append({
                    "query_index": query,
                    "query_row": int(graph.query_row[query]),
                    "query_ik14": str(graph.query_ik14[query]),
                    "query_formula": str(graph.query_formula[query]),
                    "formula_fold": int(folds[query]),
                    "near": bool(graph.query_has_near[query]),
                    # Routing and panel selection must use one immutable clean
                    # geometry.  The live gradient-batch forward above exists
                    # only to construct the action direction; using it again
                    # for clean rank made V4 disagree with P/N/A4 when batch
                    # composition changed floating-point boundary ties.
                    "baseline_rank": int(ranks_by_query[query]),
                    "baseline_margin": float(margins_by_query[query]),
                })
        meta = pd.DataFrame(query_meta)
        sequential = build_sequential_supported_boost_views(
            meta, model, graph, tensors, embeddings, embedding_index, device, args,
        )
        records: list[dict[str, object]] = []
        raw_spectra: list[torch.Tensor] = []
        for row in meta.itertuples(index=False):
            query = int(row.query_index)
            clean = tensors[int(row.query_row)]
            generated = build_recipe_views(
                clean,
                gradients[query],
                roles_by_query[query],
                seed=stable_seed(args.sample_seed, int(row.query_row)),
                baseline_margin=float(row.baseline_margin),
            )
            generated.extend(sequential.get(query, ()))
            for recipe in select_registered_v4_views(generated):
                if recipe["control"] is None:
                    incomplete_control_counts[str(recipe["recipe"])] += 1
                    continue
                action_id = _action_id(int(row.query_row), recipe)
                records.append({
                    "action_id": action_id,
                    "direction": "direct_gradient_path",
                    "query_index": query,
                    "query_row": int(row.query_row),
                    "query_ik14": str(row.query_ik14),
                    "query_formula": str(row.query_formula),
                    "near": bool(row.near),
                    "formula_fold": int(row.formula_fold),
                    "source": V4_SOURCE,
                    "reference_policy": "live_current_boundary",
                    "family": str(recipe["family"]),
                    "recipe": str(recipe["recipe"]),
                    "recipe_id": str(recipe["recipe_id"]),
                    "evidence_role": str(recipe["evidence_role"]),
                    "target_down": json.dumps(list(map(int, recipe["target_down"]))),
                    "target_up": json.dumps(list(map(int, recipe["target_up"]))),
                    "control_down": json.dumps(list(map(int, recipe["control_down"]))),
                    "control_up": json.dumps(list(map(int, recipe["control_up"]))),
                    "dose": recipe.get("total_fractional_dose"),
                    "clean_rank": int(row.baseline_rank),
                    "clean_margin": float(row.baseline_margin),
                    "control_kind": "matched_path",
                    "control_semantic": "matched_neutral",
                })
                raw_spectra.extend([recipe["target"], recipe["control"]])
                materialized_recipe_counts[str(recipe["recipe"])] += 1
        if not records:
            continue
        unique_spectra, raw_to_unique = _exact_tensor_deduplication_plan(raw_spectra)
        total_raw_encoder_views += len(raw_spectra)
        total_unique_encoder_views += len(unique_spectra)
        maximum_resident_spectra = max(maximum_resident_spectra, len(raw_spectra))
        encoded = np.empty((len(unique_spectra), embeddings.shape[1]), dtype=np.float32)
        maximum_resident_embedding_bytes = max(
            maximum_resident_embedding_bytes, int(encoded.nbytes),
        )
        with torch.inference_mode():
            for left in range(0, len(unique_spectra), args.batch_size):
                right = min(left + args.batch_size, len(unique_spectra))
                encoded[left:right] = forward_embeddings(
                    model,
                    torch.stack(unique_spectra[left:right]).to(device),
                    args.amp,
                ).float().cpu().numpy()
        scored = []
        action_arrays = []
        control_arrays = []
        for index, record in enumerate(records):
            query = int(record["query_index"])
            _, rows, ptr, _ = graph.query_block(query)
            candidate = embeddings[[embedding_index[int(row)] for row in rows]]
            action_score = score_candidate_boundary(
                rows, ptr, candidate, encoded[int(raw_to_unique[2 * index])],
            )
            control_score = score_candidate_boundary(
                rows, ptr, candidate, encoded[int(raw_to_unique[2 * index + 1])],
            )
            route = route_action(
                clean_rank=int(record["clean_rank"]),
                clean_margin=float(record["clean_margin"]),
                action_rank=action_score.rank,
                action_margin=action_score.margin,
                control_margin=control_score.margin,
                thresholds=thresholds,
            )
            scored.append({
                **record,
                "action_rank": int(action_score.rank),
                "action_margin": float(action_score.margin),
                "action_positive_row": int(action_score.positive_row),
                "action_hard_negative_molecule_index": int(
                    action_score.hard_negative_molecule_index
                ),
                "action_hard_negative_row": int(action_score.hard_negative_row),
                "control_rank": int(control_score.rank),
                "control_margin": float(control_score.margin),
                "control_positive_row": int(control_score.positive_row),
                "control_hard_negative_molecule_index": int(
                    control_score.hard_negative_molecule_index
                ),
                "control_hard_negative_row": int(control_score.hard_negative_row),
                "margin_change": float(action_score.margin - float(record["clean_margin"])),
                "paired_advantage": float(action_score.margin - control_score.margin),
                "route": route,
                "corrective_weight": float(route == "corrective"),
                "action_tensor_index": index,
            })
            action_arrays.append(raw_spectra[2 * index].detach().cpu().numpy())
            control_arrays.append(raw_spectra[2 * index + 1].detach().cpu().numpy())
        frame = pd.DataFrame(scored)
        route_counts.update(map(str, frame.route))
        total_actions += len(frame)
        per_query_frames.append(frame.groupby("query_index", sort=True).agg(
            clean_rank=("clean_rank", "first"),
            corrective_actions=("corrective_weight", "sum"),
            best_action_rank=("action_rank", "min"),
            best_margin_change=("margin_change", "max"),
            harmful_actions=("route", lambda values: int(np.sum(values == "harmful"))),
        ).reset_index())
        frontier = lossless_selector_frontier(
            frame,
            maximum_corrective_per_query=args.maximum_corrective_frontier,
            maximum_harmful_per_query=args.maximum_harmful_frontier,
            maximum_robust_per_query=args.maximum_robust_frontier,
        )
        take = frontier.action_tensor_index.to_numpy(np.int64)
        if len(frontier):
            frontier_frames.append(frontier)
            frontier_action_chunks.append(
                np.stack([action_arrays[index] for index in take]).astype(np.float32)
            )
            frontier_control_chunks.append(
                np.stack([control_arrays[index] for index in take]).astype(np.float32)
            )
        processed = chunk_left + len(chunk)
        print(
            f"[v4-best-router chunk] {processed:,}/{len(queries):,}; "
            f"actions={len(frame):,}; unique_views={len(unique_spectra):,}; "
            f"frontier={len(frontier):,}",
            flush=True,
        )

    if not frontier_frames:
        raise RuntimeError("v4 best-action routing produced no trainable frontier")
    routed = pd.concat(frontier_frames, ignore_index=True, sort=False)
    action_spectra = np.concatenate(frontier_action_chunks, axis=0)
    control_spectra = np.concatenate(frontier_control_chunks, axis=0)
    if len(routed) != len(action_spectra) or action_spectra.shape != control_spectra.shape:
        raise RuntimeError("v4 best-action frontier tensor alignment failed")
    if routed.action_id.astype(str).duplicated().any():
        raise RuntimeError("v4 best-action IDs are not unique")
    routed["action_tensor_index"] = np.arange(len(routed), dtype=np.int64)
    per_query = pd.concat(per_query_frames, ignore_index=True, sort=False)

    report: dict[str, object] = {
        "status": "noise_corrected_v4_action_router_audit_complete",
        "formal_training_authorized": bool(args.formal),
        "outer_formula_fold": int(args.outer_fold),
        "query_scope": args.query_scope,
        "configuration": {
            name: getattr(args, name)
            for name in REGISTERED_FORMAL_V4_ROUTE_CONFIGURATION
        },
        "action_lineage": action_lineage_contract(),
        "registered_recipes": [
            {
                "name": recipe.name,
                "family": recipe.family,
                "evidence_role": recipe.evidence_role,
                "development_evidence": recipe.development_evidence,
            }
            for recipe in registered_v4_action_recipes()
        ],
        "queries": int(len(queries)),
        "positive_actions": int(total_actions),
        "direction_controls": int(total_actions),
        "selector_frontier_actions_retained": int(len(routed)),
        "route_counts": {str(key): int(value) for key, value in route_counts.items()},
        "materialized_recipe_counts": dict(materialized_recipe_counts),
        "incomplete_control_counts": dict(incomplete_control_counts),
        "queries_with_corrective_action": int(per_query.corrective_actions.gt(0).sum()),
        "error_queries": int(per_query.clean_rank.gt(1).sum()),
        "error_queries_with_rank1_action": int(
            (per_query.clean_rank.gt(1) & per_query.best_action_rank.eq(1)).sum()
        ),
        "error_queries_with_corrective_action": int(
            (per_query.clean_rank.gt(1) & per_query.corrective_actions.gt(0)).sum()
        ),
        "action_panel": panel_report,
        "bounded_memory": {
            "query_chunk_size": int(args.query_chunk_size),
            "maximum_resident_spectra": int(maximum_resident_spectra),
            "maximum_resident_action_embedding_bytes": int(
                maximum_resident_embedding_bytes
            ),
            "monolithic_all_action_embedding_allocation": False,
        },
        "exact_spectrum_encoding_deduplication": {
            "materialized_action_and_control_spectra": int(total_raw_encoder_views),
            "unique_spectra_sent_to_encoder": int(total_unique_encoder_views),
            "duplicate_encoder_forwards_avoided": int(
                total_raw_encoder_views - total_unique_encoder_views
            ),
        },
        "selector_frontier": {
            "maximum_corrective_per_query": int(args.maximum_corrective_frontier),
            "maximum_harmful_per_query": int(args.maximum_harmful_frontier),
            "maximum_robust_per_query": int(args.maximum_robust_frontier),
        },
        "contracts": {
            "registered_formal_route_configuration_verified": bool(args.formal),
            "only_frozen_best_v4_recipes_materialized": bool(
                set(materialized_recipe_counts)
                <= {recipe.name for recipe in registered_v4_action_recipes()}
            ),
            "same_role_matched_neutral_controls_only": bool(
                routed.control_semantic.astype(str).eq("matched_neutral").all()
            ),
            "control_semantics_explicit": True,
            "route_mined_only_on_outer_train": True,
            "current_E8_error_queries_all_retained_in_action_panel": bool(
                panel_report[
                    "all_initial_E8_errors_retained_in_executed_panel"
                ]
            ),
            "selector_frontier_is_lossless_for_global_caps": True,
            "exact_action_control_candidate_switch_rows_recorded": True,
            "nonfrontier_action_metadata_aggregated_not_materialized": True,
            "noncorrective_weight_exact_zero": bool(
                routed.loc[
                    ~routed.route.eq("corrective"), "corrective_weight"
                ].eq(0).all()
            ),
            "outer_held_formula_consumed": False,
            "teacher_embedding_target_used": False,
            "P3_consumed": False,
        },
        "model_provenance": model_provenance,
        "provenance": {
            "candidate_graph_sha256": sha256_file(graph_path),
            "graph_report_sha256": sha256_file(graph_report_path),
            "script_sha256": sha256_file(Path(__file__)),
            "best_action_registry_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_best_action_v5.py")
            ),
            "action_expansion_audit_sha256": sha256_file(
                Path(__file__).with_name("audit_noise_corrected_action_expansion_v4.py")
            ),
            "action_expansion_core_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_action_expansion_v4.py")
            ),
            "trust_region_core_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_trust_region_action_v4.py")
            ),
            "action_routing_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_action_routing.py")
            ),
            "action_routing_v3_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_action_routing_v3.py")
            ),
            "action_panel_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_action_panel.py")
            ),
        },
        "runtime_seconds": float(time.time() - started),
        "claim_limit": "Outer-train raw-spectrum action routing; not encoder performance.",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent,
    ))
    try:
        routed.to_csv(
            staging / "routed_actions.csv.gz", index=False, compression="gzip",
        )
        per_query.to_csv(
            staging / "per_query.csv.gz", index=False, compression="gzip",
        )
        np.savez_compressed(
            staging / "action_spectra.npz",
            action_ids=np.asarray(routed.action_id.astype(str), dtype=str),
            action_spectra=action_spectra,
            control_spectra=control_spectra,
        )
        report["provenance"]["action_spectra_sha256"] = sha256_file(
            staging / "action_spectra.npz"
        )
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
