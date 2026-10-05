"""Route every mature A4 Top-50 exact peak action in corrected E8 geometry.

Only outcome-free A4 proposal metadata are reused: real token, candidate role,
gradient rank and the preregistered four attenuation doses.  Historical exact
outcomes and the later one-action teacher selection are deliberately not read.
Each proposed intervention is executed again on the raw spectrum and scored on
the corrected molecule-max graph with a matched peak-control direction.
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

import h5py
import numpy as np
import pandas as pd
import torch

from audit_noise_corrected_a4_router import choose_control_tokens
from noise_corrected_action_panel import (
    lossless_selector_frontier,
    select_initial_e8_error_boundary_queries,
)
from noise_corrected_action_routing import RoutingThresholds, route_action
from noise_corrected_action_routing_v3 import score_candidate_boundary
from noise_final_core import CandidateGraph, sha256_file, stable_fold, strict_rank
from noise_v3_core import ROLE_NAMES, attenuate_and_renormalize, stable_seed
from train_e1_identity import load_base_model, torch_load_compat
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, forward_embeddings


REGISTERED_FORMAL_A4_ROUTE_CONFIGURATION: dict[str, object] = {
    "formula_fold_seed": 20260825,
    "query_scope": "initial_error_boundary",
    "max_queries": 0,
    "sample_seed": 20260906,
    "maximum_gradient_rank": 50,
    "control_repeats": 2,
    "paired_advantage_threshold": 0.01,
    "harm_margin_threshold": 0.01,
    "robustness_slack": 0.005,
    "boundary_correct_multiplier": 2.0,
    "minimum_boundary_correct": 2048,
    "maximum_boundary_correct": 4096,
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
    for name, expected in REGISTERED_FORMAL_A4_ROUTE_CONFIGURATION.items():
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
            "formal A4 route configuration drifted: "
            + json.dumps(mismatches, sort_keys=True)
        )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--a4-scan-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-student-checkpoint", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument(
        "--query-scope",
        choices=("official_errors", "official_correct", "all", "initial_error_boundary"),
        default="all",
    )
    parser.add_argument("--max-queries", type=int, default=64)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=20260906)
    parser.add_argument("--maximum-gradient-rank", type=int, default=50)
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--paired-advantage-threshold", type=float, default=0.01)
    parser.add_argument("--harm-margin-threshold", type=float, default=0.01)
    parser.add_argument("--robustness-slack", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-chunk-size", type=int, default=32)
    parser.add_argument("--boundary-correct-multiplier", type=float, default=2.0)
    parser.add_argument("--minimum-boundary-correct", type=int, default=2048)
    parser.add_argument("--maximum-boundary-correct", type=int, default=4096)
    parser.add_argument("--maximum-corrective-frontier", type=int, default=16)
    parser.add_argument("--maximum-harmful-frontier", type=int, default=8)
    parser.add_argument("--maximum-robust-frontier", type=int, default=8)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def gradient_rank_bin(rank: int) -> str:
    for right, label in ((1, "1"), (3, "2-3"), (6, "4-6"), (12, "7-12"), (25, "13-25"), (50, "26-50")):
        if rank <= right:
            return label
    return ">50"


def action_identifier(query_row: int, token: int, dose: float) -> str:
    payload = f"corrected-v1|A4-full|{query_row}|{token}|{dose:.2f}".encode()
    return "NC-A4-" + hashlib.sha256(payload).hexdigest()[:24]


def remap_scan_queries(
    graph: CandidateGraph,
    scan: pd.DataFrame,
    folds: np.ndarray,
    *,
    outer_fold: int,
    query_scope: str,
    maximum_queries: int,
    sample_seed: int,
) -> pd.DataFrame:
    required = {"scan_position", "query_row", "query_ik14", "query_formula"}
    if missing := required - set(scan.columns):
        raise KeyError(f"A4 scan query table misses {sorted(missing)}")
    by_row = {int(row): query for query, row in enumerate(graph.query_row)}
    mapped = scan.query_row.astype(np.int64).map(by_row)
    frame = scan.loc[mapped.notna()].copy()
    frame["query_index_corrected"] = mapped[mapped.notna()].astype(np.int64).to_numpy()
    query = frame.query_index_corrected.to_numpy(np.int64)
    if (
        not np.array_equal(frame.query_row.to_numpy(np.int64), graph.query_row[query])
        or not np.array_equal(frame.query_ik14.astype(str).to_numpy(), graph.query_ik14[query])
        or not np.array_equal(frame.query_formula.astype(str).to_numpy(), graph.query_formula[query])
    ):
        raise RuntimeError("A4 scan row/identity/formula remap failed")
    frame["formula_fold"] = folds[query]
    frame = frame.loc[frame.formula_fold.ne(outer_fold)].copy()
    ranks = np.asarray([
        strict_rank(graph.official_molecule_scores(int(value)))
        for value in frame.query_index_corrected
    ], dtype=np.int16)
    if query_scope == "official_errors":
        frame = frame.loc[ranks > 1].copy()
    elif query_scope == "official_correct":
        frame = frame.loc[ranks == 1].copy()
    if maximum_queries < 0:
        raise ValueError("max-queries cannot be negative")
    if maximum_queries and len(frame) > maximum_queries:
        keep = np.sort(np.random.default_rng(sample_seed).choice(
            len(frame), maximum_queries, replace=False,
        ))
        frame = frame.iloc[keep].copy()
    return frame.sort_values("query_index_corrected", kind="stable").reset_index(drop=True)


def _panel_initial_geometry(
    graph: CandidateGraph,
    panel: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_index: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.empty(len(panel), dtype=np.int16)
    margins = np.empty(len(panel), dtype=np.float32)
    for position, row in enumerate(panel.itertuples(index=False)):
        query = int(row.query_index_corrected)
        _, candidate_rows, ptr, _ = graph.query_block(query)
        candidate = embeddings[[embedding_index[int(value)] for value in candidate_rows]]
        qvector = embeddings[embedding_index[int(row.query_row)]]
        molecule = np.maximum.reduceat(candidate @ qvector, ptr[:-1])
        ranks[position] = strict_rank(molecule)
        margins[position] = float(molecule[0] - np.max(molecule[1:]))
    return ranks, margins


def main() -> None:
    args = arguments()
    _validate_registered_formal_configuration(args)
    started = time.time()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.outer_fold not in range(5) or not 1 <= args.maximum_gradient_rank <= 50:
        raise ValueError("outer fold/maximum A4 gradient rank is invalid")
    if args.formal and (args.max_queries != 0 or args.maximum_gradient_rank != 50):
        raise ValueError("formal A4 routing must consume all Top-50 actions in scope")
    if (
        args.query_chunk_size < 1
        or min(
            args.maximum_corrective_frontier,
            args.maximum_harmful_frontier,
            args.maximum_robust_frontier,
        ) < 1
    ):
        raise ValueError("A4 routing chunk/frontier limits must be positive")
    graph_path = args.graph_dir / "candidate_graph.npz"
    graph_report_path = args.graph_dir / "report.json"
    scan_path = args.a4_scan_dir / "scan_queries.csv.gz"
    h5_path = args.a4_scan_dir / "exact_peak_scan.h5"
    a4_decision_path = args.a4_scan_dir / "decision.json"
    initial_decision_path = args.initial_student_checkpoint.parent / "decision.json"
    required = (
        graph_path, graph_report_path, scan_path, h5_path, a4_decision_path,
        args.data, args.official_checkpoint, args.architecture_checkpoint,
        args.initial_student_checkpoint, initial_decision_path,
    )
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    a4_decision = json.loads(a4_decision_path.read_text(encoding="utf-8"))
    if graph_report.get("formal_training_authorized") is not True:
        raise RuntimeError("corrected graph is not training-authorized")
    if (
        a4_decision.get("status") != "noise_v3_a4_exact_peak_scan_decision"
        or a4_decision.get("integrity", {}).get("formal") is not True
        or int(a4_decision.get("integrity", {}).get("exact_variants", -1)) != 825152
    ):
        raise RuntimeError("immutable A4 proposal artifact failed provenance")
    graph = CandidateGraph(graph_path)
    folds = np.asarray([
        stable_fold(str(value), 5, args.formula_fold_seed) for value in graph.query_formula
    ], dtype=np.int8)
    panel = remap_scan_queries(
        graph, pd.read_csv(scan_path, low_memory=False), folds,
        outer_fold=args.outer_fold, query_scope=args.query_scope,
        maximum_queries=args.max_queries, sample_seed=args.sample_seed,
    )
    if not len(panel):
        raise RuntimeError("corrected A4 full-action panel is empty")

    needed: set[int] = set(map(int, panel.query_row))
    for query in panel.query_index_corrected:
        _, rows, _, _ = graph.query_block(int(query))
        needed.update(map(int, rows))
    reachable = np.asarray(sorted(needed), dtype=np.int64)
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, _ = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device,
        args.n_highest_peaks,
    )
    package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
    initial_decision = json.loads(initial_decision_path.read_text(encoding="utf-8"))
    if (
        package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
        or package.get("inference_clean_only") is not True
        or package.get("P2b_used") is not False
        or int(package.get("outer_fold", -1)) != args.outer_fold
        or initial_decision.get("formal") is not True
    ):
        raise RuntimeError("mature E8 initialization contract failed")
    model.load_state_dict(package["model_state"], strict=True)
    model.eval()
    embeddings = encode_rows(
        model, store, reachable, device, args.batch_size, args.amp,
        "corrected-A4-full-initial",
    )
    embedding_index = {int(row): index for index, row in enumerate(reachable)}
    panel_report: dict[str, object] = {
        "selection": args.query_scope,
        "A4_eligible_outer_train_queries": int(len(panel)),
        "selected_action_queries": int(len(panel)),
    }
    if args.query_scope == "initial_error_boundary":
        initial_rank, initial_margin = _panel_initial_geometry(
            graph, panel, embeddings, embedding_index,
        )
        selected_queries, panel_report = select_initial_e8_error_boundary_queries(
            panel.query_index_corrected.to_numpy(np.int64),
            initial_rank,
            initial_margin,
            panel.query_ik14.astype(str).to_numpy(),
            boundary_multiplier=args.boundary_correct_multiplier,
            minimum_boundary_correct=args.minimum_boundary_correct,
            maximum_boundary_correct=args.maximum_boundary_correct,
        )
        panel = panel.loc[
            panel.query_index_corrected.isin(set(map(int, selected_queries)))
        ].copy().reset_index(drop=True)
        panel_report["selection"] = (
            "all_initial_E8_errors_plus_identity_diverse_boundary_correct_with_A4_proposals"
        )
    if not len(panel):
        raise RuntimeError("corrected A4 selected action panel is empty")

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
    total_actions = 0
    maximum_resident_spectra = 0
    maximum_resident_embedding_bytes = 0
    with h5py.File(h5_path, "r") as handle:
        doses = np.asarray(json.loads(handle.attrs["attenuations_json"]), dtype=np.float32)
        if not np.array_equal(doses, np.asarray([.25, .50, .75, 1.0], dtype=np.float32)):
            raise RuntimeError("A4 attenuation registry drifted")
        action_query = handle["action_query"][:]
        action_token = handle["action_token"][:]
        action_role = handle["action_role"][:]
        action_mz = handle["action_mz"][:]
        action_intensity = handle["action_intensity"][:]
        action_rank = handle["action_gradient_rank"][:]
        policy_eligible = handle["action_policy_eligible"][:]
        ptr = handle["query_action_ptr"][:]
        # No result_* dataset is opened: historical outcomes cannot route the
        # current E8 experiment.
        for chunk_left in range(0, len(panel), args.query_chunk_size):
            panel_chunk = panel.iloc[chunk_left:chunk_left + args.query_chunk_size]
            spectra: list[torch.Tensor] = []
            layouts: list[dict[str, object]] = []
            for panel_row in panel_chunk.itertuples(index=False):
                scan_position = int(panel_row.scan_position)
                query = int(panel_row.query_index_corrected)
                clean = store.one(int(panel_row.query_row))
                left, right = int(ptr[scan_position]), int(ptr[scan_position + 1])
                roles = np.full(int(clean.shape[0]), -1, dtype=np.int8)
                query_tokens = action_token[left:right].astype(np.int64)
                if np.any((query_tokens <= 0) | (query_tokens >= len(roles))):
                    raise RuntimeError("A4 proposal token is outside the preprocessed spectrum")
                roles[query_tokens] = action_role[left:right]
                chosen_actions = np.arange(left, right, dtype=np.int64)
                chosen_actions = chosen_actions[
                    policy_eligible[chosen_actions]
                    & (action_rank[chosen_actions] >= 1)
                    & (action_rank[chosen_actions] <= args.maximum_gradient_rank)
                ]
                for action_index in chosen_actions:
                    if int(action_query[action_index]) != scan_position:
                        raise RuntimeError("A4 action/query pointer drifted")
                    token = int(action_token[action_index])
                    role = int(action_role[action_index])
                    values = clean.detach().cpu().numpy()
                    if (
                        not np.isclose(values[token, 0], action_mz[action_index], atol=1e-4)
                        or not np.isclose(values[token, 1], action_intensity[action_index], atol=1e-5)
                    ):
                        raise RuntimeError("A4 proposal token differs from raw clean spectrum")
                    controls, control_kind = choose_control_tokens(
                        clean, token, roles, args.control_repeats,
                        stable_seed(args.sample_seed, int(panel_row.query_row), token),
                    )
                    for dose in doses:
                        target_index = len(spectra)
                        spectra.append(attenuate_and_renormalize(clean, token, float(dose)))
                        control_indices = []
                        for control in controls:
                            control_indices.append(len(spectra))
                            spectra.append(attenuate_and_renormalize(
                                clean, int(control), float(dose),
                            ))
                        layouts.append({
                            "query": query, "query_row": int(panel_row.query_row),
                            "query_ik14": str(panel_row.query_ik14),
                            "query_formula": str(panel_row.query_formula),
                            "formula_fold": int(panel_row.formula_fold),
                            "token": token, "role": role,
                            "gradient_rank": int(action_rank[action_index]),
                            "dose": float(dose), "target_index": target_index,
                            "control_indices": control_indices,
                            "control_tokens": controls, "control_kind": control_kind,
                        })
            if not layouts:
                continue
            maximum_resident_spectra = max(maximum_resident_spectra, len(spectra))
            encoded = np.empty((len(spectra), embeddings.shape[1]), dtype=np.float32)
            maximum_resident_embedding_bytes = max(
                maximum_resident_embedding_bytes, int(encoded.nbytes),
            )
            with torch.inference_mode():
                for left in range(0, len(spectra), args.batch_size):
                    right = min(left + args.batch_size, len(spectra))
                    encoded[left:right] = forward_embeddings(
                        model, torch.stack(spectra[left:right]).to(device), args.amp,
                    ).float().cpu().numpy()

            records: list[dict[str, object]] = []
            target_spectra: list[np.ndarray] = []
            selected_control_spectra: list[np.ndarray] = []
            for layout in layouts:
                query = int(layout["query"])
                _, rows, candidate_ptr, _ = graph.query_block(query)
                candidate = embeddings[[embedding_index[int(row)] for row in rows]]

                def score(vector: np.ndarray):
                    return score_candidate_boundary(
                        rows, candidate_ptr, candidate, vector,
                    )

                clean_score = score(
                    embeddings[embedding_index[int(layout["query_row"])]],
                )
                clean_rank, clean_margin = clean_score.rank, clean_score.margin
                target_index = int(layout["target_index"])
                action_score = score(encoded[target_index])
                action_rank_value, action_margin = action_score.rank, action_score.margin
                indices = list(map(int, layout["control_indices"]))
                if indices:
                    control_values = [score(encoded[index]) for index in indices]
                    chosen_control = int(np.argmax([
                        value.margin for value in control_values
                    ]))
                    control_score = control_values[chosen_control]
                    control_rank = control_score.rank
                    control_margin = control_score.margin
                    control_tensor = spectra[indices[chosen_control]]
                    control_token = int(layout["control_tokens"][chosen_control])
                else:
                    control_rank, control_margin = clean_rank, clean_margin
                    control_score = clean_score
                    control_tensor = store.one(int(layout["query_row"]))
                    control_token = -1
                route = route_action(
                    clean_rank=clean_rank, clean_margin=clean_margin,
                    action_rank=action_rank_value, action_margin=action_margin,
                    control_margin=control_margin, thresholds=thresholds,
                )
                identifier = action_identifier(
                    int(layout["query_row"]), int(layout["token"]), float(layout["dose"]),
                )
                tensor_index = len(target_spectra)
                target_spectra.append(
                    spectra[target_index].detach().cpu().numpy().astype(np.float32),
                )
                selected_control_spectra.append(
                    control_tensor.detach().cpu().numpy().astype(np.float32),
                )
                role = str(ROLE_NAMES[int(layout["role"])])
                records.append({
                    "action_id": identifier, "query_index": query,
                    "query_row": int(layout["query_row"]),
                    "query_ik14": str(layout["query_ik14"]),
                    "query_formula": str(layout["query_formula"]),
                    "formula_fold": int(layout["formula_fold"]),
                    "near": bool(graph.query_has_near[query]),
                    "source": "A4_exact",
                    "family": f"exact_peak_{role}|rank={gradient_rank_bin(int(layout['gradient_rank']))}",
                    "recipe_id": f"token={int(layout['token'])}|dose={float(layout['dose']):.2f}",
                    "token": int(layout["token"]), "role": int(layout["role"]),
                    "gradient_rank": int(layout["gradient_rank"]),
                    "attenuation": float(layout["dose"]),
                    "control_kind": str(layout["control_kind"]),
                    "control_semantic": (
                        "clean_fallback"
                        if str(layout["control_kind"]) == "clean_fallback"
                        else "matched_neutral"
                    ),
                    "control_token": control_token,
                    "clean_rank": clean_rank, "clean_margin": clean_margin,
                    "action_rank": action_rank_value, "action_margin": action_margin,
                    "control_rank": control_rank, "control_margin": control_margin,
                    "action_positive_row": action_score.positive_row,
                    "action_hard_negative_molecule_index": (
                        action_score.hard_negative_molecule_index
                    ),
                    "action_hard_negative_row": action_score.hard_negative_row,
                    "control_positive_row": control_score.positive_row,
                    "control_hard_negative_molecule_index": (
                        control_score.hard_negative_molecule_index
                    ),
                    "control_hard_negative_row": control_score.hard_negative_row,
                    "margin_change": action_margin - clean_margin,
                    "paired_advantage": action_margin - control_margin,
                    "route": route, "corrective_weight": float(route == "corrective"),
                    "action_tensor_index": tensor_index,
                })
            chunk_result = pd.DataFrame(records)
            route_counts.update(map(str, chunk_result.route))
            total_actions += len(chunk_result)
            per_query_frames.append(chunk_result.groupby("query_index", sort=True).agg(
                clean_rank=("clean_rank", "first"),
                corrective_actions=("corrective_weight", "sum"),
                harmful_actions=("route", lambda value: int(np.sum(value == "harmful"))),
                best_action_rank=("action_rank", "min"),
                best_margin_change=("margin_change", "max"),
            ).reset_index())
            frontier = lossless_selector_frontier(
                chunk_result,
                maximum_corrective_per_query=args.maximum_corrective_frontier,
                maximum_harmful_per_query=args.maximum_harmful_frontier,
                maximum_robust_per_query=args.maximum_robust_frontier,
            )
            take = frontier.action_tensor_index.to_numpy(np.int64)
            if len(frontier):
                target_array = np.stack(target_spectra).astype(np.float32)
                control_array = np.stack(selected_control_spectra).astype(np.float32)
                frontier_action_chunks.append(target_array[take])
                frontier_control_chunks.append(control_array[take])
                frontier_frames.append(frontier)
            processed = min(chunk_left + args.query_chunk_size, len(panel))
            print(
                f"[corrected A4 chunk] {processed:,}/{len(panel):,}; "
                f"resident_spectra={len(spectra):,}; frontier={len(frontier):,}",
                flush=True,
            )
    if not frontier_frames:
        raise RuntimeError("A4 routing produced no corrective/harmful/robust selector frontier")
    result = pd.concat(frontier_frames, ignore_index=True, sort=False)
    action_spectra = np.concatenate(frontier_action_chunks, axis=0)
    control_spectra = np.concatenate(frontier_control_chunks, axis=0)
    if len(result) != len(action_spectra) or action_spectra.shape != control_spectra.shape:
        raise RuntimeError("A4 selector frontier tensor alignment failed")
    result["action_tensor_index"] = np.arange(len(result), dtype=np.int64)
    per_query = pd.concat(per_query_frames, ignore_index=True, sort=False)
    report = {
        "status": "noise_corrected_a4_router_audit_complete",
        "formal_training_authorized": bool(args.formal),
        "router_variant": "all_outcome_free_gradient_topk_actions",
        "outer_formula_fold": args.outer_fold, "query_scope": args.query_scope,
        "configuration": {
            name: getattr(args, name)
            for name in REGISTERED_FORMAL_A4_ROUTE_CONFIGURATION
        },
        "queries": int(len(panel)), "actions": int(total_actions),
        "actions_evaluated": int(total_actions),
        "selector_frontier_actions_retained": int(len(result)),
        "maximum_gradient_rank": args.maximum_gradient_rank,
        "route_counts": {str(key): int(value) for key, value in route_counts.items()},
        "queries_with_corrective_action": int(per_query.corrective_actions.gt(0).sum()),
        "error_queries": int(per_query.clean_rank.gt(1).sum()),
        "error_queries_with_rank1_action": int(
            (per_query.clean_rank.gt(1) & per_query.best_action_rank.eq(1)).sum()
        ),
        "action_panel": panel_report,
        "bounded_memory": {
            "query_chunk_size": int(args.query_chunk_size),
            "maximum_resident_spectra": int(maximum_resident_spectra),
            "maximum_resident_action_embedding_bytes": int(maximum_resident_embedding_bytes),
            "monolithic_all_action_embedding_allocation": False,
        },
        "selector_frontier": {
            "maximum_corrective_per_query": int(args.maximum_corrective_frontier),
            "maximum_harmful_per_query": int(args.maximum_harmful_frontier),
            "maximum_robust_per_query": int(args.maximum_robust_frontier),
        },
        "contracts": {
            "registered_formal_route_configuration_verified": bool(args.formal),
            "all_A4_topk_actions_preserved_before_routing": True,
            "one_action_teacher_selection_consumed": False,
            "historical_A4_outcomes_consumed": False,
            "old_query_index_reused": False,
            "stable_row_identity_formula_remap": True,
            "current_E8_geometry_rerouted": True,
            "current_E8_error_queries_with_A4_proposals_all_retained": bool(
                args.query_scope != "initial_error_boundary"
                or panel_report["initial_E8_error_queries"]
                == int(per_query.clean_rank.gt(1).sum())
            ),
            "selector_frontier_is_lossless_for_global_caps": True,
            "exact_action_control_candidate_switch_rows_recorded": True,
            "control_semantics_explicit": True,
            "nonfrontier_action_metadata_aggregated_not_materialized": True,
            "noncorrective_weight_exact_zero": bool(
                result.loc[~result.route.eq("corrective"), "corrective_weight"].eq(0).all()
            ),
            "outer_held_formula_consumed": False,
            "teacher_embedding_target_used": False, "P3_consumed": False,
        },
        "model_provenance": {
            "initial_student_checkpoint_sha256": sha256_file(args.initial_student_checkpoint),
            "initial_decision_sha256": sha256_file(initial_decision_path),
        },
        "provenance": {
            "candidate_graph_sha256": sha256_file(graph_path),
            "graph_report_sha256": sha256_file(graph_report_path),
            "A4_scan_sha256": sha256_file(h5_path),
            "A4_decision_sha256": sha256_file(a4_decision_path),
            "script_sha256": sha256_file(Path(__file__)),
            "action_routing_v3_sha256": sha256_file(
                Path(__file__).with_name("noise_corrected_action_routing_v3.py")
            ),
        },
        "runtime_seconds": time.time() - started,
        "claim_limit": "Outer-train A4 full-action routing audit; not encoder performance.",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        result.to_csv(staging / "routed_actions.csv.gz", index=False, compression="gzip")
        per_query.to_csv(staging / "per_query.csv.gz", index=False, compression="gzip")
        np.savez_compressed(
            staging / "action_spectra.npz",
            action_ids=np.asarray(result.action_id.astype(str), dtype=str),
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
