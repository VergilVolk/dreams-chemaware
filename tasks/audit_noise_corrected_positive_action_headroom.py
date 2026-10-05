"""Re-evaluate registered E10--E12 positive-evidence actions on the corrected graph.

This is an action-headroom audit, not a training manifest.  The action recipes
are copied from the historical E10/E11/E12 implementations; only the candidate
graph and query sampling are replaced.  Outcomes are written to a separate
artifact and must never be consumed as deployment-time routing labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from audit_noise_final_e10_positive_residual_matrix import cell_variant
from audit_noise_final_e11_reference_diversity_matrix import maxmin_indices
from audit_noise_final_positive_guided_matrix import reference_profile
from audit_noise_final_positive_peak_transfer import recurrent_missing_peaks
from noise_final_core import CandidateGraph, load_embedding_cache, sha256_file, strict_rank
from train_e1_identity import load_base_model
from train_noise_final_r2_shared_encoder import SpectrumStore, forward_embeddings


CORE_AND_EXPANDED = (
    (("consensus_projection", dose, 0.0) for dose in (0.25, 0.50, 0.75, 1.00)),
    (("prevalence_attenuation", dose, 0.0) for dose in (0.25, 0.50, 0.75, 1.00)),
    (("recurrent_union_mix", dose, 0.0) for dose in (0.10, 0.25, 0.50)),
    (("matched_intensity_transport", dose, 0.0) for dose in (0.25, 0.50, 0.75, 1.00)),
    (("recurrent_peak_graft", dose, 0.0) for dose in (0.10, 0.25, 0.50)),
    (("balanced_peak_exchange", dose, 0.0) for dose in (0.10, 0.25, 0.50)),
    iter((
        ("transport_then_union", 0.50, 0.50),
        ("transport_then_union", 1.00, 0.50),
        ("consensus_then_union", 0.50, 0.50),
        ("consensus_then_union", 0.75, 0.50),
    )),
)
CORE_AND_EXPANDED = tuple(item for group in CORE_AND_EXPANDED for item in group)
E11_RECIPES = (
    ("recurrent_union_mix", 0.50, 0.0),
    ("balanced_peak_exchange", 0.50, 0.0),
    ("consensus_then_union", 0.75, 0.50),
    ("transport_then_union", 1.00, 0.50),
)
POLICIES = ("top3", "farthest3", "maxmin6")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--query-scope", choices=("official_errors", "official_correct", "all"), default="official_errors")
    parser.add_argument("--recipe-set", choices=("full", "mature_shortlist", "fixed_p1"), default="full")
    parser.add_argument("--max-queries", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=20260906)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def select_reference_rows(
    rows: np.ndarray, scores: np.ndarray, vectors: np.ndarray, policy: str,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if len(rows) == 0:
        raise RuntimeError("positive molecule has no reference spectra")
    if policy == "top3":
        chosen = np.argsort(-scores, kind="stable")[:3]
    elif policy == "farthest3":
        chosen = np.argsort(scores, kind="stable")[:3]
    elif policy == "maxmin6":
        chosen = maxmin_indices(vectors, scores, 6)
    else:
        raise ValueError(f"unregistered reference policy: {policy}")
    return rows[np.asarray(chosen, dtype=np.int64)]


def recipe_id(
    policy: str, family: str, dose: float, auxiliary: float,
    prevalence: float, maximum: int, weighted: bool,
) -> str:
    return (
        f"{policy}|{family}|dose={dose:.2f}|aux={auxiliary:.2f}|"
        f"prevalence={prevalence:.2f}|max={maximum}|weighted={int(weighted)}"
    )


def registered_recipes(recipe_set: str = "full") -> list[tuple[str, str, float, float, float, int, bool]]:
    output: list[tuple[str, str, float, float, float, int, bool]] = []
    # E10 and E10-B used top-3 references, 0.67 recurrence, and at most 5 peaks.
    for family, dose, auxiliary in CORE_AND_EXPANDED:
        output.append(("top3", family, dose, auxiliary, 0.67, 5, False))
    # E11 changed only reference selection for four frozen mature recipes.
    for policy in ("farthest3", "maxmin6"):
        for family, dose, auxiliary in E11_RECIPES:
            output.append((policy, family, dose, auxiliary, 0.67, 5, False))
    # E12-B relaxed recurrence to 0.50 prevalence and tested max 5/10, dose .25/.50.
    for policy in POLICIES:
        for maximum in (5, 10):
            for dose in (0.25, 0.50):
                output.append((policy, "recurrent_union_mix", dose, 0.0, 0.50, maximum, False))
        output.append((policy, "recurrent_union_mix", 0.50, 0.0, 0.50, 10, True))
    ids = [recipe_id(*item) for item in output]
    if len(ids) != len(set(ids)):
        raise RuntimeError("registered positive-action recipe IDs are not unique")
    if recipe_set in {"mature_shortlist", "fixed_p1"}:
        keep = {
            # Historical E10 fixed mature action.
            ("top3", "recurrent_union_mix", 0.50, 0.0, 0.67, 5, False),
            # Historical E10-B combined positive-evidence action.
            ("top3", "transport_then_union", 1.00, 0.50, 0.67, 5, False),
            # Historical intensity endpoint retained as a risk-sensitive contrast.
            ("top3", "consensus_projection", 1.00, 0.0, 0.67, 5, False),
            # E11 reference diversity and E12 relaxed recurrence representatives.
            ("farthest3", "transport_then_union", 1.00, 0.50, 0.67, 5, False),
            ("maxmin6", "transport_then_union", 1.00, 0.50, 0.67, 5, False),
            ("top3", "recurrent_union_mix", 0.25, 0.0, 0.50, 10, False),
            ("top3", "recurrent_union_mix", 0.50, 0.0, 0.50, 10, False),
            ("farthest3", "recurrent_union_mix", 0.25, 0.0, 0.50, 10, False),
            ("farthest3", "recurrent_union_mix", 0.50, 0.0, 0.50, 10, False),
            ("maxmin6", "recurrent_union_mix", 0.50, 0.0, 0.50, 10, True),
        }
        if recipe_set == "fixed_p1":
            keep = {("top3", "transport_then_union", 1.00, 0.50, 0.67, 5, False)}
        output = [item for item in output if item in keep]
        if len(output) != len(keep):
            raise RuntimeError("mature positive-action shortlist drifted from the full registry")
    elif recipe_set != "full":
        raise ValueError(f"unknown recipe set: {recipe_set}")
    return output


def replay_official_ranks(
    graph: CandidateGraph, embeddings: np.ndarray, row_index: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frozen = np.empty(graph.n_queries, dtype=np.int16)
    replayed = np.empty(graph.n_queries, dtype=np.int16)
    margins = np.empty(graph.n_queries, dtype=np.float32)
    for query in range(graph.n_queries):
        frozen[query] = strict_rank(graph.official_molecule_scores(query))
        _, rows, ptr, _ = graph.query_block(query)
        query_vector = embeddings[row_index[int(graph.query_row[query])]]
        candidate = embeddings[[row_index[int(row)] for row in rows]]
        molecule_scores = np.maximum.reduceat(candidate @ query_vector, ptr[:-1])
        replayed[query] = strict_rank(molecule_scores)
        margins[query] = float(molecule_scores[0] - np.max(molecule_scores[1:]))
    return frozen, replayed, margins


def select_queries(
    graph: CandidateGraph, scope: str, maximum: int, seed: int,
    frozen_ranks: np.ndarray, replayed_ranks: np.ndarray,
) -> np.ndarray:
    if scope == "official_errors":
        eligible = np.flatnonzero((frozen_ranks != 1) & (replayed_ranks != 1))
    elif scope == "official_correct":
        eligible = np.flatnonzero((frozen_ranks == 1) & (replayed_ranks == 1))
    else:
        eligible = np.arange(graph.n_queries, dtype=np.int64)
    if maximum < 0:
        raise ValueError("max-queries must be non-negative")
    if maximum and maximum < len(eligible):
        eligible = np.sort(np.random.default_rng(seed).choice(eligible, maximum, replace=False))
    return np.asarray(eligible, dtype=np.int64)


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output_dir}")
    graph_path = args.graph_dir / "candidate_graph.npz"
    cache_path = args.graph_dir / "official_embeddings.npz"
    graph_report_path = args.graph_dir / "report.json"
    required = (graph_path, cache_path, graph_report_path, args.data, args.official_checkpoint, args.architecture_checkpoint)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    if graph_report.get("status") != "noise_corrected_candidate_graph_complete":
        raise RuntimeError("corrected graph report is not complete")

    started = time.time()
    graph = CandidateGraph(graph_path)
    cache_rows, cache_embeddings, cache_index = load_embedding_cache(cache_path)
    frozen_ranks, replayed_ranks, replayed_margins = replay_official_ranks(
        graph, cache_embeddings, cache_index,
    )
    queries = select_queries(
        graph, args.query_scope, args.max_queries, args.sample_seed,
        frozen_ranks, replayed_ranks,
    )
    if len(queries) == 0:
        raise RuntimeError("query selection is empty")

    positive_rows_by_query: dict[int, np.ndarray] = {}
    context_by_query: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int, float]] = {}
    needed_rows: set[int] = set(map(int, graph.query_row[queries]))
    for query in queries:
        _, rows, ptr, _ = graph.query_block(int(query))
        qvector = cache_embeddings[cache_index[int(graph.query_row[int(query)])]]
        vectors = cache_embeddings[[cache_index[int(row)] for row in rows]]
        pair_scores = vectors @ qvector
        molecule_scores = np.maximum.reduceat(pair_scores, ptr[:-1])
        baseline_rank = strict_rank(molecule_scores)
        baseline_margin = float(molecule_scores[0] - np.max(molecule_scores[1:]))
        if baseline_rank != int(replayed_ranks[int(query)]):
            raise RuntimeError("selected-query replay rank drifted within the audit")
        pos_rows = np.asarray(rows[:int(ptr[1])], dtype=np.int64)
        positive_rows_by_query[int(query)] = pos_rows
        needed_rows.update(map(int, pos_rows))
        context_by_query[int(query)] = (rows, ptr, vectors, baseline_rank, baseline_margin)

    store = SpectrumStore(args.data, np.asarray(sorted(needed_rows), dtype=np.int64), args.n_highest_peaks)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    model.eval()
    recipes = registered_recipes(args.recipe_set)

    action_tensors: list[torch.Tensor] = []
    action_meta: list[dict[str, object]] = []
    for local, query_value in enumerate(queries):
        query = int(query_value)
        clean = store.one(int(graph.query_row[query]))
        rows, ptr, vectors, baseline_rank, baseline_margin = context_by_query[query]
        qvector = cache_embeddings[cache_index[int(graph.query_row[query])]]
        pair_scores = vectors @ qvector
        positive_rows = positive_rows_by_query[query]
        positive_vectors = vectors[:int(ptr[1])]
        positive_scores = pair_scores[:int(ptr[1])]
        selections = {
            policy: select_reference_rows(positive_rows, positive_scores, positive_vectors, policy)
            for policy in POLICIES
        }
        profile_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        missing_cache: dict[tuple[str, float, int], np.ndarray] = {}
        for policy, family, dose, auxiliary, prevalence_floor, maximum, weighted in recipes:
            references = [store.one(int(row)) for row in selections[policy]]
            if policy not in profile_cache:
                profile_cache[policy] = reference_profile(clean, references, args.fragment_tolerance)
            missing_key = (policy, prevalence_floor, maximum)
            if missing_key not in missing_cache:
                missing_cache[missing_key] = recurrent_missing_peaks(
                    clean, references, args.fragment_tolerance, prevalence_floor, maximum,
                )
            prevalence, target = profile_cache[policy]
            missing = np.asarray(missing_cache[missing_key], dtype=np.float32).copy()
            if weighted and len(missing):
                missing[:, 1] *= missing[:, 2]
            variant = cell_variant(clean, (prevalence, target), missing, family, dose, auxiliary)
            action_tensors.append(variant)
            action_meta.append({
                "query_index": query,
                "query_row": int(graph.query_row[query]),
                "query_ik14": str(graph.query_ik14[query]),
                "query_formula": str(graph.query_formula[query]),
                "near": bool(graph.query_has_near[query]),
                "baseline_rank": int(baseline_rank),
                "frozen_graph_baseline_rank": int(frozen_ranks[query]),
                "baseline_margin": baseline_margin,
                "reference_policy": policy,
                "reference_rows": ",".join(map(str, selections[policy])),
                "family": family,
                "dose": dose,
                "auxiliary_dose": auxiliary,
                "prevalence_floor": prevalence_floor,
                "maximum_transferred_peaks": maximum,
                "support_weighted": weighted,
                "recipe_id": recipe_id(policy, family, dose, auxiliary, prevalence_floor, maximum, weighted),
                "available_missing_peaks": int(len(missing_cache[missing_key])),
            })
        if (local + 1) % 16 == 0 or local + 1 == len(queries):
            print(f"[positive-action-build] {local + 1:,}/{len(queries):,} queries", flush=True)

    encoded = np.empty((len(action_tensors), cache_embeddings.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for left in range(0, len(action_tensors), args.batch_size):
            right = min(left + args.batch_size, len(action_tensors))
            batch = torch.stack(action_tensors[left:right]).to(device)
            result = forward_embeddings(model, batch, args.amp).float().cpu().numpy()
            if not np.all(np.isfinite(result)):
                raise RuntimeError("positive action encoding produced non-finite values")
            encoded[left:right] = result
            if right == len(action_tensors) or right % (args.batch_size * 10) == 0:
                print(f"[positive-action-encode] {right:,}/{len(action_tensors):,}", flush=True)

    outcomes: list[dict[str, object]] = []
    for meta, vector in zip(action_meta, encoded):
        query = int(meta["query_index"])
        rows, ptr, candidate_vectors, _, _ = context_by_query[query]
        molecule_scores = np.maximum.reduceat(candidate_vectors @ vector, ptr[:-1])
        rank = strict_rank(molecule_scores)
        margin = float(molecule_scores[0] - np.max(molecule_scores[1:]))
        outcomes.append({
            **meta,
            "action_rank": int(rank),
            "action_margin": margin,
            "margin_change": margin - float(meta["baseline_margin"]),
            "corrected": bool(int(meta["baseline_rank"]) != 1 and rank == 1),
            "introduced": bool(int(meta["baseline_rank"]) == 1 and rank != 1),
        })
    frame = pd.DataFrame(outcomes)

    cells = []
    for recipe, group in frame.groupby("recipe_id", sort=True):
        corrected = int(group["corrected"].sum())
        introduced = int(group["introduced"].sum())
        cells.append({
            "recipe_id": recipe,
            "queries": int(group["query_index"].nunique()),
            "corrected": corrected,
            "introduced": introduced,
            "risk_net": corrected - 2 * introduced,
            "mean_margin_change": float(group["margin_change"].mean()),
            "positive_margin_fraction": float((group["margin_change"] > 0).mean()),
        })
    per_query = frame.groupby("query_index", sort=True).agg(
        baseline_rank=("baseline_rank", "first"),
        any_correcting_action=("corrected", "max"),
        any_introducing_action=("introduced", "max"),
        best_action_rank=("action_rank", "min"),
        best_margin_change=("margin_change", "max"),
    ).reset_index()
    report = {
        "status": "noise_corrected_positive_action_headroom_complete",
        "formal": bool(args.max_queries == 0 and args.query_scope == "all"),
        "query_scope": args.query_scope,
        "recipe_set": args.recipe_set,
        "queries": int(len(queries)),
        "recipes": int(len(recipes)),
        "actions": int(len(frame)),
        "baseline_errors": int((per_query["baseline_rank"] != 1).sum()),
        "baseline_correct": int((per_query["baseline_rank"] == 1).sum()),
        "queries_with_any_correcting_action": int(per_query["any_correcting_action"].sum()),
        "baseline_correct_with_any_introducing_action": int(
            per_query.loc[per_query["baseline_rank"] == 1, "any_introducing_action"].sum()
        ),
        "selected_query_noop_oracle_correction_pp": float(100.0 * (
            (per_query["baseline_rank"] != 1) & per_query["any_correcting_action"]
        ).mean()),
        "whole_graph_noop_oracle_recall1_delta_pp": (
            float(100.0 * per_query["any_correcting_action"].mean())
            if args.query_scope == "all" and len(queries) == graph.n_queries else None
        ),
        "cells": cells,
        "contracts": {
            "historical_e10_e11_e12_recipes_unchanged": True,
            "corrected_candidate_graph": True,
            "real_same_identity_positive_references": True,
            "action_outcomes_separate_from_training": True,
            "outcome_aware_oracle_not_deployable": True,
            "teacher_or_distillation_used": False,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "model_provenance": {
            "initialization": initialization,
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
        },
        "provenance": {
            "candidate_graph_sha256": sha256_file(graph_path),
            "embedding_cache_sha256": sha256_file(cache_path),
            "graph_report_sha256": sha256_file(graph_report_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "baseline_replay": {
            "whole_graph_frozen_errors": int(np.sum(frozen_ranks != 1)),
            "whole_graph_cache_replay_errors": int(np.sum(replayed_ranks != 1)),
            "whole_graph_rank_mismatches": int(np.sum(frozen_ranks != replayed_ranks)),
            "rank1_boundary_mismatches": int(np.sum((frozen_ranks == 1) != (replayed_ranks == 1))),
            "maximum_absolute_replayed_margin_among_rank1_boundary_mismatches": float(
                np.max(np.abs(replayed_margins[(frozen_ranks == 1) != (replayed_ranks == 1)]), initial=0.0)
            ),
            "scope_requires_stable_frozen_and_replayed_status": args.query_scope != "all",
        },
        "runtime_seconds": time.time() - started,
        "claim_limit": "Action headroom only; no clean-input encoder improvement or deployable routing is claimed.",
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        frame.to_csv(temporary / "per_action.csv.gz", index=False)
        per_query.to_csv(temporary / "per_query.csv.gz", index=False)
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
