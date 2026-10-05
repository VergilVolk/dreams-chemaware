"""Full corrected-graph reachability for action-routed clean-boundary injection.

This is a frozen-official-embedding development proxy, not an encoder result.
It trains one bounded shared map per outer formula fold on every outer-train
clean query.  Mature N action availability changes per-query dose only; action
multiplicity and historical action outcomes are never consumed.
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
import torch.nn.functional as F

from audit_noise_direct_shared_metric_reachability import (
    SharedDiagonalMetric, SharedResidualMetric, transform_numpy,
)
from audit_noise_peak_gate_candidate_injection import formula_ci
from noise_corrected_fullgraph_evaluation import (
    GraphScores, expanded_indices, full_metrics, official_scores,
    paired_outcome_table, query_table,
)
from noise_final_core import CandidateGraph, load_embedding_cache, sha256_file, stable_fold


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--action-coverage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-list", default="0,1,2,3,4")
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--adapter-type", choices=("diagonal", "lowrank"), default="diagonal")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--residual-strength", type=float, default=0.15)
    parser.add_argument("--top-negatives", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--action-error-weight", type=float, default=6.0)
    parser.add_argument("--other-error-weight", type=float, default=2.0)
    parser.add_argument("--correct-weight", type=float, default=0.5)
    parser.add_argument("--safety-weight", type=float, default=8.0)
    parser.add_argument("--safety-slack", type=float, default=0.005)
    parser.add_argument("--preserve-weight", type=float, default=0.25)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--score-chunk-edges", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def representative_positions(
    graph: CandidateGraph, row_index: dict[int, int], top_negatives: int,
) -> tuple[np.ndarray, np.ndarray]:
    if top_negatives < 1:
        raise ValueError("top-negatives must be positive")
    official_pair = graph.features[:, graph.dreams_column]
    official_molecule = np.maximum.reduceat(official_pair, graph.molecule_ptr[:-1])
    best_pair = np.empty(len(graph.molecule_label), dtype=np.int64)
    for molecule, (left, right) in enumerate(zip(graph.molecule_ptr[:-1], graph.molecule_ptr[1:])):
        left, right = int(left), int(right)
        best_pair[molecule] = left + int(np.argmax(official_pair[left:right]))
    representative_row = graph.pair_candidate_row[best_pair]
    try:
        representative = np.asarray([row_index[int(row)] for row in representative_row], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError(f"embedding cache misses representative row {error}") from error
    positive = representative[graph.query_ptr[:-1]]
    negative = np.empty((graph.n_queries, top_negatives), dtype=np.int64)
    for query, (left, right) in enumerate(zip(graph.query_ptr[:-1], graph.query_ptr[1:])):
        left, right = int(left), int(right)
        wrong = np.arange(left + 1, right, dtype=np.int64)
        order = wrong[np.argsort(-official_molecule[wrong], kind="stable")]
        chosen = representative[order[:top_negatives]]
        if len(chosen) < top_negatives:
            chosen = np.pad(chosen, (0, top_negatives - len(chosen)), mode="edge")
        negative[query] = chosen
    return positive, negative


def train_fold(
    fold: int,
    train: np.ndarray,
    embeddings: np.ndarray,
    query_position: np.ndarray,
    positive_position: np.ndarray,
    negative_position: np.ndarray,
    baseline_rank: np.ndarray,
    action_query: np.ndarray,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict[str, float]]:
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    device = torch.device(args.device)
    model: torch.nn.Module
    if args.adapter_type == "diagonal":
        model = SharedDiagonalMetric(embeddings.shape[1], args.residual_strength).to(device)
    else:
        model = SharedResidualMetric(
            embeddings.shape[1], args.hidden_dim, args.residual_strength,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    is_error = baseline_rank > 1
    weight = np.where(
        action_query & is_error, args.action_error_weight,
        np.where(is_error, args.other_error_weight, args.correct_weight),
    ).astype(np.float32)
    rng = np.random.default_rng(args.seed + fold)
    last = np.zeros(6, dtype=np.float64)
    for epoch in range(args.epochs):
        order = rng.permutation(train)
        total = np.zeros(6, dtype=np.float64)
        model.train()
        for left in range(0, len(order), args.batch_size):
            index = order[left:left + args.batch_size]
            q0 = torch.as_tensor(
                np.asarray(embeddings[query_position[index]], dtype=np.float32), device=device,
            )
            p0 = torch.as_tensor(
                np.asarray(embeddings[positive_position[index]], dtype=np.float32), device=device,
            )
            n0 = torch.as_tensor(
                np.asarray(embeddings[negative_position[index]], dtype=np.float32), device=device,
            )
            q = model(q0)
            p = model(p0)
            n = model(n0.reshape(-1, n0.shape[-1])).reshape_as(n0)
            positive = torch.sum(q * p, dim=1)
            negative = torch.einsum("bd,bkd->bk", q, n)
            margin = positive - negative.max(dim=1).values
            per_query = F.softplus((args.rank_margin - margin) / args.temperature)
            local_weight = torch.as_tensor(weight[index], device=device)
            rank_loss = torch.sum(local_weight * per_query) / local_weight.sum().clamp_min(1e-8)
            initial_positive = torch.sum(q0 * p0, dim=1)
            initial_negative = torch.einsum("bd,bkd->bk", q0, n0)
            initial_margin = initial_positive - initial_negative.max(dim=1).values
            safe = torch.as_tensor(~is_error[index], device=device)
            safety = (
                F.relu(initial_margin[safe].detach() - args.safety_slack - margin[safe]).mean()
                if bool(safe.any()) else rank_loss * 0.0
            )
            current = torch.cat((q, p, n.reshape(-1, n.shape[-1])), dim=0)
            initial = torch.cat((q0, p0, n0.reshape(-1, n0.shape[-1])), dim=0)
            preserve = (1.0 - torch.sum(current * initial, dim=1)).mean()
            loss = rank_loss + args.safety_weight * safety + args.preserve_weight * preserve
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(index)
            total += np.asarray([
                float(loss.detach()), float(rank_loss.detach()), float(safety.detach()),
                float(preserve.detach()), float(grad), count,
            ]) * count
        last = total[:5] / total[5]
        print(
            f"[corrected fullgraph fold={fold} epoch={epoch + 1}] "
            f"loss={last[0]:.5f} rank={last[1]:.5f} safety={last[2]:.5f} "
            f"preserve={last[3]:.6f} grad={last[4]:.4f}", flush=True,
        )
    return model, {
        "final_loss": float(last[0]), "final_rank_loss": float(last[1]),
        "final_safety_loss": float(last[2]), "final_preserve_loss": float(last[3]),
        "final_gradient_norm": float(last[4]),
    }


def score_held_edges(
    graph: CandidateGraph,
    transformed: np.ndarray,
    row_index: dict[int, int],
    pair_query: np.ndarray,
    held: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(pair_query, held, assume_unique=False)
    edge = np.flatnonzero(mask)
    query_position = np.asarray([row_index[int(row)] for row in graph.query_row], dtype=np.int64)
    candidate_position = np.asarray(
        [row_index[int(row)] for row in graph.pair_candidate_row[edge]], dtype=np.int64,
    )
    output = np.empty(len(edge), dtype=np.float32)
    for left in range(0, len(edge), chunk):
        right = min(left + chunk, len(edge))
        q = transformed[query_position[pair_query[edge[left:right]]]]
        c = transformed[candidate_position[left:right]]
        output[left:right] = np.einsum("ij,ij->i", q, c, optimize=True)
    return edge, output


def nested_delta(candidate: dict[str, object], official: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in candidate.items():
        if key not in official:
            continue
        reference = official[key]
        if isinstance(value, dict) and isinstance(reference, dict):
            output[key] = nested_delta(value, reference)
        elif isinstance(value, (int, float)) and isinstance(reference, (int, float)):
            output[key] = float(value - reference)
    return output


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    if args.folds != 5:
        raise ValueError("corrected route freezes five formula folds")
    selected_folds = sorted({int(value) for value in args.fold_list.split(",") if value.strip()})
    if not selected_folds or any(value not in range(args.folds) for value in selected_folds):
        raise ValueError("fold-list must select a non-empty subset of 0..4")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    graph_path = args.graph_dir / "candidate_graph.npz"
    cache_path = args.graph_dir / "official_embeddings.npz"
    graph_report_path = args.graph_dir / "report.json"
    action_path = args.action_coverage_dir / "remapped_training_actions.csv.gz"
    action_report_path = args.action_coverage_dir / "report.json"
    source_manifest_path = args.source_manifest_dir / "manifest.npz"
    for path in (
        graph_path, cache_path, graph_report_path, action_path, action_report_path,
        source_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    action_report = json.loads(action_report_path.read_text(encoding="utf-8"))
    if (
        graph_report.get("formal_training_authorized") is not True
        or graph_report.get("provenance", {}).get("candidate_graph_sha256") != sha256_file(graph_path)
        or graph_report.get("provenance", {}).get("embedding_cache_sha256") != sha256_file(cache_path)
        or action_report.get("status") != "noise_corrected_action_coverage_complete"
        or action_report.get("contracts", {}).get("action_outcomes_consumed") is not False
    ):
        raise RuntimeError("graph or outcome-free action coverage provenance failed")
    graph = CandidateGraph(graph_path)
    rows, embeddings, row_index = load_embedding_cache(cache_path)
    with np.load(source_manifest_path, allow_pickle=False) as source:
        if not np.array_equal(source["query_row"], graph.query_row):
            raise RuntimeError("source manifest query order differs from graph")
        query_adduct = np.asarray(source["query_adduct"], dtype=str)
    baseline_score = official_scores(graph)
    baseline_table = query_table(graph, baseline_score)
    baseline_rank = baseline_table["rank"].to_numpy(np.int64)
    folds = np.asarray([
        stable_fold(str(formula), args.folds, args.formula_fold_seed)
        for formula in graph.query_formula
    ], dtype=np.int8)
    actions = pd.read_csv(action_path, usecols=["query_index", "query_row"], low_memory=False)
    actions = actions.drop_duplicates("query_index")
    if not np.array_equal(
        actions["query_row"].to_numpy(np.int64), graph.query_row[actions["query_index"].to_numpy(np.int64)],
    ):
        raise RuntimeError("remapped action rows no longer align to graph")
    action_query = np.zeros(graph.n_queries, dtype=bool)
    action_query[actions["query_index"].to_numpy(np.int64)] = True
    query_position = np.asarray([row_index[int(row)] for row in graph.query_row], dtype=np.int64)
    positive_position, negative_position = representative_positions(
        graph, row_index, args.top_negatives,
    )
    _, pair_query = expanded_indices(graph)
    oof_pair = np.full(len(graph.pair_candidate_row), np.nan, dtype=np.float32)
    logs: list[dict[str, object]] = []
    for fold in selected_folds:
        train = np.flatnonzero(folds != fold)
        held = np.flatnonzero(folds == fold)
        if set(graph.query_formula[train]) & set(graph.query_formula[held]):
            raise RuntimeError(f"formula leakage in fold {fold}")
        model, log = train_fold(
            fold, train, embeddings, query_position, positive_position, negative_position,
            baseline_rank, action_query, args,
        )
        transformed = transform_numpy(
            model, embeddings, torch.device(args.device), max(512, args.batch_size),
        )
        edge, values = score_held_edges(
            graph, transformed, row_index, pair_query, held, args.score_chunk_edges,
        )
        oof_pair[edge] = values
        held_query_position = query_position[held]
        logs.append({
            "fold": fold, "train_queries": int(len(train)), "held_queries": int(len(held)),
            "train_action_queries": int(np.sum(action_query[train])),
            "train_action_errors": int(np.sum(action_query[train] & (baseline_rank[train] > 1))),
            "held_embedding_cosine_mean": float(np.mean(np.einsum(
                "ij,ij->i", transformed[held_query_position], embeddings[held_query_position],
            ))),
            **log,
        })
        del transformed, model
    evaluated_queries = np.flatnonzero(np.isin(folds, selected_folds))
    evaluated_edges = np.isin(pair_query, evaluated_queries, assume_unique=False)
    if not np.all(np.isfinite(oof_pair[evaluated_edges])) or np.any(np.isfinite(oof_pair[~evaluated_edges])):
        raise RuntimeError("OOF edge coverage is incomplete or crosses selected folds")
    oof_molecule = np.maximum.reduceat(oof_pair, graph.molecule_ptr[:-1])
    candidate_score = GraphScores(pair=oof_pair, molecule=oof_molecule)
    official_metrics, official_query = full_metrics(
        graph, baseline_score, query_adduct=query_adduct, queries=evaluated_queries,
    )
    candidate_metrics, candidate_query = full_metrics(
        graph, candidate_score, query_adduct=query_adduct, queries=evaluated_queries,
    )
    paired = paired_outcome_table(official_query, candidate_query)
    recall_ci_frame = paired.rename(columns={
        "official_rank": "baseline_rank", "candidate_rank": "final_rank",
    })
    recall_ci = formula_ci(recall_ci_frame, args.bootstrap_resamples, args.seed)
    report = {
        "status": "noise_corrected_fullgraph_shared_map_complete",
        "formal": bool(selected_folds == list(range(args.folds))),
        "proxy_only": True,
        "selected_folds": selected_folds,
        "evaluated_queries": int(len(evaluated_queries)),
        "evaluated_formulas": int(len(np.unique(graph.query_formula[evaluated_queries]))),
        "official": official_metrics,
        "candidate": candidate_metrics,
        "candidate_minus_official": nested_delta(candidate_metrics, official_metrics),
        "paired_top1": {
            "corrected": int(paired.corrected.sum()),
            "introduced": int(paired.introduced.sum()),
            "risk_net": int(paired.risk_net.sum()),
            "formula_cluster_delta_recall1": recall_ci,
        },
        "folds": logs,
        "action_coverage": {
            "queries": int(action_query.sum()),
            "baseline_errors": int(np.sum(action_query & (baseline_rank > 1))),
            "all_baseline_errors": int(np.sum(baseline_rank > 1)),
        },
        "contracts": {
            "all_queries_in_selected_formula_folds_evaluated": True,
            "outer_formula_exclusion": True,
            "shared_query_and_candidate_map": True,
            "molecule_score_is_spectrum_max": True,
            "action_multiplicity_changes_dose": False,
            "historical_action_outcomes_consumed": False,
            "identity_is_only_ranking_label": True,
            "teacher_embedding_or_margin_target_used": False,
            "raw_encoder_checkpoint_produced": False,
            "massspecgym_pairwise_not_nist20_replication": True,
        },
        "configuration": vars(args),
        "runtime_seconds": time.time() - started,
        "provenance": {
            "candidate_graph_sha256": sha256_file(graph_path),
            "embedding_cache_sha256": sha256_file(cache_path),
            "action_coverage_report_sha256": sha256_file(action_report_path),
            "action_manifest_sha256": sha256_file(action_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Formula-OOF frozen-official-embedding shared-map reachability proxy; "
            "not a raw-spectrum encoder checkpoint or a server training result."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_fullgraph_proxy_", dir=args.output_dir.parent))
    try:
        paired.to_csv(staging / "paired_per_query.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
