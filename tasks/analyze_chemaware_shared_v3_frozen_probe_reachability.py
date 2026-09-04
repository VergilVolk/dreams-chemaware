"""Nested formula-held-out reachability audit for a frozen chemical probe.

This diagnostic asks whether official DreaMS spectrum embeddings contain a
linearly decodable Morgan connectivity geometry.  A ridge map is fitted from
identity-equal spectrum centroids to frozen molecule-teacher vectors using only
training formulas.  Ridge strength is selected on a disjoint inner formula
fold and evaluated once on the outer fold.  Identity-permuted teacher labels
are processed with the same nested protocol as a negative control.

The candidate fingerprints used for scoring make this a training-mechanism
diagnostic, not a deployable annotation result.  It does not change a model and
cannot support a performance claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_shared_v2_core import (  # noqa: E402
    ChemAwareTokenStore,
    MoleculeTeacherStore,
    formula_folds,
)
from noise_final_core import (  # noqa: E402
    CandidateGraph,
    json_dump,
    sha256_file,
    strict_metrics,
    strict_rank,
)
from summarize_chemaware_shared_v2_g1 import formula_bootstrap  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=1701)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0]
    )
    parser.add_argument("--permutation-seeds", type=int, nargs="+", default=[17, 41, 73])
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def identity_centroids(
    graph: CandidateGraph,
    store: ChemAwareTokenStore,
    teacher: MoleculeTeacherStore,
) -> np.ndarray:
    rows_by_identity: dict[str, set[int]] = {value: set() for value in teacher.ik14}
    for molecule_index, identity in enumerate(graph.molecule_ik14.astype(str)):
        left = int(graph.molecule_ptr[molecule_index])
        right = int(graph.molecule_ptr[molecule_index + 1])
        rows_by_identity[identity].update(
            map(int, graph.pair_candidate_row[left:right])
        )
    centroids = np.empty((len(teacher.ik14), store.dimension), dtype=np.float32)
    for index, identity in enumerate(teacher.ik14):
        rows = sorted(rows_by_identity[str(identity)])
        if not rows:
            raise RuntimeError(f"teacher identity lacks reference spectra: {identity}")
        positions = [store.position[row] for row in rows]
        value = np.mean(np.asarray(store.official_embeddings[positions]), axis=0)
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"invalid official centroid for identity {identity}")
        centroids[index] = value / norm
    return centroids


def fit_ridge(
    inputs: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    if alpha <= 0 or inputs.ndim != 2 or targets.ndim != 2 or len(inputs) != len(targets):
        raise ValueError("invalid ridge probe inputs")
    input_mean = inputs.mean(axis=0, keepdims=True)
    target_mean = targets.mean(axis=0, keepdims=True)
    centered_input = np.asarray(inputs - input_mean, dtype=np.float64)
    centered_target = np.asarray(targets - target_mean, dtype=np.float64)
    kernel = centered_input @ centered_input.T
    dual = np.linalg.solve(
        kernel + alpha * np.eye(len(kernel), dtype=np.float64), centered_target
    )
    weight = centered_input.T @ dual
    return {
        "input_mean": input_mean.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "weight": weight.astype(np.float32),
    }


def predict_probe(values: np.ndarray, probe: dict[str, np.ndarray]) -> np.ndarray:
    prediction = (
        (np.asarray(values, dtype=np.float32) - probe["input_mean"])
        @ probe["weight"] + probe["target_mean"]
    )
    norms = np.linalg.norm(prediction, axis=1, keepdims=True)
    return prediction / np.clip(norms, 1e-12, None)


def query_spectrum_embeddings(
    graph: CandidateGraph,
    store: ChemAwareTokenStore,
    queries: np.ndarray,
) -> np.ndarray:
    return np.asarray([
        store.official_embeddings[store.position[int(graph.query_row[query])]]
        for query in np.asarray(queries, dtype=np.int64)
    ], dtype=np.float32)


def probe_ranks(
    predicted: np.ndarray,
    queries: np.ndarray,
    graph: CandidateGraph,
    teacher_graph_embeddings: np.ndarray,
) -> np.ndarray:
    ranks = []
    for value, query in zip(predicted, queries):
        left, right = map(int, graph.query_ptr[int(query):int(query) + 2])
        scores = teacher_graph_embeddings[left:right] @ value
        ranks.append(strict_rank(scores))
    return np.asarray(ranks, dtype=np.int32)


def evaluate(
    probe: dict[str, np.ndarray],
    queries: np.ndarray,
    graph: CandidateGraph,
    store: ChemAwareTokenStore,
    teacher_graph_embeddings: np.ndarray,
) -> dict:
    predicted = predict_probe(query_spectrum_embeddings(graph, store, queries), probe)
    ranks = probe_ranks(predicted, queries, graph, teacher_graph_embeddings)
    near = graph.query_has_near[queries]
    same_formula_negative = np.asarray([
        np.any(
            graph.molecule_formula[
                int(graph.query_ptr[query]) + 1:int(graph.query_ptr[query + 1])
            ].astype(str) == str(graph.query_formula[query])
        )
        for query in queries
    ], dtype=bool)
    metrics = strict_metrics(ranks, near)
    return {
        "queries": int(len(queries)),
        "ranks": ranks,
        "recall1": metrics["recall1"],
        "mrr": metrics["mrr"],
        "near_queries": int(np.sum(near)),
        "near_recall1": metrics.get("near_recall1"),
        "same_formula_negative_queries": int(np.sum(same_formula_negative)),
        "same_formula_negative_recall1": (
            float(np.mean(ranks[same_formula_negative] == 1))
            if np.any(same_formula_negative) else None
        ),
    }


def nested_arm(
    labels: np.ndarray,
    centroids: np.ndarray,
    identity_fold: np.ndarray,
    query_fold: np.ndarray,
    graph: CandidateGraph,
    store: ChemAwareTokenStore,
    teacher_graph_embeddings: np.ndarray,
    folds: int,
    alphas: list[float],
    permutation_seed: int | None = None,
) -> dict:
    fold_reports = []
    query_parts, rank_parts = [], []
    for outer in range(folds):
        inner = (outer + 1) % folds
        train_identity = np.flatnonzero(
            (identity_fold != outer) & (identity_fold != inner)
        )
        refit_identity = np.flatnonzero(identity_fold != outer)
        inner_query = np.flatnonzero(query_fold == inner)
        outer_query = np.flatnonzero(query_fold == outer)
        if not all(map(len, (train_identity, refit_identity, inner_query, outer_query))):
            raise RuntimeError(f"empty nested probe split for outer fold {outer}")
        train_targets = labels[train_identity]
        refit_targets = labels[refit_identity]
        if permutation_seed is not None:
            # Permute strictly within the currently allowed identity set.  A
            # held-out formula's chemical vector must never enter a probe fit,
            # even in a pseudo-teacher control.
            train_rng = np.random.default_rng(permutation_seed + outer * 1009 + 17)
            refit_rng = np.random.default_rng(permutation_seed + outer * 1009 + 41)
            train_targets = train_targets[train_rng.permutation(len(train_targets))]
            refit_targets = refit_targets[refit_rng.permutation(len(refit_targets))]
        inner_scores = []
        for alpha in alphas:
            probe = fit_ridge(centroids[train_identity], train_targets, alpha)
            result = evaluate(
                probe, inner_query, graph, store, teacher_graph_embeddings
            )
            inner_scores.append((float(result["recall1"]), float(result["mrr"]), -alpha, alpha))
        selected_alpha = float(max(inner_scores)[-1])
        probe = fit_ridge(
            centroids[refit_identity], refit_targets, selected_alpha
        )
        outer_result = evaluate(
            probe, outer_query, graph, store, teacher_graph_embeddings
        )
        query_parts.append(outer_query)
        rank_parts.append(outer_result.pop("ranks"))
        fold_reports.append({
            "outer_fold": outer,
            "inner_fold": inner,
            "train_identities": int(len(train_identity)),
            "refit_identities": int(len(refit_identity)),
            "selected_alpha": selected_alpha,
            "outer": outer_result,
        })
    query = np.concatenate(query_parts)
    ranks = np.concatenate(rank_parts)
    order = np.argsort(query, kind="stable")
    query, ranks = query[order], ranks[order]
    if not np.array_equal(query, np.arange(graph.n_queries)):
        raise RuntimeError("nested probe does not cover every query exactly once")
    near = graph.query_has_near[query]
    metrics = strict_metrics(ranks, near)
    return {
        "folds": fold_reports,
        "queries": int(len(query)),
        "recall1": metrics["recall1"],
        "mrr": metrics["mrr"],
        "near_recall1": metrics.get("near_recall1"),
        "ranks": ranks,
    }


def paired_comparison(
    reference_rank: np.ndarray,
    candidate_rank: np.ndarray,
    graph: CandidateGraph,
    resamples: int,
    seed: int,
) -> dict:
    reference_correct = reference_rank == 1
    candidate_correct = candidate_rank == 1
    corrected = int(np.sum(~reference_correct & candidate_correct))
    introduced = int(np.sum(reference_correct & ~candidate_correct))
    near = graph.query_has_near
    return {
        **formula_bootstrap(
            reference_rank, candidate_rank, graph.query_formula, resamples, seed
        ),
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "near_delta_recall1": float(
            np.mean(candidate_correct[near]) - np.mean(reference_correct[near])
        ),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen-probe audit: {args.output}")
    if args.folds < 3 or any(alpha <= 0 for alpha in args.alphas):
        raise ValueError("probe needs at least three folds and positive ridge strengths")
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    graph = CandidateGraph(args.graph)
    store = ChemAwareTokenStore(
        args.token_dir, args.graph, args.official_checkpoint, require_formal=False
    )
    store.require_graph_coverage(graph)
    teacher = MoleculeTeacherStore(
        args.teacher_dir, args.graph, graph, require_formal=False
    )
    centroids = identity_centroids(graph, store, teacher)
    labels = np.asarray(teacher.embeddings, dtype=np.float32)
    teacher_graph_embeddings = labels[teacher.graph_index]
    identity_fold = formula_folds(teacher.formula, args.folds, args.fold_seed)
    query_fold = formula_folds(graph.query_formula, args.folds, args.fold_seed)
    correct = nested_arm(
        labels, centroids, identity_fold, query_fold, graph, store,
        teacher_graph_embeddings, args.folds, list(args.alphas),
    )
    control_reports = []
    for seed in args.permutation_seeds:
        control = nested_arm(
            labels, centroids, identity_fold, query_fold, graph, store,
            teacher_graph_embeddings, args.folds, list(args.alphas),
            permutation_seed=seed,
        )
        control_reports.append({
            "seed": seed,
            "recall1": control["recall1"],
            "mrr": control["mrr"],
            "near_recall1": control["near_recall1"],
            "folds": control["folds"],
            "correct_teacher_vs_control": paired_comparison(
                control["ranks"], correct["ranks"], graph,
                args.bootstrap_resamples, 20260902 + seed,
            ),
        })
    positive_identity = graph.molecule_ik14[graph.query_ptr[:-1]].astype(str)
    oracle_ranks = probe_ranks(
        labels[[teacher.index[str(value)] for value in positive_identity]],
        np.arange(graph.n_queries), graph, teacher_graph_embeddings,
    )
    oracle = strict_metrics(oracle_ranks, graph.query_has_near)
    control_recall = np.asarray([value["recall1"] for value in control_reports])
    control_mrr = np.asarray([value["mrr"] for value in control_reports])
    report = {
        "status": "chemaware_shared_v3_frozen_probe_reachability_complete",
        "formal": False,
        "diagnostic_only": True,
        "teacher_kind": teacher.report.get("teacher_kind", teacher.report.get("status")),
        "identities": int(len(teacher.ik14)),
        "formulas": int(len(np.unique(teacher.formula.astype(str)))),
        "queries": graph.n_queries,
        "folds": args.folds,
        "alphas": args.alphas,
        "correct_teacher": {
            **{key: value for key, value in correct.items() if key != "ranks"},
            "delta_recall1_vs_permuted_mean": float(
                correct["recall1"] - np.mean(control_recall)
            ),
            "delta_mrr_vs_permuted_mean": float(correct["mrr"] - np.mean(control_mrr)),
        },
        "identity_permuted_controls": control_reports,
        "teacher_oracle": oracle,
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "decision_hint": {
            "probe_signal_above_all_permuted_controls": bool(
                correct["recall1"] > np.max(control_recall)
                and correct["mrr"] > np.max(control_mrr)
            ),
            "formula_ci_positive_vs_all_permuted_controls": bool(all(
                value["correct_teacher_vs_control"]["delta_recall1_ci_low"] > 0
                for value in control_reports
            )),
            "risk_net_positive_vs_all_permuted_controls": bool(all(
                value["correct_teacher_vs_control"]["risk_net_lambda2"] > 0
                for value in control_reports
            )),
            "probe_recall1_at_least_0_10": bool(correct["recall1"] >= 0.10),
        },
        "claim_limit": (
            "Formula-held-out linear decodability diagnostic only. Candidate molecule "
            "fingerprints are used for scoring, so this is not deployable retrieval evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
