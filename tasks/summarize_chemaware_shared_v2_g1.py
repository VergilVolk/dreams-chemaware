"""Aggregate the complete 5-fold x 3-seed ChemAware-v2 G1 control ledger."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, json_dump, sha256_file  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_g1")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/g8r_chemaware_shared_v2_g1_summary.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 41, 73])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    return parser.parse_args()


def formula_bootstrap(
    old_rank: np.ndarray,
    new_rank: np.ndarray,
    formulas: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    unique, inverse = np.unique(formulas.astype(str), return_inverse=True)
    recall_effect = (new_rank == 1).astype(float) - (old_rank == 1).astype(float)
    mrr_effect = 1.0 / new_rank - 1.0 / old_rank
    counts = np.bincount(inverse)
    recall_sums = np.bincount(inverse, weights=recall_effect)
    mrr_sums = np.bincount(inverse, weights=mrr_effect)
    rng = np.random.default_rng(seed)
    recall_draw = np.empty(resamples, dtype=np.float64)
    mrr_draw = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.integers(0, len(unique), len(unique))
        denominator = counts[sampled].sum()
        recall_draw[index] = recall_sums[sampled].sum() / denominator
        mrr_draw[index] = mrr_sums[sampled].sum() / denominator
    return {
        "formulas": int(len(unique)),
        "delta_recall1": float(np.mean(recall_effect)),
        "delta_recall1_ci_low": float(np.quantile(recall_draw, 0.025)),
        "delta_recall1_ci_high": float(np.quantile(recall_draw, 0.975)),
        "delta_mrr": float(np.mean(mrr_effect)),
        "delta_mrr_ci_low": float(np.quantile(mrr_draw, 0.025)),
        "delta_mrr_ci_high": float(np.quantile(mrr_draw, 0.975)),
    }


def load_seed(root: Path, seed: int, folds: int, graph: CandidateGraph) -> dict:
    queries: list[np.ndarray] = []
    old_ranks: list[np.ndarray] = []
    new_ranks: list[np.ndarray] = []
    decisions: list[dict] = []
    provenance: list[dict] = []
    for fold in range(folds):
        run = root / f"seed_{seed}" / f"fold_{fold}"
        decision_path = run / "decision.json"
        predictions_path = run / "outer_predictions.npz"
        if not decision_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(f"incomplete G1 run: {run}")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if (
            decision.get("status") != "chemaware_shared_v2_g1_fold_complete"
            or decision.get("formal") is not True
            or decision.get("chemical_supervision") is not False
            or int(decision.get("seed", -1)) != seed
            or int(decision.get("outer_fold", -1)) != fold
        ):
            raise RuntimeError(f"invalid formal G1 decision: {decision_path}")
        with np.load(predictions_path) as body:
            query = np.asarray(body["query"], dtype=np.int64)
            old_rank = np.asarray(body["old_rank"], dtype=np.int32)
            new_rank = np.asarray(body["new_rank"], dtype=np.int32)
        if len(query) != len(old_rank) or len(query) != len(new_rank):
            raise RuntimeError(f"malformed G1 predictions: {predictions_path}")
        queries.append(query)
        old_ranks.append(old_rank)
        new_ranks.append(new_rank)
        decisions.append(decision)
        provenance.append({
            "fold": fold,
            "decision_sha256": sha256_file(decision_path),
            "predictions_sha256": sha256_file(predictions_path),
        })
    query = np.concatenate(queries)
    order = np.argsort(query, kind="stable")
    query = query[order]
    old_rank = np.concatenate(old_ranks)[order]
    new_rank = np.concatenate(new_ranks)[order]
    if not np.array_equal(query, np.arange(graph.n_queries)):
        raise RuntimeError(f"seed {seed} does not cover every graph query exactly once")
    near = graph.query_has_near[query]
    old_correct, new_correct = old_rank == 1, new_rank == 1
    return {
        "query": query,
        "old_rank": old_rank,
        "new_rank": new_rank,
        "formulas": graph.query_formula[query],
        "summary": {
            "seed": seed,
            "queries": int(len(query)),
            "baseline_recall1": float(np.mean(old_correct)),
            "recall1": float(np.mean(new_correct)),
            "delta_recall1": float(np.mean(new_correct) - np.mean(old_correct)),
            "baseline_mrr": float(np.mean(1.0 / old_rank)),
            "mrr": float(np.mean(1.0 / new_rank)),
            "delta_mrr": float(np.mean(1.0 / new_rank) - np.mean(1.0 / old_rank)),
            "corrected": int(np.sum(~old_correct & new_correct)),
            "introduced": int(np.sum(old_correct & ~new_correct)),
            "risk_net_lambda2": int(np.sum(~old_correct & new_correct) - 2 * np.sum(old_correct & ~new_correct)),
            "near_queries": int(np.sum(near)),
            "baseline_near_recall1": float(np.mean(old_correct[near])),
            "near_recall1": float(np.mean(new_correct[near])),
            "delta_near_recall1": float(np.mean(new_correct[near]) - np.mean(old_correct[near])),
            "selected_epoch_zero_folds": int(np.sum([item["best_epoch"] == 0 for item in decisions])),
        },
        "provenance": provenance,
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite G1 summary: {args.output}")
    graph = CandidateGraph(args.graph)
    runs = [load_seed(args.root, seed, args.folds, graph) for seed in args.seeds]
    reference = runs[0]
    for run in runs[1:]:
        if not np.array_equal(run["query"], reference["query"]):
            raise RuntimeError("G1 seed query ledgers differ")
        if not np.array_equal(run["old_rank"], reference["old_rank"]):
            raise RuntimeError("G1 seed official baselines differ")
    bootstraps = {
        str(run["summary"]["seed"]): formula_bootstrap(
            run["old_rank"], run["new_rank"], run["formulas"],
            args.bootstrap_resamples, args.bootstrap_seed + index,
        )
        for index, run in enumerate(runs)
    }
    delta_r1 = np.asarray([run["summary"]["delta_recall1"] for run in runs])
    delta_mrr = np.asarray([run["summary"]["delta_mrr"] for run in runs])
    delta_near = np.asarray([run["summary"]["delta_near_recall1"] for run in runs])
    report = {
        "status": "chemaware_shared_v2_g1_multifold_summary_complete",
        "formal": True,
        "experiment": "G1_clean_listwise_control",
        "chemical_supervision": False,
        "folds": args.folds,
        "seeds": args.seeds,
        "per_seed": [run["summary"] for run in runs],
        "formula_cluster_bootstrap": bootstraps,
        "across_seed": {
            "delta_recall1_mean": float(np.mean(delta_r1)),
            "delta_recall1_std": float(np.std(delta_r1, ddof=1)),
            "delta_mrr_mean": float(np.mean(delta_mrr)),
            "delta_mrr_std": float(np.std(delta_mrr, ddof=1)),
            "delta_near_recall1_mean": float(np.mean(delta_near)),
            "delta_near_recall1_std": float(np.std(delta_near, ddof=1)),
        },
        "gates": {
            "all_seed_risk_net_positive": bool(all(run["summary"]["risk_net_lambda2"] > 0 for run in runs)),
            "all_seed_near_nonnegative": bool(np.all(delta_near >= 0)),
            "all_seed_formula_ci_positive": bool(all(value["delta_recall1_ci_low"] > 0 for value in bootstraps.values())),
        },
        "matched_control_ready_for_G2": True,
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "runs": {str(run["summary"]["seed"]): run["provenance"] for run in runs},
        },
        "claim_limit": (
            "This is the no-chemistry continuation control. It cannot establish chemical "
            "attribution; every G2+ arm must beat this paired ledger under the same folds, "
            "seeds, capacity, sampler, steps, and selection rule."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_dump(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
