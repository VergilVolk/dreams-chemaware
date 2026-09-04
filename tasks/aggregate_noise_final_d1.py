"""Aggregate 5-fold OOF D1 results for three independent seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noise_final_core import CandidateGraph, json_dump, strict_rank


ROOT = Path(__file__).resolve().parent.parent


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/validation/g8r_noise_final_d1_adapter")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260825, 20260826, 20260827])
    parser.add_argument("--minimum-preservation", type=float, default=0.995)
    return parser.parse_args()


def formula_cluster_ci(
    base: np.ndarray, new: np.ndarray, formulas: np.ndarray,
    seed: int, resamples: int = 5000,
) -> dict[str, float]:
    unique = np.unique(formulas)
    groups = [np.flatnonzero(formulas == value) for value in unique]
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(groups), size=len(groups))
        numerator = 0.0
        denominator = 0
        for group_index in chosen:
            positions = groups[int(group_index)]
            numerator += float(np.sum(new[positions] - base[positions]))
            denominator += len(positions)
        estimates[index] = numerator / denominator
    return {
        "mean": float(np.mean(new - base)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def summarize(seed: int, root: Path, graph: CandidateGraph) -> dict:
    bodies = []
    for fold in range(5):
        path = root / f"seed_{seed}" / f"fold_{fold}" / "outer_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as body:
            bodies.append({key: body[key] for key in body.files})
    query = np.concatenate([body["query"] for body in bodies]).astype(np.int64)
    if len(query) != graph.n_queries or len(np.unique(query)) != graph.n_queries:
        raise RuntimeError(f"seed {seed} does not cover every query exactly once")
    order = np.argsort(query)
    query = query[order]
    if not np.array_equal(query, np.arange(graph.n_queries)):
        raise RuntimeError("OOF query indices are not complete")
    old_rank = np.concatenate([body["old_rank"] for body in bodies])[order]
    new_rank = np.concatenate([body["new_rank"] for body in bodies])[order]
    preservation = np.concatenate([body["preservation"] for body in bodies])[order]
    # D0's cached candidate graph is the frozen baseline contract.  Online
    # embedding recomputation can move a score by a few ulps and change a
    # strict tie-aware rank.  Never let that numerical boundary silently
    # redefine the baseline used for corrected/introduced accounting.
    frozen_rank = np.asarray([
        strict_rank(graph.official_molecule_scores(query_index))
        for query_index in range(graph.n_queries)
    ], dtype=np.int16)
    baseline_mismatch = old_rank != frozen_rank
    base, new = frozen_rank == 1, new_rank == 1
    near = graph.query_has_near
    overall_delta = formula_cluster_ci(
        base.astype(np.float64), new.astype(np.float64), graph.query_formula,
        seed=seed + 1000,
    )
    near_delta_ci = formula_cluster_ci(
        base[near].astype(np.float64), new[near].astype(np.float64),
        graph.query_formula[near], seed=seed + 2000,
    )
    return {
        "seed": seed, "n_queries": graph.n_queries,
        "baseline_recall1": float(np.mean(base)), "recall1": float(np.mean(new)),
        "delta_recall1": float(np.mean(new) - np.mean(base)),
        "baseline_mrr": float(np.mean(1.0 / frozen_rank)), "mrr": float(np.mean(1.0 / new_rank)),
        "delta_mrr": float(np.mean(1.0 / new_rank) - np.mean(1.0 / frozen_rank)),
        "corrected": int(np.sum(~base & new)), "introduced": int(np.sum(base & ~new)),
        "baseline_near_recall1": float(np.mean(base[near])), "near_recall1": float(np.mean(new[near])),
        "delta_near_recall1": float(np.mean(new[near]) - np.mean(base[near])),
        "preservation_mean": float(np.mean(preservation)), "preservation_min": float(np.min(preservation)),
        "online_baseline_mismatch_count": int(np.sum(baseline_mismatch)),
        "online_baseline_mismatch_fraction": float(np.mean(baseline_mismatch)),
        "formula_cluster_delta_recall1": overall_delta,
        "near_formula_cluster_delta_recall1": near_delta_ci,
    }


def main():
    args = arguments()
    graph = CandidateGraph(args.graph)
    per_seed = [summarize(seed, args.root, graph) for seed in args.seeds]
    delta = np.asarray([row["delta_recall1"] for row in per_seed])
    near_delta = np.asarray([row["delta_near_recall1"] for row in per_seed])
    gates = {
        "all_seeds_overall_nonnegative": bool(np.all(delta >= 0)),
        "all_seeds_near_nonnegative": bool(np.all(near_delta >= 0)),
        "all_seeds_corrected_ge_introduced": bool(all(row["corrected"] >= row["introduced"] for row in per_seed)),
        "all_seeds_preservation_ok": bool(all(row["preservation_mean"] >= args.minimum_preservation for row in per_seed)),
        "all_seed_formula_cluster_ci_not_negative": bool(all(
            row["formula_cluster_delta_recall1"]["ci_low"] >= -0.003 for row in per_seed
        )),
        "mean_clean_gain_strictly_positive": bool(delta.mean() > 0),
        "at_least_two_seeds_have_positive_net_corrections": bool(sum(
            row["corrected"] > row["introduced"] for row in per_seed
        ) >= 2),
    }
    gates["pass_to_D2"] = bool(all(gates.values()))
    report = {
        "status": "noise_final_d1_aggregate", "per_seed": per_seed,
        "mean_delta_recall1": float(delta.mean()), "mean_delta_near_recall1": float(near_delta.mean()),
        "direction_consistent": bool(np.all(np.sign(delta) == np.sign(delta[0]))),
        "gates": gates,
        "decision": "Only a passing clean adapter may receive the P-arm teacher in D2.",
    }
    json_dump(args.root / "aggregate.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not gates["pass_to_D2"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
