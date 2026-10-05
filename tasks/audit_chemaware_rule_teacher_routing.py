"""Audit where the admitted rule-mass teacher helps or harms frozen retrieval.

This audit consumes the already frozen pair-first inner ranks.  It does not
train a model and it does not touch the outer fold.  Query-wise routing is an
upper-bound diagnostic for choosing a *training-only* teacher scope; it is not
a deployable inference router and must never be reported as an embedding gain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
from train_chemaware_full_candidate_alignment import (  # noqa: E402
    formula_bootstrap, official_outcomes,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--rank-file", type=Path,
        default=ROOT / "data/validation/chemaware_rule_mass_pairfirst_full_inner_v1/inner_per_query.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument(
        "--boundary-grid", type=float, nargs="+",
        default=(0.0, 1e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1),
    )
    return parser.parse_args()


def routed_rank(old_rank: np.ndarray, teacher_rank: np.ndarray,
                active: np.ndarray) -> np.ndarray:
    if old_rank.shape != teacher_rank.shape or active.shape != old_rank.shape:
        raise ValueError("rank and routing arrays must have the same shape")
    return np.where(active, teacher_rank, old_rank)


def transition(old_rank: np.ndarray, new_rank: np.ndarray) -> dict:
    old_hit = old_rank == 1
    new_hit = new_rank == 1
    return {
        "recall1": float(np.mean(new_hit)),
        "delta_recall1": float(np.mean(new_hit) - np.mean(old_hit)),
        "mrr": float(np.mean(1.0 / new_rank)),
        "delta_mrr": float(np.mean(1.0 / new_rank) - np.mean(1.0 / old_rank)),
        "corrected": int(np.sum(~old_hit & new_hit)),
        "introduced": int(np.sum(old_hit & ~new_hit)),
    }


def finite_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "quantiles": None}
    return {
        "count": int(len(values)),
        "quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    required = (args.manifest, args.token_dir / "rows.npy",
                args.token_dir / "official_embeddings_f32.npy", args.rank_file)
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)

    with np.load(args.manifest) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(
        args.token_dir / "official_embeddings_f32.npy", mmap_mode="r",
    )
    row_position = {int(row): index for index, row in enumerate(rows)}
    with np.load(args.rank_file) as loaded:
        query = loaded["query"].astype(np.int64)
        formula = loaded["formula"].astype(str)
        old_rank = loaded["old_rank"].astype(np.int64)
        rule_rank = loaded["rule_mass_rank"].astype(np.int64)

    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    if np.any(fold[query] != args.inner_fold):
        raise RuntimeError("rank file contains a query outside the locked inner fold")
    if np.any(fold[query] == args.outer_fold):
        raise RuntimeError("outer query entered the routing audit")
    recomputed_error, recomputed_margin = official_outcomes(
        body, query, official, row_position, None,
    )
    margin = recomputed_margin[query].astype(np.float64)
    if not np.array_equal(recomputed_error[query], old_rank != 1):
        raise RuntimeError("rank-file baseline disagrees with recomputed official outcome")

    old_hit = old_rank == 1
    rule_hit = rule_rank == 1
    corrected = ~old_hit & rule_hit
    introduced = old_hit & ~rule_hit
    all_metrics = transition(old_rank, rule_rank)
    all_delta = rule_hit.astype(float) - old_hit.astype(float)
    all_metrics["formula_bootstrap"] = formula_bootstrap(
        all_delta, formula, args.seed + 701, args.bootstrap_draws,
    )

    error_only_active = ~old_hit
    error_only_rank = routed_rank(old_rank, rule_rank, error_only_active)
    error_only = transition(old_rank, error_only_rank)
    error_only["active_queries"] = int(np.sum(error_only_active))
    error_only["active_fraction"] = float(np.mean(error_only_active))
    error_only["formula_bootstrap"] = formula_bootstrap(
        (error_only_rank == 1).astype(float) - old_hit.astype(float),
        formula, args.seed + 801, args.bootstrap_draws,
    )

    boundary = []
    for offset, threshold in enumerate(map(float, args.boundary_grid)):
        active = error_only_active | (old_hit & (margin <= threshold))
        rank = routed_rank(old_rank, rule_rank, active)
        metrics = transition(old_rank, rank)
        metrics.update({
            "threshold": threshold,
            "active_queries": int(np.sum(active)),
            "active_fraction": float(np.mean(active)),
            "risk_utility": int(metrics["corrected"] - 2 * metrics["introduced"]),
            "formula_bootstrap": formula_bootstrap(
                (rank == 1).astype(float) - old_hit.astype(float),
                formula, args.seed + 901 + 31 * offset, args.bootstrap_draws,
            ),
        })
        boundary.append(metrics)
    selected = max(
        boundary,
        key=lambda item: (
            item["risk_utility"], item["delta_recall1"], item["delta_mrr"],
            -item["active_fraction"], -item["threshold"],
        ),
    )

    report = {
        "status": "AUDIT_COMPLETE",
        "scope": (
            "inner-fold teacher-routing headroom only; no model trained; "
            "not a deployable result; outer fold untouched"
        ),
        "data": {
            "queries": int(len(query)),
            "baseline_errors": int(np.sum(~old_hit)),
            "baseline_correct": int(np.sum(old_hit)),
            "formula_clusters": int(len(np.unique(formula))),
        },
        "unrouted_rule_mass": all_metrics,
        "official_error_only_oracle": error_only,
        "error_or_boundary_grid": boundary,
        "selected_training_scope_diagnostic": {
            "scope": (
                "official_error" if selected["threshold"] == 0.0
                else "official_error_or_boundary"
            ),
            **selected,
        },
        "margin_diagnostics": {
            "all_official_correct": finite_summary(margin[old_hit]),
            "rule_mass_introduced_errors": finite_summary(margin[introduced]),
            "rule_mass_preserved_correct": finite_summary(margin[old_hit & rule_hit]),
            "rule_mass_corrected_errors": finite_summary(margin[corrected]),
            "rule_mass_uncorrected_errors": finite_summary(margin[~old_hit & ~rule_hit]),
        },
        "contracts": {
            "pair_first_rank_file": True,
            "routing_uses_official_training_outcome_only": True,
            "candidate_router_absent_at_deployment": True,
            "embedding_training_was_run": False,
            "outer_fold_evaluated": False,
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "rank_file_sha256": sha256_file(args.rank_file),
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
