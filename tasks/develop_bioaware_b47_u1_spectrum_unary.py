#!/usr/bin/env python
"""Develop B47-U1 on the corrected labelled graph without opening B47 truth.

The only permitted intervention is a frozen, spectrum-only aggregation of the
official query/reference cosine scores within one candidate identity.  Recipe
selection is outer formula-fold cross-fitted.  The selected scorer can be
carried to B47 only if its OOF result passes the preregistered safety gates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_b47_u1_core import (  # noqa: E402
    CandidateGraph, choose_recipe, fixed_recipes, formula_cluster_bootstrap,
    paired_summary, query_metrics, score_recipe, sha256_file,
    stable_formula_folds, strict_ranks, unary_features,
)


EXPECTED_GRAPH_SHA256 = "8a57bb3a9cccdf69a738f2b093bfad8f6fc4393b23f1342c7a035ac54ce58fa1"
EXPECTED_GRAPH_CONTRACT = "train_primary_all_p3_disjoint_v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir", type=Path,
        default=Path("data/validation/noise_corrected_candidate_graph_v1_20260906"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260921)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, body: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp",
    ) as handle:
        json.dump(body, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_contract(graph_dir: Path) -> tuple[Path, dict]:
    graph_path = graph_dir / "candidate_graph.npz"
    report_path = graph_dir / "report.json"
    if not graph_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"corrected graph is incomplete: {graph_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("formal_training_authorized") is not True
        or report.get("data_contract") != EXPECTED_GRAPH_CONTRACT
        or report.get("provenance", {}).get("candidate_graph_sha256")
        != EXPECTED_GRAPH_SHA256
        or sha256_file(graph_path) != EXPECTED_GRAPH_SHA256
        or report.get("contracts", {}).get("P3_consumed") is not False
        or report.get("contracts", {}).get("P2b") != "forbidden"
        or report.get("contracts", {}).get("official_score_is_query_candidate_cosine") is not True
    ):
        raise RuntimeError("corrected graph provenance/scientific contract changed")
    return graph_path, report


def macro_query_auc(score: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    output = np.empty(len(query_ptr) - 1, dtype=np.float64)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        local = np.asarray(score[int(left):int(right)], dtype=np.float64)
        positive = local[0]
        negative = local[1:]
        output[query] = (
            np.sum(positive > negative) + 0.5 * np.sum(positive == negative)
        ) / len(negative)
    return output


def query_top_index(score: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    output = np.empty(len(query_ptr) - 1, dtype=np.int64)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        output[query] = int(left) + int(np.argmax(score[int(left):int(right)]))
    return output


def count_strata(
    graph: CandidateGraph,
    features,
    old_rank: np.ndarray,
    new_rank: np.ndarray,
) -> dict[str, dict[str, float | int | bool]]:
    positive_count = features.reference_count[graph.query_ptr[:-1]]
    labels = {
        "1": positive_count == 1,
        "2-3": (positive_count >= 2) & (positive_count <= 3),
        "4-9": (positive_count >= 4) & (positive_count <= 9),
        "10-31": (positive_count >= 10) & (positive_count <= 31),
        "32+": positive_count >= 32,
    }
    return {
        name: paired_summary(
            old_rank, new_rank, graph.query_formula, graph.query_has_near, mask,
        )
        for name, mask in labels.items() if np.any(mask)
    }


def main() -> None:
    args = arguments()
    if args.folds != 5:
        raise ValueError("U1 preregistration fixes exactly five formula folds")
    if args.bootstrap_resamples < 1000:
        raise ValueError("formal U1 requires at least 1,000 bootstrap resamples")
    graph_dir = args.graph_dir.resolve()
    graph_path, graph_report = load_contract(graph_dir)
    if args.preflight_only:
        graph = CandidateGraph(graph_path)
        if graph.n_queries != 83619 or len(graph.features) != 6220661:
            raise RuntimeError("corrected graph denominator changed")
        print("[BioAware B47 U1 preflight] PASS; no result written", flush=True)
        return
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite frozen U1 output: {output}")

    graph = CandidateGraph(graph_path)
    if graph.n_queries != 83619 or len(graph.features) != 6220661:
        raise RuntimeError("corrected graph denominator changed")
    pair_score = np.asarray(graph.features[:, graph.dreams_column], dtype=np.float32)
    features = unary_features(pair_score, graph.molecule_ptr)
    recipes = fixed_recipes()
    if len(recipes) != 33 or recipes[0].name != "max":
        raise RuntimeError("U1 preregistered recipe family drifted")
    score_by_name = {recipe.name: score_recipe(features, recipe) for recipe in recipes}
    rank_by_name = {
        name: strict_ranks(score, graph.query_ptr) for name, score in score_by_name.items()
    }
    baseline_rank = rank_by_name["max"]
    baseline_metrics = query_metrics(baseline_rank, graph.query_has_near)
    expected = graph_report["official_baseline"]
    if (
        not np.isclose(baseline_metrics["recall@1"], expected["recall1"], atol=1e-12)
        or not np.isclose(baseline_metrics["mrr"], expected["mrr"], atol=1e-12)
    ):
        raise RuntimeError("U1 failed to reproduce the corrected official baseline")

    formula_fold = stable_formula_folds(graph.query_formula, args.folds, args.fold_seed)
    oof_rank = np.empty(graph.n_queries, dtype=np.int16)
    oof_score = np.empty(len(graph.molecule_label), dtype=np.float32)
    outer: list[dict] = []
    for held_fold in range(args.folds):
        train = formula_fold != held_fold
        held = ~train
        train_summary = {
            name: paired_summary(
                baseline_rank, ranks, graph.query_formula, graph.query_has_near, train,
            )
            for name, ranks in rank_by_name.items()
        }
        selected = choose_recipe(train_summary)
        held_summary = paired_summary(
            baseline_rank, rank_by_name[selected], graph.query_formula,
            graph.query_has_near, held,
        )
        oof_rank[held] = rank_by_name[selected][held]
        query_indices = np.flatnonzero(held)
        for query in query_indices:
            left, right = map(int, graph.query_ptr[query:query + 2])
            oof_score[left:right] = score_by_name[selected][left:right]
        outer.append({
            "fold": int(held_fold),
            "train_queries": int(train.sum()),
            "held_queries": int(held.sum()),
            "selected_recipe": selected,
            "held_result": held_summary,
        })
        print(
            f"[U1 fold {held_fold}] recipe={selected} "
            f"dR1={held_summary['delta_recall1']:+.6f} "
            f"C/I={held_summary['corrected']}/{held_summary['introduced']}",
            flush=True,
        )

    all_mask = np.ones(graph.n_queries, dtype=bool)
    full_summary = {
        name: paired_summary(
            baseline_rank, ranks, graph.query_formula, graph.query_has_near, all_mask,
        )
        for name, ranks in rank_by_name.items()
    }
    frozen_name = choose_recipe(full_summary)
    frozen_recipe = next(recipe for recipe in recipes if recipe.name == frozen_name)
    oof_metrics = query_metrics(oof_rank, graph.query_has_near)
    oof_paired = paired_summary(
        baseline_rank, oof_rank, graph.query_formula, graph.query_has_near, all_mask,
    )
    delta_top1 = (oof_rank == 1).astype(np.float64) - (baseline_rank == 1).astype(np.float64)
    ci = formula_cluster_bootstrap(
        delta_top1, graph.query_formula, args.bootstrap_resamples, args.fold_seed + 1,
    )
    oof_auc = macro_query_auc(oof_score, graph.query_ptr)
    baseline_auc = macro_query_auc(score_by_name["max"], graph.query_ptr)
    baseline_top = query_top_index(score_by_name["max"], graph.query_ptr)
    oof_top = query_top_index(oof_score, graph.query_ptr)
    fold_nonnegative = all(
        float(item["held_result"]["delta_recall1"]) >= 0.0
        and float(item["held_result"]["near_delta_recall1"]) >= 0.0
        for item in outer
    )
    gates = {
        "frozen_recipe_is_not_max": frozen_name != "max",
        "oof_recall1_positive": float(oof_paired["delta_recall1"]) > 0.0,
        "oof_mrr_nonnegative": float(oof_paired["delta_mrr"]) >= 0.0,
        "oof_near_recall1_nonnegative": float(oof_paired["near_delta_recall1"]) >= 0.0,
        "oof_corrected_gt_2x_introduced": bool(
            oof_paired["corrected_gt_2x_introduced"]
        ),
        "oof_formula_cluster_ci_low_positive": float(ci["ci_low"]) > 0.0,
        "every_outer_fold_overall_and_near_nonnegative": bool(fold_nonnegative),
    }
    pass_to_b47 = all(gates.values())

    recipe_rows = []
    for recipe in recipes:
        row = {
            "recipe": recipe.name,
            "support": recipe.support,
            "shrinkage": recipe.shrinkage,
            "log_count_weight": recipe.log_count_weight,
            **full_summary[recipe.name],
        }
        recipe_rows.append(row)
    per_query = pd.DataFrame({
        "query_index": np.arange(graph.n_queries, dtype=np.int64),
        "query_row": graph.query_row,
        "query_ik14": graph.query_ik14,
        "query_formula": graph.query_formula,
        "formula_fold": formula_fold,
        "near": graph.query_has_near,
        "baseline_rank": baseline_rank,
        "oof_rank": oof_rank,
        "baseline_top_molecule_index": baseline_top,
        "oof_top_molecule_index": oof_top,
        "baseline_correct": baseline_rank == 1,
        "oof_correct": oof_rank == 1,
        "corrected": (baseline_rank > 1) & (oof_rank == 1),
        "introduced": (baseline_rank == 1) & (oof_rank > 1),
        "baseline_macro_query_auc": baseline_auc,
        "oof_macro_query_auc": oof_auc,
        "positive_reference_spectra": features.reference_count[graph.query_ptr[:-1]],
    })

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".bioaware_b47_u1_", dir=output.parent))
    try:
        pd.DataFrame(recipe_rows).to_csv(
            staging / "recipe_scan.csv.gz", index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        per_query.to_csv(
            staging / "per_query_oof.csv.gz", index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        frozen = {
            "status": "bioaware_b47_u1_spectrum_unary_recipe_frozen",
            "eligible_for_b47_truthblind_apply": bool(pass_to_b47),
            "name": frozen_recipe.name,
            "support": frozen_recipe.support,
            "shrinkage": frozen_recipe.shrinkage,
            "log_count_weight": frozen_recipe.log_count_weight,
            "selection_source": "all corrected development queries after formula-OOF audit",
        }
        atomic_json(staging / "frozen_recipe.json", frozen)
        report = {
            "status": "bioaware_b47_u1_spectrum_unary_development_complete",
            "formal": True,
            "development_graph_only": True,
            "candidate_recipes": int(len(recipes)),
            "baseline": baseline_metrics,
            "oof": {
                **oof_metrics,
                **oof_paired,
                "macro_query_auc": float(np.mean(oof_auc)),
                "baseline_macro_query_auc": float(np.mean(baseline_auc)),
                "delta_macro_query_auc": float(np.mean(oof_auc - baseline_auc)),
                "formula_cluster_bootstrap_recall1_delta": ci,
                "top1_identity_switches": int(np.sum(oof_top != baseline_top)),
            },
            "outer_folds": outer,
            "frozen_recipe": frozen,
            "positive_reference_count_strata": count_strata(
                graph, features, baseline_rank, oof_rank,
            ),
            "gates": gates,
            "pass_to_b47_truthblind_apply": bool(pass_to_b47),
            "contracts": {
                "only_official_spectrum_cosines_used": True,
                "formula_outer_oof": True,
                "candidate_set_unchanged": True,
                "ties_count_against_positive": True,
                "B47_truth_opened": False,
                "reaction_network_used": False,
                "phenotype_used": False,
                "P2b_used": False,
                "shared_embedding_changed": False,
                "adduct_pooling_validated": False,
            },
            "provenance": {
                "candidate_graph_sha256": sha256_file(graph_path),
                "graph_report_sha256": sha256_file(graph_dir / "report.json"),
                "script_sha256": sha256_file(Path(__file__)),
                "core_sha256": sha256_file(Path(__file__).with_name("bioaware_b47_u1_core.py")),
                "recipe_scan_sha256": sha256_file(staging / "recipe_scan.csv.gz"),
                "per_query_oof_sha256": sha256_file(staging / "per_query_oof.csv.gz"),
                "frozen_recipe_sha256": sha256_file(staging / "frozen_recipe.json"),
            },
            "parameters": {
                "folds": int(args.folds),
                "fold_seed": int(args.fold_seed),
                "bootstrap_resamples": int(args.bootstrap_resamples),
            },
            "claim_limit": (
                "Formula-OOF development result for a spectrum-only reference aggregator. "
                "It is not B47 accuracy, reaction-network gain, external transfer, or "
                "shared-embedding improvement. Unknown-adduct pooling remains unvalidated."
            ),
        }
        atomic_json(staging / "report.json", report)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
