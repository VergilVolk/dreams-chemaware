#!/usr/bin/env python
"""Audit the baseline-unique no-identity-switch safety wrapper for B47-U1."""
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
    CandidateGraph, fixed_recipes, formula_cluster_bootstrap, paired_summary,
    query_metrics, same_top_unique_veto, score_recipe, sha256_file,
    stable_formula_folds, strict_ranks, unary_features,
)


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--u1-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    graph_dir, u1_dir = args.graph_dir.resolve(), args.u1_dir.resolve()
    graph_path = graph_dir / "candidate_graph.npz"
    u1_report_path = u1_dir / "report.json"
    u1_recipe_path = u1_dir / "frozen_recipe.json"
    for path in (graph_path, graph_dir / "report.json", u1_report_path, u1_recipe_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite frozen U1b output: {output}")
    u1 = json.loads(u1_report_path.read_text(encoding="utf-8"))
    frozen = json.loads(u1_recipe_path.read_text(encoding="utf-8"))
    if (
        u1.get("status") != "bioaware_b47_u1_spectrum_unary_development_complete"
        or u1.get("formal") is not True
        or u1.get("provenance", {}).get("candidate_graph_sha256") != sha256_file(graph_path)
        or u1.get("provenance", {}).get("frozen_recipe_sha256") != sha256_file(u1_recipe_path)
        or frozen.get("name") != u1.get("frozen_recipe", {}).get("name")
    ):
        raise RuntimeError("U1 provenance/frozen recipe is inconsistent")
    graph = CandidateGraph(graph_path)
    feature = unary_features(graph.features[:, graph.dreams_column], graph.molecule_ptr)
    recipes = {recipe.name: recipe for recipe in fixed_recipes()}
    if frozen["name"] not in recipes:
        raise RuntimeError("frozen U1 recipe is absent from the preregistered family")
    baseline_score = feature.maximum
    baseline_rank = strict_ranks(baseline_score, graph.query_ptr)
    fold = stable_formula_folds(
        graph.query_formula, int(u1["parameters"]["folds"]),
        int(u1["parameters"]["fold_seed"]),
    )
    oof_calibrated = np.empty_like(baseline_score)
    for outer in u1["outer_folds"]:
        name = str(outer["selected_recipe"])
        if name not in recipes:
            raise RuntimeError(f"unknown OOF recipe: {name}")
        local_score = score_recipe(feature, recipes[name])
        for query in np.flatnonzero(fold == int(outer["fold"])):
            left, right = map(int, graph.query_ptr[query:query + 2])
            oof_calibrated[left:right] = local_score[left:right]
    safe_score, safe_gate = same_top_unique_veto(
        baseline_score, oof_calibrated, graph.query_ptr,
    )
    safe_rank = strict_ranks(safe_score, graph.query_ptr)
    selection = np.ones(graph.n_queries, dtype=bool)
    paired = paired_summary(
        baseline_rank, safe_rank, graph.query_formula, graph.query_has_near, selection,
    )
    metrics = query_metrics(safe_rank, graph.query_has_near)
    delta = (safe_rank == 1).astype(float) - (baseline_rank == 1).astype(float)
    ci = formula_cluster_bootstrap(
        delta, graph.query_formula, args.bootstrap_resamples, args.seed,
    )
    outer_result = []
    for held_fold in range(5):
        mask = fold == held_fold
        outer_result.append({
            "fold": held_fold,
            **paired_summary(
                baseline_rank, safe_rank, graph.query_formula,
                graph.query_has_near, mask,
            ),
        })
    # Independently verify the structural theorem on every query.
    baseline_top = np.asarray([
        int(left) + int(np.argmax(baseline_score[int(left):int(right)]))
        for left, right in zip(graph.query_ptr[:-1], graph.query_ptr[1:])
    ], dtype=np.int64)
    safe_top = np.asarray([
        int(left) + int(np.argmax(safe_score[int(left):int(right)]))
        for left, right in zip(graph.query_ptr[:-1], graph.query_ptr[1:])
    ], dtype=np.int64)
    identity_switches = int(np.sum(baseline_top != safe_top))
    gates = {
        "top1_identity_switches_zero": identity_switches == 0,
        "corrected_positive": int(paired["corrected"]) > 0,
        "introduced_zero": int(paired["introduced"]) == 0,
        "formula_cluster_ci_low_positive": float(ci["ci_low"]) > 0.0,
        "near_recall1_nonnegative": float(paired["near_delta_recall1"]) >= 0.0,
        "mrr_nonnegative": float(paired["delta_mrr"]) >= 0.0,
        "every_outer_fold_overall_and_near_nonnegative": all(
            float(item["delta_recall1"]) >= 0.0
            and float(item["near_delta_recall1"]) >= 0.0
            for item in outer_result
        ),
    }
    passed = all(gates.values())
    final_calibrated = score_recipe(feature, recipes[frozen["name"]])
    _, full_gate = same_top_unique_veto(
        baseline_score, final_calibrated, graph.query_ptr,
    )
    policy = {
        "status": "bioaware_b47_u1b_baseline_unique_veto_frozen",
        "eligible_for_b47_truthblind_apply": bool(passed),
        "recipe": {
            "name": frozen["name"],
            "support": frozen["support"],
            "shrinkage": float(frozen["shrinkage"]),
            "log_count_weight": float(frozen["log_count_weight"]),
        },
        "veto": (
            "use calibrated candidate scores only when original max-score Top-1 "
            "is unique, calibrated Top-1 is unique, and both identities agree"
        ),
        "numerical_tolerance": 1e-12,
    }
    per_query = pd.DataFrame({
        "query_index": np.arange(graph.n_queries),
        "query_row": graph.query_row,
        "query_ik14": graph.query_ik14,
        "query_formula": graph.query_formula,
        "formula_fold": fold,
        "near": graph.query_has_near,
        "safe_gate": safe_gate,
        "baseline_rank": baseline_rank,
        "safe_rank": safe_rank,
        "corrected": (baseline_rank > 1) & (safe_rank == 1),
        "introduced": (baseline_rank == 1) & (safe_rank > 1),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".bioaware_b47_u1b_", dir=output.parent))
    try:
        per_query.to_csv(
            staging / "per_query_safe_oof.csv.gz", index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        atomic_json(staging / "frozen_policy.json", policy)
        report = {
            "status": "bioaware_b47_u1b_safe_veto_complete",
            "formal": True,
            "development_graph_only": True,
            "oof": {
                **metrics, **paired,
                "formula_cluster_bootstrap_recall1_delta": ci,
                "safe_gate_queries": int(safe_gate.sum()),
                "top1_identity_switches": identity_switches,
            },
            "outer_folds": outer_result,
            "frozen_policy": policy,
            "full_development_policy_gate_queries": int(full_gate.sum()),
            "gates": gates,
            "pass_to_b47_truthblind_apply": bool(passed),
            "contracts": {
                "B47_truth_opened": False,
                "reaction_network_used": False,
                "phenotype_used": False,
                "P2b_used": False,
                "shared_embedding_changed": False,
                "top1_identity_change_forbidden_by_construction": True,
                "baseline_tie_resolution_forbidden": True,
                "candidate_order_used_for_tie_breaking": False,
            },
            "provenance": {
                "candidate_graph_sha256": sha256_file(graph_path),
                "u1_report_sha256": sha256_file(u1_report_path),
                "u1_frozen_recipe_sha256": sha256_file(u1_recipe_path),
                "script_sha256": sha256_file(Path(__file__)),
                "core_sha256": sha256_file(Path(__file__).with_name("bioaware_b47_u1_core.py")),
                "per_query_sha256": sha256_file(staging / "per_query_safe_oof.csv.gz"),
                "frozen_policy_sha256": sha256_file(staging / "frozen_policy.json"),
            },
            "claim_limit": (
                "A spectrum-only baseline-unique preservation audit. It cannot switch "
                "Top-1 identity or resolve baseline ties, and therefore cannot improve "
                "Top-1 retrieval by construction. It is not a B47 accuracy result, "
                "reaction-network result, or embedding improvement."
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
