#!/usr/bin/env python
"""Aggregate exactly one formula-held-out prediction per query."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, sha256_file  # noqa: E402


def formula_bootstrap(formula, old, new, repeats, seed):
    table = pd.DataFrame({
        "formula": formula,
        "delta": (new == 1).astype(float) - (old == 1).astype(float),
    })
    groups = {
        str(key): value.delta.to_numpy()
        for key, value in table.groupby("formula", sort=True)
    }
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats)
    for index in range(repeats):
        selected = rng.choice(keys, len(keys), replace=True)
        values[index] = np.mean(np.concatenate([groups[str(key)] for key in selected]))
    return {
        "mean": float(np.mean(new == 1) - np.mean(old == 1)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "clusters": len(keys), "resamples": repeats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/validation/bioaware_embedding_adapter")
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--minimum-preservation", type=float, default=0.995)
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/bioaware_embedding_adapter_oof.json")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"fail-closed: output exists: {args.output}")
    graph = CandidateGraph(args.graph)
    old = np.full(graph.n_queries, -1, dtype=np.int16)
    new = np.full(graph.n_queries, -1, dtype=np.int16)
    reports = []
    prediction_hashes = {}
    frozen_hyperparameters = None
    for fold in range(5):
        directory = args.root / f"fold_{fold}" / f"seed_{args.seed}"
        report_path = directory / "report.json"
        prediction_path = directory / "heldout_predictions.npz"
        if not report_path.exists() or not prediction_path.exists():
            raise FileNotFoundError([report_path, prediction_path])
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        current_hyperparameters = reports[-1].get("training", {}).get("frozen_hyperparameters")
        if not current_hyperparameters:
            raise RuntimeError(f"fold {fold} does not declare frozen hyperparameters")
        if frozen_hyperparameters is None:
            frozen_hyperparameters = current_hyperparameters
        elif current_hyperparameters != frozen_hyperparameters:
            raise RuntimeError("outer folds used different hyperparameters")
        with np.load(prediction_path) as body:
            query = np.asarray(body["query"], dtype=np.int64)
            if np.any(old[query] != -1):
                raise RuntimeError("query appears in more than one outer fold")
            old[query] = body["old_rank"]
            new[query] = body["new_rank"]
        prediction_hashes[str(fold)] = sha256_file(prediction_path)
    if np.any(old < 1) or np.any(new < 1):
        raise RuntimeError("outer-fold predictions do not cover every query exactly once")
    base = old == 1
    candidate = new == 1
    near = graph.query_has_near
    ci = formula_bootstrap(graph.query_formula, old, new, args.bootstrap, args.seed)
    preservation = [float(report["heldout"]["preservation_mean"]) for report in reports]
    delta_mrr = float(np.mean(1.0 / new) - np.mean(1.0 / old))
    corrected = int(np.sum(~base & candidate))
    introduced = int(np.sum(base & ~candidate))
    near_delta = float(candidate[near].mean() - base[near].mean())
    result = {
        "status": "bioaware_embedding_adapter_formula_oof_complete",
        "formal": True,
        "n_queries": graph.n_queries,
        "baseline_recall1": float(base.mean()),
        "recall1": float(candidate.mean()),
        "delta_recall1": float(candidate.mean() - base.mean()),
        "baseline_mrr": float(np.mean(1.0 / old)),
        "mrr": float(np.mean(1.0 / new)),
        "delta_mrr": delta_mrr,
        "corrected": corrected,
        "introduced": introduced,
        "baseline_near_recall1": float(base[near].mean()),
        "near_recall1": float(candidate[near].mean()),
        "delta_near_recall1": near_delta,
        "formula_cluster_bootstrap": ci,
        "fold_preservation_mean": preservation,
        "frozen_hyperparameters": frozen_hyperparameters,
        "gates": {
            "formula_ci_positive": ci["ci_low"] > 0,
            "mrr_positive": delta_mrr > 0,
            "near_nonnegative": near_delta >= 0,
            "corrected_gt_introduced": corrected > introduced,
            "all_folds_preserved": min(preservation) >= args.minimum_preservation,
        },
        "contracts": {
            "shared_embedding": True, "P2b": "forbidden", "P3": "not opened",
            "reaction_head_deployed": False,
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "predictions": prediction_hashes,
        },
        "claim_limit": "Training-space formula-OOF result; a frozen external test is required for SOTA claims.",
    }
    result["gates"]["pass"] = all(result["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
