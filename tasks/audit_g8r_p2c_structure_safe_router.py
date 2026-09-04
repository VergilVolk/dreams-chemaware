#!/usr/bin/env python
"""Freeze and audit a structure-safe router around the frozen P2b expert.

The router is label-free at deployment: if the strict candidate set contains
an MCES-near pair, retain official DreaMS; otherwise use frozen P2b.  The MCES
near definition and P2b configuration are fixed before this audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from g8r_p2_rank_fusion_core import fusion_configuration_from_mapping
from train_g8r_p2b_rank_fusion import Cache, config_key, precompute_predictions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(
    formulas: np.ndarray,
    baseline: np.ndarray,
    final: np.ndarray,
    repeats: int,
    seed: int,
) -> dict:
    table = pd.DataFrame(
        {"formula": formulas.astype(str), "baseline": baseline, "final": final}
    )
    groups = {key: value for key, value in table.groupby("formula", sort=True)}
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    delta = np.empty(repeats)
    for index in range(repeats):
        selected = rng.choice(keys, size=len(keys), replace=True)
        sample = pd.concat([groups[str(key)] for key in selected], ignore_index=True)
        delta[index] = sample.final.mean() - sample.baseline.mean()
    return {
        "mean": float(final.mean() - baseline.mean()),
        "ci_low": float(np.quantile(delta, 0.025)),
        "ci_high": float(np.quantile(delta, 0.975)),
        "clusters": len(keys),
        "resamples": repeats,
    }


def metrics(
    baseline_top1: np.ndarray,
    baseline_mrr: np.ndarray,
    p2b_top1: np.ndarray,
    p2b_mrr: np.ndarray,
    use_p2b: np.ndarray,
    mask: np.ndarray,
) -> dict:
    base = baseline_top1[mask]
    base_mrr = baseline_mrr[mask]
    p2 = p2b_top1[mask]
    p2_mrr = p2b_mrr[mask]
    use = use_p2b[mask]
    final = np.where(use, p2, base)
    final_mrr = np.where(use, p2_mrr, base_mrr)
    return {
        "n_queries": int(mask.sum()),
        "baseline_recall1": float(base.mean()),
        "recall1": float(final.mean()),
        "delta_recall1": float(final.mean() - base.mean()),
        "baseline_mrr": float(base_mrr.mean()),
        "mrr": float(final_mrr.mean()),
        "delta_mrr": float(final_mrr.mean() - base_mrr.mean()),
        "corrected": int(np.sum((~base) & final)),
        "introduced": int(np.sum(base & (~final))),
        "p2b_usage_rate": float(use.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path,
        default=Path("data/validation/g8r_p2_listwise_cache.npz"),
    )
    parser.add_argument(
        "--artifact", type=Path,
        default=Path("data/validation/g8r_p2b_rank_fusion.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/validation/g8r_p2c_structure_safe_router.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    for path in (args.cache, args.artifact):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise RuntimeError(f"fail-closed: output exists: {args.output}")
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    configuration_body = artifact.get("frozen_configuration", artifact.get("configuration"))
    if configuration_body is None:
        raise RuntimeError("frozen P2b configuration missing")
    configuration = fusion_configuration_from_mapping(configuration_body)
    cache = Cache(args.cache)
    baseline, predictions = precompute_predictions(cache, [configuration])
    p2b_top1 = predictions["top1"][0]
    p2b_mrr = predictions["mrr"][0]
    use_p2b = ~cache.query_has_near
    final_top1 = np.where(use_p2b, p2b_top1, baseline["top1"])
    full = np.ones(cache.n_queries, dtype=bool)
    near = cache.query_has_near
    nonnear = ~near
    ci = bootstrap(
        cache.query_formula, baseline["top1"], final_top1,
        args.bootstrap, args.seed,
    )
    report = {
        "status": "g8r_p2c_structure_safe_router_complete",
        "formal": True,
        "router": {
            "rule": "use frozen P2b iff strict candidate group contains no MCES 0-2 pair",
            "uses_truth": False,
            "uses_candidate_structures": True,
            "threshold_selected_here": False,
            "frozen_p2b_configuration": configuration_body,
            "configuration_key": config_key(configuration),
        },
        "overall": metrics(
            baseline["top1"], baseline["mrr"], p2b_top1, p2b_mrr,
            use_p2b, full,
        ),
        "near_candidate_groups": metrics(
            baseline["top1"], baseline["mrr"], p2b_top1, p2b_mrr,
            use_p2b, near,
        ),
        "nonnear_candidate_groups": metrics(
            baseline["top1"], baseline["mrr"], p2b_top1, p2b_mrr,
            use_p2b, nonnear,
        ),
        "formula_cluster_bootstrap": ci,
        "gates": {
            "overall_positive": ci["ci_low"] > 0,
            "near_nondegrading": bool(np.all(final_top1[near] == baseline["top1"][near])),
            "corrected_gt_introduced": int(np.sum((~baseline["top1"]) & final_top1))
            > int(np.sum(baseline["top1"] & (~final_top1))),
        },
        "contracts": {
            "P2b": "frozen; not refit",
            "near_definition": "MCES grade 0-2 from preregistered pairs cache",
            "deployment": "candidate structures are required",
            "P3": "not opened by this script",
        },
        "provenance": {
            "cache_sha256": sha256(args.cache),
            "p2b_artifact_sha256": sha256(args.artifact),
        },
        "claim_limit": (
            "P2 development audit. This freezes a future router; a new sealed "
            "external panel is required for replacement/SOTA claims."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
