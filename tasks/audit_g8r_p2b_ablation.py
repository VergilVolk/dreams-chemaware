"""Locked P2b ablation on the consumed P2 development cache.

This audit does not select or alter the frozen P2b artifact.  It quantifies the
increment over each constituent score and compares the nested-OOF procedure
with the strongest single feature using paired formula-cluster bootstrap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from g8r_p2_listwise_core import deterministic_formula_fold
from g8r_p2_rank_fusion_core import (
    FusionConfiguration,
    fusion_configuration_from_mapping,
)
from train_g8r_p2b_rank_fusion import (
    Cache,
    metrics_for,
    pooled_metrics,
    precompute_predictions,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "data/validation/g8r_p2_listwise_cache.npz"
DEFAULT_ARTIFACT = ROOT / "data/validation/g8r_p2b_rank_fusion.json"
DEFAULT_SELECTION = ROOT / "data/validation/g8r_p2b_rank_fusion.selection.json"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_p2b_ablation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def canonical_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar(corrected: int, introduced: int) -> float:
    discordant = corrected + introduced
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(corrected, introduced) + 1))
    return float(min(1.0, 2.0 * tail / (2 ** discordant)))


def paired_formula_bootstrap(
    formulas: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    grouped: dict[str, np.ndarray] = {}
    formulas = np.asarray(formulas, dtype=object)
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    for formula in sorted(set(map(str, formulas))):
        grouped[formula] = difference[formulas == formula]
    names = sorted(grouped)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        indices = rng.integers(0, len(names), len(names))
        values = np.concatenate([grouped[names[index]] for index in indices])
        draws[iteration] = values.mean()
    return {
        "mean": float(difference.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    for path in (args.cache, args.artifact, args.selection):
        if not path.is_file():
            raise FileNotFoundError(path)
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if artifact.get("status") != "g8r_p2b_rank_fusion_frozen":
        raise RuntimeError("P2b artifact is not frozen")
    if selection.get("status") != "g8r_p2b_selection_passed" or not selection.get("gates", {}).get("pass"):
        raise RuntimeError("P2b selection did not pass")
    if artifact.get("selection_report_sha256") != canonical_hash(args.selection):
        raise RuntimeError("artifact/selection hash mismatch")
    if artifact.get("cache_sha256") != sha256_file(args.cache):
        raise RuntimeError("artifact/cache hash mismatch")
    if artifact.get("p3_used_for_training_or_selection") is not False:
        raise RuntimeError("artifact does not explicitly exclude P3")

    named = {
        "sqrt_cosine_only": FusionConfiguration("absolute", (0.0, 1.0, 0.0, 0.0), 0, 0.0),
        "entropy_only": FusionConfiguration("absolute", (0.0, 0.0, 1.0, 0.0), 0, 0.0),
        "neutral_loss_only": FusionConfiguration("absolute", (0.0, 0.0, 0.0, 1.0), 0, 0.0),
        "dreams_neutral_loss_10_90": FusionConfiguration("absolute", (0.1, 0.0, 0.0, 0.9), 1, 0.0),
        "frozen_full_development": fusion_configuration_from_mapping(
            artifact["configuration"]
        ),
    }
    outer_rows = selection["outer_folds"]
    if sorted(int(row["outer_fold"]) for row in outer_rows) != list(range(args.folds)):
        raise RuntimeError("selection report does not contain exactly one row per outer fold")
    outer_configurations = [
        fusion_configuration_from_mapping(row["selected_configuration"])
        for row in outer_rows
    ]
    unique: list[FusionConfiguration] = []
    for configuration in [*named.values(), *outer_configurations]:
        if configuration not in unique:
            unique.append(configuration)
    index = {configuration: position for position, configuration in enumerate(unique)}

    cache = Cache(args.cache)
    baseline, predictions = precompute_predictions(cache, unique)
    all_mask = np.ones(cache.n_queries, dtype=bool)
    methods = {
        name: metrics_for(cache, baseline, predictions, index[configuration], all_mask)
        for name, configuration in named.items()
    }
    fold = np.asarray([
        deterministic_formula_fold(str(formula), args.folds) for formula in cache.query_formula
    ])
    selected_by_query = np.full(cache.n_queries, -1, dtype=np.int32)
    for row, configuration in zip(outer_rows, outer_configurations):
        selected_by_query[fold == int(row["outer_fold"])] = index[configuration]
    if np.any(selected_by_query < 0):
        raise RuntimeError("OOF selection mapping is incomplete")
    oof_metrics, _ = pooled_metrics(cache, baseline, predictions, selected_by_query)
    methods["nested_oof_p2b"] = oof_metrics

    baseline_top1 = baseline["top1"]
    method_top1: dict[str, np.ndarray] = {}
    for name, configuration in named.items():
        method_top1[name] = predictions["top1"][index[configuration]]
    nested_top1 = np.asarray([
        predictions["top1"][selected_by_query[query], query]
        for query in range(cache.n_queries)
    ], dtype=bool)
    method_top1["nested_oof_p2b"] = nested_top1

    comparisons = {}
    for name, values in method_top1.items():
        corrected = int(np.sum((~baseline_top1) & values))
        introduced = int(np.sum(baseline_top1 & (~values)))
        comparisons[f"{name}_vs_dreams"] = {
            "paired_formula_bootstrap": paired_formula_bootstrap(
                cache.query_formula, values, baseline_top1, args.bootstrap, args.seed,
            ),
            "corrected": corrected,
            "introduced": introduced,
            "mcnemar_exact_p": exact_mcnemar(corrected, introduced),
        }
    neutral = method_top1["neutral_loss_only"]
    corrected = int(np.sum((~neutral) & nested_top1))
    introduced = int(np.sum(neutral & (~nested_top1)))
    comparisons["nested_oof_p2b_vs_neutral_loss_only"] = {
        "paired_formula_bootstrap": paired_formula_bootstrap(
            cache.query_formula, nested_top1, neutral, args.bootstrap, args.seed + 1,
        ),
        "corrected": corrected,
        "introduced": introduced,
        "mcnemar_exact_p": exact_mcnemar(corrected, introduced),
    }

    report = {
        "status": "g8r_p2b_locked_ablation_complete",
        "artifact_sha256": canonical_hash(args.artifact),
        "selection_sha256": canonical_hash(args.selection),
        "cache_sha256": sha256_file(args.cache),
        "methods": methods,
        "comparisons": comparisons,
        "p3_read": False,
        "interpretation_rule": (
            "P2b adds evidence beyond neutral-loss-only only if the paired formula-cluster "
            "CI for nested_oof_p2b_vs_neutral_loss_only is above zero."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
