"""P2b: nested formula-isolated, molecule-listwise rank fusion.

This is the deliberately simple successor to the P2 neural residual model.
It combines four deployment-available scores at the spectrum-pair level:
DreaMS cosine, modified cosine, entropy similarity and neutral-loss cosine.
Hyperparameters are selected only on the outer-training formulas.  The sealed
P3 manifests are neither accepted nor read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from g8r_p2_listwise_core import deterministic_formula_fold
from g8r_p2_rank_fusion_core import (
    FusionConfiguration,
    grouped_max,
    normalize_pair_features,
    strict_rank,
    unique_top_index,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "data/validation/g8r_p2_listwise_cache.npz"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_p2b_rank_fusion.json"
SELECTED_FEATURES = (
    "dreams_similarity",
    "sqrt_cosine",
    "entropy_similarity",
    "neutral_loss_sqrt_cosine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--weight-denominator", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


class Cache:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=True) as body:
            for name in body.files:
                setattr(self, name, body[name])
        self.features = np.asarray(self.features, dtype=np.float64)
        self.feature_names = list(map(str, self.feature_names))
        self.query_ptr = np.asarray(self.query_ptr, dtype=np.int64)
        self.molecule_ptr = np.asarray(self.molecule_ptr, dtype=np.int64)
        self.query_formula = np.asarray(self.query_formula, dtype=object)
        self.query_has_near = np.asarray(self.query_has_near, dtype=bool)
        self.n_queries = len(self.query_ptr) - 1
        if len(self.query_formula) != self.n_queries or len(self.query_has_near) != self.n_queries:
            raise RuntimeError("query metadata is not aligned")
        if self.query_ptr[0] != 0 or self.query_ptr[-1] != len(self.molecule_label):
            raise RuntimeError("query_ptr is invalid")
        if self.molecule_ptr[0] != 0 or self.molecule_ptr[-1] != len(self.features):
            raise RuntimeError("molecule_ptr is invalid")
        if np.any(np.diff(self.query_ptr) < 2) or np.any(np.diff(self.molecule_ptr) < 1):
            raise RuntimeError("every query needs >=2 molecules and every molecule >=1 spectrum pair")
        missing = [name for name in SELECTED_FEATURES if name not in self.feature_names]
        if missing:
            raise RuntimeError(f"cache is missing P2b features: {missing}")


def weight_grid(denominator: int) -> list[tuple[float, ...]]:
    if denominator < 2:
        raise ValueError("weight denominator must be >=2")
    values = []
    for a in range(denominator + 1):
        for b in range(denominator - a + 1):
            for c in range(denominator - a - b + 1):
                d = denominator - a - b - c
                values.append(tuple(value / denominator for value in (a, b, c, d)))
    return values


def configuration_grid(denominator: int) -> list[FusionConfiguration]:
    gates = ((0, 0.0), (1, 0.0), (2, 0.0), (2, 0.05))
    configurations = []
    for normalization in ("absolute", "query_minmax"):
        for weights in weight_grid(denominator):
            # Pure DreaMS is the common baseline, not a candidate intervention.
            if weights == (1.0, 0.0, 0.0, 0.0):
                continue
            for min_support, min_advantage in gates:
                configurations.append(FusionConfiguration(
                    normalization=normalization,
                    weights=weights,
                    min_support=min_support,
                    min_advantage=min_advantage,
                ))
    return configurations


def config_key(configuration: FusionConfiguration) -> str:
    weights = "-".join(f"{value:.2f}" for value in configuration.weights)
    return (f"{configuration.normalization}|w={weights}|"
            f"support={configuration.min_support}|adv={configuration.min_advantage:.2f}")


def precompute_predictions(cache: Cache, configurations: list[FusionConfiguration]):
    selected_indices = [cache.feature_names.index(name) for name in SELECTED_FEATURES]
    selected_pairs = cache.features[:, selected_indices]
    query_pair_ptr = cache.molecule_ptr[cache.query_ptr]
    baseline_molecule = grouped_max(cache.features[:, selected_indices[0]], cache.molecule_ptr)
    n_config = len(configurations)
    n_query = cache.n_queries
    top1 = np.empty((n_config, n_query), dtype=bool)
    mrr = np.empty((n_config, n_query), dtype=np.float32)
    margin = np.empty((n_config, n_query), dtype=np.float32)
    intervened = np.empty((n_config, n_query), dtype=bool)

    baseline_top1 = np.empty(n_query, dtype=bool)
    baseline_mrr = np.empty(n_query, dtype=np.float32)
    baseline_margin = np.empty(n_query, dtype=np.float32)
    baseline_top = np.empty(n_query, dtype=np.int16)
    for query, (left, right) in enumerate(zip(cache.query_ptr[:-1], cache.query_ptr[1:])):
        rank, reciprocal, gap = strict_rank(baseline_molecule[left:right], 0)
        baseline_top1[query] = rank == 1
        baseline_mrr[query] = reciprocal
        baseline_margin[query] = gap
        winner = unique_top_index(baseline_molecule[left:right])
        baseline_top[query] = -1 if winner is None else winner

    by_normalization: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for normalization in {configuration.normalization for configuration in configurations}:
        normalized_pairs = normalize_pair_features(selected_pairs, query_pair_ptr, normalization)
        raw_molecule = np.stack(
            [grouped_max(normalized_pairs[:, column], cache.molecule_ptr) for column in (1, 2, 3)],
            axis=1,
        )
        raw_top = np.full((n_query, 3), -1, dtype=np.int16)
        for query, (left, right) in enumerate(zip(cache.query_ptr[:-1], cache.query_ptr[1:])):
            for raw_column in range(3):
                winner = unique_top_index(raw_molecule[left:right, raw_column])
                raw_top[query, raw_column] = -1 if winner is None else winner
        by_normalization[normalization] = (normalized_pairs, raw_top)

    for index, configuration in enumerate(configurations):
        normalized_pairs, raw_top = by_normalization[configuration.normalization]
        fused_pair = normalized_pairs @ np.asarray(configuration.weights, dtype=np.float64)
        fused_molecule = grouped_max(fused_pair, cache.molecule_ptr)
        for query, (left, right) in enumerate(zip(cache.query_ptr[:-1], cache.query_ptr[1:])):
            fused_block = fused_molecule[left:right]
            fused_winner = unique_top_index(fused_block)
            support = 0 if fused_winner is None else int(np.sum(raw_top[query] == fused_winner))
            if fused_winner is None:
                advantage = -np.inf
            elif baseline_top[query] < 0:
                advantage = np.inf
            elif fused_winner == baseline_top[query]:
                advantage = 0.0
            else:
                advantage = float(fused_block[fused_winner] - fused_block[baseline_top[query]])
            use = (support >= configuration.min_support
                   and advantage + 1e-12 >= configuration.min_advantage)
            scores = fused_block if use else baseline_molecule[left:right]
            rank, reciprocal, gap = strict_rank(scores, 0)
            top1[index, query] = rank == 1
            mrr[index, query] = reciprocal
            margin[index, query] = gap
            intervened[index, query] = use
        if (index + 1) % 200 == 0 or index + 1 == n_config:
            print(f"[fusion] {index + 1:,}/{n_config:,} configurations", flush=True)
    baseline = {
        "top1": baseline_top1,
        "mrr": baseline_mrr,
        "margin": baseline_margin,
    }
    predictions = {"top1": top1, "mrr": mrr, "margin": margin, "intervened": intervened}
    return baseline, predictions


def metrics_for(
    cache: Cache,
    baseline: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    configuration_index: int,
    query_mask: np.ndarray,
) -> dict:
    query_mask = np.asarray(query_mask, dtype=bool)
    base = baseline["top1"][query_mask]
    final = predictions["top1"][configuration_index, query_mask]
    base_mrr = baseline["mrr"][query_mask]
    final_mrr = predictions["mrr"][configuration_index, query_mask]
    near = cache.query_has_near[query_mask]
    result = {
        "n_queries": int(query_mask.sum()),
        "baseline_recall1": float(base.mean()),
        "recall1": float(final.mean()),
        "delta_recall1": float(final.mean() - base.mean()),
        "baseline_mrr": float(base_mrr.mean()),
        "mrr": float(final_mrr.mean()),
        "delta_mrr": float(final_mrr.mean() - base_mrr.mean()),
        "corrected": int(np.sum((~base) & final)),
        "introduced": int(np.sum(base & (~final))),
        "intervention_rate": float(predictions["intervened"][configuration_index, query_mask].mean()),
        "mean_margin_delta": float(np.mean(
            predictions["margin"][configuration_index, query_mask] - baseline["margin"][query_mask]
        )),
    }
    if near.any():
        result.update({
            "n_near": int(near.sum()),
            "baseline_near_recall1": float(base[near].mean()),
            "near_recall1": float(final[near].mean()),
            "delta_near_recall1": float(final[near].mean() - base[near].mean()),
        })
    else:
        result.update({"n_near": 0, "baseline_near_recall1": None,
                       "near_recall1": None, "delta_near_recall1": None})
    return result


def selection_tuple(metrics: dict) -> tuple[float, ...]:
    near = metrics["delta_near_recall1"]
    safe = (metrics["delta_recall1"] >= 0 and near is not None and near >= 0
            and metrics["delta_mrr"] >= 0
            and metrics["corrected"] >= metrics["introduced"])
    return (
        float(safe),
        metrics["delta_recall1"],
        near if near is not None else -1.0,
        metrics["corrected"] - metrics["introduced"],
        metrics["delta_mrr"],
        -metrics["intervention_rate"],
    )


def choose_configuration(
    cache: Cache,
    baseline: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    query_mask: np.ndarray,
) -> tuple[int, dict]:
    best_index = -1
    best_metrics = None
    for index in range(len(predictions["top1"])):
        metrics = metrics_for(cache, baseline, predictions, index, query_mask)
        if best_metrics is None or selection_tuple(metrics) > selection_tuple(best_metrics):
            best_index, best_metrics = index, metrics
    if best_metrics is None:
        raise RuntimeError("no fusion configuration could be evaluated")
    return best_index, best_metrics


def pooled_metrics(cache: Cache, baseline: dict, predictions: dict,
                   selected_by_query: np.ndarray) -> tuple[dict, list[dict]]:
    final_top1 = np.empty(cache.n_queries, dtype=bool)
    final_mrr = np.empty(cache.n_queries, dtype=np.float32)
    final_margin = np.empty(cache.n_queries, dtype=np.float32)
    final_intervened = np.empty(cache.n_queries, dtype=bool)
    for query, configuration in enumerate(selected_by_query):
        final_top1[query] = predictions["top1"][configuration, query]
        final_mrr[query] = predictions["mrr"][configuration, query]
        final_margin[query] = predictions["margin"][configuration, query]
        final_intervened[query] = predictions["intervened"][configuration, query]
    base = baseline["top1"]
    near = cache.query_has_near
    records = [{
        "query": query,
        "formula": str(cache.query_formula[query]),
        "near": bool(near[query]),
        "base_top1": bool(base[query]),
        "top1": bool(final_top1[query]),
    } for query in range(cache.n_queries)]
    metrics = {
        "n_queries": cache.n_queries,
        "baseline_recall1": float(base.mean()),
        "recall1": float(final_top1.mean()),
        "delta_recall1": float(final_top1.mean() - base.mean()),
        "baseline_mrr": float(baseline["mrr"].mean()),
        "mrr": float(final_mrr.mean()),
        "delta_mrr": float(final_mrr.mean() - baseline["mrr"].mean()),
        "corrected": int(np.sum((~base) & final_top1)),
        "introduced": int(np.sum(base & (~final_top1))),
        "intervention_rate": float(final_intervened.mean()),
        "mean_margin_delta": float(np.mean(final_margin - baseline["margin"])),
        "n_near": int(near.sum()),
        "baseline_near_recall1": float(base[near].mean()),
        "near_recall1": float(final_top1[near].mean()),
        "delta_near_recall1": float(final_top1[near].mean() - base[near].mean()),
    }
    return metrics, records


def formula_cluster_ci(records: list[dict], n_bootstrap: int, seed: int) -> dict:
    groups: dict[str, list[float]] = {}
    for record in records:
        groups.setdefault(record["formula"], []).append(
            float(record["top1"]) - float(record["base_top1"])
        )
    formulas = sorted(groups)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for iteration in range(n_bootstrap):
        sampled = rng.integers(0, len(formulas), len(formulas))
        values = np.concatenate([np.asarray(groups[formulas[index]]) for index in sampled])
        draws[iteration] = values.mean()
    return {
        "mean": float(np.mean([value for group in groups.values() for value in group])),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def main() -> None:
    args = parse_args()
    selection_path = args.output.with_suffix(".selection.json")
    if (args.output.exists() or selection_path.exists()) and not args.overwrite:
        raise FileExistsError("P2b output already exists; refusing to overwrite")
    audit_path = args.cache.with_suffix(".json")
    if not args.cache.is_file() or not audit_path.is_file():
        raise FileNotFoundError("P2 cache or cache audit is missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "g8r_p2_listwise_cache_built":
        raise RuntimeError("invalid P2 cache audit")
    if audit.get("cache_sha256") != sha256_file(args.cache):
        raise RuntimeError("P2 cache hash mismatch")

    cache = Cache(args.cache)
    configurations = configuration_grid(args.weight_denominator)
    print(f"P2b: {cache.n_queries:,} queries; {len(configurations):,} preregistered fusions")
    baseline, predictions = precompute_predictions(cache, configurations)
    fold = np.asarray([
        deterministic_formula_fold(str(formula), args.folds) for formula in cache.query_formula
    ])
    for held in range(args.folds):
        train_formulas = set(map(str, cache.query_formula[fold != held]))
        test_formulas = set(map(str, cache.query_formula[fold == held]))
        if train_formulas & test_formulas:
            raise RuntimeError("formula leakage across outer folds")

    selected_by_query = np.full(cache.n_queries, -1, dtype=np.int32)
    fold_reports = []
    for held in range(args.folds):
        train_mask = fold != held
        test_mask = fold == held
        selected, train_metrics = choose_configuration(
            cache, baseline, predictions, train_mask,
        )
        test_metrics = metrics_for(cache, baseline, predictions, selected, test_mask)
        selected_by_query[test_mask] = selected
        fold_reports.append({
            "outer_fold": held,
            "selected_configuration_index": selected,
            "selected_configuration": asdict(configurations[selected]),
            "selection_on_outer_train": train_metrics,
            "evaluation_on_outer_test": test_metrics,
        })
        print(
            f"[outer {held}] config={config_key(configurations[selected])} "
            f"dR1={test_metrics['delta_recall1']:+.4f} "
            f"near={test_metrics['delta_near_recall1']:+.4f} "
            f"C/I={test_metrics['corrected']}/{test_metrics['introduced']}",
            flush=True,
        )
    if np.any(selected_by_query < 0):
        raise RuntimeError("outer OOF predictions are incomplete")

    oof, records = pooled_metrics(cache, baseline, predictions, selected_by_query)
    oof["formula_cluster_bootstrap"] = formula_cluster_ci(records, args.bootstrap, args.seed)
    final_index, final_dev_metrics = choose_configuration(
        cache, baseline, predictions, np.ones(cache.n_queries, dtype=bool),
    )
    every_fold_safe = all(
        report["evaluation_on_outer_test"]["delta_recall1"] >= 0
        and report["evaluation_on_outer_test"]["delta_near_recall1"] >= 0
        for report in fold_reports
    )
    gates = {
        "overall_gain_at_least_three_points": oof["delta_recall1"] >= 0.03,
        "near_recall1_nonnegative": oof["delta_near_recall1"] >= 0.0,
        "mrr_nonnegative": oof["delta_mrr"] >= 0.0,
        "corrected_ge_introduced": oof["corrected"] >= oof["introduced"],
        "formula_cluster_ci_positive": oof["formula_cluster_bootstrap"]["ci_low"] > 0.0,
        "every_outer_fold_overall_and_near_nonnegative": every_fold_safe,
    }
    gates["pass"] = all(gates.values())
    selection = {
        "status": "g8r_p2b_selection_passed" if gates["pass"] else "g8r_p2b_selection_failed",
        "cache_sha256": sha256_file(args.cache),
        "cache_audit_sha256": sha256_file(audit_path),
        "selected_features": list(SELECTED_FEATURES),
        "n_preregistered_configurations": len(configurations),
        "outer_formula_isolated_oof": oof,
        "outer_folds": fold_reports,
        "full_development_selected_configuration_index": final_index,
        "full_development_selected_configuration": asdict(configurations[final_index]),
        "full_development_metrics_post_selection": final_dev_metrics,
        "gates": gates,
        "sealed_p3_read": False,
        "important_limit": (
            "Outer OOF estimates the selection procedure. The full-development configuration "
            "is frozen only after the OOF gate passes; sealed P3 remains the final evidence."
        ),
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"oof": oof, "gates": gates}, ensure_ascii=False, indent=2))
    if not gates["pass"]:
        raise RuntimeError(f"P2b OOF gates failed; see {selection_path}")

    artifact = {
        "status": "g8r_p2b_rank_fusion_frozen",
        "configuration": asdict(configurations[final_index]),
        "selected_features": list(SELECTED_FEATURES),
        "cache_sha256": sha256_file(args.cache),
        "cache_audit_sha256": sha256_file(audit_path),
        "selection_report_sha256": sha256_file(selection_path),
        "selection_script_sha256": sha256_file(Path(__file__)),
        "outer_formula_isolated_oof": oof,
        "full_development_metrics_post_selection": final_dev_metrics,
        "p3_used_for_training_or_selection": False,
    }
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[frozen] {args.output}")


if __name__ == "__main__":
    main()

