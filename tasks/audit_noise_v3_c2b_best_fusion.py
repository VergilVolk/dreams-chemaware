"""Formula-OOF incremental token fusion on top of the frozen P2b strategy.

This is a no-training strategy audit.  It first reproduces frozen P2b on the
complete P3-disjoint graph, then asks whether a small candidate-conditioned
token residual can improve P2b without sacrificing near retrieval or safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from build_g8r_real_error_atlas import Cache
from g8r_p2_listwise_core import deterministic_formula_fold
from g8r_p2_rank_fusion_core import (
    fuse_one_query, fusion_configuration_from_mapping, grouped_max,
    normalize_pair_features, strict_rank, unique_top_index,
)


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TokenFusion:
    weight: float
    maximum_p2b_top_gap: float
    minimum_advantage: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c2b_token_pair_cache.npz")
    p.add_argument("--artifact", type=Path, default=ROOT / "data/validation/g8r_p2b_rank_fusion.json")
    p.add_argument("--output", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c2b_best_fusion.json")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260825)
    return p.parse_args()


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block): digest.update(chunk)
    return digest.hexdigest()


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    return (values - low) / (high - low) if high - low > 1e-12 else np.zeros_like(values)


def metrics(base: np.ndarray, final: np.ndarray, near: np.ndarray, mask: np.ndarray) -> dict:
    b, f, n = base[mask], final[mask], near[mask]
    return {
        "queries": int(mask.sum()), "baseline_recall1": float(b.mean()),
        "recall1": float(f.mean()), "delta_recall1": float(f.mean() - b.mean()),
        "corrected": int(np.sum((~b) & f)), "introduced": int(np.sum(b & (~f))),
        "risk_net_lambda2": int(np.sum((~b) & f) - 2 * np.sum(b & (~f))),
        "near_queries": int(n.sum()),
        "baseline_near_recall1": float(b[n].mean()) if n.any() else None,
        "near_recall1": float(f[n].mean()) if n.any() else None,
        "near_delta_recall1": float(f[n].mean() - b[n].mean()) if n.any() else None,
    }


def cluster_ci(formulas: np.ndarray, contribution: np.ndarray, n: int, seed: int) -> dict:
    unique = np.unique(formulas.astype(str)); groups = [contribution[formulas.astype(str) == value] for value in unique]
    rng = np.random.default_rng(seed); draws = np.empty(n, float)
    for iteration in range(n):
        sampled = rng.integers(0, len(groups), len(groups))
        draws[iteration] = np.concatenate([groups[index] for index in sampled]).mean()
    return {"mean": float(contribution.mean()), "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5))}


def main() -> None:
    args = parse_args()
    for path in (args.cache, args.cache.with_suffix(".json"), args.artifact):
        if not path.is_file(): raise FileNotFoundError(path)
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite {args.output}")
    cache_report = json.loads(args.cache.with_suffix(".json").read_text(encoding="utf-8"))
    if not cache_report.get("formal") or not cache_report.get("gates", {}).get("pass"):
        raise RuntimeError("formal C2-B0 cache has not passed")
    cache = Cache(args.cache)
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    frozen = fusion_configuration_from_mapping(artifact["configuration"])
    selected_names = list(map(str, artifact["selected_features"]))
    expected = ["dreams_similarity", "sqrt_cosine", "entropy_similarity", "neutral_loss_sqrt_cosine"]
    if selected_names != expected: raise RuntimeError(f"unexpected P2b schema: {selected_names}")
    selected = cache.features[:, [cache.feature_names.index(name) for name in selected_names]]
    token = cache.features[:, cache.feature_names.index("token_cosine_weighted")]
    query_pair_ptr = cache.molecule_ptr[cache.query_ptr]
    normalized = normalize_pair_features(selected, query_pair_ptr, frozen.normalization)
    p2b_top1, p2b_top_gap, official_top1, near = [], [], [], []
    p2b_molecule_blocks, token_molecule_blocks = [], []
    for query in range(cache.n_queries):
        ml, mr = map(int, cache.query_ptr[query:query + 2]); pl, pr = int(cache.molecule_ptr[ml]), int(cache.molecule_ptr[mr])
        ptr = cache.molecule_ptr[ml:mr + 1] - pl
        p2b, _, _ = fuse_one_query(
            normalized[pl:pr], selected[pl:pr, 0], ptr,
            np.asarray(frozen.weights), (1, 2, 3), frozen.min_support, frozen.min_advantage,
        )
        official = grouped_max(selected[pl:pr, 0], ptr)
        token_molecule = grouped_max((token[pl:pr] + 1.0) / 2.0, ptr)
        orank, _, _ = strict_rank(official, 0); prank, _, _ = strict_rank(p2b, 0)
        sorted_p2b = np.sort(np.asarray(p2b, dtype=np.float64))
        # Deployment-available confidence only: score gap between the model's
        # top two candidates.  It never references the positive molecule.
        top_gap = float(sorted_p2b[-1] - sorted_p2b[-2])
        official_top1.append(orank == 1); p2b_top1.append(prank == 1); p2b_top_gap.append(top_gap)
        near.append(bool(cache.query_has_near[query])); p2b_molecule_blocks.append(minmax(p2b)); token_molecule_blocks.append(minmax(token_molecule))
        if (query + 1) % 2000 == 0 or query + 1 == cache.n_queries:
            print(f"[C2-B best] prepared {query + 1:,}/{cache.n_queries:,}", flush=True)
    official_top1 = np.asarray(official_top1, bool); p2b_top1 = np.asarray(p2b_top1, bool)
    p2b_top_gap = np.asarray(p2b_top_gap, float); near = np.asarray(near, bool)

    configs = [TokenFusion(0.0, float("inf"), 0.0)] + [TokenFusion(weight, top_gap, advantage)
               for weight in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
               for top_gap in (0.005, 0.01, 0.02, 0.05, 0.10, float("inf"))
               for advantage in (0.0, 0.005, 0.01, 0.02)]
    predictions = np.empty((len(configs), cache.n_queries), bool)
    interventions = np.zeros_like(predictions)
    for ci, config in enumerate(configs):
        for query, (p2b, tok) in enumerate(zip(p2b_molecule_blocks, token_molecule_blocks)):
            combined = (1.0 - config.weight) * p2b + config.weight * tok
            base_top, new_top = unique_top_index(p2b), unique_top_index(combined)
            advantage = (0.0 if new_top is None or base_top is None or new_top == base_top
                         else float(combined[new_top] - combined[base_top]))
            use = (p2b_top_gap[query] <= config.maximum_p2b_top_gap + 1e-12
                   and new_top is not None and new_top != base_top
                   and advantage + 1e-12 >= config.minimum_advantage)
            rank, _, _ = strict_rank(combined if use else p2b, 0)
            predictions[ci, query] = rank == 1; interventions[ci, query] = use

    folds = np.asarray([deterministic_formula_fold(str(value), args.folds) for value in cache.query_formula])
    for held in range(args.folds):
        if set(map(str, cache.query_formula[folds == held])) & set(map(str, cache.query_formula[folds != held])):
            raise RuntimeError("formula leakage across C2-B fusion folds")
    oof = np.empty(cache.n_queries, bool); oof_intervention = np.empty(cache.n_queries, bool); chosen = []
    for held in range(args.folds):
        train = folds != held; test = folds == held
        best_index, best_key = 0, None
        for ci in range(len(configs)):
            value = metrics(p2b_top1, predictions[ci], near, train)
            safe = value["near_delta_recall1"] >= 0 and value["introduced"] <= value["corrected"]
            key = (int(safe), value["delta_recall1"], value["risk_net_lambda2"], -value["introduced"])
            if best_key is None or key > best_key: best_index, best_key = ci, key
        oof[test] = predictions[best_index, test]; oof_intervention[test] = interventions[best_index, test]
        chosen.append({"held_formula_fold": held, "configuration": asdict(configs[best_index]),
                       "training_metrics": metrics(p2b_top1, predictions[best_index], near, train),
                       "held_metrics": metrics(p2b_top1, predictions[best_index], near, test)})

    all_mask = np.ones(cache.n_queries, bool)
    p2b_vs_official = metrics(official_top1, p2b_top1, near, all_mask)
    token_vs_p2b = metrics(p2b_top1, oof, near, all_mask)
    final_vs_official = metrics(official_top1, oof, near, all_mask)
    formulas = np.asarray(cache.query_formula, object)
    report = {
        "status": "noise_v3_c2b_best_fusion_complete", "protocol": "formula-OOF token increment on frozen P2b",
        "queries": int(cache.n_queries), "formulas": int(len(np.unique(formulas))),
        "frozen_p2b_vs_official": p2b_vs_official,
        "token_increment_vs_p2b": token_vs_p2b,
        "final_strategy_vs_official": final_vs_official,
        "token_increment_formula_cluster_ci": cluster_ci(formulas, oof.astype(float) - p2b_top1.astype(float), args.bootstrap, args.seed),
        "final_vs_official_formula_cluster_ci": cluster_ci(formulas, oof.astype(float) - official_top1.astype(float), args.bootstrap, args.seed + 1),
        "intervention_rate": float(oof_intervention.mean()), "fold_selection": chosen,
        "gates": {
            "p2b_reproduced_positive": bool(p2b_vs_official["delta_recall1"] > 0),
            "token_increment_positive": bool(token_vs_p2b["delta_recall1"] > 0),
            "token_increment_ci_positive": False,
            "final_near_nonnegative_vs_official": bool(final_vs_official["near_delta_recall1"] >= 0),
            "final_corrected_ge_introduced": bool(final_vs_official["corrected"] >= final_vs_official["introduced"]),
        },
        "claim_limit": "Training-graph formula-OOF strategy audit; not sealed-P3 efficacy.",
        "provenance": {"cache_sha256": sha256_file(args.cache), "artifact_sha256": sha256_file(args.artifact)},
    }
    report["gates"]["token_increment_ci_positive"] = bool(report["token_increment_formula_cluster_ci"]["ci_low"] > 0)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
