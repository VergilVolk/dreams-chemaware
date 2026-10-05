"""Audit a shared PSD kernel from rule-center response minus local background.

Every spectrum is mapped independently.  The primary feature is the normalized
difference between the curated rule response and three shared-coordinate local
background responses.  Each background is also treated as a pseudo-center and
given its own discovery-selected mass/rule weight as a matched negative arm.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from audit_chemaware_independent_mass_rule_mkl import fused_ranks, paired
from audit_chemaware_mass_kernel_embedding import KernelCache, metric, score_queries
from chemaware_iceberg_direct_core import stable_formula_folds
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import formula_bootstrap, identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
RULE_ARMS = {
    "centered_local": "rule_response_centered_local",
    "centered_positive": "rule_response_centered_positive",
    "raw_rule": "rule_response",
    "local_contrast_a": "rule_response_local_contrast_a",
    "local_contrast_b": "rule_response_local_contrast_b",
    "local_contrast_c": "rule_response_local_contrast_c",
}


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
        "--rule-library", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_centered_local_witness_kernel_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--discovery-natural-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--multi-bin-widths", type=float, nargs="+", default=(0.01, 0.02, 0.05))
    parser.add_argument("--uniform-channel-weight", type=float, default=1.0)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--rule-channel-weight", type=float, default=1.0)
    parser.add_argument(
        "--beta", type=float, nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def ranks_for_variant(
    scored: dict[str, np.ndarray], variant: str,
    mass_beta: float, rule_beta: float,
) -> np.ndarray:
    alias = dict(scored)
    alias["rule_response"] = scored[variant]
    return fused_ranks(alias, mass_beta, rule_beta)


def select_arm(
    scored: dict[str, np.ndarray], variant: str, beta: tuple[float, ...] | list[float],
) -> tuple[dict[str, object], dict[tuple[float, float], np.ndarray]]:
    grid: list[dict[str, object]] = []
    ranks: dict[tuple[float, float], np.ndarray] = {}
    for mass_beta in beta:
        for rule_beta in beta:
            key = (float(mass_beta), float(rule_beta))
            rank = ranks_for_variant(scored, variant, *key)
            ranks[key] = rank
            result = metric(scored["old_rank"], rank)
            grid.append({
                "mass_beta": key[0], "rule_beta": key[1], **result,
                "risk_utility_at_1": int(result["corrected"] - 2 * result["introduced"]),
            })
    selected = max(
        grid,
        key=lambda row: (
            int(row["risk_utility_at_1"]), -int(row["introduced"]),
            float(row["delta_mrr"]),
            -(float(row["mass_beta"]) + float(row["rule_beta"])),
        ),
    )
    return {"selected": selected, "grid": grid}, ranks


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.smoke:
        args.discovery_natural_identities = min(args.discovery_natural_identities, 192)
        args.max_inner_identities = 128
        args.bootstrap_draws = min(args.bootstrap_draws, 300)
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    discovery_pool = np.flatnonzero((fold != args.inner_fold) & (fold != args.outer_fold))
    inner_pool = np.flatnonzero(fold == args.inner_fold)
    discovery = identity_balanced_queries(
        discovery_pool, body["query_ik14"], np.random.default_rng(args.seed + 72),
        args.discovery_natural_identities,
    )
    inner = identity_balanced_queries(
        inner_pool, body["query_ik14"], np.random.default_rng(args.seed + 19),
        args.max_inner_identities,
    )
    discovery_formula = body["query_formula"][discovery].astype(str)
    inner_formula = body["query_formula"][inner].astype(str)
    if set(discovery_formula) & set(inner_formula):
        raise RuntimeError("discovery and inner formulas overlap")
    variants = ("mass", *tuple(RULE_ARMS.values()))
    cache = KernelCache(args, row_position, variants=variants)
    discovery_scored = score_queries(discovery, body, official, row_position, cache, variants)
    print(f"scored discovery queries={len(discovery)} rows={len(cache.cache)}", flush=True)
    inner_scored = score_queries(inner, body, official, row_position, cache, variants)
    print(f"scored inner queries={len(inner)} rows={len(cache.cache)}", flush=True)

    selections: dict[str, dict[str, object]] = {}
    inner_results: dict[str, dict[str, object]] = {}
    inner_ranks: dict[str, np.ndarray] = {}
    for name, variant in RULE_ARMS.items():
        selection, _ = select_arm(discovery_scored, variant, args.beta)
        selections[name] = selection
        chosen = selection["selected"]
        rank = ranks_for_variant(
            inner_scored, variant,
            float(chosen["mass_beta"]), float(chosen["rule_beta"]),
        )
        inner_ranks[name] = rank
        result = metric(inner_scored["old_rank"], rank)
        delta = (rank == 1).astype(float) - (inner_scored["old_rank"] == 1).astype(float)
        inner_results[name] = {
            **result,
            "risk_utility_at_1": int(result["corrected"] - 2 * result["introduced"]),
            **formula_bootstrap(delta, inner_formula, args.seed + 500, args.bootstrap_draws),
        }
    primary = inner_ranks["centered_local"]
    comparisons = {
        f"centered_local_minus_{name}": paired(
            primary, rank, inner_formula, args.seed + 600 + index, args.bootstrap_draws,
        )
        for index, (name, rank) in enumerate(inner_ranks.items())
        if name != "centered_local"
    }
    primary_result = inner_results["centered_local"]
    report = {
        "status": "CHEMAWARE_CENTERED_LOCAL_WITNESS_KERNEL_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "shared_embedding_result": True,
        "scope": "natural folds 0-2 selection; used inner fold 3; outer fold 4 sealed",
        "claim_limit": "Frozen explicit shared-kernel development result, not DreaMS parameter fine-tuning or external confirmation.",
        "method": {
            "primary_map": "normalize(rule_response - mean(local_background_0.071,0.137,0.223))",
            "shared_coordinate_nulls": [0.071, 0.137, 0.223],
            "candidate_free_feature_map": True,
            "same_feature_function_for_query_and_reference": True,
            "euclidean_psd_by_construction": True,
            "embedding_dimension": int(official.shape[1] + args.kernel_dim + len(cache.nl_rules) + len(cache.cf_rules)),
        },
        "data": {
            "discovery_queries": int(len(discovery)), "inner_queries": int(len(inner)),
            "outer_queries_untouched": int(np.sum(fold == args.outer_fold)),
            "cached_rows": int(len(cache.cache)), "formula_overlap": 0,
        },
        "selection_on_discovery": selections,
        "held_inner": inner_results,
        "paired_inner": comparisons,
        "gates": {
            "absolute_formula_ci_positive": primary_result["formula_cluster_bootstrap_95ci"][0] > 0,
            "corrected_exceeds_twice_introduced": primary_result["corrected"] > 2 * primary_result["introduced"],
            "beats_raw_rule_ci": comparisons["centered_local_minus_raw_rule"]["formula_cluster_bootstrap_95ci"][0] > 0,
            "beats_positive_center_ci": comparisons["centered_local_minus_centered_positive"]["formula_cluster_bootstrap_95ci"][0] > 0,
            **{
                f"beats_{name}_ci": comparisons[f"centered_local_minus_{name}"]["formula_cluster_bootstrap_95ci"][0] > 0
                for name in ("local_contrast_a", "local_contrast_b", "local_contrast_c")
            },
            "outer_fold_untouched": True,
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "rule_library_sha256": sha256_file(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_centered_kernel_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_ranks.npz", query=inner, formula=inner_formula,
            baseline_rank=inner_scored["old_rank"],
            **{f"{name}_rank": rank for name, rank in inner_ranks.items()},
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"],
        "selected": {name: body["selected"] for name, body in selections.items()},
        "held_inner": inner_results, "paired_inner": comparisons,
        "gates": report["gates"], "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
