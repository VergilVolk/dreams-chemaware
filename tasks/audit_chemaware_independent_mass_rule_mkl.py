"""Test independent physical-mass and curated-rule weights in one shared embedding."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from audit_chemaware_mass_kernel_embedding import KernelCache, metric, score_queries, strict_rank
from chemaware_iceberg_direct_core import stable_formula_folds
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import formula_bootstrap, identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]


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
        default=ROOT / "data/validation/chemaware_independent_mass_rule_mkl_v1",
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
    return parser.parse_args()


def fused_ranks(scored: dict, mass_beta: float, rule_beta: float) -> np.ndarray:
    output = np.empty(len(scored["query"]), dtype=np.int16)
    for index in range(len(output)):
        pair_score = (
            np.asarray(scored["global"][index], dtype=np.float32)
            + float(mass_beta) * np.asarray(scored["mass"][index], dtype=np.float32)
            + float(rule_beta) * np.asarray(scored["rule_response"][index], dtype=np.float32)
        )
        pointer = np.asarray(scored["reference_ptr"][index], dtype=np.int64)
        molecule_score = np.maximum.reduceat(pair_score, pointer[:-1])
        output[index] = strict_rank(
            molecule_score, np.asarray(scored["labels"][index], dtype=bool),
        )
    return output


def grid(scored: dict, beta: tuple[float, ...] | list[float]) -> tuple[list[dict], dict]:
    rows = []
    ranks = {}
    for mass_beta in beta:
        for rule_beta in beta:
            rank = fused_ranks(scored, mass_beta, rule_beta)
            key = (float(mass_beta), float(rule_beta))
            ranks[key] = rank
            item = metric(scored["old_rank"], rank)
            rows.append({
                "mass_beta": key[0], "rule_beta": key[1], **item,
                "risk_utility": int(item["corrected"] - 2 * item["introduced"]),
            })
    return rows, ranks


def paired(left: np.ndarray, right: np.ndarray, formula: np.ndarray, seed: int, draws: int) -> dict:
    delta = (left == 1).astype(float) - (right == 1).astype(float)
    return {
        "delta_recall1": float(delta.mean()),
        **formula_bootstrap(delta, formula, seed, draws),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    train_pool = np.flatnonzero((fold != args.inner_fold) & (fold != args.outer_fold))
    inner_pool = np.flatnonzero(fold == args.inner_fold)
    outer_pool = np.flatnonzero(fold == args.outer_fold)
    discovery = identity_balanced_queries(
        train_pool, body["query_ik14"], np.random.default_rng(args.seed + 72),
        args.discovery_natural_identities,
    )
    inner = identity_balanced_queries(
        inner_pool, body["query_ik14"], np.random.default_rng(args.seed + 19),
        args.max_inner_identities,
    )
    if set(body["query_formula"][discovery].astype(str)) & set(body["query_formula"][inner].astype(str)):
        raise RuntimeError("discovery and inner formulas overlap")
    cache = KernelCache(args, row_position, variants=("mass", "rule_response"))
    discovery_scored = score_queries(
        discovery, body, official, row_position, cache, ("mass", "rule_response"),
    )
    print(f"scored discovery queries={len(discovery)} rows={len(cache.cache)}", flush=True)
    inner_scored = score_queries(
        inner, body, official, row_position, cache, ("mass", "rule_response"),
    )
    print(f"scored inner queries={len(inner)} rows={len(cache.cache)}", flush=True)
    discovery_grid, _ = grid(discovery_scored, args.beta)
    selected = max(
        discovery_grid,
        key=lambda item: (
            int(item["risk_utility"]), -int(item["introduced"]), float(item["delta_mrr"]),
            -(float(item["mass_beta"]) + float(item["rule_beta"])),
        ),
    )
    inner_grid, inner_ranks = grid(inner_scored, args.beta)
    key = (float(selected["mass_beta"]), float(selected["rule_beta"]))
    selected_rank = inner_ranks[key]
    rule_key = max(
        (item for item in discovery_grid if item["mass_beta"] == 0.0),
        key=lambda item: (
            int(item["risk_utility"]), -int(item["introduced"]), float(item["delta_mrr"]),
            -float(item["rule_beta"]),
        ),
    )
    mass_key = max(
        (item for item in discovery_grid if item["rule_beta"] == 0.0),
        key=lambda item: (
            int(item["risk_utility"]), -int(item["introduced"]), float(item["delta_mrr"]),
            -float(item["mass_beta"]),
        ),
    )
    rule_rank = inner_ranks[(0.0, float(rule_key["rule_beta"]))]
    mass_rank = inner_ranks[(float(mass_key["mass_beta"]), 0.0)]
    all_rank = np.stack(list(inner_ranks.values()), axis=1)
    oracle_rank = np.min(all_rank, axis=1)
    formula = inner_scored["formula"].astype(str)
    selected_metric = metric(inner_scored["old_rank"], selected_rank)
    delta = (selected_rank == 1).astype(float) - (inner_scored["old_rank"] == 1).astype(float)
    absolute_ci = formula_bootstrap(delta, formula, args.seed + 500, args.bootstrap_draws)
    versus_rule = paired(selected_rank, rule_rank, formula, args.seed + 600, args.bootstrap_draws)
    versus_mass = paired(selected_rank, mass_rank, formula, args.seed + 700, args.bootstrap_draws)
    report = {
        "status": "CHEMAWARE_INDEPENDENT_MASS_RULE_MKL_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "natural training-formula selection; used inner fold 3; outer fold 4 untouched",
        "claim_limit": "Frozen shared-feature development result, not DreaMS fine-tuning or external confirmation.",
        "data": {
            "discovery_queries": int(len(discovery)), "inner_queries": int(len(inner)),
            "outer_queries_untouched": int(len(outer_pool)), "cached_rows": int(len(cache.cache)),
        },
        "shared_embedding": {
            "map": "[official, sqrt(mass_beta)*mass, sqrt(rule_beta)*rule_response]",
            "dimension": int(official.shape[1] + args.kernel_dim + len(cache.nl_rules) + len(cache.cf_rules)),
            "candidate_set_used_by_map": False, "pair_fusion_before_molecule_max": True,
        },
        "selection_on_discovery": {"selected": selected, "grid": discovery_grid},
        "held_inner": {
            "selected_independent_mkl": selected_metric,
            "rule_only": metric(inner_scored["old_rank"], rule_rank),
            "mass_only": metric(inner_scored["old_rank"], mass_rank),
            "no_op_aware_grid_oracle": {
                **metric(inner_scored["old_rank"], oracle_rank),
                "claim_limit": "weights selected per query after truth; headroom only",
            },
            "all_grid": inner_grid,
        },
        "held_inner_absolute_formula_bootstrap": absolute_ci,
        "paired_inner": {
            "independent_mkl_minus_rule_only": versus_rule,
            "independent_mkl_minus_mass_only": versus_mass,
        },
        "gates": {
            "absolute_ci_positive": absolute_ci["formula_cluster_bootstrap_95ci"][0] > 0,
            "increment_over_rule_only_ci_positive": versus_rule["formula_cluster_bootstrap_95ci"][0] > 0,
            "increment_over_mass_only_ci_positive": versus_mass["formula_cluster_bootstrap_95ci"][0] > 0,
            "outer_fold_untouched": True,
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "rule_library_sha256": sha256_file(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_independent_mkl_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_ranks.npz", query=inner, formula=formula,
            baseline_rank=inner_scored["old_rank"], selected_rank=selected_rank,
            rule_rank=rule_rank, mass_rank=mass_rank, oracle_rank=oracle_rank,
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "selected_weights": key,
        "selected": selected_metric,
        "rule_only": report["held_inner"]["rule_only"],
        "mass_only": report["held_inner"]["mass_only"],
        "oracle": report["held_inner"]["no_op_aware_grid_oracle"],
        "paired_inner": report["paired_inner"], "gates": report["gates"],
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
