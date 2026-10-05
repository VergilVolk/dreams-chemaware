"""Audit a formula-conditional Fisher metric over centered chemical rules."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_chemaware_empirical_rule_reliability_kernel import fused_ranks, grid, paired_formula_ci
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries
from audit_chemaware_observable_tangent_metric import formula_bootstrap, retrieval
from audit_chemaware_whitened_centered_rule_kernel import CENTER_NAMES, FourCenterCache, TransformCache
from audit_chemaware_within_identity_whitened_rule_kernel import (
    permute_identity_within_formula,
    row_identity_registry,
)
from chemaware_conditional_fisher_core import (
    fit_conditional_fisher_map,
    permute_identity_formulas,
)
from chemaware_iceberg_direct_core import stable_formula_folds
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
CONTROL_ARMS = ("local_a", "local_b", "local_c", "identity_permuted", "formula_permuted", "global_between")
ARMS = ("true", *CONTROL_ARMS)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-ranks", type=Path,
        default=ROOT / "data/validation/chemaware_whitened_centered_rule_kernel_v1/inner_ranks.npz",
    )
    parser.add_argument(
        "--within-ranks", type=Path,
        default=ROOT / "data/validation/chemaware_within_identity_whitened_rule_kernel_v1/inner_ranks.npz",
    )
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
        default=ROOT / "data/validation/chemaware_conditional_fisher_rule_kernel_v1",
    )
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--fit-identities", type=int, default=4096)
    parser.add_argument("--validation-identities", type=int, default=0)
    parser.add_argument("--final-fit-identities", type=int, default=8192)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--within-shrinkage", type=float, default=0.25)
    parser.add_argument("--rank", type=int, nargs="+", default=(16, 32, 64, 128, 256, 316))
    parser.add_argument(
        "--beta", type=float, nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _row_formulas_for_permuted_identity_formulas(
    identity: np.ndarray,
    formula: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    unique_identity, first = np.unique(identity.astype(str), return_index=True)
    actual = formula.astype(str)[first]
    permuted, report = permute_identity_formulas(actual, seed)
    registry = {name: value for name, value in zip(unique_identity, permuted)}
    output = np.asarray([registry[value] for value in identity.astype(str)])
    return output, report


def fit_fisher_transforms(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    cache: FourCenterCache,
    ranks: tuple[int, ...] | list[int],
    shrinkage: float,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, str], dict[str, object]]:
    rows, identity, formula, registry_report = row_identity_registry(queries, body)
    permuted_identity, identity_permutation_report = permute_identity_within_formula(
        identity, formula, seed + 1,
    )
    permuted_formula, formula_permutation_report = _row_formulas_for_permuted_identity_formulas(
        identity, formula, seed + 2,
    )
    specifications = {
        "true": ("true", identity, formula, "conditional"),
        "local_a": ("local_a", identity, formula, "conditional"),
        "local_b": ("local_b", identity, formula, "conditional"),
        "local_c": ("local_c", identity, formula, "conditional"),
        "identity_permuted": ("true", permuted_identity, formula, "conditional"),
        "formula_permuted": ("true", identity, permuted_formula, "conditional"),
        "global_between": ("true", identity, formula, "global"),
    }
    features = {
        center: np.stack([
            np.asarray(cache.get(int(row))[center], dtype=np.float32) for row in rows
        ])
        for center in CENTER_NAMES
    }
    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    variants: dict[str, str] = {}
    reports: dict[str, object] = {}
    maximum_rank = max(map(int, ranks))
    for arm, (center, labels, groups, mode) in specifications.items():
        mean, full_transform, report = fit_conditional_fisher_map(
            features[center], labels, groups, shrinkage, maximum_rank,
            between_mode=mode,
        )
        eigenvalues = np.asarray(report.pop("generalized_eigenvalues"), dtype=np.float64)
        total = max(float(eigenvalues.sum()), 1e-12)
        for rank in map(int, ranks):
            variant = f"fisher_{arm}_k{rank}"
            transforms[variant] = (mean, full_transform[:, :rank])
            variants[variant] = center
            reports[variant] = {
                **report,
                "rank": rank,
                "smallest_retained_generalized_eigenvalue": float(eigenvalues[rank - 1]),
                "retained_generalized_eigenvalue_fraction": float(eigenvalues[:rank].sum() / total),
            }
    return transforms, variants, {
        "registry": registry_report,
        "identity_permutation": identity_permutation_report,
        "formula_permutation": formula_permutation_report,
        "transforms": reports,
    }


def select_arm(
    scored: dict[str, np.ndarray],
    arm: str,
    variants: dict[str, str],
    beta: tuple[float, ...] | list[float],
) -> tuple[dict[str, object], dict[str, object]]:
    eligible = [name for name in variants if name.startswith(f"fisher_{arm}_k")]
    tables = {}
    candidates = []
    for variant in eligible:
        selected, rows = grid(scored, variant, list(map(float, beta)))
        tables[variant] = {"selected": selected, "grid": rows}
        candidates.append({"variant": variant, **selected})
    chosen = max(
        candidates,
        key=lambda row: (
            int(row["risk_utility_at_1"]), -int(row["introduced_at_1"]),
            float(row["delta_mrr"]), -int(row["variant"].rsplit("k", 1)[1]),
            -float(row["mass_beta"]), -float(row["rule_beta"]),
        ),
    )
    return chosen, tables


def _aligned_parent_rank(
    path: Path,
    inner: np.ndarray,
    formula: np.ndarray,
    baseline: np.ndarray,
    key: str,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as parent:
        position = {int(query): index for index, query in enumerate(parent["query"])}
        if any(int(query) not in position for query in inner):
            raise RuntimeError(f"inner query absent from parent ledger: {path}")
        aligned = np.asarray([position[int(query)] for query in inner], dtype=np.int64)
        if not (
            np.array_equal(parent["formula"][aligned].astype(str), formula)
            and np.array_equal(parent["baseline_rank"][aligned], baseline)
        ):
            raise RuntimeError(f"parent ledger does not match current inner protocol: {path}")
        return np.asarray(parent[key])[aligned]


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.smoke:
        args.fit_identities = min(args.fit_identities, 512)
        args.validation_identities = 256
        args.final_fit_identities = min(args.final_fit_identities, 1024)
        args.max_inner_identities = 256
        args.rank = (16, 64, 128, 316)
        args.beta = (0.0, 0.1, 0.2, 0.4, 0.8)
        args.bootstrap_draws = min(args.bootstrap_draws, 500)
    ranks = tuple(sorted(set(map(int, args.rank))))
    if not ranks or ranks[-1] > 316:
        raise ValueError("requested Fisher rank exceeds the 316 rule channels")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], 5, args.fold_seed)
    fit_queries = identity_balanced_queries(
        np.flatnonzero(np.isin(fold, [0, 1])), body["query_ik14"],
        np.random.default_rng(args.seed + 11), args.fit_identities,
    )
    validation = identity_balanced_queries(
        np.flatnonzero(fold == 2), body["query_ik14"],
        np.random.default_rng(args.seed + 17), args.validation_identities,
    )
    final_fit = identity_balanced_queries(
        np.flatnonzero(np.isin(fold, [0, 1, 2])), body["query_ik14"],
        np.random.default_rng(args.seed + 23), args.final_fit_identities,
    )
    inner = identity_balanced_queries(
        np.flatnonzero(fold == 3), body["query_ik14"],
        np.random.default_rng(args.fold_seed + 19), args.max_inner_identities,
    )
    if set(body["query_formula"][final_fit].astype(str)) & set(body["query_formula"][inner].astype(str)):
        raise RuntimeError("fit and inner formulas overlap")
    kernel_args = SimpleNamespace(
        token_dir=args.token_dir, rule_library=args.rule_library,
        top_peaks=args.top_peaks, kernel_dim=args.kernel_dim,
        bin_width=args.bin_width, grid_offsets=args.grid_offsets,
        intensity_power=args.intensity_power, mass_shift_da=args.mass_shift_da,
        pair_weight=0.25, multi_bin_widths=(0.01, 0.02, 0.05),
        uniform_channel_weight=1.0, rule_tolerance=args.rule_tolerance,
        rule_channel_weight=1.0,
    )
    base = KernelCache(
        kernel_args, row_position,
        variants=(
            "mass", "rule_response", "rule_response_local_background_a",
            "rule_response_local_background_b", "rule_response_local_background_c",
        ),
    )
    centers = FourCenterCache(base)
    fit_transform, fit_variants, fit_report = fit_fisher_transforms(
        fit_queries, body, centers, ranks, args.within_shrinkage, args.seed + 101,
    )
    validation_cache = TransformCache(centers, fit_transform, fit_variants)
    validation_scored = score_queries(
        validation, body, official, row_position, validation_cache,
        ("mass", *fit_variants),
    )
    selection = {}
    selection_grid = {}
    for arm in ARMS:
        selection[arm], selection_grid[arm] = select_arm(
            validation_scored, arm, fit_variants, args.beta,
        )
    print(json.dumps({"fold2_selection": selection}, indent=2), flush=True)
    del validation_scored, validation_cache, fit_transform
    gc.collect()

    selected_ranks = tuple(sorted({
        int(selection[arm]["variant"].rsplit("k", 1)[1]) for arm in ARMS
    }))
    final_transform_all, final_variants_all, final_fit_report = fit_fisher_transforms(
        final_fit, body, centers, selected_ranks, args.within_shrinkage, args.seed + 201,
    )
    chosen_variants = {selection[arm]["variant"]: final_variants_all[selection[arm]["variant"]] for arm in ARMS}
    final_transform = {variant: final_transform_all[variant] for variant in chosen_variants}
    inner_cache = TransformCache(centers, final_transform, chosen_variants)
    inner_scored = score_queries(
        inner, body, official, row_position, inner_cache, ("mass", *chosen_variants),
    )
    baseline = np.asarray(inner_scored["old_rank"])
    formula = np.asarray(inner_scored["formula"]).astype(str)
    held = {}
    arm_ranks = {}
    for arm in ARMS:
        chosen = selection[arm]
        rank = fused_ranks(
            inner_scored, chosen["variant"], chosen["mass_beta"], chosen["rule_beta"],
        )
        arm_ranks[arm] = rank
        held[arm] = {
            "selected": chosen,
            "retrieval": retrieval(baseline, rank),
            "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
                formula, baseline, rank, draws=args.bootstrap_draws,
                seed=args.seed + 300 + ARMS.index(arm),
            ),
        }
    marginal_rank = _aligned_parent_rank(
        args.parent_ranks, inner, formula, baseline, "whitened_true_rank",
    )
    within_rank = _aligned_parent_rank(
        args.within_ranks, inner, formula, baseline, "within_true_rank",
    )
    primary = arm_ranks["true"]
    comparison_ranks = {"marginal_whitening": marginal_rank, "within_whitening": within_rank}
    comparison_ranks.update({arm: arm_ranks[arm] for arm in CONTROL_ARMS})
    comparisons = {
        f"fisher_true_minus_{name}": paired_formula_ci(
            formula, primary, control, args, 400 + index,
        )
        for index, (name, control) in enumerate(comparison_ranks.items())
    }
    target = held["true"]
    gates = {
        "absolute_formula_ci_positive": target["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
        "corrected_exceeds_twice_introduced": (
            int(target["retrieval"]["corrected_at_1"])
            > 2 * int(target["retrieval"]["introduced_at_1"])
        ),
        **{
            f"beats_{name}_ci": comparisons[f"fisher_true_minus_{name}"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0
            for name in comparison_ranks
        },
        "outer_fold_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_CONDITIONAL_FISHER_RULE_KERNEL_PASS"
            if all(gates.values()) else "CHEMAWARE_CONDITIONAL_FISHER_RULE_KERNEL_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "folds 0-1 metric fit; fold 2 rank/fusion selection; folds 0-2 refit; inner fold 3 evaluation; outer fold 4 sealed",
        "claim_limit": "Shared-kernel development evidence on an already-used inner fold; not DreaMS fine-tuning or external confirmation.",
        "method": {
            "objective": "top generalized eigenvectors of S_between_within_formula v = lambda S_within_identity v",
            "map": "[official, sqrt(mass_beta)*mass, sqrt(rule_beta)*unit((centered_rule-mu)@T_fisher)]",
            "within_shrinkage_fixed_from_prior_audit": float(args.within_shrinkage),
            "same_map_for_query_and_reference": True,
            "candidate_formula_identity_free_at_deployment": True,
            "formula_used_only_to_fit_metric_on_development_folds": True,
        },
        "initial_fit": fit_report,
        "fold2_selection": {
            arm: {"selected": selection[arm], "grid": selection_grid[arm]} for arm in ARMS
        },
        "final_fit": {
            "registry": final_fit_report["registry"],
            "identity_permutation": final_fit_report["identity_permutation"],
            "formula_permutation": final_fit_report["formula_permutation"],
            "selected_transform_reports": {
                selection[arm]["variant"]: final_fit_report["transforms"][selection[arm]["variant"]]
                for arm in ARMS
            },
        },
        "held_inner": held,
        "parent_retrieval": {
            "marginal_whitening": retrieval(baseline, marginal_rank),
            "within_whitening": retrieval(baseline, within_rank),
        },
        "paired_inner": comparisons,
        "gates": gates,
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "rule_library_sha256": sha256_file(args.rule_library),
            "marginal_parent_ranks_sha256": sha256_file(args.parent_ranks),
            "within_parent_ranks_sha256": sha256_file(args.within_ranks),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_conditional_fisher_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_ranks.npz", query=inner, formula=formula,
            baseline_rank=baseline, fisher_true_rank=primary,
            marginal_whitening_rank=marginal_rank, within_whitening_rank=within_rank,
            **{f"{arm}_rank": arm_ranks[arm] for arm in CONTROL_ARMS},
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "selection": selection,
        "held_inner": held, "parent_retrieval": report["parent_retrieval"],
        "paired_inner": comparisons, "gates": gates,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
