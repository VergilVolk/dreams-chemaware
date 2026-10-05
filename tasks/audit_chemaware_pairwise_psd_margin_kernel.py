"""Audit a cross-fit pairwise-margin PSD update of the best ChemAware kernel."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_chemaware_empirical_rule_reliability_kernel import fused_ranks, paired_formula_ci
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries
from audit_chemaware_observable_tangent_metric import formula_bootstrap, retrieval
from audit_chemaware_whitened_centered_rule_kernel import CENTER_NAMES, FourCenterCache, TransformCache
from chemaware_formula_rule_core import rows_for_queries
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_pairwise_psd_core import (
    compose_psd_update,
    cross_split_coordinate_consensus,
    formula_balanced_pair_operator,
)
from chemaware_shrinkage_whitening_core import apply_whitener, fit_shrinkage_whitener
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
CONTROL_ARMS = ("local_a", "local_b", "local_c", "query_permuted", "role_reversed")
ARMS = ("true", *CONTROL_ARMS)


def stable_formula_binary_split(formula: np.ndarray, seed: int) -> np.ndarray:
    output = np.asarray([
        int.from_bytes(
            hashlib.sha256(f"{seed}|{str(value)}".encode()).digest()[:8], "little",
        ) % 2
        for value in np.asarray(formula).astype(str)
    ], dtype=np.int8)
    registry: dict[str, int] = {}
    for name, split in zip(np.asarray(formula).astype(str), output):
        if name in registry and registry[name] != int(split):
            raise RuntimeError("one formula crossed the binary consensus split")
        registry[name] = int(split)
    return output


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-ranks", type=Path,
        default=ROOT / "data/validation/chemaware_whitened_centered_rule_kernel_v1/inner_ranks.npz",
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
        default=ROOT / "data/validation/chemaware_pairwise_psd_margin_kernel_v1",
    )
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260905)
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
    parser.add_argument("--base-shrinkage", type=float, default=0.50)
    parser.add_argument("--mass-beta", type=float, default=0.40)
    parser.add_argument("--rule-beta", type=float, default=0.80)
    parser.add_argument(
        "--eta", type=float, nargs="+", default=(0.0, 0.10, 0.25, 0.50, 0.75, 0.90),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def fit_base_transforms(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    cache: FourCenterCache,
    shrinkage: float,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, str], dict[str, object]]:
    rows = rows_for_queries(queries, body)
    transforms = {}
    variants = {}
    reports = {}
    for center in CENTER_NAMES:
        feature = np.stack([
            np.asarray(cache.get(int(row))[center], dtype=np.float32) for row in rows
        ])
        mean, transform, report = fit_shrinkage_whitener(feature, shrinkage)
        variant = f"base_{center}"
        transforms[variant] = (mean, transform)
        variants[variant] = center
        reports[variant] = report
    return transforms, variants, {"spectrum_rows": int(len(rows)), "transforms": reports}


def mine_boundary_pairs(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    official: np.ndarray,
    row_position: dict[int, int],
    cache: TransformCache,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    qrows = np.empty(len(queries), dtype=np.int64)
    prows = np.empty(len(queries), dtype=np.int64)
    nrows = np.empty(len(queries), dtype=np.int64)
    margin = np.empty(len(queries), dtype=np.float32)
    for out_index, query in enumerate(map(int, queries)):
        qrow = int(body["query_row"][query])
        qrows[out_index] = qrow
        qpos = row_position[qrow]
        qfeature = cache.get(qrow)
        left, right = map(int, body["query_ptr"][query:query + 2])
        labels = np.asarray(body["molecule_label"][left:right], dtype=bool)
        pair_left = int(body["molecule_ptr"][left])
        pair_right = int(body["molecule_ptr"][right])
        reference_rows = np.asarray(body["pair_candidate_row"][pair_left:pair_right], dtype=np.int64)
        reference_position = np.asarray([row_position[int(row)] for row in reference_rows])
        pair_score = official[reference_position] @ official[qpos]
        reference_mass = np.stack([
            np.asarray(cache.get(int(row))["mass"], dtype=np.float32) for row in reference_rows
        ])
        reference_rule = np.stack([
            np.asarray(cache.get(int(row))["base_true"], dtype=np.float32) for row in reference_rows
        ])
        pair_score = (
            pair_score
            + float(args.mass_beta) * (reference_mass @ np.asarray(qfeature["mass"], dtype=np.float32))
            + float(args.rule_beta) * (reference_rule @ np.asarray(qfeature["base_true"], dtype=np.float32))
        )
        pointer = np.asarray(body["molecule_ptr"][left:right + 1], dtype=np.int64) - pair_left
        best_score = np.empty(right - left, dtype=np.float32)
        best_row = np.empty(right - left, dtype=np.int64)
        for molecule, (start, stop) in enumerate(zip(pointer[:-1], pointer[1:])):
            best = int(start + np.argmax(pair_score[start:stop]))
            best_score[molecule] = float(pair_score[best])
            best_row[molecule] = int(reference_rows[best])
        positive = np.flatnonzero(labels)
        negative = np.flatnonzero(~labels)
        if not len(positive) or not len(negative):
            raise RuntimeError("pair mining requires positive and negative molecules")
        pindex = int(positive[np.argmax(best_score[positive])])
        nindex = int(negative[np.argmax(best_score[negative])])
        prows[out_index] = best_row[pindex]
        nrows[out_index] = best_row[nindex]
        margin[out_index] = best_score[pindex] - best_score[nindex]
        if (out_index + 1) % 256 == 0:
            print(f"mined boundary pairs {out_index + 1}/{len(queries)}", flush=True)
    return {
        "query": np.asarray(queries, dtype=np.int64),
        "query_row": qrows,
        "positive_row": prows,
        "negative_row": nrows,
        "formula": np.asarray(body["query_formula"][queries]).astype(str),
        "base_margin": margin,
    }


def logistic_boundary_weight(margin: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    margin = np.asarray(margin, dtype=np.float64)
    positive_scale = np.median(np.abs(margin - np.median(margin)))
    scale = max(float(positive_scale), 1e-3)
    weight = 1.0 / (1.0 + np.exp(np.clip(margin / scale, -40.0, 40.0)))
    weight /= max(float(weight.mean()), 1e-12)
    return weight.astype(np.float32), {
        "scale_is_margin_mad": True,
        "scale": scale,
        "mean": float(weight.mean()),
        "quantiles": np.quantile(weight, (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)).astype(float).tolist(),
        "base_error_fraction": float(np.mean(margin <= 0.0)),
    }


def permute_query_rows_within_precursor_bins(
    query_row: np.ndarray,
    precursor_mz: np.ndarray,
    seed: int,
    bins: int = 10,
) -> tuple[np.ndarray, dict[str, object]]:
    output = np.asarray(query_row, dtype=np.int64).copy()
    precursor_mz = np.asarray(precursor_mz, dtype=np.float64)
    if precursor_mz.shape != output.shape or not np.isfinite(precursor_mz).all():
        raise ValueError("query permutation precursor values are invalid")
    rng = np.random.default_rng(seed)
    edge = np.unique(np.quantile(precursor_mz, np.linspace(0.0, 1.0, bins + 1)))
    assignment = np.clip(np.searchsorted(edge[1:-1], precursor_mz, side="right"), 0, bins - 1)
    for value in np.unique(assignment):
        index = np.flatnonzero(assignment == value)
        if len(index) >= 2:
            permutation = rng.permutation(index)
            if np.any(permutation == index):
                permutation = np.roll(index, 1)
            output[index] = output[permutation]
    return output, {
        "seed": int(seed),
        "within_precursor_quantile_bins": int(len(edge) - 1),
        "precursor_distribution_preserved_exactly": True,
        "changed_query_fraction": float(np.mean(output != query_row)),
    }


def _features(
    rows: np.ndarray,
    cache: TransformCache,
    variant: str,
) -> np.ndarray:
    return np.stack([
        np.asarray(cache.get(int(row))[variant], dtype=np.float32) for row in rows
    ])


def fit_update_family(
    queries: np.ndarray,
    split: np.ndarray,
    body: dict[str, np.ndarray],
    official: np.ndarray,
    row_position: dict[int, int],
    centers: FourCenterCache,
    base_transforms: dict[str, tuple[np.ndarray, np.ndarray]],
    base_variants: dict[str, str],
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, str], dict[str, object]]:
    base_cache = TransformCache(centers, base_transforms, base_variants)
    ledger = mine_boundary_pairs(queries, body, official, row_position, base_cache, args)
    query_position = np.asarray([row_position[int(row)] for row in ledger["query_row"]], dtype=np.int64)
    permuted_query, permutation_report = permute_query_rows_within_precursor_bins(
        ledger["query_row"], centers.base.precursor[query_position], seed + 1,
    )
    arm_center = {
        "true": "true", "local_a": "local_a", "local_b": "local_b", "local_c": "local_c",
        "query_permuted": "true", "role_reversed": "true",
    }
    split_reports = {}
    consensus_reports = {}
    operators = {}
    for arm, center in arm_center.items():
        variant = f"base_{center}"
        split_operator = []
        local_reports = []
        for split_value in (0, 1):
            index = np.flatnonzero(split == split_value)
            qrow = permuted_query[index] if arm == "query_permuted" else ledger["query_row"][index]
            query_feature = _features(qrow, base_cache, variant)
            positive_feature = _features(ledger["positive_row"][index], base_cache, variant)
            negative_feature = _features(ledger["negative_row"][index], base_cache, variant)
            if arm == "role_reversed":
                positive_feature, negative_feature = negative_feature, positive_feature
            weight, weight_report = logistic_boundary_weight(ledger["base_margin"][index])
            operator, operator_report = formula_balanced_pair_operator(
                query_feature, positive_feature, negative_feature,
                ledger["formula"][index], weight,
            )
            split_operator.append(operator)
            local_reports.append({
                "split": split_value, "queries": int(len(index)),
                "weight": weight_report, "operator": operator_report,
            })
        consensus, consensus_report = cross_split_coordinate_consensus(
            split_operator[0], split_operator[1],
        )
        operators[arm] = consensus
        split_reports[arm] = local_reports
        consensus_reports[arm] = consensus_report
    transforms = {}
    variants = {}
    transform_reports = {}
    for arm, center in arm_center.items():
        mean, base_transform = base_transforms[f"base_{center}"]
        for eta in map(float, args.eta):
            token = str(eta).replace(".", "p")
            variant = f"pairpsd_{arm}_e{token}"
            transform, report = compose_psd_update(base_transform, operators[arm], eta)
            transforms[variant] = (mean, transform)
            variants[variant] = center
            transform_reports[variant] = report
    return transforms, variants, {
        "pair_ledger": {
            "queries": int(len(ledger["query"])),
            "base_margin_quantiles": np.quantile(
                ledger["base_margin"], (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1),
            ).astype(float).tolist(),
            "base_error_fraction": float(np.mean(ledger["base_margin"] <= 0)),
        },
        "query_permutation": permutation_report,
        "split_operator": split_reports,
        "spectral_consensus": consensus_reports,
        "transforms": transform_reports,
    }


def select_eta(
    scored: dict[str, np.ndarray],
    arm: str,
    variants: dict[str, str],
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    for variant in variants:
        if not variant.startswith(f"pairpsd_{arm}_e"):
            continue
        eta = float(variant.rsplit("e", 1)[1].replace("p", "."))
        rank = fused_ranks(scored, variant, args.mass_beta, args.rule_beta)
        rows.append({"variant": variant, "eta": eta, **retrieval(scored["old_rank"], rank)})
    selected = max(
        rows,
        key=lambda row: (
            int(row["risk_utility_at_1"]), -int(row["introduced_at_1"]),
            float(row["delta_mrr"]), -float(row["eta"]),
        ),
    )
    return selected, rows


def aligned_parent_rank(
    path: Path,
    inner: np.ndarray,
    formula: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as parent:
        position = {int(query): index for index, query in enumerate(parent["query"])}
        aligned = np.asarray([position[int(query)] for query in inner], dtype=np.int64)
        if not (
            np.array_equal(parent["formula"][aligned].astype(str), formula)
            and np.array_equal(parent["baseline_rank"][aligned], baseline)
        ):
            raise RuntimeError("parent whitening rank ledger is not aligned")
        return np.asarray(parent["whitened_true_rank"])[aligned]


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.smoke:
        args.fit_identities = min(args.fit_identities, 512)
        args.validation_identities = 256
        args.final_fit_identities = min(args.final_fit_identities, 1024)
        args.max_inner_identities = 256
        args.eta = (0.0, 0.25, 0.50, 0.90)
        args.bootstrap_draws = min(args.bootstrap_draws, 500)
    if any(not 0 <= float(eta) < 1 for eta in args.eta):
        raise ValueError("every eta must be in [0, 1)")
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
    fit_split = stable_formula_binary_split(body["query_formula"][fit_queries], args.seed + 501)
    final_split = stable_formula_binary_split(body["query_formula"][final_fit], args.seed + 601)
    if min(np.bincount(fit_split)) < 2 or min(np.bincount(final_split)) < 2:
        raise RuntimeError("cross-fit formula split is empty")
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
    fit_base, fit_base_variants, fit_base_report = fit_base_transforms(
        fit_queries, body, centers, args.base_shrinkage,
    )
    fit_update, fit_variants, fit_report = fit_update_family(
        fit_queries, fit_split, body, official, row_position, centers,
        fit_base, fit_base_variants, args, args.seed + 701,
    )
    validation_cache = TransformCache(centers, fit_update, fit_variants)
    validation_scored = score_queries(
        validation, body, official, row_position, validation_cache,
        ("mass", *fit_variants),
    )
    selection = {}
    selection_grid = {}
    for arm in ARMS:
        selection[arm], selection_grid[arm] = select_eta(
            validation_scored, arm, fit_variants, args,
        )
    print(json.dumps({"fold2_selection": selection}, indent=2), flush=True)
    del validation_scored, validation_cache, fit_update, fit_base
    gc.collect()

    chosen_eta = tuple(sorted({float(selection[arm]["eta"]) for arm in ARMS} | {0.0}))
    original_eta = args.eta
    args.eta = chosen_eta
    final_base, final_base_variants, final_base_report = fit_base_transforms(
        final_fit, body, centers, args.base_shrinkage,
    )
    final_update_all, final_variants_all, final_fit_report = fit_update_family(
        final_fit, final_split, body, official, row_position, centers,
        final_base, final_base_variants, args, args.seed + 801,
    )
    args.eta = original_eta
    chosen_variants = {selection[arm]["variant"]: final_variants_all[selection[arm]["variant"]] for arm in ARMS}
    eta0_variant = "pairpsd_true_e0p0"
    chosen_variants[eta0_variant] = final_variants_all[eta0_variant]
    final_update = {variant: final_update_all[variant] for variant in chosen_variants}
    inner_cache = TransformCache(centers, final_update, chosen_variants)
    inner_scored = score_queries(
        inner, body, official, row_position, inner_cache, ("mass", *chosen_variants),
    )
    baseline = np.asarray(inner_scored["old_rank"])
    formula = np.asarray(inner_scored["formula"]).astype(str)
    eta0_rank = fused_ranks(inner_scored, eta0_variant, args.mass_beta, args.rule_beta)
    parent_rank = aligned_parent_rank(args.parent_ranks, inner, formula, baseline)
    if not args.smoke and not np.array_equal(eta0_rank, parent_rank):
        raise RuntimeError("eta=0 does not exactly reproduce the frozen marginal-whitening parent")
    held = {}
    arm_ranks = {}
    for arm in ARMS:
        chosen = selection[arm]
        rank = fused_ranks(
            inner_scored, chosen["variant"], args.mass_beta, args.rule_beta,
        )
        arm_ranks[arm] = rank
        held[arm] = {
            "selected": chosen,
            "retrieval": retrieval(baseline, rank),
            "increment_over_eta0": retrieval(eta0_rank, rank),
            "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
                formula, baseline, rank, draws=args.bootstrap_draws,
                seed=args.seed + 900 + ARMS.index(arm),
            ),
        }
    primary = arm_ranks["true"]
    comparisons = {
        "pairpsd_true_minus_eta0_parent": paired_formula_ci(
            formula, primary, eta0_rank, args, 1000,
        ),
        **{
            f"pairpsd_true_minus_{arm}": paired_formula_ci(
                formula, primary, arm_ranks[arm], args, 1010 + index,
            )
            for index, arm in enumerate(CONTROL_ARMS)
        },
    }
    target = held["true"]
    gates = {
        "selected_nonzero_eta": float(selection["true"]["eta"]) > 0,
        "absolute_formula_ci_positive": target["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
        "corrected_exceeds_twice_introduced": (
            int(target["retrieval"]["corrected_at_1"])
            > 2 * int(target["retrieval"]["introduced_at_1"])
        ),
        "beats_eta0_parent_ci": comparisons["pairpsd_true_minus_eta0_parent"][
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ][0] > 0,
        **{
            f"beats_{arm}_ci": comparisons[f"pairpsd_true_minus_{arm}"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0
            for arm in CONTROL_ARMS
        },
        "outer_fold_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_PAIRWISE_PSD_MARGIN_KERNEL_PASS"
            if all(gates.values()) else "CHEMAWARE_PAIRWISE_PSD_MARGIN_KERNEL_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "folds 0-1 operator fit; fold 2 eta selection; folds 0-2 refit; inner fold 3 evaluation; outer fold 4 sealed",
        "claim_limit": "Shared-kernel development evidence on an already-used inner fold; not DreaMS fine-tuning or external confirmation.",
        "method": {
            "objective": "fixed-rule-coordinate cross-fit consensus of formula-balanced sym(q(pos-neg)^T) at the frozen retrieval boundary",
            "metric": "M_eta = I + eta*A/||A||_2, eta < 1",
            "map": "[official, sqrt(0.4)*mass, sqrt(0.8)*unit((centered_rule-mu)@W_base@sqrt(M_eta))]",
            "boundary_weight": "parameter-free logistic weight using split-local margin MAD",
            "base_fusion_fixed": {"mass_beta": args.mass_beta, "rule_beta": args.rule_beta},
            "same_map_for_query_and_reference": True,
            "candidate_formula_identity_free_at_deployment": True,
        },
        "data": {
            "fit_queries": int(len(fit_queries)), "validation_queries": int(len(validation)),
            "final_fit_queries": int(len(final_fit)), "inner_queries": int(len(inner)),
            "outer_queries_untouched": int(np.sum(fold == 4)),
            "fit_cross_split_counts": np.bincount(fit_split).astype(int).tolist(),
            "final_cross_split_counts": np.bincount(final_split).astype(int).tolist(),
        },
        "initial_base_fit": fit_base_report,
        "initial_update_fit": fit_report,
        "fold2_selection": {
            arm: {"selected": selection[arm], "grid": selection_grid[arm]} for arm in ARMS
        },
        "final_base_fit": final_base_report,
        "final_update_fit": final_fit_report,
        "held_inner": held,
        "eta0_parent_retrieval": retrieval(baseline, eta0_rank),
        "paired_inner": comparisons,
        "gates": gates,
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "rule_library_sha256": sha256_file(args.rule_library),
            "parent_ranks_sha256": sha256_file(args.parent_ranks),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_pairwise_psd_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_ranks.npz", query=inner, formula=formula,
            baseline_rank=baseline, eta0_parent_rank=eta0_rank,
            pairpsd_true_rank=primary,
            **{f"{arm}_rank": arm_ranks[arm] for arm in CONTROL_ARMS},
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "selection": selection,
        "held_inner": held, "eta0_parent": report["eta0_parent_retrieval"],
        "paired_inner": comparisons, "gates": gates,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
