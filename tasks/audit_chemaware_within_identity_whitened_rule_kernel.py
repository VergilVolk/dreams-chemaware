"""Audit identity-balanced within-spectrum whitening of centered rules."""
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
from audit_chemaware_whitened_centered_rule_kernel import (
    CENTER_NAMES,
    FourCenterCache,
    TransformCache,
)
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_shrinkage_whitening_core import fit_balanced_within_whitener
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
PERMUTED_ARM = "identity_permuted_true"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-report", type=Path,
        default=ROOT / "data/validation/chemaware_whitened_centered_rule_kernel_v1/report.json",
    )
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
        default=ROOT / "data/validation/chemaware_within_identity_whitened_rule_kernel_v1",
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
    parser.add_argument(
        "--shrinkage", type=float, nargs="+", default=(0.01, 0.05, 0.10, 0.25, 0.50, 1.0),
    )
    parser.add_argument(
        "--beta", type=float, nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def row_identity_registry(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    registry: dict[int, tuple[str, str]] = {}
    conflicts = 0
    for query in map(int, queries):
        qrow = int(body["query_row"][query])
        qvalue = (str(body["query_ik14"][query]), str(body["query_formula"][query]))
        conflicts += int(qrow in registry and registry[qrow] != qvalue)
        registry[qrow] = qvalue
        left, right = map(int, body["query_ptr"][query:query + 2])
        for molecule in range(left, right):
            value = (str(body["molecule_ik14"][molecule]), str(body["molecule_formula"][molecule]))
            pair_left, pair_right = map(int, body["molecule_ptr"][molecule:molecule + 2])
            for row in body["pair_candidate_row"][pair_left:pair_right]:
                row = int(row)
                conflicts += int(row in registry and registry[row] != value)
                registry[row] = value
    if conflicts:
        raise RuntimeError(f"row identity/formula registry has {conflicts} conflicts")
    rows = np.asarray(sorted(registry), dtype=np.int64)
    identity = np.asarray([registry[int(row)][0] for row in rows])
    formula = np.asarray([registry[int(row)][1] for row in rows])
    _, count = np.unique(identity, return_counts=True)
    return rows, identity, formula, {
        "rows": int(len(rows)), "identities": int(len(count)),
        "identities_ge2": int(np.sum(count >= 2)),
        "identities_ge3": int(np.sum(count >= 3)),
        "rows_in_replicated_identities": int(np.sum(count[count >= 2])),
        "median_spectra_per_identity": float(np.median(count)),
        "max_spectra_per_identity": int(np.max(count)), "registry_conflicts": 0,
    }


def permute_identity_within_formula(
    identity: np.ndarray,
    formula: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    identity = np.asarray(identity).astype(str)
    formula = np.asarray(formula).astype(str)
    rng = np.random.default_rng(seed)
    output = identity.copy()
    for value in np.unique(formula):
        index = np.flatnonzero(formula == value)
        if len(index) >= 2:
            output[index] = identity[rng.permutation(index)]
    return output, {
        "within_formula_only": True,
        "identity_label_multiset_preserved_within_formula": True,
        "changed_row_fraction": float(np.mean(output != identity)),
    }


def fit_within_transforms(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    cache: FourCenterCache,
    shrinkage: tuple[float, ...] | list[float],
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, str], dict[str, object]]:
    rows, identity, formula, registry_report = row_identity_registry(queries, body)
    permuted_identity, permutation_report = permute_identity_within_formula(identity, formula, seed)
    transforms = {}
    variants = {}
    reports = {}
    for center_name in CENTER_NAMES:
        feature = np.stack([
            np.asarray(cache.get(int(row))[center_name], dtype=np.float32) for row in rows
        ])
        for value in map(float, shrinkage):
            token = str(value).replace(".", "p")
            variant = f"within_{center_name}_s{token}"
            mean, transform, report = fit_balanced_within_whitener(feature, identity, value)
            transforms[variant] = (mean, transform)
            variants[variant] = center_name
            reports[variant] = report
            if center_name == "true":
                control = f"within_{PERMUTED_ARM}_s{token}"
                pmean, ptransform, preport = fit_balanced_within_whitener(
                    feature, permuted_identity, value,
                )
                transforms[control] = (pmean, ptransform)
                variants[control] = "true"
                reports[control] = preport
        del feature
    return transforms, variants, {
        "registry": registry_report, "permutation": permutation_report,
        "transforms": reports,
    }


def select_arm(
    scored: dict[str, np.ndarray],
    arm: str,
    variants: dict[str, str],
    beta: tuple[float, ...] | list[float],
) -> tuple[dict[str, object], dict[str, object]]:
    if arm == PERMUTED_ARM:
        prefix = f"within_{PERMUTED_ARM}_"
        eligible = [variant for variant in variants if variant.startswith(prefix)]
    else:
        prefix = f"within_{arm}_"
        eligible = [variant for variant in variants if variant.startswith(prefix)]
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
            float(row["delta_mrr"]), -float(row["mass_beta"]), -float(row["rule_beta"]),
        ),
    )
    return chosen, tables


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.smoke:
        args.fit_identities = min(args.fit_identities, 256)
        args.validation_identities = 128
        args.final_fit_identities = min(args.final_fit_identities, 512)
        args.max_inner_identities = 128
        args.shrinkage = (0.05, 0.25, 1.0)
        args.beta = (0.0, 0.1, 0.2, 0.4)
        args.bootstrap_draws = min(args.bootstrap_draws, 300)
    parent_report = json.loads(args.parent_report.read_text(encoding="utf-8"))
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
    fit_transform, fit_variants, fit_report = fit_within_transforms(
        fit_queries, body, centers, args.shrinkage, args.seed + 101,
    )
    validation_cache = TransformCache(centers, fit_transform, fit_variants)
    validation_scored = score_queries(
        validation, body, official, row_position, validation_cache,
        ("mass", *fit_variants),
    )
    arms = (*CENTER_NAMES, PERMUTED_ARM)
    selection = {}
    selection_grid = {}
    for arm in arms:
        chosen, tables = select_arm(validation_scored, arm, fit_variants, args.beta)
        selection[arm] = chosen; selection_grid[arm] = tables
    print(json.dumps({"fold2_selection": selection}, indent=2), flush=True)
    del validation_scored, validation_cache, fit_transform
    gc.collect()

    selected_shrinkage = tuple(sorted({
        float(selection[arm]["variant"].rsplit("_s", 1)[1].replace("p", ".")) for arm in arms
    }))
    final_transform_all, final_variants_all, final_fit_report = fit_within_transforms(
        final_fit, body, centers, selected_shrinkage, args.seed + 201,
    )
    chosen_variants = {selection[arm]["variant"]: (
        "true" if arm == PERMUTED_ARM else arm
    ) for arm in arms}
    final_transform = {variant: final_transform_all[variant] for variant in chosen_variants}
    inner_cache = TransformCache(centers, final_transform, chosen_variants)
    inner_scored = score_queries(
        inner, body, official, row_position, inner_cache, ("mass", *chosen_variants),
    )
    baseline = np.asarray(inner_scored["old_rank"])
    formula = np.asarray(inner_scored["formula"]).astype(str)
    ranks = {}
    held = {}
    for arm in arms:
        chosen = selection[arm]
        rank = fused_ranks(
            inner_scored, chosen["variant"], chosen["mass_beta"], chosen["rule_beta"],
        )
        ranks[arm] = rank
        held[arm] = {
            "selected": chosen, "retrieval": retrieval(baseline, rank),
            "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
                formula, baseline, rank, draws=args.bootstrap_draws,
                seed=args.seed + 300 + arms.index(arm),
            ),
        }
    parent = np.load(args.parent_ranks, allow_pickle=False)
    parent_position = {int(query): index for index, query in enumerate(parent["query"])}
    if any(int(query) not in parent_position for query in inner):
        raise RuntimeError("current inner queries are absent from parent marginal-whitening ledger")
    aligned = np.asarray([parent_position[int(query)] for query in inner], dtype=np.int64)
    if not (
        np.array_equal(parent["formula"][aligned].astype(str), formula)
        and np.array_equal(parent["baseline_rank"][aligned], baseline)
    ):
        raise RuntimeError("parent marginal-whitening inner ledger does not match")
    marginal_rank = np.asarray(parent["whitened_true_rank"])[aligned]
    primary = ranks["true"]
    comparisons = {
        "within_true_minus_marginal_true": paired_formula_ci(
            formula, primary, marginal_rank, args, 400,
        ),
        **{
            f"within_true_minus_{arm}": paired_formula_ci(
                formula, primary, ranks[arm], args, 410 + index,
            )
            for index, arm in enumerate(("local_a", "local_b", "local_c", PERMUTED_ARM))
        },
    }
    target = held["true"]
    gates = {
        "absolute_formula_ci_positive": target[
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ][0] > 0,
        "corrected_exceeds_twice_introduced": (
            int(target["retrieval"]["corrected_at_1"])
            > 2 * int(target["retrieval"]["introduced_at_1"])
        ),
        "beats_marginal_whitening_ci": comparisons["within_true_minus_marginal_true"][
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ][0] > 0,
        **{
            f"beats_{arm}_ci": comparisons[f"within_true_minus_{arm}"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0
            for arm in ("local_a", "local_b", "local_c", PERMUTED_ARM)
        },
        "outer_fold_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_WITHIN_IDENTITY_WHITENED_RULE_KERNEL_PASS"
            if all(gates.values()) else "CHEMAWARE_WITHIN_IDENTITY_WHITENED_RULE_KERNEL_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "folds 0-1 within covariance fit; fold 2 selection; folds 0-2 refit; inner fold 3 evaluation; outer fold 4 sealed",
        "claim_limit": "Frozen shared metric development result on an already-used inner fold; not DreaMS fine-tuning or external confirmation.",
        "method": {
            "within_covariance": "identity-balanced mean of per-identity residual covariance",
            "map": "[official, sqrt(mass_beta)*mass, sqrt(rule_beta)*unit((centered_rule-mu)@(within_cov_shrunk^-1/2))]",
            "same_map_for_query_and_reference": True,
            "candidate_formula_identity_free_at_deployment": True,
            "identity_permutation_control": "row identities permuted within formula with group-size multiset preserved",
        },
        "initial_fit": fit_report,
        "fold2_selection": {
            arm: {"selected": selection[arm], "grid": selection_grid[arm]} for arm in arms
        },
        "final_fit": {
            "registry": final_fit_report["registry"],
            "permutation": final_fit_report["permutation"],
            "selected_transform_reports": {
                selection[arm]["variant"]: final_fit_report["transforms"][selection[arm]["variant"]]
                for arm in arms
            },
        },
        "held_inner": held,
        "marginal_whitening_parent": {
            "report": str(args.parent_report), "report_sha256": sha256_file(args.parent_report),
            "ranks": str(args.parent_ranks), "ranks_sha256": sha256_file(args.parent_ranks),
            "retrieval": retrieval(baseline, marginal_rank),
        },
        "paired_inner": comparisons,
        "gates": gates,
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "rule_library_sha256": sha256_file(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_within_whitening_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_ranks.npz", query=inner, formula=formula,
            baseline_rank=baseline, within_true_rank=primary,
            marginal_true_rank=marginal_rank, local_a_rank=ranks["local_a"],
            local_b_rank=ranks["local_b"], local_c_rank=ranks["local_c"],
            identity_permuted_rank=ranks[PERMUTED_ARM],
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "selection": selection,
        "held_inner": held, "parent": report["marginal_whitening_parent"],
        "paired_inner": comparisons, "gates": gates,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
