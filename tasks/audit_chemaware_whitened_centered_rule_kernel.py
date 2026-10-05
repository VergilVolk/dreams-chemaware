"""Audit covariance-whitened centered chemical rules as a shared embedding."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_chemaware_empirical_rule_reliability_kernel import fused_ranks, grid, paired_formula_ci, unit
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries
from audit_chemaware_observable_tangent_metric import formula_bootstrap, retrieval
from chemaware_formula_rule_core import rows_for_queries
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_shrinkage_whitening_core import apply_whitener, fit_shrinkage_whitener
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
CENTER_NAMES = ("true", "local_a", "local_b", "local_c")


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
        default=ROOT / "data/validation/chemaware_whitened_centered_rule_kernel_v1",
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


class FourCenterCache:
    """Symmetric residual feature for each of four shared-coordinate centers."""

    def __init__(self, base: KernelCache) -> None:
        self.base = base
        self.cache: dict[int, dict[str, np.ndarray]] = {}

    def get(self, row: int) -> dict[str, np.ndarray]:
        row = int(row)
        if row not in self.cache:
            raw = self.base.get(row)
            centers = [
                np.asarray(raw[name], dtype=np.float32)
                for name in (
                    "rule_response", "rule_response_local_background_a",
                    "rule_response_local_background_b", "rule_response_local_background_c",
                )
            ]
            output = {"mass": raw["mass"], "raw_rule": raw["rule_response"]}
            for index, name in enumerate(CENTER_NAMES):
                others = [value for other, value in enumerate(centers) if other != index]
                output[name] = unit(centers[index] - np.mean(np.stack(others), axis=0))
            self.cache[row] = output
        return self.cache[row]


class TransformCache:
    def __init__(
        self,
        base: FourCenterCache,
        transforms: dict[str, tuple[np.ndarray, np.ndarray]],
        variants: dict[str, str],
    ) -> None:
        self.base = base
        self.transforms = transforms
        self.variants = variants
        self.cache: dict[int, dict[str, np.ndarray]] = {}

    def get(self, row: int) -> dict[str, np.ndarray]:
        row = int(row)
        if row not in self.cache:
            source = self.base.get(row)
            output = {
                "mass": source["mass"], "uniform_centered": source["true"],
                "raw_rule": source["raw_rule"],
            }
            for variant, center_name in self.variants.items():
                mean, transform = self.transforms[variant]
                output[variant] = apply_whitener(
                    np.asarray(source[center_name], dtype=np.float32), mean, transform,
                ).astype(np.float16)
            self.cache[row] = output
        return self.cache[row]


def fit_transforms(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    cache: FourCenterCache,
    shrinkage: tuple[float, ...] | list[float],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, str], dict[str, object]]:
    rows = rows_for_queries(queries, body)
    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    variants: dict[str, str] = {}
    reports: dict[str, object] = {}
    for center_name in CENTER_NAMES:
        feature = np.stack([
            np.asarray(cache.get(int(row))[center_name], dtype=np.float32) for row in rows
        ])
        for value in map(float, shrinkage):
            token = str(value).replace(".", "p")
            variant = f"whitened_{center_name}_s{token}"
            mean, transform, report = fit_shrinkage_whitener(feature, value)
            transforms[variant] = (mean, transform)
            variants[variant] = center_name
            reports[variant] = report
        del feature
    return transforms, variants, {"spectrum_rows": int(len(rows)), "transforms": reports}


def select_whitened_arm(
    scored: dict[str, np.ndarray],
    center_name: str,
    variants: dict[str, str],
    beta: tuple[float, ...] | list[float],
) -> tuple[dict[str, object], dict[str, object]]:
    candidate = {}
    all_rows = []
    for variant, owner in variants.items():
        if owner != center_name:
            continue
        selected, rows = grid(scored, variant, list(map(float, beta)))
        candidate[variant] = {"selected": selected, "grid": rows}
        all_rows.append({"variant": variant, **selected})
    chosen = max(
        all_rows,
        key=lambda row: (
            int(row["risk_utility_at_1"]), -int(row["introduced_at_1"]),
            float(row["delta_mrr"]), -float(row["mass_beta"]), -float(row["rule_beta"]),
        ),
    )
    return chosen, candidate


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
    fit_transform, fit_variants, fit_report = fit_transforms(
        fit_queries, body, centers, args.shrinkage,
    )
    validation_cache = TransformCache(centers, fit_transform, fit_variants)
    validation_scored = score_queries(
        validation, body, official, row_position, validation_cache,
        ("mass", "uniform_centered", "raw_rule", *fit_variants),
    )
    selection = {}
    selection_grid = {}
    for center_name in CENTER_NAMES:
        chosen, grid_body = select_whitened_arm(
            validation_scored, center_name, fit_variants, args.beta,
        )
        selection[center_name] = chosen; selection_grid[center_name] = grid_body
    uniform_selected, uniform_grid = grid(
        validation_scored, "uniform_centered", list(map(float, args.beta)),
    )
    raw_selected, raw_grid = grid(validation_scored, "raw_rule", list(map(float, args.beta)))
    print(json.dumps({
        "selected_whitened": selection, "uniform": uniform_selected, "raw": raw_selected,
    }, indent=2), flush=True)
    del validation_scored, validation_cache, fit_transform
    gc.collect()

    final_transform_all, final_variants_all, final_fit_report = fit_transforms(
        final_fit, body, centers, args.shrinkage,
    )
    chosen_variants = {selection[name]["variant"]: name for name in CENTER_NAMES}
    final_transform = {name: final_transform_all[name] for name in chosen_variants}
    inner_cache = TransformCache(centers, final_transform, chosen_variants)
    inner_scored = score_queries(
        inner, body, official, row_position, inner_cache,
        ("mass", "uniform_centered", "raw_rule", *chosen_variants),
    )
    baseline = np.asarray(inner_scored["old_rank"])
    formula = np.asarray(inner_scored["formula"]).astype(str)
    ranks = {}
    held = {}
    for name in CENTER_NAMES:
        chosen = selection[name]
        rank = fused_ranks(
            inner_scored, chosen["variant"], chosen["mass_beta"], chosen["rule_beta"],
        )
        ranks[name] = rank
        held[name] = {
            "selected": chosen,
            "retrieval": retrieval(baseline, rank),
            "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
                formula, baseline, rank, draws=args.bootstrap_draws,
                seed=args.seed + 300 + CENTER_NAMES.index(name),
            ),
        }
    uniform_rank = fused_ranks(
        inner_scored, "uniform_centered", uniform_selected["mass_beta"], uniform_selected["rule_beta"],
    )
    raw_rank = fused_ranks(
        inner_scored, "raw_rule", raw_selected["mass_beta"], raw_selected["rule_beta"],
    )
    held["uniform_centered"] = {
        "selected": uniform_selected, "retrieval": retrieval(baseline, uniform_rank),
        "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
            formula, baseline, uniform_rank, draws=args.bootstrap_draws, seed=args.seed + 310,
        ),
    }
    held["raw_rule"] = {
        "selected": raw_selected, "retrieval": retrieval(baseline, raw_rank),
        "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
            formula, baseline, raw_rank, draws=args.bootstrap_draws, seed=args.seed + 311,
        ),
    }
    primary = ranks["true"]
    comparison_rank = {"uniform_centered": uniform_rank, "raw_rule": raw_rank}
    comparison_rank.update({name: ranks[name] for name in ("local_a", "local_b", "local_c")})
    paired = {
        f"whitened_true_minus_{name}": paired_formula_ci(
            formula, primary, rank, args, 400 + index,
        )
        for index, (name, rank) in enumerate(comparison_rank.items())
    }
    primary_held = held["true"]
    gates = {
        "absolute_formula_ci_positive": primary_held[
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ][0] > 0,
        "corrected_exceeds_twice_introduced": (
            int(primary_held["retrieval"]["corrected_at_1"])
            > 2 * int(primary_held["retrieval"]["introduced_at_1"])
        ),
        **{
            f"beats_{name}_ci": paired[f"whitened_true_minus_{name}"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0
            for name in comparison_rank
        },
        "outer_fold_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_WHITENED_CENTERED_RULE_KERNEL_PASS"
            if all(gates.values()) else "CHEMAWARE_WHITENED_CENTERED_RULE_KERNEL_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "folds 0-1 covariance fit; fold 2 selection; folds 0-2 refit; inner fold 3 evaluation; outer fold 4 sealed",
        "claim_limit": "Frozen explicit shared-kernel development result, not DreaMS fine-tuning or external confirmation.",
        "method": {
            "map": "[official, sqrt(mass_beta)*mass, sqrt(rule_beta)*unit((centered_rule-mu)@(Sigma_shrunk^-1/2))]",
            "covariance_fit_uses_labels": False,
            "same_map_for_query_and_reference": True,
            "candidate_formula_identity_free_at_deployment": True,
            "symmetric_four_center_controls": True,
        },
        "data": {
            "fit_queries": int(len(fit_queries)), "validation_queries": int(len(validation)),
            "final_fit_queries": int(len(final_fit)), "inner_queries": int(len(inner)),
            "outer_queries_untouched": int(np.sum(fold == 4)),
        },
        "initial_covariance_fit": fit_report,
        "fold2_selection": {
            "whitened": selection, "whitened_grid": selection_grid,
            "uniform_centered": {"selected": uniform_selected, "grid": uniform_grid},
            "raw_rule": {"selected": raw_selected, "grid": raw_grid},
        },
        "final_covariance_fit": {
            "spectrum_rows": final_fit_report["spectrum_rows"],
            "chosen_transforms": {
                variant: final_fit_report["transforms"][variant] for variant in chosen_variants
            },
        },
        "held_inner": held,
        "paired_inner": paired,
        "gates": gates,
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "rule_library_sha256": sha256_file(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_whitened_rule_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_ranks.npz", query=inner, formula=formula,
            baseline_rank=baseline, whitened_true_rank=primary,
            uniform_centered_rank=uniform_rank, raw_rule_rank=raw_rank,
            local_a_rank=ranks["local_a"], local_b_rank=ranks["local_b"],
            local_c_rank=ranks["local_c"],
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "selection": selection,
        "held_inner": held, "paired_inner": paired, "gates": gates,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
