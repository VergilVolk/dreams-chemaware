"""Audit a repeatable-chemical-subspace kernel nested in the frozen parent."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_chemaware_empirical_rule_reliability_kernel import fused_ranks, paired_formula_ci, unit
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries
from audit_chemaware_observable_tangent_metric import formula_bootstrap, retrieval
from audit_chemaware_pairwise_psd_margin_kernel import aligned_parent_rank, fit_base_transforms
from audit_chemaware_whitened_centered_rule_kernel import CENTER_NAMES, FourCenterCache
from audit_chemaware_within_identity_whitened_rule_kernel import (
    permute_identity_within_formula,
    row_identity_registry,
)
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_replicate_agreement_core import (
    agreement_transform,
    apply_agreement_transform,
    fit_formula_conditional_contrast_basis,
    fit_replicate_agreement_basis,
)
from chemaware_shrinkage_whitening_core import apply_whitener
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ("constant_parent", "identity_permuted", "local_a", "local_b", "local_c")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-ranks", type=Path, default=ROOT / "data/validation/chemaware_whitened_centered_rule_kernel_v1/inner_ranks.npz")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--rule-library", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_replicate_agreement_kernel_v1")
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--fit-identities", type=int, default=4096)
    parser.add_argument("--validation-identities", type=int, default=0)
    parser.add_argument("--final-fit-identities", type=int, default=8192)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--base-shrinkage", type=float, default=0.50)
    parser.add_argument("--mass-beta", type=float, default=0.40)
    parser.add_argument("--rule-beta", type=float, default=0.80)
    parser.add_argument("--min-replicates", type=int, default=3)
    parser.add_argument("--agreement-floor", type=float, nargs="+", default=(0.0, 0.10, 0.25, 0.50, 0.75))
    parser.add_argument("--agreement-power", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--conditional-contrast", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def fit_bases(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    centers: FourCenterCache,
    base_transforms: dict[str, tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    rows, identity, formula, registry = row_identity_registry(queries, body)
    bases = {}
    reports = {}
    features = {}
    for center in CENTER_NAMES:
        mean, transform = base_transforms[f"base_{center}"]
        feature = np.stack([
            apply_whitener(np.asarray(centers.get(int(row))[center], dtype=np.float32), mean, transform)
            for row in rows
        ])
        features[center] = feature
        if args.conditional_contrast:
            bases[center], reports[center] = fit_formula_conditional_contrast_basis(
                feature, identity, formula, min_replicates=args.min_replicates,
            )
        else:
            bases[center], reports[center] = fit_replicate_agreement_basis(
                feature, identity, min_replicates=args.min_replicates,
            )
    permuted_identity, permutation = permute_identity_within_formula(identity, formula, seed)
    if args.conditional_contrast:
        bases["identity_permuted"], reports["identity_permuted"] = fit_formula_conditional_contrast_basis(
            features["true"], permuted_identity, formula, min_replicates=args.min_replicates,
        )
        bases["role_reversed"], reports["role_reversed"] = fit_formula_conditional_contrast_basis(
            features["true"], identity, formula, min_replicates=args.min_replicates, reverse=True,
        )
    else:
        bases["identity_permuted"], reports["identity_permuted"] = fit_replicate_agreement_basis(
            features["true"], permuted_identity, min_replicates=args.min_replicates,
        )
    return bases, {"registry": registry, "identity_permutation": permutation, "bases": reports}


class AgreementCache:
    def __init__(self, centers, base_transforms, bases, configurations):
        self.centers = centers
        self.base_transforms = base_transforms
        self.configurations = configurations
        self.transforms = {
            name: agreement_transform(bases[basis], floor=floor, power=power)
            for name, (_center, basis, floor, power) in configurations.items()
            if basis != "constant"
        }
        self.cache = {}

    def get(self, row: int) -> dict[str, np.ndarray]:
        row = int(row)
        if row in self.cache:
            return self.cache[row]
        source = self.centers.get(row)
        whitened = {}
        for center in set(value[0] for value in self.configurations.values()):
            mean, transform = self.base_transforms[f"base_{center}"]
            whitened[center] = apply_whitener(
                np.asarray(source[center], dtype=np.float32), mean, transform,
            )
        output = {"mass": source["mass"]}
        for name, (center, basis, _floor, _power) in self.configurations.items():
            output[name] = (
                whitened[center].astype(np.float16) if basis == "constant"
                else apply_agreement_transform(whitened[center], self.transforms[name]).astype(np.float16)
            )
        self.cache[row] = output
        return output


def selection_configurations(args: argparse.Namespace) -> dict[str, tuple[str, str, float, float]]:
    output = {"agreement_constant": ("true", "constant", 1.0, 1.0)}
    for floor in map(float, args.agreement_floor):
        for power in map(float, args.agreement_power):
            token = f"f{floor:g}_p{power:g}".replace(".", "p")
            output[f"agreement_{token}"] = ("true", "true", floor, power)
    return output


def select_configuration(scored, configurations, args):
    table = []
    for name, (_center, basis, floor, power) in configurations.items():
        rank = fused_ranks(scored, name, args.mass_beta, args.rule_beta)
        table.append({"variant": name, "basis": basis, "floor": floor, "power": power, **retrieval(scored["old_rank"], rank)})
    selected = max(table, key=lambda row: (
        int(row["risk_utility_at_1"]), -int(row["introduced_at_1"]),
        float(row["delta_mrr"]), int(row["basis"] == "constant"), float(row["floor"]),
    ))
    return selected, table


def final_configurations(selected, conditional_contrast: bool = False):
    floor, power = float(selected["floor"]), float(selected["power"])
    output = {
        "agreement_true": ("true", "true", floor, power),
        "constant_parent": ("true", "constant", 1.0, 1.0),
        "identity_permuted": ("true", "identity_permuted", floor, power),
        "local_a": ("local_a", "local_a", floor, power),
        "local_b": ("local_b", "local_b", floor, power),
        "local_c": ("local_c", "local_c", floor, power),
    }
    if conditional_contrast:
        output["role_reversed"] = ("true", "role_reversed", floor, power)
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.smoke:
        args.fit_identities = min(args.fit_identities, 1024)
        args.validation_identities = 512
        args.final_fit_identities = min(args.final_fit_identities, 2048)
        args.max_inner_identities = 512
        args.agreement_floor = (0.0, 0.25, 0.50, 0.75)
        args.agreement_power = (0.5, 1.0, 2.0)
        args.bootstrap_draws = min(args.bootstrap_draws, 1000)
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], 5, args.fold_seed)
    fit = identity_balanced_queries(np.flatnonzero(np.isin(fold, [0, 1])), body["query_ik14"], np.random.default_rng(args.seed + 11), args.fit_identities)
    validation = identity_balanced_queries(np.flatnonzero(fold == 2), body["query_ik14"], np.random.default_rng(args.seed + 17), args.validation_identities)
    final_fit = identity_balanced_queries(np.flatnonzero(np.isin(fold, [0, 1, 2])), body["query_ik14"], np.random.default_rng(args.seed + 23), args.final_fit_identities)
    inner = identity_balanced_queries(np.flatnonzero(fold == 3), body["query_ik14"], np.random.default_rng(args.fold_seed + 19), args.max_inner_identities)
    if set(body["query_formula"][final_fit].astype(str)) & set(body["query_formula"][inner].astype(str)):
        raise RuntimeError("agreement fit and inner formulas overlap")
    kernel_args = SimpleNamespace(
        token_dir=args.token_dir, rule_library=args.rule_library, top_peaks=32,
        kernel_dim=2048, bin_width=0.02, grid_offsets=4, intensity_power=0.5,
        mass_shift_da=0.137, pair_weight=0.25, multi_bin_widths=(0.01, 0.02, 0.05),
        uniform_channel_weight=1.0, rule_tolerance=0.02, rule_channel_weight=1.0,
    )
    base = KernelCache(kernel_args, row_position, variants=(
        "mass", "rule_response", "rule_response_local_background_a",
        "rule_response_local_background_b", "rule_response_local_background_c",
    ))
    centers = FourCenterCache(base)
    initial_base, _variants, initial_base_report = fit_base_transforms(fit, body, centers, args.base_shrinkage)
    initial_bases, initial_report = fit_bases(fit, body, centers, initial_base, args, args.seed + 101)
    selection_configs = selection_configurations(args)
    validation_cache = AgreementCache(centers, initial_base, initial_bases, selection_configs)
    validation_scored = score_queries(validation, body, official, row_position, validation_cache, ("mass", *selection_configs))
    selected, selection_grid = select_configuration(validation_scored, selection_configs, args)
    print(json.dumps({"fold2_selected_agreement": selected}, indent=2), flush=True)
    del validation_cache, validation_scored, initial_base, initial_bases
    gc.collect()

    final_base, _variants, final_base_report = fit_base_transforms(final_fit, body, centers, args.base_shrinkage)
    final_bases, final_fit_report = fit_bases(final_fit, body, centers, final_base, args, args.seed + 201)
    configurations = final_configurations(selected, args.conditional_contrast)
    inner_cache = AgreementCache(centers, final_base, final_bases, configurations)
    scored = score_queries(inner, body, official, row_position, inner_cache, ("mass", *configurations))
    baseline = np.asarray(scored["old_rank"])
    formula = np.asarray(scored["formula"]).astype(str)
    ranks = {name: fused_ranks(scored, name, args.mass_beta, args.rule_beta) for name in configurations}
    parent = aligned_parent_rank(args.parent_ranks, inner, formula, baseline)
    if not args.smoke and not np.array_equal(ranks["constant_parent"], parent):
        raise RuntimeError("agreement constant arm does not reproduce the frozen parent")
    held = {name: {
        "retrieval": retrieval(baseline, rank),
        "increment_over_constant_parent": retrieval(ranks["constant_parent"], rank),
        "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
            formula, baseline, rank, draws=args.bootstrap_draws,
            seed=args.seed + 300 + list(configurations).index(name),
        ),
    } for name, rank in ranks.items()}
    controls = CONTROLS + (("role_reversed",) if args.conditional_contrast else ())
    comparisons = {f"agreement_true_minus_{name}": paired_formula_ci(
        formula, ranks["agreement_true"], ranks[name], args, 400 + index,
    ) for index, name in enumerate(controls)}
    target = held["agreement_true"]
    gates = {
        "nonconstant_agreement_selected": selected["basis"] != "constant",
        "absolute_formula_ci_positive": target["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
        "corrected_exceeds_twice_introduced": target["retrieval"]["corrected_at_1"] > 2 * target["retrieval"]["introduced_at_1"],
        **{f"beats_{name}_ci": comparisons[f"agreement_true_minus_{name}"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0 for name in controls},
        "outer_fold_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_FORMULA_CONDITIONAL_REPLICATE_CONTRAST_KERNEL_PASS"
            if args.conditional_contrast and all(gates.values())
            else "CHEMAWARE_FORMULA_CONDITIONAL_REPLICATE_CONTRAST_KERNEL_FAIL"
            if args.conditional_contrast
            else "CHEMAWARE_REPLICATE_AGREEMENT_KERNEL_PASS"
            if all(gates.values()) else "CHEMAWARE_REPLICATE_AGREEMENT_KERNEL_FAIL"
        ),
        "formal_training_authorized": False, "weights_updated": False,
        "scope": "folds 0-1 agreement fit; fold 2 map selection; folds 0-2 refit; inner fold 3 evaluation; outer fold 4 sealed",
        "claim_limit": "Closed-form shared-kernel development evidence; not DreaMS fine-tuning or external confirmation.",
        "method": {
            "model": "centered_rule(h,i)=latent_chemistry(h)+acquisition_noise(h,i)",
            "objective": (
                "formula-equal within-identity distinct-spectrum cross moment minus same-formula different-identity centroid cross moment"
                if args.conditional_contrast else
                "identity-equal mean cross-covariance of distinct same-identity spectra"
            ),
            "map": "parent-whitened centered rules rotated and nonnegatively weighted by repeat-agreement eigenvalue",
            "nested_parent": "agreement floor 1 is exactly the frozen marginal-whitening cosine kernel",
            "negative_control": "identity labels permuted within formula with multiplicities preserved before basis fitting",
            "same_map_for_query_and_reference": True,
            "candidate_formula_identity_free_at_deployment": True,
        },
        "data": {"fit_queries": len(fit), "validation_queries": len(validation), "final_fit_queries": len(final_fit), "inner_queries": len(inner), "outer_queries_untouched": int(np.sum(fold == 4))},
        "initial_base_fit": initial_base_report, "initial_agreement_fit": initial_report,
        "fold2_selection": {"selected": selected, "grid": selection_grid},
        "final_base_fit": final_base_report, "final_agreement_fit": final_fit_report,
        "held_inner": held, "paired_inner": comparisons, "gates": gates,
        "provenance": {"manifest_sha256": sha256_file(args.manifest), "token_report_sha256": sha256_file(args.token_dir / "report.json"), "rule_library_sha256": sha256_file(args.rule_library), "parent_ranks_sha256": sha256_file(args.parent_ranks)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_replicate_agreement_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(temporary / "inner_ranks.npz", query=inner, formula=formula, baseline_rank=baseline, **{f"{name}_rank": rank for name, rank in ranks.items()})
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "selected": selected, "held_inner": held, "paired_inner": comparisons, "gates": gates, "output": str((args.output / "report.json").resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
