"""Stack the orthogonal candidate residual on the frozen shared ChemAware parent."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import audit_chemaware_candidate_evidence_policy as candidate_policy
import audit_chemaware_orthogonal_rule_residual_policy as orthogonal
from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries, strict_rank
from audit_chemaware_whitened_centered_rule_kernel import FourCenterCache
from chemaware_formula_rule_core import rows_for_queries
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_orthogonal_rule_policy_core import (
    base_and_chemical_feature_indices,
    signed_rule_contrast,
    validate_matched_tables,
)
from chemaware_shrinkage_whitening_core import apply_whitener, fit_shrinkage_whitener
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
CONTROL_NAMES = ("zero_contrast", "reversed_contrast", "alignment_permuted")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-ranks", type=Path, default=ROOT / "data/validation/chemaware_whitened_centered_rule_kernel_v1/inner_ranks.npz")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--rule-library", type=Path, default=ROOT / "dreams/models/chem_aware/chem_rules_data.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_parent_conditioned_residual_policy_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--sampling-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--train-identities", type=int, default=4096)
    parser.add_argument("--validation-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--parent-fit-identities", type=int, default=4096)
    parser.add_argument("--parent-final-fit-identities", type=int, default=8192)
    parser.add_argument("--base-shrinkage", type=float, default=0.50)
    parser.add_argument("--parent-mass-beta", type=float, default=0.40)
    parser.add_argument("--parent-rule-beta", type=float, default=0.80)
    parser.add_argument("--beta", type=float, nargs="+", default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60))
    parser.add_argument("--residual-dose", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0))
    parser.add_argument("--risk-penalty", type=float, default=2.0)
    parser.add_argument("--min-selected-formulas", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class ParentFeatureCache:
    def __init__(self, centers: FourCenterCache, mean: np.ndarray, transform: np.ndarray):
        self.centers = centers
        self.mean = mean
        self.transform = transform
        self.cache = {}

    def get(self, row: int) -> dict[str, np.ndarray]:
        row = int(row)
        if row not in self.cache:
            raw = self.centers.base.get(row)
            centered = self.centers.get(row)
            self.cache[row] = {
                "mass": raw["mass"], "rule_response": raw["rule_response"],
                "rule_response_content_permuted": raw["rule_response_content_permuted"],
                "parent_rule": apply_whitener(
                    np.asarray(centered["true"], dtype=np.float32), self.mean, self.transform,
                ).astype(np.float16),
            }
        return self.cache[row]


def fit_parent_transform(queries, body, centers, shrinkage):
    rows = rows_for_queries(queries, body)
    feature = np.stack([
        np.asarray(centers.get(int(row))["true"], dtype=np.float32) for row in rows
    ])
    mean, transform, report = fit_shrinkage_whitener(feature, float(shrinkage))
    return mean, transform, {"spectrum_rows": int(len(rows)), **report}


def condition_on_parent(scored: dict[str, np.ndarray], mass_beta: float, rule_beta: float) -> dict[str, np.ndarray]:
    output = dict(scored)
    output["global"] = np.asarray([
        np.asarray(global_score, dtype=np.float32)
        + float(mass_beta) * np.asarray(mass, dtype=np.float32)
        + float(rule_beta) * np.asarray(rule, dtype=np.float32)
        for global_score, mass, rule in zip(
            scored["global"], scored["mass"], scored["parent_rule"], strict=True,
        )
    ], dtype=object)
    rank = np.empty(len(output["global"]), dtype=np.int16)
    for index, pair_score in enumerate(output["global"]):
        pointer = np.asarray(scored["reference_ptr"][index], dtype=np.int64)
        molecule_score = np.maximum.reduceat(np.asarray(pair_score, dtype=np.float32), pointer[:-1])
        rank[index] = strict_rank(molecule_score, np.asarray(scored["labels"][index], dtype=bool))
    output["old_rank"] = rank
    return output


def build_tables(queries, body, official, row_position, cache, actions, args):
    scored = score_queries(
        queries, body, official, row_position, cache,
        ("mass", "rule_response", "rule_response_content_permuted", "parent_rule"),
    )
    parent_scored = condition_on_parent(scored, args.parent_mass_beta, args.parent_rule_beta)
    correct = candidate_policy.build_candidate_table(parent_scored, actions, 0, "rule_response")
    control = candidate_policy.build_candidate_table(
        parent_scored, actions, 0, "rule_response_content_permuted",
    )
    validate_matched_tables(correct, control)
    return correct, control


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.smoke:
        args.train_identities = min(args.train_identities, 192)
        args.validation_identities = min(args.validation_identities, 128)
        args.max_inner_identities = 128
        args.parent_fit_identities = min(args.parent_fit_identities, 256)
        args.parent_final_fit_identities = min(args.parent_final_fit_identities, 512)
        args.max_iter = min(args.max_iter, 30)
        args.min_samples_leaf = min(args.min_samples_leaf, 20)
        args.min_selected_formulas = min(args.min_selected_formulas, 10)
        args.bootstrap_draws = min(args.bootstrap_draws, 300)
    actions = [(float(mass), float(rule)) for mass in args.beta for rule in args.beta]
    if actions[0] != (0.0, 0.0):
        raise RuntimeError("parent-conditioned action grid must begin with the no-op")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {name: np.asarray(loaded[name]) for name in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    pools = [np.flatnonzero(np.isin(fold, [0, 1])), np.flatnonzero(fold == 2), np.flatnonzero(fold == 3)]
    queries = [
        identity_balanced_queries(pools[0], body["query_ik14"], np.random.default_rng(args.sampling_seed + 1), args.train_identities),
        identity_balanced_queries(pools[1], body["query_ik14"], np.random.default_rng(args.sampling_seed + 2), args.validation_identities),
        identity_balanced_queries(pools[2], body["query_ik14"], np.random.default_rng(args.sampling_seed + 19), args.max_inner_identities),
    ]
    formulas = [body["query_formula"][query].astype(str) for query in queries]
    if any(set(formulas[i]) & set(formulas[j]) for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("formula split leaked")
    train_formula_fold = fold[queries[0]]
    parent_fit = identity_balanced_queries(
        pools[0], body["query_ik14"], np.random.default_rng(args.fold_seed + 11), args.parent_fit_identities,
    )
    parent_final_fit = identity_balanced_queries(
        np.flatnonzero(np.isin(fold, [0, 1, 2])), body["query_ik14"],
        np.random.default_rng(args.fold_seed + 23), args.parent_final_fit_identities,
    )
    kernel_args = SimpleNamespace(
        token_dir=args.token_dir, rule_library=args.rule_library, top_peaks=32,
        kernel_dim=2048, bin_width=0.02, grid_offsets=4, intensity_power=0.5,
        mass_shift_da=0.137, pair_weight=0.25, multi_bin_widths=(0.01, 0.02, 0.05),
        uniform_channel_weight=1.0, rule_tolerance=0.02, rule_channel_weight=1.0,
    )
    base = KernelCache(kernel_args, row_position, variants=(
        "mass", "rule_response", "rule_response_content_permuted",
        "rule_response_local_background_a", "rule_response_local_background_b",
        "rule_response_local_background_c",
    ))
    centers = FourCenterCache(base)
    initial_mean, initial_transform, initial_transform_report = fit_parent_transform(
        parent_fit, body, centers, args.base_shrinkage,
    )
    initial_cache = ParentFeatureCache(centers, initial_mean, initial_transform)
    train_table, train_control = build_tables(
        queries[0], body, official, row_position, initial_cache, actions, args,
    )
    validation_table, validation_control = build_tables(
        queries[1], body, official, row_position, initial_cache, actions, args,
    )
    del initial_cache
    gc.collect()
    final_mean, final_transform, final_transform_report = fit_parent_transform(
        parent_final_fit, body, centers, args.base_shrinkage,
    )
    inner_cache = ParentFeatureCache(centers, final_mean, final_transform)
    inner_table, inner_control = build_tables(
        queries[2], body, official, row_position, inner_cache, actions, args,
    )
    formal_protocol = (
        args.parent_final_fit_identities == 8192 and args.max_inner_identities == 0
    )
    if formal_protocol:
        frozen = np.load(args.parent_ranks)
        if not np.array_equal(frozen["query"], queries[2]) or not np.array_equal(
            frozen["whitened_true_rank"], inner_table["baseline_rank"],
        ):
            raise RuntimeError("parent-conditioned baseline does not reproduce frozen parent ranks")
    base_indices, chemical_indices = base_and_chemical_feature_indices(candidate_policy.FEATURE_NAMES)
    correct_tables = (train_table, validation_table, inner_table)
    control_tables = (train_control, validation_control, inner_control)
    base_features = [table["feature"][..., base_indices] for table in correct_tables]
    if any(not np.array_equal(base_feature, control["feature"][..., base_indices]) for base_feature, control in zip(base_features, control_tables, strict=True)):
        raise RuntimeError("frozen parent nuisance block changed across chemical controls")
    contrasts = [
        signed_rule_contrast(correct, control, chemical_indices)
        for correct, control in zip(correct_tables, control_tables, strict=True)
    ]
    channels = {
        target: orthogonal.fit_channel(
            train_table, formulas[0], train_formula_fold, base_features[0], contrasts[0],
            target, args, args.seed + (100 if target == "benefit" else 200),
        ) for target in ("benefit", "harmful")
    }
    contrast_arms = []
    for index in range(3):
        arms, _source = orthogonal.make_contrast_arms(
            contrasts[index], correct_tables[index], args.seed + 300 + index,
        )
        contrast_arms.append(arms)
    validation_utilities = {
        float(dose): {
            name: orthogonal.utility_for_contrast(
                channels, base_features[1], contrast, validation_table["valid"],
                float(dose), args.risk_penalty,
            )[0] for name, contrast in contrast_arms[1].items()
        } for dose in args.residual_dose
    }
    selected, selection_grid = orthogonal.select_specific_setting(
        validation_table, formulas[1], validation_utilities,
        args.min_selected_formulas, CONTROL_NAMES,
    )
    evaluated = {}
    for name, contrast in contrast_arms[2].items():
        utility, _detail = orthogonal.utility_for_contrast(
            channels, base_features[2], contrast, inner_table["valid"],
            float(selected["dose"]), args.risk_penalty,
        )
        rank, selected_candidate, best = orthogonal.rank_at_threshold(
            inner_table, utility, float(selected["threshold"]),
        )
        evaluated[name] = {
            "rank": rank, "selected": selected_candidate, "best": best,
            "metric": retrieval(inner_table["baseline_rank"], rank),
        }
    primary = evaluated["correct"]
    comparisons = {
        f"correct_minus_{name}": paired_rank_comparison(
            primary["rank"], evaluated[name]["rank"], formulas[2],
            draws=args.bootstrap_draws, seed=args.seed + 600 + index,
        ) for index, name in enumerate(CONTROL_NAMES)
    }
    parent_to_stacked = primary["metric"]
    if not formal_protocol:
        absolute = None
        absolute_ci = None
    else:
        absolute_baseline = np.load(args.parent_ranks)["baseline_rank"]
        if absolute_baseline.shape != primary["rank"].shape:
            raise RuntimeError("formal absolute baseline and stacked ranks are not aligned")
        absolute = retrieval(absolute_baseline, primary["rank"])
        absolute_ci = bootstrap(
            formulas[2], absolute_baseline, primary["rank"],
            args.bootstrap_draws, args.seed + 500,
        )
    gates = {
        "selected_nonzero_chemical_dose": float(selected["dose"]) > 0,
        "validation_specific_advantage_positive": int(selected["minimum_specific_risk_advantage"]) > 0,
        "increment_over_parent_positive": float(parent_to_stacked["delta_recall1"]) > 0,
        "increment_corrected_exceeds_twice_introduced": int(parent_to_stacked["corrected_at_1"]) > 2 * int(parent_to_stacked["introduced_at_1"]),
        **{f"beats_{name}_ci": comparisons[f"correct_minus_{name}"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0 for name in CONTROL_NAMES},
        "outer_fold_untouched": True,
    }
    report = {
        "status": "CHEMAWARE_PARENT_CONDITIONED_RESIDUAL_POLICY_PASS" if all(gates.values()) else "CHEMAWARE_PARENT_CONDITIONED_RESIDUAL_POLICY_FAIL",
        "formal_training_authorized": False, "weights_updated": False,
        "candidate_conditioned": True, "shared_embedding_result": False,
        "scope": "frozen shared parent plus candidate residual; folds 0-1 fit, fold 2 select, fold 3 inner, fold 4 sealed",
        "claim_limit": "Development stacked policy; only a positive increment over the reproduced frozen parent supports added value.",
        "method": {
            "stage1": "official + 0.4 mass + 0.8 marginal-whitened centered rules",
            "stage2": "benefit-minus-2*harm candidate abstention fit to residual labels relative to stage1",
            "incremental_actions": "nonnegative mass/rule grid including an explicit zero increment",
            "chemical_control": "content-permuted rule response changes only the stage2 chemical contrast; stage1 parent stays fixed",
        },
        "data": {"train_queries": len(queries[0]), "validation_queries": len(queries[1]), "inner_queries": len(queries[2]), "outer_queries_untouched": int(np.sum(fold == 4))},
        "parent_transform": {"initial": initial_transform_report, "final": final_transform_report},
        "crossfit_diagnostics": {name: channel["diagnostics"] for name, channel in channels.items()},
        "selection": selected, "selection_grid_size": len(selection_grid),
        "held_inner_increment_over_parent": {name: value["metric"] for name, value in evaluated.items()},
        "held_inner_absolute_from_official": absolute,
        "held_inner_absolute_formula_ci95": absolute_ci,
        "paired_inner": comparisons, "gates": gates,
        "provenance": {"parent_ranks_sha256": sha256_file(args.parent_ranks), "manifest_sha256": sha256_file(args.manifest), "rule_library_sha256": sha256_file(args.rule_library)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_parent_residual_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(temporary / "inner_policy.npz", query=queries[2], formula=formulas[2], parent_rank=inner_table["baseline_rank"], **{f"{name}_rank": value["rank"] for name, value in evaluated.items()}, selected_candidate_slot=primary["selected"], best_predicted_utility=primary["best"])
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "selection": selected, "held_inner_increment_over_parent": report["held_inner_increment_over_parent"], "held_inner_absolute_from_official": absolute, "paired_inner": comparisons, "gates": gates, "output": str((args.output / "report.json").resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
