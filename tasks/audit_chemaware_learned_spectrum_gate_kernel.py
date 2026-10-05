"""Audit a learned spectrum-only gate over the best shared ChemAware kernel."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_chemaware_empirical_rule_reliability_kernel import fused_ranks, paired_formula_ci
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries
from audit_chemaware_observable_tangent_metric import formula_bootstrap, retrieval
from audit_chemaware_pairwise_psd_margin_kernel import aligned_parent_rank, fit_base_transforms
from audit_chemaware_whitened_centered_rule_kernel import CENTER_NAMES, FourCenterCache, TransformCache
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_shrinkage_whitening_core import apply_whitener
from chemaware_spectrum_gate_core import (
    SCALAR_NAMES,
    fit_cross_split_consensus_gate,
    fit_logistic_gate,
    gate_amplitude,
    gate_probability,
    spectrum_scalar_vector,
)
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
CONTROL_ARMS = ("constant_parent", "label_permuted", "gate_reversed", "local_a", "local_b", "local_c")


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
        default=ROOT / "data/validation/chemaware_learned_spectrum_gate_kernel_v1",
    )
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--fit-identities", type=int, default=4096)
    parser.add_argument("--validation-identities", type=int, default=0)
    parser.add_argument("--final-fit-identities", type=int, default=8192)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--base-shrinkage", type=float, default=0.50)
    parser.add_argument("--mass-beta", type=float, default=0.40)
    parser.add_argument("--rule-beta", type=float, default=0.80)
    parser.add_argument("--gate-floor", type=float, nargs="+", default=(0.0, 0.25, 0.50, 0.75, 0.90))
    parser.add_argument("--gate-power", type=float, nargs="+", default=(0.5, 1.0, 2.0, 4.0))
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class LearnedGateCache:
    """Materialize shared gated features from one spectrum at a time."""

    def __init__(
        self,
        centers: FourCenterCache,
        base_transforms: dict[str, tuple[np.ndarray, np.ndarray]],
        models: dict[str, dict[str, np.ndarray | float]],
        configurations: dict[str, tuple[str, str, float, float]],
    ) -> None:
        self.centers = centers
        self.base_transforms = base_transforms
        self.models = models
        self.configurations = configurations
        self.cache: dict[int, dict[str, np.ndarray]] = {}

    def get(self, row: int) -> dict[str, np.ndarray]:
        row = int(row)
        if row in self.cache:
            return self.cache[row]
        source = self.centers.get(row)
        scalar = spectrum_scalar_vector(self.centers.base, row)[None, :]
        probability = {
            name: float(gate_probability(scalar, model)[0]) for name, model in self.models.items()
        }
        chemical = {}
        for center in set(config[0] for config in self.configurations.values()):
            mean, transform = self.base_transforms[f"base_{center}"]
            chemical[center] = apply_whitener(
                np.asarray(source[center], dtype=np.float32), mean, transform,
            ).astype(np.float16)
        output = {"mass": source["mass"]}
        for variant, (center, model_kind, floor, power) in self.configurations.items():
            if model_kind == "constant":
                amplitude = 1.0
            else:
                value = probability[model_kind]
                if model_kind == "reversed":
                    value = 1.0 - value
                amplitude = float(gate_amplitude(np.asarray([value]), floor, power)[0])
            output[variant] = (amplitude * np.asarray(chemical[center], dtype=np.float32)).astype(np.float16)
        self.cache[row] = output
        return output


def fit_models_from_outcomes(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    scored: dict[str, np.ndarray],
    cache: KernelCache,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray | float]], dict[str, object]]:
    chemical_rank = fused_ranks(scored, "base_true", args.mass_beta, args.rule_beta)
    baseline_rank = np.asarray(scored["old_rank"])
    corrected = (baseline_rank > 1) & (chemical_rank == 1)
    introduced = (baseline_rank == 1) & (chemical_rank > 1)
    changed = corrected | introduced
    if int(corrected.sum()) < 10 or int(introduced.sum()) < 5:
        raise RuntimeError("too few correction/harm outcomes to fit a spectrum gate")
    query_rows = np.asarray(body["query_row"][queries], dtype=np.int64)
    feature = np.stack([spectrum_scalar_vector(cache, int(row)) for row in query_rows[changed]])
    label = corrected[changed].astype(np.int8)
    formula = np.asarray(body["query_formula"][queries]).astype(str)[changed]
    actual, actual_report = fit_cross_split_consensus_gate(
        feature, label, formula, seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    permuted_label = label[rng.permutation(len(label))]
    permuted, permuted_report = fit_cross_split_consensus_gate(
        feature, permuted_label, formula, seed=seed + 2,
    )
    coefficient = np.asarray(actual_report["consensus_coefficient"], dtype=np.float64)
    coefficient_rows = sorted(
        ({"feature": name, "coefficient": float(value)} for name, value in zip(SCALAR_NAMES, coefficient)),
        key=lambda row: abs(row["coefficient"]), reverse=True,
    )
    return {"actual": actual, "permuted": permuted, "reversed": actual}, {
        "queries": int(len(queries)), "corrected": int(corrected.sum()),
        "introduced": int(introduced.sum()), "changed": int(changed.sum()),
        "actual": actual_report, "label_permuted": permuted_report,
        "actual_coefficients_ranked": coefficient_rows,
        "permuted_label_count_preserved": bool(np.sum(permuted_label) == np.sum(label)),
    }


def selection_configurations(args: argparse.Namespace) -> dict[str, tuple[str, str, float, float]]:
    output = {"gate_true_constant": ("true", "constant", 1.0, 1.0)}
    for floor in map(float, args.gate_floor):
        for power in map(float, args.gate_power):
            ftoken = str(floor).replace(".", "p")
            ptoken = str(power).replace(".", "p")
            output[f"gate_true_f{ftoken}_p{ptoken}"] = ("true", "actual", floor, power)
    return output


def select_gate(
    scored: dict[str, np.ndarray],
    configurations: dict[str, tuple[str, str, float, float]],
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    for variant, (_, kind, floor, power) in configurations.items():
        rank = fused_ranks(scored, variant, args.mass_beta, args.rule_beta)
        rows.append({
            "variant": variant, "kind": kind, "floor": floor, "power": power,
            **retrieval(scored["old_rank"], rank),
        })
    selected = max(
        rows,
        key=lambda row: (
            int(row["risk_utility_at_1"]), -int(row["introduced_at_1"]),
            float(row["delta_mrr"]), int(row["kind"] == "constant"),
            float(row["floor"]), -float(row["power"]),
        ),
    )
    return selected, rows


def final_configurations(selected: dict[str, object]) -> dict[str, tuple[str, str, float, float]]:
    floor = float(selected["floor"])
    power = float(selected["power"])
    return {
        "learned_true": ("true", "actual", floor, power),
        "constant_parent": ("true", "constant", 1.0, 1.0),
        "label_permuted": ("true", "permuted", floor, power),
        "gate_reversed": ("true", "reversed", floor, power),
        "local_a": ("local_a", "actual", floor, power),
        "local_b": ("local_b", "actual", floor, power),
        "local_c": ("local_c", "actual", floor, power),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.smoke:
        args.fit_identities = min(args.fit_identities, 1024)
        args.validation_identities = 512
        args.final_fit_identities = min(args.final_fit_identities, 2048)
        args.max_inner_identities = 512
        args.gate_floor = (0.0, 0.5, 0.9)
        args.gate_power = (0.5, 1.0, 2.0)
        args.bootstrap_draws = min(args.bootstrap_draws, 1000)
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], 5, args.fold_seed)
    fit_queries = identity_balanced_queries(
        np.flatnonzero(np.isin(fold, [0, 1])), body["query_ik14"],
        np.random.default_rng(20260905 + 11), args.fit_identities,
    )
    validation = identity_balanced_queries(
        np.flatnonzero(fold == 2), body["query_ik14"],
        np.random.default_rng(20260905 + 17), args.validation_identities,
    )
    final_fit = identity_balanced_queries(
        np.flatnonzero(np.isin(fold, [0, 1, 2])), body["query_ik14"],
        np.random.default_rng(20260905 + 23), args.final_fit_identities,
    )
    inner = identity_balanced_queries(
        np.flatnonzero(fold == 3), body["query_ik14"],
        np.random.default_rng(args.fold_seed + 19), args.max_inner_identities,
    )
    if set(body["query_formula"][final_fit].astype(str)) & set(body["query_formula"][inner].astype(str)):
        raise RuntimeError("gate fit and inner formulas overlap")
    kernel_args = SimpleNamespace(
        token_dir=args.token_dir, rule_library=args.rule_library,
        top_peaks=32, kernel_dim=2048, bin_width=0.02, grid_offsets=4,
        intensity_power=0.5, mass_shift_da=0.137, pair_weight=0.25,
        multi_bin_widths=(0.01, 0.02, 0.05), uniform_channel_weight=1.0,
        rule_tolerance=0.02, rule_channel_weight=1.0,
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
    fit_base_cache = TransformCache(centers, fit_base, fit_base_variants)
    fit_scored = score_queries(
        fit_queries, body, official, row_position, fit_base_cache, ("mass", "base_true"),
    )
    fit_models, fit_model_report = fit_models_from_outcomes(
        fit_queries, body, fit_scored, base, args, args.seed + 101,
    )
    configurations = selection_configurations(args)
    validation_cache = LearnedGateCache(centers, fit_base, fit_models, configurations)
    validation_scored = score_queries(
        validation, body, official, row_position, validation_cache,
        ("mass", *configurations),
    )
    selected, selection_grid = select_gate(validation_scored, configurations, args)
    print(json.dumps({"fold2_selected_gate": selected}, indent=2), flush=True)
    del fit_scored, fit_base_cache, validation_scored, validation_cache, fit_base, fit_models
    gc.collect()

    final_base, final_base_variants, final_base_report = fit_base_transforms(
        final_fit, body, centers, args.base_shrinkage,
    )
    final_base_cache = TransformCache(centers, final_base, final_base_variants)
    final_fit_scored = score_queries(
        final_fit, body, official, row_position, final_base_cache, ("mass", "base_true"),
    )
    final_models, final_model_report = fit_models_from_outcomes(
        final_fit, body, final_fit_scored, base, args, args.seed + 201,
    )
    final_configs = final_configurations(selected)
    inner_cache = LearnedGateCache(centers, final_base, final_models, final_configs)
    inner_scored = score_queries(
        inner, body, official, row_position, inner_cache, ("mass", *final_configs),
    )
    baseline = np.asarray(inner_scored["old_rank"])
    formula = np.asarray(inner_scored["formula"]).astype(str)
    ranks = {
        name: fused_ranks(inner_scored, name, args.mass_beta, args.rule_beta)
        for name in final_configs
    }
    parent_rank = aligned_parent_rank(args.parent_ranks, inner, formula, baseline)
    if not args.smoke and not np.array_equal(ranks["constant_parent"], parent_rank):
        raise RuntimeError("constant learned-gate arm does not reproduce the frozen parent")
    held = {
        name: {
            "retrieval": retrieval(baseline, rank),
            "increment_over_constant_parent": retrieval(ranks["constant_parent"], rank),
            "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
                formula, baseline, rank, draws=args.bootstrap_draws,
                seed=args.seed + 300 + list(final_configs).index(name),
            ),
        }
        for name, rank in ranks.items()
    }
    primary = ranks["learned_true"]
    comparisons = {
        f"learned_true_minus_{name}": paired_formula_ci(
            formula, primary, ranks[name], args, 400 + index,
        )
        for index, name in enumerate(CONTROL_ARMS)
    }
    target = held["learned_true"]
    gates = {
        "nonconstant_gate_selected": selected["kind"] != "constant",
        "absolute_formula_ci_positive": target["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
        "corrected_exceeds_twice_introduced": (
            int(target["retrieval"]["corrected_at_1"])
            > 2 * int(target["retrieval"]["introduced_at_1"])
        ),
        **{
            f"beats_{name}_ci": comparisons[f"learned_true_minus_{name}"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0
            for name in CONTROL_ARMS
        },
        "outer_fold_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_LEARNED_SPECTRUM_GATE_KERNEL_PASS"
            if all(gates.values()) else "CHEMAWARE_LEARNED_SPECTRUM_GATE_KERNEL_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "folds 0-1 gate fit; fold 2 gate-shape selection; folds 0-2 refit; inner fold 3 evaluation; outer fold 4 sealed",
        "claim_limit": "Shared-kernel development evidence on an already-used inner fold; not DreaMS fine-tuning or external confirmation.",
        "method": {
            "map": "[official, sqrt(0.4)*mass, sqrt(0.8)*sqrt(g(x))*whitened_centered_rule]",
            "gate_input": list(SCALAR_NAMES),
            "gate_model": "formula-binary cross-fit sign-consensus of fixed-C balanced logistic coefficients",
            "gate_strength": "floor + (1-floor)*P(correction|single spectrum)^power",
            "same_map_for_query_and_reference": True,
            "candidate_formula_identity_free_at_deployment": True,
            "base_fusion_fixed": {"mass_beta": args.mass_beta, "rule_beta": args.rule_beta},
        },
        "data": {
            "fit_queries": int(len(fit_queries)), "validation_queries": int(len(validation)),
            "final_fit_queries": int(len(final_fit)), "inner_queries": int(len(inner)),
            "outer_queries_untouched": int(np.sum(fold == 4)),
        },
        "initial_base_fit": fit_base_report,
        "initial_gate_fit": fit_model_report,
        "fold2_selection": {"selected": selected, "grid": selection_grid},
        "final_base_fit": final_base_report,
        "final_gate_fit": final_model_report,
        "held_inner": held,
        "constant_parent_retrieval": retrieval(baseline, ranks["constant_parent"]),
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
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_learned_gate_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_ranks.npz", query=inner, formula=formula,
            baseline_rank=baseline, **{f"{name}_rank": rank for name, rank in ranks.items()},
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "fit_gate_outcomes": fit_model_report,
        "selected": selected, "final_gate_outcomes": final_model_report,
        "held_inner": held, "paired_inner": comparisons, "gates": gates,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
