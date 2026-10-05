"""Test degree-two interactions on the frozen whitened ChemAware kernel."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval
from audit_chemaware_empirical_rule_reliability_kernel import paired_formula_ci
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries, strict_rank
from audit_chemaware_whitened_centered_rule_kernel import (
    CENTER_NAMES,
    FourCenterCache,
    TransformCache,
    fit_transforms,
)
from chemaware_iceberg_direct_core import stable_formula_folds
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-report", type=Path,
        default=ROOT / "data/validation/chemaware_whitened_centered_rule_kernel_v1/report.json",
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
        default=ROOT / "data/validation/chemaware_whitened_rule_interaction_kernel_v1",
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
        "--interaction-beta", type=float, nargs="+",
        default=(0.0, 0.0125, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def interaction_ranks(
    scored: dict[str, np.ndarray],
    variant: str,
    *,
    mass_beta: float,
    linear_beta: float,
    interaction_beta: float,
) -> np.ndarray:
    output = np.empty(len(scored["query"]), dtype=np.int16)
    for index in range(len(output)):
        linear = np.asarray(scored[variant][index], dtype=np.float32)
        score = (
            np.asarray(scored["global"][index], dtype=np.float32)
            + float(mass_beta) * np.asarray(scored["mass"][index], dtype=np.float32)
            + float(linear_beta) * linear
            + float(interaction_beta) * linear * linear
        )
        pointer = np.asarray(scored["reference_ptr"][index], dtype=np.int64)
        molecule = np.maximum.reduceat(score, pointer[:-1])
        output[index] = strict_rank(
            molecule, np.asarray(scored["labels"][index], dtype=bool),
        )
    return output


def select_interaction(
    scored: dict[str, np.ndarray],
    variant: str,
    parent: dict[str, object],
    grid: tuple[float, ...] | list[float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    for beta in map(float, grid):
        rank = interaction_ranks(
            scored, variant,
            mass_beta=float(parent["mass_beta"]),
            linear_beta=float(parent["rule_beta"]),
            interaction_beta=beta,
        )
        rows.append({
            "interaction_beta": beta,
            **retrieval(np.asarray(scored["old_rank"]), rank),
        })
    selected = max(
        rows,
        key=lambda row: (
            int(row["risk_utility_at_1"]), -int(row["introduced_at_1"]),
            float(row["delta_mrr"]), -float(row["interaction_beta"]),
        ),
    )
    return selected, rows


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.smoke:
        args.fit_identities = min(args.fit_identities, 256)
        args.validation_identities = 128
        args.final_fit_identities = min(args.final_fit_identities, 512)
        args.max_inner_identities = 128
        args.interaction_beta = (0.0, 0.05, 0.2, 0.8)
        args.bootstrap_draws = min(args.bootstrap_draws, 300)
    parent_report = json.loads(args.parent_report.read_text(encoding="utf-8"))
    parent_selection = parent_report["fold2_selection"]["whitened"]
    if not all(parent_report["gates"][name] for name in (
        "absolute_formula_ci_positive", "corrected_exceeds_twice_introduced",
        "beats_raw_rule_ci", "beats_local_a_ci", "beats_local_b_ci", "beats_local_c_ci",
        "outer_fold_untouched",
    )):
        raise RuntimeError("parent whitened kernel lacks required chemistry/safety evidence")
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
    shrinkage = tuple(sorted({
        float(parent_report["final_covariance_fit"]["chosen_transforms"][
            parent_selection[name]["variant"]
        ]["shrinkage"])
        for name in CENTER_NAMES
    }))
    fit_transform_all, fit_variants_all, _ = fit_transforms(
        fit_queries, body, centers, shrinkage,
    )
    chosen_variants = {parent_selection[name]["variant"]: name for name in CENTER_NAMES}
    fit_transform = {name: fit_transform_all[name] for name in chosen_variants}
    validation_cache = TransformCache(centers, fit_transform, chosen_variants)
    validation_scored = score_queries(
        validation, body, official, row_position, validation_cache,
        ("mass", *chosen_variants),
    )
    selection = {}
    grids = {}
    for name in CENTER_NAMES:
        variant = parent_selection[name]["variant"]
        selected, table = select_interaction(
            validation_scored, variant, parent_selection[name], args.interaction_beta,
        )
        selection[name] = selected; grids[name] = table
    print(json.dumps({"interaction_selection": selection}, indent=2), flush=True)

    final_transform_all, _, _ = fit_transforms(final_fit, body, centers, shrinkage)
    final_transform = {name: final_transform_all[name] for name in chosen_variants}
    inner_cache = TransformCache(centers, final_transform, chosen_variants)
    inner_scored = score_queries(
        inner, body, official, row_position, inner_cache, ("mass", *chosen_variants),
    )
    baseline = np.asarray(inner_scored["old_rank"])
    formula = np.asarray(inner_scored["formula"]).astype(str)
    ranks = {}
    linear_ranks = {}
    held = {}
    for name in CENTER_NAMES:
        parent = parent_selection[name]
        variant = parent["variant"]
        selected = selection[name]
        rank = interaction_ranks(
            inner_scored, variant,
            mass_beta=parent["mass_beta"], linear_beta=parent["rule_beta"],
            interaction_beta=selected["interaction_beta"],
        )
        linear = interaction_ranks(
            inner_scored, variant,
            mass_beta=parent["mass_beta"], linear_beta=parent["rule_beta"],
            interaction_beta=0.0,
        )
        ranks[name] = rank; linear_ranks[name] = linear
        held[name] = {
            "parent": parent, "interaction_selected": selected,
            "retrieval": retrieval(baseline, rank),
            "formula_cluster_bootstrap_delta_recall1_ci95": bootstrap(
                formula, baseline, rank, args.bootstrap_draws,
                args.seed + 300 + CENTER_NAMES.index(name),
            ),
        }
    primary = ranks["true"]
    paired = {
        "interaction_true_minus_linear_true": paired_formula_ci(
            formula, primary, linear_ranks["true"], args, 400,
        ),
        **{
            f"interaction_true_minus_interaction_{name}": paired_formula_ci(
                formula, primary, ranks[name], args, 410 + index,
            )
            for index, name in enumerate(("local_a", "local_b", "local_c"))
        },
    }
    primary_held = held["true"]
    gates = {
        "nonzero_interaction_selected": selection["true"]["interaction_beta"] > 0,
        "absolute_formula_ci_positive": primary_held[
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ][0] > 0,
        "corrected_exceeds_twice_introduced": (
            int(primary_held["retrieval"]["corrected_at_1"])
            > 2 * int(primary_held["retrieval"]["introduced_at_1"])
        ),
        "beats_linear_parent_ci": paired["interaction_true_minus_linear_true"][
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ][0] > 0,
        **{
            f"beats_{name}_ci": paired[f"interaction_true_minus_interaction_{name}"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0
            for name in ("local_a", "local_b", "local_c")
        },
        "outer_fold_untouched": True,
    }
    dimension = len(base.nl_rules) + len(base.cf_rules)
    report = {
        "status": (
            "CHEMAWARE_WHITENED_RULE_INTERACTION_KERNEL_PASS"
            if all(gates.values()) else "CHEMAWARE_WHITENED_RULE_INTERACTION_KERNEL_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "parent fold-2 linear selection plus fold-2 interaction selection; inner fold 3 evaluation; outer fold 4 sealed",
        "claim_limit": "Exploratory degree-two shared-kernel result on an already-used inner fold; not compact embedding fine-tuning or external confirmation.",
        "method": {
            "kernel": "official + beta_m*mass + beta_1*(u.v) + beta_2*(u.v)^2",
            "primal_map": "[official, sqrt(beta_m)*mass, sqrt(beta_1)*u, sqrt(beta_2)*vec(u tensor u)]",
            "psd_by_construction": True,
            "candidate_formula_identity_free": True,
            "whitened_rule_dimension": int(dimension),
            "exact_degree_two_dimension": int(dimension * dimension),
        },
        "parent": {
            "report": str(args.parent_report), "report_sha256": sha256_file(args.parent_report),
            "selection": parent_selection,
        },
        "fold2_interaction_selection": {
            name: {"selected": selection[name], "grid": grids[name]} for name in CENTER_NAMES
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
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_whitened_interaction_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_ranks.npz", query=inner, formula=formula,
            baseline_rank=baseline, interaction_true_rank=primary,
            linear_true_rank=linear_ranks["true"],
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
