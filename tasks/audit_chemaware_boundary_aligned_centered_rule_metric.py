"""Audit robust rule-channel weights at the actual retrieval boundary.

Unlike average-positive/average-negative reliability, this audit estimates
each centered chemical channel on the best positive reference versus the
hardest negative molecule.  Two independently sampled formula-disjoint fits
must agree before a channel receives positive weight.  The resulting
nonnegative diagonal metric remains a deployable shared PSD embedding.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_chemaware_empirical_rule_reliability_kernel import (
    fused_ranks,
    grid,
    paired_formula_ci,
    unit,
)
from audit_chemaware_mass_kernel_embedding import KernelCache, score_queries
from audit_chemaware_observable_tangent_metric import formula_bootstrap, retrieval
from chemaware_empirical_rule_reliability_core import (
    aggregate_formula_contrasts,
    prevalence_matched_permutation,
    reliability_weights,
)
from chemaware_formula_rule_core import rows_for_queries
from chemaware_iceberg_direct_core import stable_formula_folds
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "boundary_consensus_centered",
    "uniform_centered",
    "idf_centered",
    "prevalence_matched_weight_permutation",
    "raw_rule",
)


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
        default=ROOT / "data/validation/chemaware_boundary_aligned_centered_rule_metric_v1",
    )
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--fit-identities", type=int, default=2048)
    parser.add_argument("--validation-identities", type=int, default=0)
    parser.add_argument("--final-fit-identities", type=int, default=4096)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--base-mass-beta", type=float, default=0.2)
    parser.add_argument("--base-rule-beta", type=float, default=0.4)
    parser.add_argument(
        "--beta", type=float, nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class CenteredRuleCache:
    """Create true-center residuals without changing the frozen base cache."""

    def __init__(self, base: KernelCache) -> None:
        self.base = base
        self.cache: dict[int, dict[str, np.ndarray]] = {}

    def get(self, row: int) -> dict[str, np.ndarray]:
        row = int(row)
        if row not in self.cache:
            raw = self.base.get(row)
            center = np.asarray(raw["rule_response"], dtype=np.float32)
            background = np.mean(np.stack((
                np.asarray(raw["rule_response_local_background_a"], dtype=np.float32),
                np.asarray(raw["rule_response_local_background_b"], dtype=np.float32),
                np.asarray(raw["rule_response_local_background_c"], dtype=np.float32),
            )), axis=0)
            self.cache[row] = {
                "mass": raw["mass"],
                "centered": unit(center - background),
                "raw_rule": raw["rule_response"],
            }
        return self.cache[row]


class WeightedCenteredCache:
    """Expose matched weighted centered-rule maps to the frozen scorer."""

    def __init__(
        self,
        base: CenteredRuleCache,
        empirical: np.ndarray,
        idf: np.ndarray,
        permuted: np.ndarray,
    ) -> None:
        self.base = base
        self.sqrt_weight = {
            "boundary_consensus_centered": np.sqrt(np.maximum(empirical, 0.0)),
            "uniform_centered": np.ones_like(empirical, dtype=np.float32),
            "idf_centered": np.sqrt(np.maximum(idf, 0.0)),
            "prevalence_matched_weight_permutation": np.sqrt(np.maximum(permuted, 0.0)),
        }
        self.cache: dict[int, dict[str, np.ndarray]] = {}

    def get(self, row: int) -> dict[str, np.ndarray]:
        row = int(row)
        if row not in self.cache:
            source = self.base.get(row)
            centered = np.asarray(source["centered"], dtype=np.float32)
            output = {"mass": source["mass"], "raw_rule": source["raw_rule"]}
            for name, sqrt_weight in self.sqrt_weight.items():
                output[name] = unit(centered * sqrt_weight)
            self.cache[row] = output
        return self.cache[row]


def hard_boundary_channel_contrasts(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    official: np.ndarray,
    row_position: dict[int, int],
    cache: CenteredRuleCache,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return per-query channel evidence at best-positive/hard-negative pairs."""
    dimension = len(cache.base.nl_rules) + len(cache.base.cf_rules)
    output = np.zeros((len(queries), dimension), dtype=np.float32)
    base_margin = np.empty(len(queries), dtype=np.float32)
    positive_pair_score = np.empty(len(queries), dtype=np.float32)
    negative_pair_score = np.empty(len(queries), dtype=np.float32)
    for out_index, query in enumerate(map(int, queries)):
        qrow = int(body["query_row"][query])
        qpos = row_position[qrow]
        q = cache.get(qrow)
        qmass = np.asarray(q["mass"], dtype=np.float32)
        qrule = np.asarray(q["centered"], dtype=np.float32)
        left, right = map(int, body["query_ptr"][query:query + 2])
        labels = body["molecule_label"][left:right].astype(bool)
        pair_left = int(body["molecule_ptr"][left])
        pair_right = int(body["molecule_ptr"][right])
        reference_rows = body["pair_candidate_row"][pair_left:pair_right]
        reference_official = official[[row_position[int(row)] for row in reference_rows]]
        reference_mass = np.stack([
            np.asarray(cache.get(int(row))["mass"], dtype=np.float32) for row in reference_rows
        ])
        reference_rule = np.stack([
            np.asarray(cache.get(int(row))["centered"], dtype=np.float32) for row in reference_rows
        ])
        pair_score = (
            reference_official @ official[qpos]
            + float(args.base_mass_beta) * (reference_mass @ qmass)
            + float(args.base_rule_beta) * (reference_rule @ qrule)
        )
        local_pointer = body["molecule_ptr"][left:right + 1].astype(np.int64) - pair_left
        best_scores = np.empty(right - left, dtype=np.float32)
        best_rules = np.empty((right - left, dimension), dtype=np.float32)
        for offset, (start, stop) in enumerate(zip(local_pointer[:-1], local_pointer[1:])):
            best = int(start + np.argmax(pair_score[start:stop]))
            best_scores[offset] = float(pair_score[best])
            best_rules[offset] = reference_rule[best]
        positive_index = np.flatnonzero(labels)
        negative_index = np.flatnonzero(~labels)
        if not len(positive_index) or not len(negative_index):
            raise RuntimeError("each fit query needs positive and negative molecules")
        positive = int(positive_index[np.argmax(best_scores[positive_index])])
        negative = int(negative_index[np.argmax(best_scores[negative_index])])
        output[out_index] = qrule * (best_rules[positive] - best_rules[negative])
        positive_pair_score[out_index] = best_scores[positive]
        negative_pair_score[out_index] = best_scores[negative]
        base_margin[out_index] = best_scores[positive] - best_scores[negative]
        if (out_index + 1) % 256 == 0:
            print(f"hard-boundary contrasts {out_index + 1}/{len(queries)}", flush=True)
    return output, {
        "queries": int(len(queries)),
        "base_margin_quantiles": np.quantile(
            base_margin, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
        ).astype(float).tolist(),
        "baseline_boundary_error_fraction": float(np.mean(base_margin <= 0.0)),
        "positive_pair_score_mean": float(positive_pair_score.mean()),
        "negative_pair_score_mean": float(negative_pair_score.mean()),
    }


def fit_one(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    official: np.ndarray,
    row_position: dict[int, int],
    cache: CenteredRuleCache,
    args: argparse.Namespace,
    *,
    seed: int,
    document_frequency: np.ndarray,
    documents: int,
) -> tuple[np.ndarray, dict[str, object]]:
    contrast, boundary_report = hard_boundary_channel_contrasts(
        queries, body, official, row_position, cache, args,
    )
    formula_contrast, formula_names = aggregate_formula_contrasts(
        contrast, body["query_formula"][queries],
    )
    weight, weight_report = reliability_weights(
        formula_contrast, formula_names, document_frequency, documents, seed=seed,
    )
    return weight, {
        "queries": int(len(queries)), "formulas": int(len(formula_names)),
        "boundary": boundary_report, "weight": weight_report,
    }


def robust_fit(
    first_queries: np.ndarray,
    second_queries: np.ndarray,
    body: dict[str, np.ndarray],
    official: np.ndarray,
    row_position: dict[int, int],
    cache: CenteredRuleCache,
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train_rows = rows_for_queries(np.unique(np.concatenate((first_queries, second_queries))), body)
    response = np.stack([
        np.abs(np.asarray(cache.get(int(row))["centered"], dtype=np.float32))
        for row in train_rows
    ])
    document_frequency = np.sum(response > 1e-8, axis=0)
    documents = len(response)
    idf = (np.log((documents + 1.0) / (document_frequency + 1.0)) + 1.0).astype(np.float32)
    first, first_report = fit_one(
        first_queries, body, official, row_position, cache, args, seed=seed + 1,
        document_frequency=document_frequency, documents=documents,
    )
    second, second_report = fit_one(
        second_queries, body, official, row_position, cache, args, seed=seed + 2,
        document_frequency=document_frequency, documents=documents,
    )
    consensus = np.sqrt(np.maximum(first, 0.0) * np.maximum(second, 0.0)).astype(np.float32)
    positive = consensus > 0
    if np.any(positive):
        consensus /= np.median(consensus[positive])
    permuted, permutation_report = prevalence_matched_permutation(
        consensus, document_frequency, len(cache.base.nl_rules), seed=seed + 3,
    )
    return {
        "empirical": consensus, "idf": idf, "permuted": permuted,
        "document_frequency": document_frequency.astype(np.int64),
    }, {
        "first": first_report, "second": second_report,
        "independent_positive_intersection_channels": int(np.sum(positive)),
        "first_positive_channels": int(np.sum(first > 0)),
        "second_positive_channels": int(np.sum(second > 0)),
        "positive_jaccard": float(
            np.sum((first > 0) & (second > 0)) / max(1, np.sum((first > 0) | (second > 0)))
        ),
        "consensus_weight_quantiles": (
            np.quantile(consensus[positive], (0, 0.25, 0.5, 0.75, 1)).astype(float).tolist()
            if np.any(positive) else [0.0] * 5
        ),
        "permutation": permutation_report,
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.smoke:
        args.fit_identities = min(args.fit_identities, 128)
        args.validation_identities = 128
        args.final_fit_identities = min(args.final_fit_identities, 256)
        args.max_inner_identities = 128
        args.beta = (0.0, 0.1, 0.2, 0.4)
        args.bootstrap_draws = min(args.bootstrap_draws, 300)
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], 5, args.fold_seed)
    fold01 = np.flatnonzero(np.isin(fold, [0, 1]))
    fold012 = np.flatnonzero(np.isin(fold, [0, 1, 2]))
    fold2 = np.flatnonzero(fold == 2)
    inner_pool = np.flatnonzero(fold == 3)
    rng_a = np.random.default_rng(args.seed + 11)
    rng_b = np.random.default_rng(args.seed + 29)
    fit_a = identity_balanced_queries(fold01, body["query_ik14"], rng_a, args.fit_identities)
    fit_b = identity_balanced_queries(fold01, body["query_ik14"], rng_b, args.fit_identities)
    validation = identity_balanced_queries(
        fold2, body["query_ik14"], np.random.default_rng(args.seed + 37),
        args.validation_identities,
    )
    final_a = identity_balanced_queries(
        fold012, body["query_ik14"], np.random.default_rng(args.seed + 41),
        args.final_fit_identities,
    )
    final_b = identity_balanced_queries(
        fold012, body["query_ik14"], np.random.default_rng(args.seed + 53),
        args.final_fit_identities,
    )
    inner = identity_balanced_queries(
        inner_pool, body["query_ik14"], np.random.default_rng(args.fold_seed + 19),
        args.max_inner_identities,
    )
    if set(body["query_formula"][np.concatenate((final_a, final_b))].astype(str)) & set(
        body["query_formula"][inner].astype(str)
    ):
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
    centered = CenteredRuleCache(base)
    fit_weight, fit_report = robust_fit(
        fit_a, fit_b, body, official, row_position, centered, args, seed=args.seed + 100,
    )
    validation_cache = WeightedCenteredCache(
        centered, fit_weight["empirical"], fit_weight["idf"], fit_weight["permuted"],
    )
    validation_scored = score_queries(
        validation, body, official, row_position, validation_cache, ("mass", *ARMS),
    )
    selection = {}
    for arm in ARMS:
        selected, table = grid(validation_scored, arm, list(map(float, args.beta)))
        selection[arm] = {"selected": selected, "grid": table}
    print("completed fold-2 fusion selection", flush=True)
    del validation_scored, validation_cache
    gc.collect()

    final_weight, final_fit_report = robust_fit(
        final_a, final_b, body, official, row_position, centered, args, seed=args.seed + 200,
    )
    inner_cache = WeightedCenteredCache(
        centered, final_weight["empirical"], final_weight["idf"], final_weight["permuted"],
    )
    inner_scored = score_queries(
        inner, body, official, row_position, inner_cache, ("mass", *ARMS),
    )
    formula = np.asarray(inner_scored["formula"]).astype(str)
    baseline = np.asarray(inner_scored["old_rank"])
    ranks = {}
    held = {}
    for arm in ARMS:
        chosen = selection[arm]["selected"]
        rank = fused_ranks(inner_scored, arm, chosen["mass_beta"], chosen["rule_beta"])
        ranks[arm] = rank
        held[arm] = {
            "selected_mass_beta": chosen["mass_beta"],
            "selected_rule_beta": chosen["rule_beta"],
            "retrieval": retrieval(baseline, rank),
            "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
                formula, baseline, rank, draws=args.bootstrap_draws,
                seed=args.seed + 300 + ARMS.index(arm),
            ),
        }
    target = ranks["boundary_consensus_centered"]
    paired = {
        f"boundary_consensus_minus_{arm}": paired_formula_ci(
            formula, target, ranks[arm], args, 400 + ARMS.index(arm),
        )
        for arm in ARMS if arm != "boundary_consensus_centered"
    }
    target_held = held["boundary_consensus_centered"]
    target_retrieval = target_held["retrieval"]
    gates = {
        "absolute_formula_ci_positive": target_held[
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ][0] > 0,
        "corrected_exceeds_twice_introduced": (
            int(target_retrieval["corrected_at_1"]) > 2 * int(target_retrieval["introduced_at_1"])
        ),
        **{
            f"beats_{arm}_ci": paired[f"boundary_consensus_minus_{arm}"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0
            for arm in ARMS if arm != "boundary_consensus_centered"
        },
        "outer_fold_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_BOUNDARY_ALIGNED_CENTERED_RULE_METRIC_PASS"
            if all(gates.values()) else "CHEMAWARE_BOUNDARY_ALIGNED_CENTERED_RULE_METRIC_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "folds 0-1 fit; fold 2 selection; folds 0-2 refit; inner fold 3 evaluation; outer fold 4 sealed",
        "claim_limit": "Frozen explicit shared-metric development result, not DreaMS fine-tuning or external confirmation.",
        "method": {
            "feature_map": "[official, sqrt(mass_beta)*mass, sqrt(rule_beta)*sqrt(channel_weight)*centered_rule]",
            "centered_rule": "true curated center minus mean of three shared-coordinate local backgrounds",
            "channel_target": "best-positive-reference contribution minus hardest-negative-reference contribution",
            "two_independent_fit_consensus": "geometric mean; unsupported channels receive zero",
            "nonnegative_psd_channel_weights": True,
            "candidate_formula_identity_free_at_deployment": True,
        },
        "data": {
            "fit_a_queries": int(len(fit_a)), "fit_b_queries": int(len(fit_b)),
            "validation_queries": int(len(validation)),
            "final_fit_a_queries": int(len(final_a)), "final_fit_b_queries": int(len(final_b)),
            "inner_queries": int(len(inner)), "outer_queries_untouched": int(np.sum(fold == 4)),
        },
        "initial_fit": fit_report,
        "fold2_selection": selection,
        "final_fit": final_fit_report,
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
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_boundary_metric_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "rule_weights.npz",
            boundary_consensus_weight=final_weight["empirical"],
            idf_weight=final_weight["idf"], permuted_weight=final_weight["permuted"],
            document_frequency=final_weight["document_frequency"],
        )
        np.savez_compressed(
            temporary / "inner_ranks.npz", query=inner, formula=formula,
            baseline_rank=baseline,
            **{f"{arm}_rank": rank for arm, rank in ranks.items()},
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "final_fit": final_fit_report,
        "held_inner": held, "paired_inner": paired, "gates": gates,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
