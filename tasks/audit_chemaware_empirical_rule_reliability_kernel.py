"""Learn reliable rule channels, then evaluate a shared mass-rule embedding.

The original rule kernel gives every curated neutral-loss and fragment channel
equal status.  This audit estimates, using only training-formula query/reference
pairs, which channels reproducibly overlap more for the correct identity than
for same-formula negatives.  Nonnegative weights preserve an explicit PSD
feature map.  Formula fold 2 selects fusion weights; fold 3 is evaluated once;
fold 4 stays untouched.
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

ROOT = Path(__file__).resolve().parents[1]

from audit_chemaware_mass_kernel_embedding import (  # noqa: E402
    KernelCache,
    score_queries,
    strict_rank,
)
from audit_chemaware_observable_tangent_metric import (  # noqa: E402
    formula_bootstrap,
    retrieval,
)
from chemaware_empirical_rule_reliability_core import (  # noqa: E402
    aggregate_formula_contrasts,
    prevalence_matched_permutation,
    reliability_weights,
)
from chemaware_formula_rule_core import rows_for_queries  # noqa: E402
from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
from train_chemaware_full_candidate_alignment import identity_balanced_queries  # noqa: E402


ARMS = (
    "empirical_reliability",
    "uniform_rule",
    "idf_rule",
    "prevalence_matched_weight_permutation",
    "shifted_rule_mass",
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
        default=ROOT / "data/validation/chemaware_empirical_rule_reliability_kernel_v1",
    )
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--fit-identities", type=int, default=2048)
    parser.add_argument("--validation-identities", type=int, default=2048)
    parser.add_argument("--final-fit-identities", type=int, default=4096)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--beta", type=float, nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def unit(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return (vector / norm if norm else vector).astype(np.float16)


class WeightedRuleCache:
    """Expose several matched, spectrum-local PSD rule feature maps."""

    def __init__(
        self,
        base: KernelCache,
        empirical: np.ndarray,
        idf: np.ndarray,
        permuted: np.ndarray,
    ) -> None:
        self.base = base
        self.sqrt_weight = {
            "empirical_reliability": np.sqrt(np.maximum(empirical, 0.0)),
            "uniform_rule": np.ones_like(empirical, dtype=np.float32),
            "idf_rule": np.sqrt(np.maximum(idf, 0.0)),
            "prevalence_matched_weight_permutation": np.sqrt(np.maximum(permuted, 0.0)),
            "shifted_rule_mass": np.sqrt(np.maximum(empirical, 0.0)),
        }
        self.cache: dict[int, dict[str, np.ndarray]] = {}

    def get(self, row: int) -> dict[str, np.ndarray]:
        row = int(row)
        if row in self.cache:
            return self.cache[row]
        raw = self.base.get(row)
        correct = np.asarray(raw["rule_response"], dtype=np.float32)
        shifted = np.asarray(raw["rule_response_shifted"], dtype=np.float32)
        output: dict[str, np.ndarray] = {"mass": raw["mass"]}
        for arm, weight in self.sqrt_weight.items():
            source = shifted if arm == "shifted_rule_mass" else correct
            output[arm] = unit(source * weight)
        self.cache[row] = output
        return output


def query_channel_contrasts(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    cache: KernelCache,
) -> np.ndarray:
    output = np.zeros((len(queries), len(cache.nl_rules) + len(cache.cf_rules)), dtype=np.float32)
    for out_index, query in enumerate(map(int, queries)):
        q = np.asarray(cache.get(int(body["query_row"][query]))["rule_response"], dtype=np.float32)
        left, right = map(int, body["query_ptr"][query : query + 2])
        label = np.asarray(body["molecule_label"][left:right], dtype=bool)
        molecule_mean = []
        for molecule in range(left, right):
            rleft, rright = map(int, body["molecule_ptr"][molecule : molecule + 2])
            rows = body["pair_candidate_row"][rleft:rright]
            molecule_mean.append(np.mean(np.stack([
                np.asarray(cache.get(int(row))["rule_response"], dtype=np.float32)
                for row in rows
            ]), axis=0))
        molecule_mean_array = np.stack(molecule_mean)
        positive = molecule_mean_array[np.flatnonzero(label)[0]]
        negative = np.mean(molecule_mean_array[~label], axis=0)
        output[out_index] = q * (positive - negative)
        if (out_index + 1) % 256 == 0:
            print(f"rule-channel contrasts {out_index + 1}/{len(queries)}", flush=True)
    return output


def fit_rule_weights(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    cache: KernelCache,
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    train_rows = rows_for_queries(queries, body)
    response = np.stack([
        np.asarray(cache.get(int(row))["rule_response"], dtype=np.float32)
        for row in train_rows
    ])
    document_frequency = np.sum(response > 0, axis=0)
    idf = (np.log((len(response) + 1.0) / (document_frequency + 1.0)) + 1.0).astype(np.float32)
    contrast = query_channel_contrasts(queries, body, cache)
    formula_contrast, formula_names = aggregate_formula_contrasts(
        contrast, body["query_formula"][queries],
    )
    empirical, empirical_report = reliability_weights(
        formula_contrast, formula_names, document_frequency, len(response), seed=seed,
    )
    permuted, permutation_report = prevalence_matched_permutation(
        empirical, document_frequency, len(cache.nl_rules), seed=seed + 1,
    )
    return {
        "empirical": empirical,
        "idf": idf,
        "permuted": permuted,
        "document_frequency": document_frequency.astype(np.int64),
    }, {
        "training_queries": int(len(queries)),
        "training_formulas": int(len(np.unique(body["query_formula"][queries].astype(str)))),
        "training_spectrum_rows": int(len(train_rows)),
        "neutral_loss_channels": int(len(cache.nl_rules)),
        "fragment_channels": int(len(cache.cf_rules)),
        "empirical": empirical_report,
        "permutation": permutation_report,
        "document_frequency_quantiles": np.quantile(
            document_frequency, (0.0, 0.25, 0.5, 0.75, 1.0),
        ).astype(float).tolist(),
    }


def fused_ranks(
    scored: dict,
    arm: str,
    mass_beta: float,
    rule_beta: float,
) -> np.ndarray:
    output = np.empty(len(scored["query"]), dtype=np.int16)
    for index in range(len(output)):
        pair_score = (
            np.asarray(scored["global"][index], dtype=np.float32)
            + float(mass_beta) * np.asarray(scored["mass"][index], dtype=np.float32)
            + float(rule_beta) * np.asarray(scored[arm][index], dtype=np.float32)
        )
        pointer = np.asarray(scored["reference_ptr"][index], dtype=np.int64)
        molecule_score = np.maximum.reduceat(pair_score, pointer[:-1])
        output[index] = strict_rank(
            molecule_score, np.asarray(scored["labels"][index], dtype=bool),
        )
    return output


def grid(scored: dict, arm: str, beta: list[float]) -> tuple[dict[str, object], list[dict[str, object]]]:
    table = []
    for mass_beta in beta:
        for rule_beta in beta:
            rank = fused_ranks(scored, arm, mass_beta, rule_beta)
            summary = retrieval(np.asarray(scored["old_rank"]), rank)
            table.append({
                "mass_beta": float(mass_beta), "rule_beta": float(rule_beta),
                **summary,
            })
    selected = max(
        table,
        key=lambda item: (
            int(item["risk_utility_at_1"]),
            -int(item["introduced_at_1"]),
            float(item["delta_mrr"]),
            -(float(item["mass_beta"]) + float(item["rule_beta"])),
        ),
    )
    return selected, table


def paired_formula_ci(
    formula: np.ndarray,
    target_rank: np.ndarray,
    control_rank: np.ndarray,
    args: argparse.Namespace,
    offset: int,
) -> dict[str, object]:
    target_rank = np.asarray(target_rank)
    control_rank = np.asarray(control_rank)
    return {
        "delta_recall1": float(np.mean(target_rank == 1) - np.mean(control_rank == 1)),
        "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
            np.asarray(formula).astype(str), control_rank, target_rank,
            draws=args.bootstrap_draws, seed=args.seed + offset,
        ),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.smoke:
        args.fit_identities = min(args.fit_identities, 256)
        args.validation_identities = min(args.validation_identities, 256)
        args.final_fit_identities = min(args.final_fit_identities, 512)
        args.max_inner_identities = 256
        args.beta = [0.0, 0.1, 0.2]
        args.bootstrap_draws = min(args.bootstrap_draws, 200)
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], 5, args.fold_seed)
    rng = np.random.default_rng(args.seed)
    fit_queries = identity_balanced_queries(
        np.flatnonzero(np.isin(fold, [0, 1])), body["query_ik14"], rng,
        args.fit_identities,
    )
    validation_queries = identity_balanced_queries(
        np.flatnonzero(fold == 2), body["query_ik14"], rng,
        args.validation_identities,
    )
    final_fit_queries = identity_balanced_queries(
        np.flatnonzero(np.isin(fold, [0, 1, 2])), body["query_ik14"], rng,
        args.final_fit_identities,
    )
    inner_queries = identity_balanced_queries(
        np.flatnonzero(fold == 3), body["query_ik14"],
        np.random.default_rng(args.fold_seed + 19), args.max_inner_identities,
    )
    outer_queries = np.flatnonzero(fold == 4)
    if set(body["query_formula"][final_fit_queries].astype(str)) & set(
        body["query_formula"][inner_queries].astype(str)
    ):
        raise RuntimeError("final fit and inner formulas overlap")

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
        variants=("mass", "rule_response", "rule_response_shifted"),
    )
    fit_weight, fit_report = fit_rule_weights(
        fit_queries, body, base, args, seed=args.seed + 10,
    )
    validation_cache = WeightedRuleCache(
        base, fit_weight["empirical"], fit_weight["idf"], fit_weight["permuted"],
    )
    validation_scored = score_queries(
        validation_queries, body, official, row_position, validation_cache,
        ("mass", *ARMS),
    )
    selection = {}
    for arm in ARMS:
        selected, table = grid(validation_scored, arm, list(map(float, args.beta)))
        selection[arm] = {"selected": selected, "grid": table}
    print("completed frozen fold-2 fusion selection", flush=True)
    del validation_scored, validation_cache
    gc.collect()

    final_weight, final_fit_report = fit_rule_weights(
        final_fit_queries, body, base, args, seed=args.seed + 20,
    )
    inner_cache = WeightedRuleCache(
        base, final_weight["empirical"], final_weight["idf"], final_weight["permuted"],
    )
    inner_scored = score_queries(
        inner_queries, body, official, row_position, inner_cache,
        ("mass", *ARMS),
    )
    inner_ranks = {}
    inner_report = {}
    formula = np.asarray(inner_scored["formula"]).astype(str)
    baseline = np.asarray(inner_scored["old_rank"])
    for arm in ARMS:
        chosen = selection[arm]["selected"]
        rank = fused_ranks(
            inner_scored, arm, chosen["mass_beta"], chosen["rule_beta"],
        )
        inner_ranks[arm] = rank
        inner_report[arm] = {
            "selected_mass_beta": chosen["mass_beta"],
            "selected_rule_beta": chosen["rule_beta"],
            "retrieval": retrieval(baseline, rank),
            "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
                formula, baseline, rank, draws=args.bootstrap_draws,
                seed=args.seed + 100 + ARMS.index(arm),
            ),
        }
    target = inner_ranks["empirical_reliability"]
    paired = {
        f"empirical_minus_{arm}": paired_formula_ci(
            formula, target, inner_ranks[arm], args, 200 + ARMS.index(arm),
        )
        for arm in ARMS if arm != "empirical_reliability"
    }
    gates = {
        "absolute_formula_ci_positive": (
            inner_report["empirical_reliability"]
            ["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0
        ),
        "risk_utility_positive": (
            inner_report["empirical_reliability"]["retrieval"]["risk_utility_at_1"] > 0
        ),
        "beats_uniform_rule_ci": paired["empirical_minus_uniform_rule"]
        ["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
        "beats_idf_rule_ci": paired["empirical_minus_idf_rule"]
        ["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
        "beats_prevalence_matched_permutation_ci": paired[
            "empirical_minus_prevalence_matched_weight_permutation"
        ]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
        "beats_shifted_rule_ci": paired["empirical_minus_shifted_rule_mass"]
        ["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
        "outer_fold_untouched": True,
    }
    report = {
        "status": (
            "CHEMAWARE_EMPIRICAL_RULE_RELIABILITY_KERNEL_PASS"
            if all(gates.values()) else "CHEMAWARE_EMPIRICAL_RULE_RELIABILITY_KERNEL_FAIL"
        ),
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "folds 0-1 fit; fold 2 fusion selection; fold 3 evaluation; fold 4 sealed",
        "claim_limit": "Frozen explicit shared-feature development result, not DreaMS fine-tuning.",
        "method": {
            "feature_map": "[official, sqrt(mass_beta)*mass, sqrt(rule_beta)*sqrt(channel_weight)*rule_response]",
            "channel_target": "formula-balanced correct-identity overlap minus same-formula negative overlap",
            "nonnegative_psd_channel_weights": True,
            "candidate_or_formula_input_at_deployment": False,
            "matched_controls_get_independent_fold2_fusion_selection": True,
        },
        "data": {
            "fit_queries_folds01": int(len(fit_queries)),
            "validation_queries_fold2": int(len(validation_queries)),
            "final_fit_queries_folds012": int(len(final_fit_queries)),
            "inner_queries_fold3": int(len(inner_queries)),
            "outer_queries_untouched": int(len(outer_queries)),
        },
        "initial_weight_fit": fit_report,
        "fold2_selection": selection,
        "final_weight_fit": final_fit_report,
        "held_inner": inner_report,
        "paired_inner": paired,
        "gates": gates,
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "rule_library_sha256": sha256_file(args.rule_library),
        },
    }
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_rule_reliability_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        np.savez_compressed(
            temporary / "inner_ranks.npz",
            query=inner_queries, formula=formula, baseline_rank=baseline,
            **{f"{arm}_rank": rank for arm, rank in inner_ranks.items()},
        )
        np.savez_compressed(
            temporary / "rule_weights.npz",
            empirical_weight=final_weight["empirical"], idf_weight=final_weight["idf"],
            permuted_weight=final_weight["permuted"],
            document_frequency=final_weight["document_frequency"],
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"],
        "held_inner": inner_report,
        "paired_inner": paired,
        "gates": gates,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
