#!/usr/bin/env python
"""Leakage-controlled effect tests for the raw MSV000096214 feature table.

Implements contract v1.3. The primary inferential unit is a genotype/lineage,
not repeated scores of the same biological sample.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


WT = "WT_Col0"
TT7 = ("tt7_115_EMS", "tt7_8_SALK_039417", "tt7_9_SALK_seq_064306")
ALTERNATIVES = ("tt3_SALK_099848C", "tt4_11_SALK_020583", "fls1_3_EMS")
DOUBLES = {
    "tt7_115__tt4_11": "tt7_115_EMS",
    "tt7_8__tt4_11": "tt7_8_SALK_039417",
    "tt7_9__tt4_11": "tt7_9_SALK_seq_064306",
}
N_BOOT = 2000
RNG_SEED = 20261012


def genotype_map(manifest: pd.DataFrame) -> dict[str, str]:
    return {
        Path(row.filename).stem: row.genotype
        for row in manifest.loc[manifest["role"] == "study"].itertuples(index=False)
    }


def prepare_log2(matrix: pd.DataFrame, median_center: bool) -> pd.DataFrame:
    numeric = matrix.astype(float).where(matrix.astype(float) > 0)
    transformed = np.log2(numeric)
    if median_center:
        transformed = transformed.subtract(transformed.median(axis=0, skipna=True), axis=1)
    return transformed


def group_columns(genotypes: dict[str, str], matrix: pd.DataFrame, group: str) -> list[str]:
    return [column for column, label in genotypes.items() if label == group and column in matrix.columns]


def group_signature(log2: pd.DataFrame, group: str, genotypes: dict[str, str]) -> pd.Series:
    mutant_columns = group_columns(genotypes, log2, group)
    wt_columns = group_columns(genotypes, log2, WT)
    mutant_threshold = 3 if group == "fls1_3_EMS" else 4
    valid = (
        (log2[mutant_columns].notna().sum(axis=1) >= mutant_threshold)
        & (log2[wt_columns].notna().sum(axis=1) >= 4)
    )
    result = log2[mutant_columns].median(axis=1) - log2[wt_columns].median(axis=1)
    return result.where(valid)


def training_signature(
    log2: pd.DataFrame,
    genotypes: dict[str, str],
    train_targets: tuple[str, str],
    excluded_wt: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    wt_columns = group_columns(genotypes, log2, WT)
    if excluded_wt is not None:
        wt_columns = [column for column in wt_columns if column != excluded_wt]
    first = group_columns(genotypes, log2, train_targets[0])
    second = group_columns(genotypes, log2, train_targets[1])
    target_columns = first + second
    valid = (
        (log2[first].notna().sum(axis=1) >= 4)
        & (log2[second].notna().sum(axis=1) >= 4)
        & (log2[wt_columns].notna().sum(axis=1) >= max(3, len(wt_columns) - 1))
    )
    wt_center = log2[wt_columns].median(axis=1)
    weights = log2[target_columns].median(axis=1) - wt_center
    return wt_center, weights.where(valid)


def score_sample(sample: pd.Series, wt_center: pd.Series, weights: pd.Series) -> float:
    delta = sample - wt_center
    valid = delta.notna() & weights.notna()
    if int(valid.sum()) < 10:
        return float("nan")
    rho, _ = stats.spearmanr(delta[valid], weights[valid])
    return float(rho)


def scores_for_model(
    log2: pd.DataFrame,
    genotypes: dict[str, str],
    train_targets: tuple[str, str],
) -> dict[str, float]:
    full_center, full_weights = training_signature(log2, genotypes, train_targets)
    scores = {}
    for sample in log2.columns:
        if sample not in genotypes:
            continue
        if genotypes[sample] == WT:
            center, weights = training_signature(log2, genotypes, train_targets, sample)
        else:
            center, weights = full_center, full_weights
        scores[sample] = score_sample(log2[sample], center, weights)
    return scores


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    values = np.concatenate([positive, negative])
    ranks = stats.rankdata(values)
    rank_sum = ranks[:len(positive)].sum()
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def values_for_group(scores: dict[str, float], genotypes: dict[str, str], group: str) -> np.ndarray:
    return np.asarray([score for sample, score in scores.items() if genotypes[sample] == group], dtype=float)


def evaluate_assignment(
    log2: pd.DataFrame,
    genotypes: dict[str, str],
    targets: tuple[str, str, str],
    singles: tuple[str, ...],
) -> dict[str, object]:
    complement = tuple(group for group in singles if group not in targets)
    held_results = {}
    pairwise = []
    for held in targets:
        train = tuple(group for group in targets if group != held)
        scores = scores_for_model(log2, genotypes, train)  # type: ignore[arg-type]
        held_values = values_for_group(scores, genotypes, held)
        comparisons = {}
        for comparator in (WT, *complement):
            comparator_values = values_for_group(scores, genotypes, comparator)
            pair_auc = auc(held_values, comparator_values)
            pairwise.append(pair_auc)
            comparisons[comparator] = {
                "n_held": int(np.isfinite(held_values).sum()),
                "n_comparator": int(np.isfinite(comparator_values).sum()),
                "auroc": pair_auc,
                "mean_margin": float(np.nanmean(held_values) - np.nanmean(comparator_values)),
            }
        held_results[held] = {"trained_on": list(train), "comparisons": comparisons}
    return {
        "targets": list(targets),
        "complement": list(complement),
        "held_results": held_results,
        "joint_min_pairwise_auroc": float(np.nanmin(pairwise)),
        "joint_mean_pairwise_auroc": float(np.nanmean(pairwise)),
    }


def exact_t2(log2: pd.DataFrame, genotypes: dict[str, str]) -> dict[str, object]:
    singles = tuple(sorted((*TT7, *ALTERNATIVES)))
    assignments = []
    observed = None
    for targets in itertools.combinations(singles, 3):
        result = evaluate_assignment(log2, genotypes, targets, singles)
        assignments.append({
            "targets": list(targets),
            "joint_min_pairwise_auroc": result["joint_min_pairwise_auroc"],
            "joint_mean_pairwise_auroc": result["joint_mean_pairwise_auroc"],
        })
        if set(targets) == set(TT7):
            observed = result
    if observed is None:
        raise AssertionError("true target assignment missing")
    statistic = float(observed["joint_min_pairwise_auroc"])
    p_right = sum(float(item["joint_min_pairwise_auroc"]) >= statistic - 1e-12 for item in assignments) / 20
    assignments.sort(key=lambda item: (-float(item["joint_min_pairwise_auroc"]), item["targets"]))
    held = observed["held_results"]
    gates = {
        "all_three_held_vs_wt_auroc_above_half": all(
            float(held[group]["comparisons"][WT]["auroc"]) > 0.5 for group in TT7
        ),
        "all_nine_held_vs_alternative_margins_positive": all(
            float(held[group]["comparisons"][alt]["mean_margin"]) > 0
            for group in TT7 for alt in ALTERNATIVES
        ),
        "true_assignment_uniquely_best_exact_p_0_05": p_right == 0.05,
    }
    return {
        "observed": observed,
        "assignments_ranked": assignments,
        "exact_cluster_test": {
            "n_assignments": 20,
            "observed_statistic": statistic,
            "observed_rank": next(i + 1 for i, item in enumerate(assignments) if set(item["targets"]) == set(TT7)),
            "right_tail_p": p_right,
            "minimum_attainable_p": 0.05,
        },
        "gates": gates,
        "pass": all(gates.values()),
    }


def spearman_pairwise(first: pd.Series, second: pd.Series) -> float:
    valid = first.notna() & second.notna()
    if int(valid.sum()) < 10:
        return float("nan")
    rho, _ = stats.spearmanr(first[valid], second[valid])
    return float(rho)


def bootstrap_delta(first: pd.Series, second: pd.Series, third: pd.Series,
                    rng: np.random.Generator) -> dict[str, object]:
    frame = pd.concat([first, second, third], axis=1).dropna()
    deltas = []
    for _ in range(N_BOOT):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        deltas.append(
            stats.spearmanr(sample.iloc[:, 0], sample.iloc[:, 1]).statistic
            - stats.spearmanr(sample.iloc[:, 0], sample.iloc[:, 2]).statistic
        )
    return {
        "conditional_feature_bootstrap_median": float(np.median(deltas)),
        "conditional_feature_bootstrap_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "inference_limit": "Feature-axis stability only; correlated ions/adducts are not independent biological units.",
    }


def evaluate_world(log2: pd.DataFrame, genotypes: dict[str, str],
                   rng: np.random.Generator) -> dict[str, object]:
    signatures = {
        group: group_signature(log2, group, genotypes)
        for group in set(genotypes.values()) if group != WT
    }
    tt7_mean = pd.concat([signatures[group] for group in TT7], axis=1).mean(axis=1)
    t1 = {}
    for double, paired in DOUBLES.items():
        r_tt4 = spearman_pairwise(signatures[double], signatures["tt4_11_SALK_020583"])
        r_tt7 = spearman_pairwise(signatures[double], tt7_mean)
        t1[double] = {
            "paired_allele": paired,
            "rho_vs_tt4": r_tt4,
            "rho_vs_tt7mean": r_tt7,
            "rho_delta": r_tt4 - r_tt7,
            "bootstrap": bootstrap_delta(signatures[double], signatures["tt4_11_SALK_020583"], tt7_mean, rng),
        }
    t2 = exact_t2(log2, genotypes)
    t3_deltas = {
        double: spearman_pairwise(signatures[double], signatures["tt4_11_SALK_020583"])
        - spearman_pairwise(signatures[double], signatures[paired])
        for double, paired in DOUBLES.items()
    }
    tt7_pair_rho = [spearman_pairwise(signatures[a], signatures[b]) for a, b in itertools.combinations(TT7, 2)]
    tt7_tt3 = [spearman_pairwise(signatures[a], signatures["tt3_SALK_099848C"]) for a in TT7]
    tt7_fls1 = [spearman_pairwise(signatures[a], signatures["fls1_3_EMS"]) for a in TT7]
    t3 = {
        "paired_epistasis_deltas": t3_deltas,
        "all_three_paired_deltas_positive": all(value > 0 for value in t3_deltas.values()),
        "median_rho_tt7_pairs": float(np.nanmedian(tt7_pair_rho)),
        "median_rho_tt7_tt3": float(np.nanmedian(tt7_tt3)),
        "median_rho_tt7_fls1": float(np.nanmedian(tt7_fls1)),
    }
    passed = all(value["rho_delta"] > 0 for value in t1.values()) and bool(t2["pass"]) and t3["all_three_paired_deltas_positive"]
    return {"T1_epistasis": t1, "T1_pass_3of3": all(value["rho_delta"] > 0 for value in t1.values()),
            "T2_exact_heldout": t2, "T3_topology": t3, "pass": bool(passed)}


def missingness_audit(matrix: pd.DataFrame, genotypes: dict[str, str]) -> dict[str, object]:
    fractions = matrix.isna().mean(axis=0)
    by_group = {}
    for group in sorted(set(genotypes.values())):
        columns = group_columns(genotypes, matrix, group)
        values = fractions[columns]
        by_group[group] = {"samples": len(columns), "median_missing_fraction": float(values.median()),
                           "samples_above_50pct_missing": int((values > 0.5).sum())}
    gate = all(value["samples_above_50pct_missing"] <= 1 for value in by_group.values())
    return {"by_genotype": by_group, "no_genotype_has_more_than_one_bad_sample": gate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus-dir", type=Path, default=Path("data/external/msv000096214/consensus"))
    parser.add_argument("--readiness-manifest", type=Path, default=Path("data/validation/msv000096214_readiness_v2/sample_manifest.csv.gz"))
    parser.add_argument("--audit", type=Path, default=Path("data/validation/msv000096214_table_audit/report.json"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/msv000096214_effect_stage"))
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit.get("pass_to_effect_stage"):
        raise SystemExit("EFFECT_STAGE_BLOCKED: feature-table audit did not pass")
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = pd.read_csv(args.readiness_manifest)
    genotype_lookup = genotype_map(manifest)
    loaded = {}
    for polarity in ("positive", "negative"):
        matrix = pd.read_csv(args.consensus_dir / polarity / "matrix_filled.csv.gz", index_col="group_id")
        genotypes = {column: label for column, label in genotype_lookup.items() if column in matrix.columns}
        missing = missingness_audit(matrix, genotypes)
        if not missing["no_genotype_has_more_than_one_bad_sample"]:
            raise SystemExit(f"EFFECT_STAGE_BLOCKED: genotype-concentrated missingness in {polarity}")
        loaded[polarity] = (matrix, genotypes, missing)

    args.output.mkdir(parents=True)
    rng = np.random.default_rng(RNG_SEED)
    report: dict[str, object] = {
        "status": "MSV000096214_EFFECT_STAGE_T1_T3_V1_3", "formal": True,
        "data_role": "development/consumed", "sealed": False,
        "contract": "docs/GLM_MSV000096214_STRUCTURE_HIDDEN_CONTRACT_20261012.md v1.3",
        "n_conditional_feature_bootstrap": N_BOOT, "rng_seed": RNG_SEED, "polarity": {},
    }
    for polarity, (matrix, genotypes, missing) in loaded.items():
        primary = evaluate_world(prepare_log2(matrix, True), genotypes, rng)
        sensitivity = evaluate_world(prepare_log2(matrix, False), genotypes, rng)
        report["polarity"][polarity] = {
            "features": int(len(matrix)), "samples": int(len(genotypes)),
            "missingness_after_genotype_join": missing,
            "primary_median_centered_log2": primary,
            "raw_log2_sensitivity": sensitivity,
            "pass_primary_and_sensitivity": bool(primary["pass"] and sensitivity["pass"]),
        }
    report["pass_to_posthoc_t4"] = all(value["pass_primary_and_sensitivity"] for value in report["polarity"].values())
    report["claim_limit"] = (
        "Same-study, same-instrument, development evidence. Exact p resolution is 0.05 from 20 genotype-cluster assignments. "
        "Feature bootstraps are conditional stability only; run order remains genotype-blocked."
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        polarity: {"primary_pass": body["primary_median_centered_log2"]["pass"],
                   "primary_t2": body["primary_median_centered_log2"]["T2_exact_heldout"]["exact_cluster_test"],
                   "sensitivity_pass": body["raw_log2_sensitivity"]["pass"]}
        for polarity, body in report["polarity"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
