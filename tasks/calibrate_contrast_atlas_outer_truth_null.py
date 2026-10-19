#!/usr/bin/env python
"""Matched outer truth-label null for frozen Contrast Atlas retrieval.

Biological relevance labels are permuted within each query and predeclared
technical stratum while candidate scores, coverage, abstention and associations
remain frozen.  This tests whether retrieval exceeds technically matched chance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adjudicate_contrast_atlas_unknown_increment import metrics
from build_contrast_atlas_query_outcomes import build_outcomes, validate_inputs


def outer_truth_null(
    candidates_raw: pd.DataFrame,
    registry: pd.DataFrame,
    truth: pd.DataFrame,
    strata_columns: list[str],
    draws: int = 999,
    seed: int = 20261013,
    fdr_threshold: float = 0.10,
    world: str = "all",
    endpoint: str = "balanced_endpoint",
) -> tuple[dict[str, object], pd.DataFrame]:
    if draws < 99:
        raise ValueError("at least 99 outer-null draws are required")
    missing_strata = set(strata_columns) - set(candidates_raw.columns)
    if missing_strata:
        raise RuntimeError(f"candidate table missing null strata: {sorted(missing_strata)}")
    validated, _, _ = validate_inputs(candidates_raw, registry, truth)
    if world not in set(validated["world"]):
        raise RuntimeError(f"requested world absent: {world}")

    target_strata = validated[["query_id", "target_study_id", "target_contrast_id", *strata_columns]].drop_duplicates()
    if target_strata.duplicated(["query_id", "target_study_id", "target_contrast_id"]).any():
        raise RuntimeError("null stratum changes across evidence worlds")
    truth_with_strata = truth.merge(
        target_strata,
        on=["query_id", "target_study_id", "target_contrast_id"],
        how="left",
        validate="one_to_one",
    )
    if truth_with_strata[strata_columns].isna().any().any():
        raise RuntimeError("truth target lacks a null stratum")

    group_columns = ["query_id", *strata_columns]
    grouped_indices = []
    for _, group in truth_with_strata.groupby(group_columns, sort=False, dropna=False):
        if len(group) > 1 and group["expected_association"].nunique() > 1:
            grouped_indices.append(group.index.to_numpy())
    movable_rows = int(sum(len(indices) for indices in grouped_indices))
    if len(grouped_indices) < 3:
        report = {
            "status": "ABSTAIN_INSUFFICIENT_MATCHED_OUTER_NULL",
            "world": world,
            "endpoint": endpoint,
            "draws": draws,
            "matched_movable_strata": len(grouped_indices),
            "matched_movable_rows": movable_rows,
            "pass_outer_truth_null": False,
            "claim_limit": "At least three query-specific matched strata with label variation are required.",
        }
        return report, pd.DataFrame()

    observed_outcomes, _ = build_outcomes(candidates_raw, registry, truth, fdr_threshold)
    observed_subset = observed_outcomes.loc[observed_outcomes["world"] == world]
    observed_metrics = metrics(observed_subset)
    observed = float(observed_metrics[endpoint])

    rng = np.random.default_rng(seed)
    null_values = np.empty(draws, dtype=float)
    for draw in range(draws):
        permuted = truth_with_strata.copy()
        labels = permuted["expected_association"].to_numpy(copy=True)
        for indices in grouped_indices:
            labels[indices] = rng.permutation(labels[indices])
        permuted["expected_association"] = labels
        # Preserve the preregistration/provenance fields required by the truth
        # contract.  Only the relevance label is permuted under this null.
        permuted_truth = permuted[list(truth.columns)]
        outcomes, _ = build_outcomes(candidates_raw, registry, permuted_truth, fdr_threshold)
        null_values[draw] = float(metrics(outcomes.loc[outcomes["world"] == world])[endpoint])
    p_right = float((1 + np.sum(null_values >= observed)) / (draws + 1))
    null_table = pd.DataFrame({"draw": np.arange(draws), "endpoint": null_values})
    report = {
        "status": "CONTRAST_ATLAS_MATCHED_OUTER_TRUTH_NULL_COMPLETE",
        "formal": True,
        "world": world,
        "endpoint": endpoint,
        "draws": draws,
        "seed": seed,
        "fdr_threshold": fdr_threshold,
        "strata_columns": strata_columns,
        "matched_movable_strata": len(grouped_indices),
        "matched_movable_rows": movable_rows,
        "observed_metrics": observed_metrics,
        "observed_endpoint": observed,
        "null_mean": float(np.mean(null_values)),
        "null_q025": float(np.quantile(null_values, 0.025)),
        "null_q975": float(np.quantile(null_values, 0.975)),
        "right_tail_p": p_right,
        "pass_outer_truth_null": bool(p_right < 0.05 and observed > np.quantile(null_values, 0.975)),
        "claim_limit": "Matched relevance-label null only; it does not replace held-out source validation or establish mechanism.",
    }
    return report, null_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrated-candidates", type=Path, required=True)
    parser.add_argument("--query-registry", type=Path, required=True)
    parser.add_argument("--truth-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strata-columns", required=True, help="Comma-separated phenotype-blind technical strata")
    parser.add_argument("--draws", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--fdr-threshold", type=float, default=0.10)
    parser.add_argument("--world", default="all")
    parser.add_argument("--endpoint", default="balanced_endpoint", choices=["balanced_endpoint", "match_recall1", "match_mrr"])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    strata = [item.strip() for item in args.strata_columns.split(",") if item.strip()]
    if not strata:
        raise ValueError("at least one null stratum is required")
    candidates = pd.read_csv(args.calibrated_candidates)
    registry = pd.read_csv(args.query_registry)
    truth = pd.read_csv(args.truth_ledger)
    report, null = outer_truth_null(
        candidates, registry, truth, strata, args.draws, args.seed,
        args.fdr_threshold, args.world, args.endpoint,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not null.empty:
        null.to_csv(args.output_dir / "null_distribution.csv.gz", index=False, compression="gzip")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
