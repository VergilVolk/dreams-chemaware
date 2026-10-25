#!/usr/bin/env python3
"""Aggregate the MSV000095331 retrieval endpoint across microbial contexts.

The primary application run contains five related microbial contexts for each
drug mixture.  They are useful biological strata, but they are not independent
replications.  This script gives every context equal weight, forms one target
rank for each mixture x candidate-study unit, and reports mean-score and
median-score aggregation without changing any entity links or response models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


STATUS = "MSV000095331_CONTEXT_ROBUST_RETRIEVAL_COMPLETE"
TARGET_BY_MIXTURE = {
    "1": "drug_sertraline",
    "3": "drug_omeprazole",
    "4": "drug_loratadine",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--response-scores",
        type=Path,
        default=Path(
            "data/validation/"
            "msv000095331_entity_method_increment_multiencoder_run_2404230/"
            "response_retrieval_scores.tsv.gz"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    scores = pd.read_csv(args.response_scores, sep="\t")
    required = {
        "method_name", "query_contrast_id", "candidate_study_id",
        "candidate_contrast_id", "matched_entity_pairs",
        "coverage_sensitive_response_score",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"response score table misses columns: {sorted(missing)}")

    scores["mixture_id"] = scores["query_contrast_id"].str.extract(
        r"drug_mix_(\d+)_", expand=False
    )
    panel = scores.loc[scores["mixture_id"].isin(TARGET_BY_MIXTURE)].copy()
    if panel.empty:
        raise RuntimeError("no registered mixture contexts found")

    context_counts = panel.groupby(
        ["method_name", "mixture_id", "candidate_study_id", "candidate_contrast_id"]
    )["query_contrast_id"].nunique()
    if not context_counts.eq(5).all():
        bad = context_counts.loc[~context_counts.eq(5)]
        raise RuntimeError(f"expected five contexts per candidate; found {bad.head()}")

    aggregate = panel.groupby(
        ["method_name", "mixture_id", "candidate_study_id", "candidate_contrast_id"],
        as_index=False,
    ).agg(
        microbial_contexts=("query_contrast_id", "nunique"),
        mean_response_score=("coverage_sensitive_response_score", "mean"),
        median_response_score=("coverage_sensitive_response_score", "median"),
        minimum_response_score=("coverage_sensitive_response_score", "min"),
        maximum_response_score=("coverage_sensitive_response_score", "max"),
        mean_matched_entity_pairs=("matched_entity_pairs", "mean"),
        minimum_matched_entity_pairs=("matched_entity_pairs", "min"),
        maximum_matched_entity_pairs=("matched_entity_pairs", "max"),
    )
    group = ["method_name", "mixture_id", "candidate_study_id"]
    aggregate["mean_aggregate_rank"] = aggregate.groupby(group)[
        "mean_response_score"
    ].rank(ascending=False, method="max")
    aggregate["median_aggregate_rank"] = aggregate.groupby(group)[
        "median_response_score"
    ].rank(ascending=False, method="max")
    aggregate["registered_target"] = [
        str(candidate) == TARGET_BY_MIXTURE[str(mixture)]
        for mixture, candidate in zip(
            aggregate["mixture_id"], aggregate["candidate_contrast_id"]
        )
    ]

    targets = aggregate.loc[aggregate["registered_target"]].copy()
    # Not every candidate study assayed every registered drug.  The evaluable
    # denominator is therefore the frozen mixture-study units in which the
    # prespecified target is actually present, not mixtures times all studies.
    target_units = targets[["mixture_id", "candidate_study_id"]].drop_duplicates()
    expected_units = int(len(target_units))
    unit_counts = targets.groupby("method_name").size()
    if not unit_counts.eq(expected_units).all():
        raise RuntimeError(
            f"each method must have {expected_units} target units; got {unit_counts.to_dict()}"
        )

    noise = targets.loc[targets["method_name"].eq("noise_v1"), [
        "mixture_id", "candidate_study_id", "mean_aggregate_rank",
        "median_aggregate_rank",
    ]].rename(columns={
        "mean_aggregate_rank": "noise_mean_aggregate_rank",
        "median_aggregate_rank": "noise_median_aggregate_rank",
    })
    targets = targets.merge(
        noise,
        on=["mixture_id", "candidate_study_id"],
        how="left",
        validate="many_to_one",
    )
    targets["mean_rank_change_vs_noise"] = (
        targets["noise_mean_aggregate_rank"] - targets["mean_aggregate_rank"]
    )
    targets["median_rank_change_vs_noise"] = (
        targets["noise_median_aggregate_rank"] - targets["median_aggregate_rank"]
    )

    summary = targets.groupby("method_name", as_index=False).agg(
        mixture_study_units=("registered_target", "size"),
        mean_aggregate_top1_units=(
            "mean_aggregate_rank", lambda value: int(value.eq(1).sum())
        ),
        median_aggregate_top1_units=(
            "median_aggregate_rank", lambda value: int(value.eq(1).sum())
        ),
        median_target_rank_mean_aggregation=("mean_aggregate_rank", "median"),
        mean_target_rank_mean_aggregation=("mean_aggregate_rank", "mean"),
        median_target_rank_median_aggregation=("median_aggregate_rank", "median"),
        improved_units_vs_noise_mean=(
            "mean_rank_change_vs_noise", lambda value: int(value.gt(0).sum())
        ),
        worsened_units_vs_noise_mean=(
            "mean_rank_change_vs_noise", lambda value: int(value.lt(0).sum())
        ),
        unchanged_units_vs_noise_mean=(
            "mean_rank_change_vs_noise", lambda value: int(value.eq(0).sum())
        ),
    )

    args.output_dir.mkdir(parents=True)
    outputs = {
        "aggregate_candidates": args.output_dir / "aggregate_candidate_scores.tsv",
        "registered_targets": args.output_dir / "registered_target_results.tsv",
        "method_summary": args.output_dir / "method_summary.tsv",
    }
    aggregate.to_csv(outputs["aggregate_candidates"], sep="\t", index=False)
    targets.to_csv(outputs["registered_targets"], sep="\t", index=False)
    summary.to_csv(outputs["method_summary"], sep="\t", index=False)

    report = {
        "status": STATUS,
        "dataset": "MSV000095331 plus two ChemProp candidate studies",
        "dataset_role": "development_consumed_not_external_validation",
        "aggregation": {
            "primary": "equal mean of five microbial-context response scores",
            "sensitivity": "median of five microbial-context response scores",
            "ranking_unit": "mixture x candidate study",
            "spectral_scores_reused_as_biological_weights": False,
        },
        "denominators": {
            "registered_mixtures": len(TARGET_BY_MIXTURE),
            "candidate_studies_with_registered_target": int(
                target_units["candidate_study_id"].nunique()
            ),
            "microbial_contexts_per_unit": 5,
            "mixture_study_units_per_method": expected_units,
            "methods": int(targets["method_name"].nunique()),
        },
        "method_summary": summary.to_dict(orient="records"),
        "inputs": {
            "response_scores": {
                "path": str(args.response_scores),
                "sha256": sha256(args.response_scores),
            }
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "claim_boundary": [
            "Five microbial contexts are related strata, not independent replications.",
            "The six mixture-study units are descriptive development endpoints and do not support a significance claim.",
            "A context-specific Top-1 gain is not a context-robust gain unless it survives aggregation.",
            "The analysis reuses frozen entity decisions and response estimates and does not tune a spectral model.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
