#!/usr/bin/env python3
"""Build the compact biology verdict for the frozen unified ranker.

This is evaluation-only. It combines the already frozen identity decisions,
response endpoint, context aggregation and edge attribution; it never fits or
selects a spectral model from biological outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATUS = "MSV000095331_UNIFIED_BIOLOGY_VERDICT_COMPLETE"
REFERENCE = "noise_v1"
UNIFIED = "unified_candidate_ranker"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--increment-dir", type=Path,
        default=Path(
            "data/validation/"
            "msv000095331_entity_method_increment_unified_local_20261026"
        ),
    )
    parser.add_argument(
        "--context-dir", type=Path,
        default=Path(
            "data/validation/msv000095331_context_robust_unified_local_20261026"
        ),
    )
    parser.add_argument(
        "--attribution-dir", type=Path,
        default=Path(
            "data/validation/"
            "msv000095331_unified_response_gain_attribution_local_20261026"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    increment_report_path = args.increment_dir / "report.json"
    overlap_path = args.increment_dir / "prespecified_overlap_results.tsv"
    context_summary_path = args.context_dir / "method_summary.tsv"
    context_targets_path = args.context_dir / "registered_target_results.tsv"
    attribution_report_path = args.attribution_dir / "report.json"
    gain_path = args.attribution_dir / "top1_gain_contexts.tsv"

    increment_report = json.loads(increment_report_path.read_text(encoding="utf-8"))
    attribution_report = json.loads(
        attribution_report_path.read_text(encoding="utf-8")
    )
    if increment_report.get("dataset_role") != "development_consumed_not_external_validation":
        raise RuntimeError("unexpected increment dataset role")
    overlaps = pd.read_csv(overlap_path, sep="\t")
    context_summary = pd.read_csv(context_summary_path, sep="\t")
    context_targets = pd.read_csv(context_targets_path, sep="\t")
    gains = pd.read_csv(gain_path, sep="\t")
    required_methods = {REFERENCE, UNIFIED, "weighted_spectral_entropy"}
    if not required_methods.issubset(set(overlaps["method_name"])):
        raise RuntimeError("required biology methods are missing")

    raw_summary = overlaps.groupby("method_name", as_index=False).agg(
        registered_contexts=("candidate_rank", "size"),
        raw_target_top1_contexts=("candidate_rank", lambda value: int(value.eq(1).sum())),
        raw_target_median_rank=("candidate_rank", "median"),
        raw_target_mean_rank=("candidate_rank", "mean"),
        raw_mean_matched_entity_pairs=("matched_entity_pairs", "mean"),
    )
    method_summary = raw_summary.merge(
        context_summary,
        on="method_name",
        how="left",
        validate="one_to_one",
    )

    keys = ["query_contrast_id", "candidate_study_id", "candidate_contrast_id"]
    noise = overlaps.loc[overlaps["method_name"].eq(REFERENCE)].set_index(keys)
    unified = overlaps.loc[overlaps["method_name"].eq(UNIFIED)].set_index(keys)
    delta = noise[[
        "candidate_rank", "coverage_sensitive_response_score", "matched_entity_pairs"
    ]].join(
        unified[[
            "candidate_rank", "coverage_sensitive_response_score", "matched_entity_pairs"
        ]],
        lsuffix="_noise", rsuffix="_unified", how="inner",
    ).reset_index()
    delta["rank_improvement_vs_noise"] = (
        delta["candidate_rank_noise"] - delta["candidate_rank_unified"]
    )
    delta["target_score_delta_vs_noise"] = (
        delta["coverage_sensitive_response_score_unified"]
        - delta["coverage_sensitive_response_score_noise"]
    )
    delta["matched_pair_delta_vs_noise"] = (
        delta["matched_entity_pairs_unified"] - delta["matched_entity_pairs_noise"]
    )
    if len(delta) != 30:
        raise RuntimeError(f"expected 30 registered contexts, got {len(delta)}")

    target_context = context_targets.loc[
        context_targets["method_name"].isin([REFERENCE, UNIFIED])
    ].copy()
    if target_context.groupby("method_name").size().to_dict() != {
        REFERENCE: 6, UNIFIED: 6,
    }:
        raise RuntimeError("unexpected context-aggregate denominator")
    if len(gains) != 1:
        raise RuntimeError(f"expected one unified Top-1 gain, got {len(gains)}")
    gain = gains.iloc[0]

    args.output_dir.mkdir(parents=True)
    outputs = {
        "method_summary": args.output_dir / "method_biology_summary.tsv",
        "unified_context_deltas": args.output_dir / "unified_vs_noise_contexts.tsv",
        "aggregate_targets": args.output_dir / "aggregate_target_comparison.tsv",
    }
    method_summary.to_csv(outputs["method_summary"], sep="\t", index=False)
    delta.to_csv(outputs["unified_context_deltas"], sep="\t", index=False)
    target_context.to_csv(outputs["aggregate_targets"], sep="\t", index=False)

    unified_method = method_summary.loc[method_summary["method_name"].eq(UNIFIED)].iloc[0]
    noise_method = method_summary.loc[method_summary["method_name"].eq(REFERENCE)].iloc[0]
    report = {
        "status": STATUS,
        "dataset_role": "development_consumed_not_external_validation",
        "fixed_denominators": {
            "directional_queries_per_method": int(
                increment_report["denominators"]["directional_entity_queries_per_method"]
            ),
            "identity_candidate_pairs": int(
                increment_report["denominators"]["identity_candidate_pairs"]
            ),
            "registered_response_contexts": 30,
            "microbial_contexts_per_mixture": 5,
            "aggregate_mixture_study_units": 6,
        },
        "unified_vs_noise": {
            "raw_target_top1_contexts": {
                "unified": int(unified_method["raw_target_top1_contexts"]),
                "noise": int(noise_method["raw_target_top1_contexts"]),
            },
            "raw_target_mean_rank": {
                "unified": float(unified_method["raw_target_mean_rank"]),
                "noise": float(noise_method["raw_target_mean_rank"]),
            },
            "rank_contexts_improved_worsened_unchanged": [
                int(delta["rank_improvement_vs_noise"].gt(0).sum()),
                int(delta["rank_improvement_vs_noise"].lt(0).sum()),
                int(delta["rank_improvement_vs_noise"].eq(0).sum()),
            ],
            "target_score_contexts_higher_lower": [
                int(delta["target_score_delta_vs_noise"].gt(0).sum()),
                int(delta["target_score_delta_vs_noise"].lt(0).sum()),
            ],
            "mean_target_score_delta": float(
                delta["target_score_delta_vs_noise"].mean()
            ),
            "matched_pair_contexts_lower": int(
                delta["matched_pair_delta_vs_noise"].lt(0).sum()
            ),
            "mean_matched_pair_delta": float(
                delta["matched_pair_delta_vs_noise"].mean()
            ),
            "aggregate_target_top1_units": {
                "unified": int(unified_method["mean_aggregate_top1_units"]),
                "noise": int(noise_method["mean_aggregate_top1_units"]),
            },
            "aggregate_target_mean_rank": {
                "unified": float(unified_method["mean_target_rank_mean_aggregation"]),
                "noise": float(noise_method["mean_target_rank_mean_aggregation"]),
            },
            "aggregate_units_improved_worsened_unchanged": [
                int(unified_method["improved_units_vs_noise_mean"]),
                int(unified_method["worsened_units_vs_noise_mean"]),
                int(unified_method["unchanged_units_vs_noise_mean"]),
            ],
        },
        "sole_raw_top1_gain": {
            "query_contrast_id": str(gain["query_contrast_id"]),
            "candidate_study_id": str(gain["candidate_study_id"]),
            "candidate_contrast_id": str(gain["candidate_contrast_id"]),
            "rank_noise_to_unified": [
                float(gain["candidate_rank_noise"]),
                float(gain["candidate_rank_method"]),
            ],
            "target_score_noise_to_unified": [
                float(gain["coverage_sensitive_response_score_noise"]),
                float(gain["coverage_sensitive_response_score_method"]),
            ],
            "matched_pairs_noise_to_unified": [
                int(gain["matched_entity_pairs_noise"]),
                int(gain["matched_entity_pairs_method"]),
            ],
            "changed_edges": int(
                attribution_report["denominators"]["changed_edges_in_gain_contexts"]
            ),
            "helpful_harmful_changed_edges": [
                int(attribution_report["denominators"]["helpful_changed_edges"]),
                int(attribution_report["denominators"]["harmful_changed_edges"]),
            ],
            "net_response_numerator_delta": float(gain["net_response_numerator_delta"]),
        },
        "verdict": "LOCAL_RANK_CHANGES_WITHOUT_CONTEXT_ROBUST_TOP1_OR_EVIDENCE_GAIN",
        "inputs": {
            "increment_report": {"path": str(increment_report_path), "sha256": sha256(increment_report_path)},
            "overlap_results": {"path": str(overlap_path), "sha256": sha256(overlap_path)},
            "context_summary": {"path": str(context_summary_path), "sha256": sha256(context_summary_path)},
            "context_targets": {"path": str(context_targets_path), "sha256": sha256(context_targets_path)},
            "attribution_report": {"path": str(attribution_report_path), "sha256": sha256(attribution_report_path)},
            "top1_gain_context": {"path": str(gain_path), "sha256": sha256(gain_path)},
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "claim_boundary": [
            "The model and identity scores were frozen before biological responses were opened.",
            "The application panel is consumed development evidence, not external validation.",
            "Five microbial contexts per mixture are related strata, not independent studies.",
            "A rank gain caused by stronger suppression of alternatives is not evidence that target response support increased.",
            "ChemAware V2 structure evidence and BioAware sample events were unavailable in this application score.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
