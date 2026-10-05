#!/usr/bin/env python
"""Decompose the surviving BioAware opportunity action into library and graph priors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b3_reaction_coabundance_action import (  # noqa: E402
    atomic_json,
    enforce_negative_only,
)
from audit_bioaware_b3b_negative_coabundance_action import negative_source_loso  # noqa: E402
from develop_bioaware_b1_multisource_action import (  # noqa: E402
    EXPECTED_SOURCES,
    paired_cluster_bootstrap,
    paired_transition,
    run_source_loso,
    sha256,
    summarize,
)


RECIPES = {
    "spectral_only": ["spectral_score"],
    "reference_only": ["spectral_score", "log_reference_spectra"],
    "graph_only": [
        "spectral_score", "network_member", "known_log_degree",
        "known_mass_candidate_fraction",
    ],
    "combined_opportunity": [
        "spectral_score", "log_reference_spectra", "network_member",
        "known_log_degree", "known_mass_candidate_fraction",
    ],
}


def run_protocol(
    candidates: pd.DataFrame, training: str
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    results: dict[str, pd.DataFrame] = {}
    reports: dict[str, dict] = {}
    for recipe, features in RECIPES.items():
        if training == "mixed_polarity":
            result, folds = run_source_loso(candidates, recipe, features)
            result = enforce_negative_only(result)
        elif training == "negative_only":
            result, folds = negative_source_loso(candidates, recipe, features)
        else:
            raise ValueError(training)
        results[recipe] = result
        reports[recipe] = {
            "features": features,
            "pooled": summarize(result),
            "negative": summarize(result.loc[result.polarity.eq("negative")]),
            "by_source": {
                source: summarize(result.loc[result.source.eq(source)])
                for source in EXPECTED_SOURCES
            },
            "folds": folds,
        }
    return results, reports


def compare(
    left: pd.DataFrame,
    right: pd.DataFrame,
    repeats: int,
    seed: int,
) -> dict:
    by_source = {}
    for source in EXPECTED_SOURCES:
        left_source = left.loc[left.source.eq(source)]
        right_source = right.loc[right.source.eq(source)]
        by_source[source] = {
            **paired_transition(left_source, right_source),
            "delta": float(
                left_source.gated_correct.mean() - right_source.gated_correct.mean()
            ),
        }
    return {
        "transition": paired_transition(left, right),
        "identity_cluster_bootstrap": paired_cluster_bootstrap(
            left, right, "truth_candidate_id", repeats, seed
        ),
        "formula_cluster_bootstrap": paired_cluster_bootstrap(
            left, right, "truth_formula", repeats, seed + 1
        ),
        "by_source": by_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-features", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if not args.candidate_features.is_file():
        raise FileNotFoundError(args.candidate_features)
    candidates = pd.read_csv(args.candidate_features)
    required = set().union(*RECIPES.values()) | {
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "source", "polarity", "baseline_correct",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise RuntimeError(f"candidate cache missing columns: {sorted(missing)}")
    queries = candidates[[
        "query_id", "truth_candidate_id", "truth_formula", "source",
        "polarity", "baseline_correct",
    ]].drop_duplicates()
    negative_queries = queries.loc[queries.polarity.eq("negative")]
    if len(queries) != 1426 or len(negative_queries) != 548:
        raise RuntimeError("B4 candidate protocol changed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_reports: dict[str, dict] = {}
    for offset, training in enumerate(("mixed_polarity", "negative_only")):
        print(f"[B4] training={training}", flush=True)
        results, recipes = run_protocol(candidates, training)
        for recipe, result in results.items():
            path = args.output_dir / f"{training}__{recipe}__transitions.csv.gz"
            result.to_csv(path, index=False, compression="gzip")
            recipes[recipe]["transitions_sha256"] = sha256(path)
        negative_results = {
            recipe: result.loc[result.polarity.eq("negative")].copy()
            for recipe, result in results.items()
        }
        protocol_reports[training] = {
            "recipes": recipes,
            "combined_vs_reference_primary": compare(
                negative_results["combined_opportunity"], negative_results["reference_only"],
                args.bootstrap_resamples, args.seed + 10 * offset,
            ),
            "graph_only_vs_spectral": compare(
                negative_results["graph_only"], negative_results["spectral_only"],
                args.bootstrap_resamples, args.seed + 10 * offset + 2,
            ),
            "reference_only_vs_spectral": compare(
                negative_results["reference_only"], negative_results["spectral_only"],
                args.bootstrap_resamples, args.seed + 10 * offset + 4,
            ),
        }

    historical = protocol_reports["mixed_polarity"]
    combined = historical["recipes"]["combined_opportunity"]["negative"]
    graph_increment = historical["combined_vs_reference_primary"]
    gates = {
        "combined_gain_ge_3pp": combined["gated_delta_recall1"] >= 0.03,
        "combined_corrected_gt_2x_introduced": combined["corrected"] > 2 * combined["introduced"],
        "combined_corrected_identities_ge_10": combined["corrected_identities"] >= 10,
        "graph_increment_identity_ci_low_positive": graph_increment["identity_cluster_bootstrap"]["ci_low"] > 0,
        "graph_increment_formula_ci_low_positive": graph_increment["formula_cluster_bootstrap"]["ci_low"] > 0,
        "graph_increment_all_sources_nonnegative": all(
            item["delta"] >= 0 for item in graph_increment["by_source"].values()
        ),
    }
    report = {
        "status": "bioaware_b4_opportunity_decomposition_complete",
        "formal": True,
        "candidate_protocol": {
            "all_queries": int(len(queries)),
            "negative_queries": int(len(negative_queries)),
            "negative_identities": int(negative_queries.truth_candidate_id.nunique()),
            "negative_formulas": int(negative_queries.truth_formula.nunique()),
            "sources": int(negative_queries.source.nunique()),
            "baseline_negative_recall1": float(negative_queries.baseline_correct.mean()),
        },
        "protocols": protocol_reports,
        "gates": gates,
        "graph_component_is_bioaware": bool(all(gates.values())),
        "contracts": {
            "primary_action_negative_only": True,
            "positive_interventions": 0,
            "held_source_truth_identity_and_formula_purged": True,
            "reference_only_is_primary_comparator": True,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "candidate_features": sha256(args.candidate_features),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": "This decomposes an opened-development reranking action. Only gain beyond reference-only is attributable to the graph; no result here establishes shared-embedding improvement or SOTA.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
