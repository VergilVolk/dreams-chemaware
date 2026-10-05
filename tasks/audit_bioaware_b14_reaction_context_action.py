#!/usr/bin/env python
"""Test reaction-context actions beyond the multicohort catalogue prior.

B12 established a broad catalogue-opportunity action across six opened
development domains.  B14 asks whether reaction-path availability or bounded
path-strength features add a second action.  Ranker recipe and intervention
gate are selected only from inner leave-domain-out predictions.  Truth
identities and formulae of each held domain are purged before every fit.

This is opened action discovery.  It does not change the shared embedding and
does not establish reaction mechanism or external generalisation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b11_catalog_interaction_action import (  # noqa: E402
    FEATURE_RECIPES as B12_RECIPES,
    apply_gate,
    atomic_json,
    cluster_bootstrap,
    score_queries,
    sha256,
    summarize,
)
from audit_bioaware_b12_multicohort_catalog_action import (  # noqa: E402
    EXPECTED_DOMAINS,
    GATE_GRID,
    build_universe,
    split_domain,
)


RISK_PENALTY = 2
B12_RISK_NET = 42
B12_INTRODUCED = 8

CATALOGUE = list(B12_RECIPES["linear_b4_replay"])
REACTION_AVAILABILITY = [
    *CATALOGUE,
    "known_path_present",
    "edge0_present",
]
REACTION_STRENGTH = [
    *REACTION_AVAILABILITY,
    "known_path_per_degree",
    "known_inverse_depth_mean",
    "known_seed_per_degree",
    "edge0_complete_fraction",
    "edge0_bottleneck_mean",
    "edge0_reliability",
]

# B12 recipes are retained verbatim.  The two additional recipes are exactly
# the previously specified B1 reaction action families, not an unrestricted
# feature search.
FEATURE_RECIPES: dict[str, list[str]] = {
    **{name: list(features) for name, features in B12_RECIPES.items()},
    "reaction_availability": REACTION_AVAILABILITY,
    "reaction_strength": REACTION_STRENGTH,
}


def add_reaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    numeric = [
        "known_path_fraction",
        "known_inverse_depth_mean",
        "known_log_seed_support_mean",
        "known_log_degree",
        "edge0_complete_fraction",
        "edge0_bottleneck_mean",
    ]
    missing = set(numeric) - set(output.columns)
    if missing:
        raise RuntimeError(f"B14 candidate cache missing columns: {sorted(missing)}")
    for column in numeric:
        output[column] = pd.to_numeric(output[column], errors="raise").astype(float)
    degree_scale = 1.0 + output["known_log_degree"].clip(lower=0.0)
    output["known_path_present"] = (
        output["known_path_fraction"] > 0.0
    ).astype(float)
    output["edge0_present"] = (
        output["edge0_complete_fraction"] > 0.0
    ).astype(float)
    output["known_path_per_degree"] = (
        output["known_path_fraction"].clip(lower=0.0) / degree_scale
    )
    output["known_seed_per_degree"] = (
        output["known_log_seed_support_mean"].clip(lower=0.0) / degree_scale
    )
    output["edge0_reliability"] = (
        output["edge0_complete_fraction"].clip(lower=0.0, upper=1.0)
        * output["edge0_bottleneck_mean"].clip(lower=0.0, upper=1.0)
    )
    all_features = sorted(set().union(*FEATURE_RECIPES.values()))
    values = output[all_features].to_numpy(float)
    if not np.isfinite(values).all():
        bad = {
            column: int((~np.isfinite(output[column].to_numpy(float))).sum())
            for column in all_features
            if not np.isfinite(output[column].to_numpy(float)).all()
        }
        raise RuntimeError(f"B14 non-finite model features: {bad}")
    return output


def choose(inner_scores: dict[str, pd.DataFrame]) -> tuple[dict, list[dict]]:
    ledger: list[dict] = []
    for recipe, scores in inner_scores.items():
        for margin, probability in GATE_GRID:
            result = apply_gate(scores, margin, probability)
            overall = summarize(result)
            per_domain = {
                domain: summarize(result.loc[result["source"].eq(domain)])
                for domain in sorted(result["source"].unique())
            }
            safe = all(
                item["risk_net_lambda2"] >= 0 for item in per_domain.values()
            )
            ledger.append({
                "recipe": recipe,
                "margin": margin,
                "probability": probability,
                **overall,
                "every_inner_domain_risk_nonnegative": safe,
            })
    eligible = [
        row for row in ledger
        if row["every_inner_domain_risk_nonnegative"]
        and row["corrected"] > RISK_PENALTY * row["introduced"]
    ]
    if not eligible:
        selected = min(
            ledger,
            key=lambda row: (
                row["introduced"], -row["corrected"], row["intervention_rate"]
            ),
        )
        return {**selected, "selection_reason": "no_safe_configuration__minimum_harm"}, ledger
    selected = max(
        eligible,
        key=lambda row: (
            row["risk_net_lambda2"] / max(1, row["queries"]),
            -row["introduced"] / max(1, row["queries"]),
            row["delta_recall1"],
            -row["intervention_rate"],
            row["recipe"] == "linear_b4_replay",
        ),
    )
    return {**selected, "selection_reason": "maximum_safe_inner_domain_oof_risk_net_rate"}, ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-queries", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz",
    )
    parser.add_argument(
        "--kgmn-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--kgmn-seeds", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    for path in (
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    candidates, universe_report = build_universe(args)
    candidates = add_reaction_features(candidates)
    evaluated = candidates.loc[candidates["polarity"].eq("negative")]
    outer_results: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    for outer_domain in EXPECTED_DOMAINS:
        outer_train, outer_test = split_domain(candidates, outer_domain)
        outer_scores: dict[str, pd.DataFrame] = {}
        inner_scores: dict[str, pd.DataFrame] = {}
        fit_reports: dict[str, dict] = {}
        for recipe, features in FEATURE_RECIPES.items():
            scored, outer_fit = score_queries(
                outer_train, outer_test, features, outer_domain
            )
            outer_scores[recipe] = scored
            inner_tables: list[pd.DataFrame] = []
            inner_fits: list[dict] = []
            for inner_domain in EXPECTED_DOMAINS:
                if inner_domain == outer_domain:
                    continue
                inner_train, inner_test = split_domain(outer_train, inner_domain)
                if inner_test["query_id"].nunique() < 10:
                    raise RuntimeError(
                        f"{outer_domain}/{inner_domain}: fewer than 10 inner queries"
                    )
                inner_scored, inner_fit = score_queries(
                    inner_train,
                    inner_test,
                    features,
                    f"outer={outer_domain}|inner={inner_domain}",
                )
                inner_tables.append(inner_scored)
                inner_fits.append(inner_fit)
            inner_scores[recipe] = pd.concat(inner_tables, ignore_index=True)
            fit_reports[recipe] = {"outer": outer_fit, "inner": inner_fits}

        selected, ledger = choose(inner_scores)
        result = apply_gate(
            outer_scores[selected["recipe"]],
            float(selected["margin"]),
            float(selected["probability"]),
        )
        result["selected_recipe"] = selected["recipe"]
        outer_results.append(result)
        fold_reports.append({
            "outer_domain": outer_domain,
            "selected": selected,
            "outer_result": summarize(result),
            "inner_selection_ledger": ledger,
            "fit_reports": fit_reports,
        })
        print(
            f"[B14 {outer_domain}] {selected['recipe']} "
            f"m={selected['margin']:.2f} p={selected['probability']:.2f} "
            f"{summarize(result)}",
            flush=True,
        )

    result = pd.concat(outer_results, ignore_index=True)
    if len(result) != 860 or result["query_id"].nunique() != 860:
        raise RuntimeError("B14 outer OOF coverage changed")
    overall = summarize(result)
    formula_ci = cluster_bootstrap(
        result, "truth_formula", args.bootstrap_resamples, args.seed + 1
    )
    identity_ci = cluster_bootstrap(
        result, "truth_candidate_id", args.bootstrap_resamples, args.seed + 2
    )
    by_domain = {
        domain: summarize(result.loc[result["source"].eq(domain)])
        for domain in EXPECTED_DOMAINS
    }
    corrected = result.loc[result["corrected"]]
    introduced = result.loc[result["introduced"]]
    gates = {
        "gain_ge_5pp": overall["delta_recall1"] >= 0.05,
        "formula_ci_low_positive": formula_ci["ci_low"] > 0,
        "identity_ci_low_positive": identity_ci["ci_low"] > 0,
        "corrected_gt_2x_introduced": overall["corrected"] > 2 * overall["introduced"],
        "corrected_identities_ge_25": corrected["truth_candidate_id"].nunique() >= 25,
        "corrected_formulas_ge_25": corrected["truth_formula"].nunique() >= 25,
        "every_outer_domain_nonnegative": all(
            item["delta_recall1"] >= 0 for item in by_domain.values()
        ),
        "risk_net_strictly_better_than_b12": overall["risk_net_lambda2"] > B12_RISK_NET,
        "introduced_no_more_than_b12": overall["introduced"] <= B12_INTRODUCED,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transitions_path = args.output_dir / "nested_domain_loso_transitions.csv.gz"
    result.to_csv(transitions_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b14_reaction_context_action_complete",
        "formal": True,
        "protocol": (
            "six opened development domains; nested domain-OOF recipe/gate "
            "selection; held truth identity and formula purged"
        ),
        "b12_reference": {
            "risk_net_lambda2": B12_RISK_NET,
            "introduced": B12_INTRODUCED,
        },
        "universe": {
            "queries": int(evaluated["query_id"].nunique()),
            "identities": int(evaluated["truth_candidate_id"].nunique()),
            "formulas": int(evaluated["truth_formula"].nunique()),
            **universe_report["query_counts"],
        },
        "nested_oof": {
            **overall,
            "corrected_identities": int(corrected["truth_candidate_id"].nunique()),
            "corrected_formulas": int(corrected["truth_formula"].nunique()),
            "introduced_identities": int(introduced["truth_candidate_id"].nunique()),
            "introduced_formulas": int(introduced["truth_formula"].nunique()),
            "identity_cluster_bootstrap": identity_ci,
            "formula_cluster_bootstrap": formula_ci,
            "by_domain": by_domain,
            "selected_recipe_query_counts": dict(
                Counter(result["selected_recipe"].astype(str))
            ),
        },
        "feature_recipes": FEATURE_RECIPES,
        "folds": fold_reports,
        "gates": gates,
        "pass_to_reaction_context_action_construction": bool(all(gates.values())),
        "contracts": {
            "all_domains_opened_development": True,
            "outer_domain_outcomes_used_for_selection": False,
            "held_truth_identity_and_formula_purged": True,
            "candidate_identity_as_feature": False,
            "truth_or_formula_as_feature": False,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            **universe_report["provenance"],
            "transitions_sha256": sha256(transitions_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Opened multicohort reaction-context action discovery. Passing "
            "does not establish reaction mechanism, external confirmation, "
            "SOTA, or shared-embedding gain."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["pass_to_reaction_context_action_construction"]:
        raise RuntimeError(f"B14 scientific gate failed: {gates}")


if __name__ == "__main__":
    main()
