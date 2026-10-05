#!/usr/bin/env python
"""Expand BioAware catalogue-prior action discovery across six cohorts.

The primary universe combines the four Full16 biological sources with the
opened ST001154 same-formula/10-ppm and KGMN-200STD hidden-seed panels.  These
are all development resources.  Every outer domain is held out wholesale and
its truth identities and formulae are removed before model fitting.  Recipe
and gate selection use only inner-domain OOF predictions.

This experiment expands chemical coverage of the B4/B11 catalogue-prior
action.  It cannot establish reaction specificity, external confirmation, or
shared-embedding improvement.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from audit_bioaware_b5_graph_prior_external_transfer import (  # noqa: E402
    load_kgmn_panel,
    load_st_panel,
)
from audit_bioaware_b11_catalog_interaction_action import (  # noqa: E402
    FEATURE_RECIPES,
    apply_gate,
    atomic_json,
    cluster_bootstrap,
    enrich,
    score_queries,
    sha256,
    summarize,
)


INTERNAL_DOMAINS = ("BV2cell", "Mouse_brain", "Mouse_liver", "NIST_plasma")
EXTERNAL_DOMAINS = ("ST001154_same_formula_10ppm", "KGMN200STD_hidden_seed")
EXPECTED_DOMAINS = (*INTERNAL_DOMAINS, *EXTERNAL_DOMAINS)
RISK_PENALTY = 2

# The wider probability range is fixed from the already-open B5 observation
# that absolute B4 confidence did not calibrate across cohorts.  Selection is
# nevertheless fully nested; no outer-domain outcome selects a configuration.
GATE_GRID = tuple(
    (margin, probability)
    for margin in (0.04, 0.05, 0.08)
    for probability in (0.55, 0.60, 0.65, 0.70, 0.75)
)


def prefix_domain(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    output = frame.copy()
    output["query_id"] = domain + "::" + output["query_id"].astype(str)
    output["source"] = domain
    output["unit_id"] = domain
    output["polarity"] = "negative"
    return output


def build_universe(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    internal = pd.read_csv(args.internal_candidates)
    st = prefix_domain(
        load_st_panel(
            args.st_candidates,
            args.st_queries,
            "ST001154_same_formula_10ppm",
        ),
        "ST001154_same_formula_10ppm",
    )
    kgmn = prefix_domain(
        load_kgmn_panel(
            args.kgmn_candidates,
            args.kgmn_seeds,
            "KGMN200STD_hidden_seed",
        ),
        "KGMN200STD_hidden_seed",
    )
    if internal["query_id"].astype(str).str.contains("::repeat=", regex=False).any():
        raise RuntimeError("internal query IDs unexpectedly contain external repeat suffix")
    combined = enrich(pd.concat([internal, st, kgmn], ignore_index=True, sort=False))
    evaluated = combined.loc[combined["polarity"].eq("negative")]
    domains = tuple(sorted(evaluated["source"].astype(str).unique()))
    if set(domains) != set(EXPECTED_DOMAINS):
        raise RuntimeError(f"B12 domains changed: {domains}")
    query_counts = evaluated.groupby("source")["query_id"].nunique().to_dict()
    expected_counts = {
        "BV2cell": 95,
        "Mouse_brain": 131,
        "Mouse_liver": 176,
        "NIST_plasma": 146,
        "ST001154_same_formula_10ppm": 150,
        "KGMN200STD_hidden_seed": 162,
    }
    if query_counts != expected_counts:
        raise RuntimeError(f"B12 query counts changed: {query_counts}")
    provenance = {
        "internal_candidates": sha256(args.internal_candidates),
        "st_candidates": sha256(args.st_candidates),
        "st_queries": sha256(args.st_queries),
        "kgmn_candidates": sha256(args.kgmn_candidates),
        "kgmn_seeds": sha256(args.kgmn_seeds),
    }
    return combined, {"query_counts": query_counts, "provenance": provenance}


def split_domain(frame: pd.DataFrame, held_domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    held_all = frame.loc[frame["source"].eq(held_domain)].copy()
    test = held_all.loc[held_all["polarity"].eq("negative")].copy()
    identities = set(held_all["truth_candidate_id"].astype(str))
    formulas = set(held_all["truth_formula"].astype(str))
    train = frame.loc[
        ~frame["source"].eq(held_domain)
        & ~frame["truth_candidate_id"].astype(str).isin(identities)
        & ~frame["truth_formula"].astype(str).isin(formulas)
    ].copy()
    if set(train["truth_candidate_id"].astype(str)) & identities:
        raise RuntimeError(f"{held_domain}: truth-identity leakage")
    if set(train["truth_formula"].astype(str)) & formulas:
        raise RuntimeError(f"{held_domain}: truth-formula leakage")
    return train, test


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
            every_domain_nonnegative = all(
                item["risk_net_lambda2"] >= 0 for item in per_domain.values()
            )
            ledger.append({
                "recipe": recipe,
                "margin": margin,
                "probability": probability,
                **overall,
                "every_inner_domain_risk_nonnegative": every_domain_nonnegative,
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
    return {
        **selected,
        "selection_reason": "maximum_safe_inner_domain_oof_risk_net_rate",
    }, ledger


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
            "outer_test_queries": int(outer_test["query_id"].nunique()),
            "outer_train_queries_after_purge": int(outer_train["query_id"].nunique()),
            "selected": selected,
            "outer_result": summarize(result),
            "inner_selection_ledger": ledger,
            "fit_reports": fit_reports,
        })
        print(
            f"[B12 {outer_domain}] {selected['recipe']} "
            f"m={selected['margin']:.2f} p={selected['probability']:.2f} "
            f"{summarize(result)}",
            flush=True,
        )

    result = pd.concat(outer_results, ignore_index=True)
    if len(result) != 860 or result["query_id"].nunique() != 860:
        raise RuntimeError("B12 outer OOF coverage changed")
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
        "gain_ge_3pp": overall["delta_recall1"] >= 0.03,
        "formula_ci_low_positive": formula_ci["ci_low"] > 0,
        "identity_ci_low_positive": identity_ci["ci_low"] > 0,
        "corrected_gt_2x_introduced": overall["corrected"] > 2 * overall["introduced"],
        "corrected_identities_ge_20": corrected["truth_candidate_id"].nunique() >= 20,
        "corrected_formulas_ge_20": corrected["truth_formula"].nunique() >= 20,
        "every_outer_domain_nonnegative": all(
            item["delta_recall1"] >= 0 for item in by_domain.values()
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transition_path = args.output_dir / "nested_domain_loso_transitions.csv.gz"
    result.to_csv(transition_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b12_multicohort_catalog_action_complete",
        "formal": True,
        "protocol": (
            "six opened development domains; nested leave-domain-out recipe/gate "
            "selection; held truth identity and formula purged"
        ),
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
        "gate_grid": [
            {"margin": margin, "probability": probability}
            for margin, probability in GATE_GRID
        ],
        "folds": fold_reports,
        "gates": gates,
        "pass_to_shared_embedding_action_construction": bool(all(gates.values())),
        "contracts": {
            "all_domains_opened_development": True,
            "outer_domain_outcomes_used_for_selection": False,
            "held_truth_identity_and_formula_purged": True,
            "identity_equal_pairwise_training": True,
            "candidate_identity_as_feature": False,
            "truth_or_formula_as_feature": False,
            "reaction_specificity_claim": False,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            **universe_report["provenance"],
            "transitions_sha256": sha256(transition_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Opened multicohort action discovery.  Passing establishes broad "
            "catalogue-prior training actions, not reaction mechanism, blind "
            "generalization, SOTA, or improved shared embeddings."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["pass_to_shared_embedding_action_construction"]:
        raise RuntimeError(f"B12 scientific gate failed: {gates}")


if __name__ == "__main__":
    main()
