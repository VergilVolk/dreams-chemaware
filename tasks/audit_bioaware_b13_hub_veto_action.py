#!/usr/bin/env python
"""Cross-fit a deployment-visible hub-overpromotion veto for BioAware.

B12 showed that catalogue opportunity is a broad, useful action, but its harm
clusters when a proposal gains only graph degree and no new graph-membership or
mass-candidate coverage evidence.  B13 evaluates that mechanism without using
candidate identity: recipe, gate and veto threshold are chosen solely from
inner-domain OOF rows for every held outer domain.

All six cohorts are already opened development resources.  This is an action
and safety discovery audit, not external confirmation or embedding training.
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

from audit_bioaware_b11_catalog_interaction_action import (  # noqa: E402
    FEATURE_RECIPES,
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


VETOES: tuple[tuple[str, float | None], ...] = (
    ("none", None),
    ("degree_only_jump_ge_1_5", 1.5),
    ("degree_only_jump_ge_2_0", 2.0),
    ("degree_only_jump_ge_2_5", 2.5),
)
RISK_PENALTY = 2
B12_RISK_NET = 42
B12_INTRODUCED = 8


def apply_veto(
    scored: pd.DataFrame,
    margin: float,
    probability: float,
    veto_name: str,
    degree_threshold: float | None,
) -> pd.DataFrame:
    output = apply_gate(scored, margin, probability)
    required = {
        "delta_network_member", "delta_known_log_degree",
        "delta_known_mass_candidate_fraction",
    }
    missing = required - set(output.columns)
    if missing:
        raise RuntimeError(f"B13 score table missing veto columns: {sorted(missing)}")
    if veto_name == "none":
        veto = pd.Series(False, index=output.index)
    else:
        if degree_threshold is None:
            raise RuntimeError(f"{veto_name}: degree threshold is missing")
        veto = (
            output["intervene"].astype(bool)
            & output["delta_network_member"].astype(float).le(1e-12)
            & output["delta_known_mass_candidate_fraction"].astype(float).le(1e-12)
            & output["delta_known_log_degree"].astype(float).ge(
                float(degree_threshold) - 1e-12
            )
        )
    output["veto_applied"] = veto
    output.loc[veto, "intervene"] = False
    output.loc[veto, "final_candidate_id"] = output.loc[
        veto, "baseline_candidate_id"
    ].astype(str)
    output["final_correct"] = output["final_candidate_id"].astype(str).eq(
        output["truth_candidate_id"].astype(str)
    )
    output["corrected"] = ~output["baseline_correct"].astype(bool) & output["final_correct"]
    output["introduced"] = output["baseline_correct"].astype(bool) & ~output["final_correct"]
    output["delta"] = (
        output["final_correct"].astype(int) - output["baseline_correct"].astype(int)
    )
    output["veto_name"] = veto_name
    output["veto_degree_threshold"] = (
        float(degree_threshold) if degree_threshold is not None else -1.0
    )
    return output


def choose(inner_scores: dict[str, pd.DataFrame]) -> tuple[dict, list[dict]]:
    ledger: list[dict] = []
    for recipe, scores in inner_scores.items():
        for margin, probability in GATE_GRID:
            for veto_name, threshold in VETOES:
                result = apply_veto(
                    scores, margin, probability, veto_name, threshold
                )
                overall = summarize(result)
                by_domain = {
                    domain: summarize(result.loc[result["source"].eq(domain)])
                    for domain in sorted(result["source"].unique())
                }
                ledger.append({
                    "recipe": recipe,
                    "margin": margin,
                    "probability": probability,
                    "veto_name": veto_name,
                    "veto_degree_threshold": (
                        float(threshold) if threshold is not None else -1.0
                    ),
                    "vetoed": int(result["veto_applied"].sum()),
                    **overall,
                    "every_inner_domain_risk_nonnegative": all(
                        item["risk_net_lambda2"] >= 0
                        for item in by_domain.values()
                    ),
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
            row["veto_name"] == "none",
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
    results: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    for outer_domain in EXPECTED_DOMAINS:
        outer_train, outer_test = split_domain(candidates, outer_domain)
        outer_scores: dict[str, pd.DataFrame] = {}
        inner_scores: dict[str, pd.DataFrame] = {}
        for recipe, features in FEATURE_RECIPES.items():
            outer_scores[recipe], _ = score_queries(
                outer_train, outer_test, features, outer_domain
            )
            inner_tables: list[pd.DataFrame] = []
            for inner_domain in EXPECTED_DOMAINS:
                if inner_domain == outer_domain:
                    continue
                inner_train, inner_test = split_domain(outer_train, inner_domain)
                if inner_test["query_id"].nunique() < 10:
                    raise RuntimeError(
                        f"{outer_domain}/{inner_domain}: fewer than 10 inner queries"
                    )
                inner_scored, _ = score_queries(
                    inner_train,
                    inner_test,
                    features,
                    f"outer={outer_domain}|inner={inner_domain}",
                )
                inner_tables.append(inner_scored)
            inner_scores[recipe] = pd.concat(inner_tables, ignore_index=True)
        selected, ledger = choose(inner_scores)
        threshold = (
            None
            if selected["veto_name"] == "none"
            else float(selected["veto_degree_threshold"])
        )
        result = apply_veto(
            outer_scores[selected["recipe"]],
            float(selected["margin"]),
            float(selected["probability"]),
            str(selected["veto_name"]),
            threshold,
        )
        result["selected_recipe"] = selected["recipe"]
        results.append(result)
        fold_reports.append({
            "outer_domain": outer_domain,
            "selected": selected,
            "outer_result": {**summarize(result), "vetoed": int(result["veto_applied"].sum())},
            "inner_selection_ledger": ledger,
        })
        print(
            f"[B13 {outer_domain}] {selected['recipe']} "
            f"m={selected['margin']:.2f} p={selected['probability']:.2f} "
            f"veto={selected['veto_name']} {summarize(result)}",
            flush=True,
        )

    result = pd.concat(results, ignore_index=True)
    if len(result) != 860 or result["query_id"].nunique() != 860:
        raise RuntimeError("B13 outer OOF coverage changed")
    overall = summarize(result)
    identity_ci = cluster_bootstrap(
        result, "truth_candidate_id", args.bootstrap_resamples, args.seed + 1
    )
    formula_ci = cluster_bootstrap(
        result, "truth_formula", args.bootstrap_resamples, args.seed + 2
    )
    by_domain = {
        domain: summarize(result.loc[result["source"].eq(domain)])
        for domain in EXPECTED_DOMAINS
    }
    corrected = result.loc[result["corrected"]]
    introduced = result.loc[result["introduced"]]
    gates = {
        "gain_ge_3pp": overall["delta_recall1"] >= 0.03,
        "identity_ci_low_positive": identity_ci["ci_low"] > 0,
        "formula_ci_low_positive": formula_ci["ci_low"] > 0,
        "corrected_gt_2x_introduced": overall["corrected"] > 2 * overall["introduced"],
        "corrected_identities_ge_20": corrected["truth_candidate_id"].nunique() >= 20,
        "corrected_formulas_ge_20": corrected["truth_formula"].nunique() >= 20,
        "every_outer_domain_nonnegative": all(
            item["delta_recall1"] >= 0 for item in by_domain.values()
        ),
        "risk_net_strictly_better_than_b12": overall["risk_net_lambda2"] > B12_RISK_NET,
        "introduced_strictly_fewer_than_b12": overall["introduced"] < B12_INTRODUCED,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transitions_path = args.output_dir / "nested_domain_loso_transitions.csv.gz"
    result.to_csv(transitions_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b13_hub_veto_action_complete",
        "formal": True,
        "protocol": (
            "six opened development domains; nested domain-OOF ranker, gate and "
            "identity-free hub-veto selection"
        ),
        "universe": {
            "queries": int(evaluated["query_id"].nunique()),
            "identities": int(evaluated["truth_candidate_id"].nunique()),
            "formulas": int(evaluated["truth_formula"].nunique()),
            **universe_report["query_counts"],
        },
        "nested_oof": {
            **overall,
            "vetoed": int(result["veto_applied"].sum()),
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
            "selected_veto_query_counts": dict(
                Counter(result["veto_name"].astype(str))
            ),
        },
        "b12_reference": {
            "risk_net_lambda2": B12_RISK_NET,
            "introduced": B12_INTRODUCED,
            "note": "fixed from completed B12 nested OOF before B13",
        },
        "feature_recipes": FEATURE_RECIPES,
        "vetoes": [
            {"name": name, "degree_threshold": threshold}
            for name, threshold in VETOES
        ],
        "gate_grid": [
            {"margin": margin, "probability": probability}
            for margin, probability in GATE_GRID
        ],
        "folds": fold_reports,
        "gates": gates,
        "pass_to_hub_safe_action_construction": bool(all(gates.values())),
        "contracts": {
            "all_domains_opened_development": True,
            "outer_domain_outcomes_used_for_selection": False,
            "held_truth_identity_and_formula_purged": True,
            "candidate_identity_used_by_veto": False,
            "veto_uses_deployment_visible_evidence_deltas_only": True,
            "reaction_specificity_claim": False,
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
            "Opened action-safety discovery. Passing qualifies cross-fitted "
            "catalogue-prior actions for direct embedding experiments; it is not "
            "external confirmation, reaction mechanism, SOTA, or embedding gain."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["pass_to_hub_safe_action_construction"]:
        raise RuntimeError(f"B13 scientific gate failed: {gates}")


if __name__ == "__main__":
    main()
