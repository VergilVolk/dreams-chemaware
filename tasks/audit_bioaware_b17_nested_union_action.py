#!/usr/bin/env python
"""Test a nested union of B12 linear and B16 nonlinear BioAware actions.

The union is deliberately asymmetric: the calibrated B12 action has priority;
B16 may act only when B12 abstains.  Whether to use B12, B16, their agreement,
or that union is selected exclusively from inner leave-domain-out predictions
inside each outer held domain.  This prevents the opened aggregate observation
that a union has oracle headroom from selecting the held-domain action.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b11_catalog_interaction_action import (  # noqa: E402
    FEATURE_RECIPES,
    apply_gate as apply_linear_gate,
    atomic_json,
    cluster_bootstrap,
    score_queries as score_linear,
    sha256,
    summarize,
)
from audit_bioaware_b12_multicohort_catalog_action import (  # noqa: E402
    EXPECTED_DOMAINS,
    build_universe,
    choose as choose_linear,
    split_domain,
)
from audit_bioaware_b16_pairwise_nonlinear_action import (  # noqa: E402
    FEATURE_FAMILIES,
    apply_gate as apply_nonlinear_gate,
    choose as choose_nonlinear,
    prepare,
    score_queries as score_nonlinear,
)


POLICIES = (
    "b12_only", "b16_only", "agreement_only", "union_b12_priority",
)


def combine(linear: pd.DataFrame, nonlinear: pd.DataFrame, policy: str) -> pd.DataFrame:
    columns = [
        "query_id", "held_label", "source", "truth_candidate_id",
        "truth_formula", "baseline_candidate_id", "baseline_correct",
        "intervene", "final_candidate_id",
    ]
    joined = linear[columns].merge(
        nonlinear[columns], on="query_id", suffixes=("_b12", "_b16"),
        validate="one_to_one",
    )
    for column in (
        "source", "truth_candidate_id", "truth_formula",
        "baseline_candidate_id", "baseline_correct",
    ):
        left, right = joined[f"{column}_b12"], joined[f"{column}_b16"]
        if not left.astype(str).equals(right.astype(str)):
            raise RuntimeError(f"B17 B12/B16 mismatch: {column}")
    baseline = joined["baseline_candidate_id_b12"].astype(str)
    b12_final = joined["final_candidate_id_b12"].astype(str)
    b16_final = joined["final_candidate_id_b16"].astype(str)
    if policy == "b12_only":
        final = b12_final
    elif policy == "b16_only":
        final = b16_final
    elif policy == "agreement_only":
        final = baseline.where(~(
            joined["intervene_b12"].astype(bool)
            & joined["intervene_b16"].astype(bool)
            & b12_final.eq(b16_final)
        ), b12_final)
    elif policy == "union_b12_priority":
        final = baseline.where(
            ~joined["intervene_b16"].astype(bool), b16_final
        ).where(~joined["intervene_b12"].astype(bool), b12_final)
    else:
        raise ValueError(policy)
    output = pd.DataFrame({
        "query_id": joined["query_id"].astype(str),
        "held_label": joined["held_label_b12"].astype(str),
        "source": joined["source_b12"].astype(str),
        "truth_candidate_id": joined["truth_candidate_id_b12"].astype(str),
        "truth_formula": joined["truth_formula_b12"].astype(str),
        "baseline_candidate_id": baseline,
        "baseline_correct": joined["baseline_correct_b12"].astype(bool),
        "b12_intervene": joined["intervene_b12"].astype(bool),
        "b16_intervene": joined["intervene_b16"].astype(bool),
        "b12_final_candidate_id": b12_final,
        "b16_final_candidate_id": b16_final,
        "final_candidate_id": final.astype(str),
    })
    output["intervene"] = output["final_candidate_id"].ne(output["baseline_candidate_id"])
    output["final_correct"] = output["final_candidate_id"].eq(output["truth_candidate_id"])
    output["corrected"] = ~output["baseline_correct"] & output["final_correct"]
    output["introduced"] = output["baseline_correct"] & ~output["final_correct"]
    output["delta"] = output["final_correct"].astype(int) - output["baseline_correct"].astype(int)
    output["combination_policy"] = policy
    return output


def choose_policy(linear: pd.DataFrame, nonlinear: pd.DataFrame) -> tuple[str, list[dict]]:
    ledger: list[dict] = []
    for policy in POLICIES:
        result = combine(linear, nonlinear, policy)
        overall = summarize(result)
        per_domain = {
            source: summarize(result.loc[result["source"].eq(source)])
            for source in sorted(result["source"].unique())
        }
        safe = all(item["risk_net_lambda2"] >= 0 for item in per_domain.values())
        ledger.append({
            "policy": policy,
            **overall,
            "every_inner_domain_risk_nonnegative": safe,
        })
    eligible = [
        item for item in ledger
        if item["every_inner_domain_risk_nonnegative"]
        and item["corrected"] > 2 * item["introduced"]
    ]
    if not eligible:
        return "b12_only", ledger
    selected = max(eligible, key=lambda item: (
        item["risk_net_lambda2"] / max(1, item["queries"]),
        -item["introduced"] / max(1, item["queries"]),
        item["delta_recall1"],
        item["policy"] == "b12_only",
    ))
    return str(selected["policy"]), ledger


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
    parser.add_argument(
        "--b12-transitions", type=Path,
        default=ROOT / "data/validation/bioaware_b12_multicohort_catalog_localcheck_20260907_v1/nested_domain_loso_transitions.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    for path in (
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds, args.b12_transitions,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    candidates, provenance = build_universe(args)
    candidates = prepare(candidates)
    outer_results: list[pd.DataFrame] = []
    b12_replay_results: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    for outer_index, outer_domain in enumerate(EXPECTED_DOMAINS):
        outer_train, outer_test = split_domain(candidates, outer_domain)

        linear_outer: dict[str, pd.DataFrame] = {}
        linear_inner: dict[str, pd.DataFrame] = {}
        for recipe, features in FEATURE_RECIPES.items():
            linear_outer[recipe], _ = score_linear(
                outer_train, outer_test, features, outer_domain
            )
            parts: list[pd.DataFrame] = []
            for inner_domain in EXPECTED_DOMAINS:
                if inner_domain == outer_domain:
                    continue
                train, test = split_domain(outer_train, inner_domain)
                scored, _ = score_linear(
                    train, test, features,
                    f"outer={outer_domain}|inner={inner_domain}",
                )
                parts.append(scored)
            linear_inner[recipe] = pd.concat(parts, ignore_index=True)
        linear_selected, _ = choose_linear(linear_inner)
        linear_outer_result = apply_linear_gate(
            linear_outer[linear_selected["recipe"]],
            float(linear_selected["margin"]), float(linear_selected["probability"]),
        )
        linear_inner_result = apply_linear_gate(
            linear_inner[linear_selected["recipe"]],
            float(linear_selected["margin"]), float(linear_selected["probability"]),
        )
        b12_replay_results.append(linear_outer_result)

        nonlinear_outer: dict[str, pd.DataFrame] = {}
        nonlinear_inner: dict[str, pd.DataFrame] = {}
        for family, features in FEATURE_FAMILIES.items():
            nonlinear_outer[family], _ = score_nonlinear(
                outer_train, outer_test, features, outer_domain,
                args.seed + 100 * outer_index,
            )
            parts = []
            for inner_index, inner_domain in enumerate(EXPECTED_DOMAINS):
                if inner_domain == outer_domain:
                    continue
                train, test = split_domain(outer_train, inner_domain)
                scored, _ = score_nonlinear(
                    train, test, features,
                    f"outer={outer_domain}|inner={inner_domain}",
                    args.seed + 100 * outer_index + inner_index + 1,
                )
                parts.append(scored)
            nonlinear_inner[family] = pd.concat(parts, ignore_index=True)
        nonlinear_selected, _ = choose_nonlinear(nonlinear_inner)
        nonlinear_outer_result = apply_nonlinear_gate(
            nonlinear_outer[nonlinear_selected["feature_family"]],
            float(nonlinear_selected["margin"]),
            float(nonlinear_selected["probability"]),
        )
        nonlinear_inner_result = apply_nonlinear_gate(
            nonlinear_inner[nonlinear_selected["feature_family"]],
            float(nonlinear_selected["margin"]),
            float(nonlinear_selected["probability"]),
        )

        selected_policy, policy_ledger = choose_policy(
            linear_inner_result, nonlinear_inner_result
        )
        outer = combine(linear_outer_result, nonlinear_outer_result, selected_policy)
        outer_results.append(outer)
        fold_reports.append({
            "outer_domain": outer_domain,
            "linear_selected": linear_selected,
            "nonlinear_selected": nonlinear_selected,
            "policy_selected": selected_policy,
            "inner_policy_ledger": policy_ledger,
            "outer_result": summarize(outer),
        })
        print(
            f"[B17 {outer_domain}] policy={selected_policy} "
            f"linear={linear_selected['recipe']} "
            f"nonlinear={nonlinear_selected['feature_family']} "
            f"{summarize(outer)}",
            flush=True,
        )

    result = pd.concat(outer_results, ignore_index=True)
    b12_replay = pd.concat(b12_replay_results, ignore_index=True)
    if len(result) != 860 or result["query_id"].nunique() != 860:
        raise RuntimeError("B17 outer coverage changed")
    b12 = pd.read_csv(args.b12_transitions)
    check = b12_replay[[
        "query_id", "baseline_correct", "final_candidate_id", "intervene",
        "corrected", "introduced",
    ]].merge(
        b12[[
            "query_id", "baseline_correct", "final_candidate_id", "intervene",
            "corrected", "introduced",
        ]], on="query_id", suffixes=("_replay", "_frozen"), validate="one_to_one",
    )
    replay_mismatches = {
        "baseline_correct": int(check["baseline_correct_replay"].astype(bool).ne(
            check["baseline_correct_frozen"].astype(bool)
        ).sum()),
        "final_candidate_id": int(check["final_candidate_id_replay"].astype(str).ne(
            check["final_candidate_id_frozen"].astype(str)
        ).sum()),
        "intervene": int(check["intervene_replay"].astype(bool).ne(
            check["intervene_frozen"].astype(bool)
        ).sum()),
        "corrected": int(check["corrected_replay"].astype(bool).ne(
            check["corrected_frozen"].astype(bool)
        ).sum()),
        "introduced": int(check["introduced_replay"].astype(bool).ne(
            check["introduced_frozen"].astype(bool)
        ).sum()),
    }
    if any(replay_mismatches.values()):
        raise RuntimeError(f"B17 frozen B12 replay mismatch: {replay_mismatches}")
    b12_summary = summarize(b12)

    overall = summarize(result)
    corrected = result.loc[result["corrected"]]
    introduced = result.loc[result["introduced"]]
    identity_ci = cluster_bootstrap(
        result, "truth_candidate_id", args.bootstrap_resamples, args.seed + 1
    )
    formula_ci = cluster_bootstrap(
        result, "truth_formula", args.bootstrap_resamples, args.seed + 2
    )
    by_domain = {
        source: summarize(result.loc[result["source"].eq(source)])
        for source in EXPECTED_DOMAINS
    }
    gates = {
        "gain_ge_5pp": overall["delta_recall1"] >= 0.05,
        "identity_ci_low_positive": identity_ci["ci_low"] > 0,
        "formula_ci_low_positive": formula_ci["ci_low"] > 0,
        "corrected_gt_2x_introduced": overall["corrected"] > 2 * overall["introduced"],
        "corrected_identities_ge_25": corrected["truth_candidate_id"].nunique() >= 25,
        "corrected_formulas_ge_25": corrected["truth_formula"].nunique() >= 25,
        "every_outer_domain_nonnegative": all(
            item["delta_recall1"] >= 0 for item in by_domain.values()
        ),
        "risk_net_strictly_beats_b12": (
            overall["risk_net_lambda2"] > b12_summary["risk_net_lambda2"]
        ),
        "delta_not_below_b12": (
            overall["delta_recall1"] >= b12_summary["delta_recall1"] - 1e-15
        ),
        "introduced_no_more_than_b12": (
            overall["introduced"] <= b12_summary["introduced"]
        ),
        "frozen_b12_querywise_replay_exact": not any(replay_mismatches.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "nested_domain_loso_transitions.csv.gz"
    result.to_csv(path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b17_nested_union_action_complete",
        "formal": True,
        "protocol": (
            "six opened domains; B12 and B16 independently nested; action "
            "combination policy selected on inner domain-OOF predictions only"
        ),
        "nested_oof": {
            **overall,
            "corrected_identities": int(corrected["truth_candidate_id"].nunique()),
            "corrected_formulas": int(corrected["truth_formula"].nunique()),
            "introduced_identities": int(introduced["truth_candidate_id"].nunique()),
            "introduced_formulas": int(introduced["truth_formula"].nunique()),
            "identity_cluster_bootstrap": identity_ci,
            "formula_cluster_bootstrap": formula_ci,
            "by_domain": by_domain,
        },
        "policies": list(POLICIES),
        "frozen_B12_comparator": {
            **b12_summary,
            "querywise_replay_mismatches": replay_mismatches,
        },
        "folds": fold_reports,
        "gates": gates,
        "strictly_better_action_than_B12": bool(all(gates.values())),
        "contracts": {
            "outer_outcome_used_for_component_or_policy_selection": False,
            "held_truth_identity_and_formula_purged": True,
            "posthoc_global_union_used_for_outer_prediction": False,
            "candidate_identity_as_feature": False,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            **provenance["provenance"],
            "B12_transitions_sha256": sha256(args.b12_transitions),
            "transitions_sha256": sha256(path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Opened nested action-combination test. Passing would replace B12 "
            "only as an action router; it is not blind validation, reaction "
            "mechanism, SOTA, or shared-embedding improvement."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["strictly_better_action_than_B12"]:
        raise RuntimeError(f"B17 did not strictly improve B12: {gates}")


if __name__ == "__main__":
    main()
