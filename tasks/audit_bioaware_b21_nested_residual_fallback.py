#!/usr/bin/env python
"""Test a fully nested residual fallback beyond the frozen B17 action.

B19 found one low-complexity residual hypothesis: when B17 abstains, a unique
maximum in the mass-window catalogue-coverage fraction may recover a small
number of additional errors.  B21 does not use that opened aggregate outcome.
For every held domain it reconstructs B17, then selects no fallback or one of
the three already-open margin gates exclusively on inner-domain OOF rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
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
from audit_bioaware_b17_nested_union_action import combine, choose_policy  # noqa: E402


FALLBACKS: tuple[float | None, ...] = (None, 0.04, 0.05, 0.08)
B17_RISK_NET = 43
B17_DELTA = 50 / 860
B17_INTRODUCED = 7


def mass_coverage_proposals(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for query_id, group in frame.groupby("query_id", sort=False):
        maximum = float(group["known_mass_candidate_fraction"].max())
        top = group.loc[np.isclose(
            group["known_mass_candidate_fraction"], maximum, rtol=0, atol=1e-12
        )].sort_values("candidate_id", kind="stable")
        rows.append({
            "query_id": str(query_id),
            "mass_proposal_candidate_id": str(top["candidate_id"].iloc[0]),
            "mass_proposal_unique": bool(len(top) == 1),
            "mass_proposal_value": maximum,
            "fallback_baseline_gap": float(group["baseline_gap"].iloc[0]),
        })
    return pd.DataFrame(rows)


def apply_fallback(
    base: pd.DataFrame,
    candidates: pd.DataFrame,
    margin: float | None,
) -> pd.DataFrame:
    output = base.merge(
        mass_coverage_proposals(candidates), on="query_id", validate="one_to_one"
    )
    if margin is None:
        use = pd.Series(False, index=output.index)
    else:
        use = (
            ~output["intervene"].astype(bool)
            & output["mass_proposal_unique"].astype(bool)
            & output["mass_proposal_candidate_id"].astype(str).ne(
                output["baseline_candidate_id"].astype(str)
            )
            & output["fallback_baseline_gap"].astype(float).le(float(margin) + 1e-15)
        )
    output["base_intervene"] = output["intervene"].astype(bool)
    output["base_final_candidate_id"] = output["final_candidate_id"].astype(str)
    output["fallback_intervene"] = use.astype(bool)
    output["intervene"] = output["base_intervene"] | output["fallback_intervene"]
    output["final_candidate_id"] = np.where(
        output["fallback_intervene"],
        output["mass_proposal_candidate_id"],
        output["base_final_candidate_id"],
    )
    output["final_correct"] = output["final_candidate_id"].astype(str).eq(
        output["truth_candidate_id"].astype(str)
    )
    output["corrected"] = ~output["baseline_correct"].astype(bool) & output["final_correct"]
    output["introduced"] = output["baseline_correct"].astype(bool) & ~output["final_correct"]
    output["delta"] = output["final_correct"].astype(int) - output["baseline_correct"].astype(int)
    output["fallback_margin"] = "none" if margin is None else f"{margin:.2f}"
    return output


def choose_fallback(
    base: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[float | None, list[dict]]:
    ledger: list[dict] = []
    for margin in FALLBACKS:
        result = apply_fallback(base, candidates, margin)
        overall = summarize(result)
        by_domain = {
            source: summarize(result.loc[result["source"].eq(source)])
            for source in sorted(result["source"].unique())
        }
        ledger.append({
            "fallback_margin": "none" if margin is None else f"{margin:.2f}",
            **overall,
            "every_inner_domain_risk_nonnegative": all(
                item["risk_net_lambda2"] >= 0 for item in by_domain.values()
            ),
        })
    eligible = [
        item for item in ledger
        if item["every_inner_domain_risk_nonnegative"]
        and item["corrected"] > 2 * item["introduced"]
    ]
    pool = eligible if eligible else ledger
    selected = max(pool, key=lambda item: (
        item["risk_net_lambda2"] / max(1, item["queries"]),
        -item["introduced"] / max(1, item["queries"]),
        item["delta_recall1"],
        item["fallback_margin"] == "none",
    ))
    value = None if selected["fallback_margin"] == "none" else float(
        selected["fallback_margin"]
    )
    return value, ledger


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
        "--b17-transitions", type=Path,
        default=ROOT / "data/validation/bioaware_b17_nested_union_localcheck_20260907_v2/nested_domain_loso_transitions.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    for path in (
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds, args.b17_transitions,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    candidates, provenance = build_universe(args)
    candidates = prepare(candidates)
    outer_results: list[pd.DataFrame] = []
    base_outer_results: list[pd.DataFrame] = []
    folds: list[dict] = []
    for outer_index, outer_domain in enumerate(EXPECTED_DOMAINS):
        outer_train, outer_test = split_domain(candidates, outer_domain)
        inner_candidate_parts: list[pd.DataFrame] = []

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
                if recipe == next(iter(FEATURE_RECIPES)):
                    inner_candidate_parts.append(test)
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
        policy, _ = choose_policy(linear_inner_result, nonlinear_inner_result)
        base_outer = combine(linear_outer_result, nonlinear_outer_result, policy)
        base_inner = combine(linear_inner_result, nonlinear_inner_result, policy)
        inner_candidates = pd.concat(inner_candidate_parts, ignore_index=True)
        if inner_candidates.duplicated(["query_id", "candidate_id"]).any():
            raise RuntimeError(f"{outer_domain}: duplicate inner query/candidate rows")
        if set(inner_candidates["query_id"].astype(str)) != set(
            base_inner["query_id"].astype(str)
        ):
            raise RuntimeError(f"{outer_domain}: inner candidate/query coverage mismatch")
        selected_margin, fallback_ledger = choose_fallback(
            base_inner, inner_candidates
        )
        outer = apply_fallback(base_outer, outer_test, selected_margin)
        base_outer_results.append(base_outer)
        outer_results.append(outer)
        folds.append({
            "outer_domain": outer_domain,
            "B17_policy": policy,
            "fallback_selected": "none" if selected_margin is None else f"{selected_margin:.2f}",
            "inner_fallback_ledger": fallback_ledger,
            "outer_B17_replay": summarize(base_outer),
            "outer_result": summarize(outer),
        })
        print(
            f"[B21 {outer_domain}] fallback="
            f"{'none' if selected_margin is None else f'{selected_margin:.2f}'} "
            f"{summarize(outer)}", flush=True,
        )

    result = pd.concat(outer_results, ignore_index=True)
    base_replay = pd.concat(base_outer_results, ignore_index=True)
    b17 = pd.read_csv(args.b17_transitions)
    check = base_replay[[
        "query_id", "final_candidate_id", "corrected", "introduced"
    ]].merge(
        b17[["query_id", "final_candidate_id", "corrected", "introduced"]],
        on="query_id", suffixes=("_replay", "_frozen"), validate="one_to_one",
    )
    replay_mismatches = {
        column: int(check[f"{column}_replay"].astype(str).ne(
            check[f"{column}_frozen"].astype(str)
        ).sum())
        for column in ("final_candidate_id", "corrected", "introduced")
    }
    if any(replay_mismatches.values()):
        raise RuntimeError(f"B21 B17 replay mismatch: {replay_mismatches}")
    if len(result) != 860 or result["query_id"].nunique() != 860:
        raise RuntimeError("B21 OOF coverage changed")

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
        "risk_net_strictly_beats_b17": overall["risk_net_lambda2"] > B17_RISK_NET,
        "delta_not_below_b17": overall["delta_recall1"] >= B17_DELTA - 1e-15,
        "introduced_no_more_than_b17": overall["introduced"] <= B17_INTRODUCED,
        "frozen_B17_querywise_replay_exact": not any(replay_mismatches.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transition_path = args.output_dir / "nested_domain_loso_transitions.csv.gz"
    result.to_csv(transition_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b21_nested_residual_fallback_complete",
        "formal": True,
        "protocol": (
            "B17 reconstructed per outer domain; none/0.04/0.05/0.08 mass-"
            "coverage fallback selected only on inner domain-OOF rows"
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
        "frozen_B17_comparator": {
            "delta_recall1": B17_DELTA,
            "risk_net_lambda2": B17_RISK_NET,
            "introduced": B17_INTRODUCED,
            "querywise_replay_mismatches": replay_mismatches,
        },
        "folds": folds,
        "gates": gates,
        "strictly_better_action_than_B17": bool(all(gates.values())),
        "contracts": {
            "B19_opened_outcomes_used_for_outer_selection": False,
            "outer_outcomes_used_for_fallback_selection": False,
            "fallback_only_when_B17_abstains": True,
            "held_truth_identity_and_formula_purged": True,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            **provenance["provenance"],
            "B17_transitions_sha256": sha256(args.b17_transitions),
            "transitions_sha256": sha256(transition_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Opened nested residual-action test. Passing replaces B17 only as "
            "an action router; it is not blind validation, reaction mechanism, "
            "SOTA, or shared-embedding improvement."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["strictly_better_action_than_B17"]:
        raise RuntimeError(f"B21 did not strictly improve B17: {gates}")


if __name__ == "__main__":
    main()
