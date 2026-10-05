#!/usr/bin/env python
"""Cross-fit a bounded selector over residual BioAware action proposals.

B19 showed large but conflicting residual action headroom.  B22 freezes a small
mechanism dictionary and asks whether query/candidate evidence can select among
those proposed candidates without seeing the held domain.  For each outer
domain B17 is reconstructed exactly.  A shallow selector is trained on inner
domain-OOF B17 states; its intervention gate is chosen by an additional inner
leave-domain-out layer.  Held truth identities and formulae are purged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b11_catalog_interaction_action import (  # noqa: E402
    FEATURE_RECIPES, apply_gate as apply_linear_gate, atomic_json,
    cluster_bootstrap, score_queries as score_linear, sha256, summarize,
)
from audit_bioaware_b12_multicohort_catalog_action import (  # noqa: E402
    EXPECTED_DOMAINS, build_universe, choose as choose_linear, split_domain,
)
from audit_bioaware_b14_reaction_context_action import add_reaction_features  # noqa: E402
from audit_bioaware_b16_pairwise_nonlinear_action import (  # noqa: E402
    FEATURE_FAMILIES, apply_gate as apply_nonlinear_gate,
    choose as choose_nonlinear, prepare, score_queries as score_nonlinear,
)
from audit_bioaware_b17_nested_union_action import combine, choose_policy  # noqa: E402


ACTION_RULES = {
    "catalogue_mass_coverage": ("known_mass_candidate_fraction", "max"),
    "catalogue_membership": ("network_member", "max"),
    "catalogue_degree": ("known_log_degree", "max"),
    "reaction_path": ("known_path_fraction", "max"),
    "reaction_edge0_reliability": ("edge0_reliability", "max"),
    "reaction_edge1_bottleneck": ("edge1_bottleneck_mean", "max"),
    "predicted_edge_increment": ("predicted_edge_increment", "max"),
    "coabundance_neighbour_support": ("coabundance_log_neighbours_mean", "max"),
    "coabundance_multiwitness": ("coabundance_multiwitness_fraction", "max"),
}
CANDIDATE_FEATURES = (
    "spectral_score", "network_member", "known_log_degree",
    "known_mass_candidate_fraction", "log_reference_spectra",
    "known_path_fraction", "known_inverse_depth_mean",
    "known_log_seed_support_mean", "edge0_complete_fraction",
    "edge0_bottleneck_mean", "edge0_reliability",
    "edge1_complete_fraction", "edge1_bottleneck_mean",
    "predicted_edge_increment", "coabundance_log_neighbours_mean",
    "coabundance_multiwitness_fraction",
    "coabundance_actual_sign_stability_top3_mean",
)
GATE_GRID = tuple(
    (margin, advantage)
    for margin in (0.04, 0.05, 0.08)
    for advantage in (0.0, 0.025, 0.05, 0.10)
)
B17_RISK_NET = 43
B17_DELTA = 50 / 860
B17_INTRODUCED = 7


def reconstruct_b17_fold(
    candidates: pd.DataFrame,
    outer_domain: str,
    outer_index: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    outer_train, outer_test = split_domain(candidates, outer_domain)
    inner_candidate_parts: list[pd.DataFrame] = []
    linear_outer: dict[str, pd.DataFrame] = {}
    linear_inner: dict[str, pd.DataFrame] = {}
    for recipe_index, (recipe, features) in enumerate(FEATURE_RECIPES.items()):
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
            if recipe_index == 0:
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
            seed + 100 * outer_index,
        )
        parts = []
        for inner_index, inner_domain in enumerate(EXPECTED_DOMAINS):
            if inner_domain == outer_domain:
                continue
            train, test = split_domain(outer_train, inner_domain)
            scored, _ = score_nonlinear(
                train, test, features,
                f"outer={outer_domain}|inner={inner_domain}",
                seed + 100 * outer_index + inner_index + 1,
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
        raise RuntimeError(f"{outer_domain}: duplicate inner candidate rows")
    if set(inner_candidates["query_id"].astype(str)) != set(base_inner["query_id"].astype(str)):
        raise RuntimeError(f"{outer_domain}: inner query mismatch")
    return base_outer, outer_test, base_inner, inner_candidates, {
        "linear": linear_selected,
        "nonlinear": nonlinear_selected,
        "policy": policy,
    }


def build_alternatives(
    candidates: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.DataFrame:
    required = set(CANDIDATE_FEATURES) | {
        feature for feature, _ in ACTION_RULES.values()
    }
    if missing := required - set(candidates):
        raise RuntimeError(f"B22 candidates miss: {sorted(missing)}")
    rows: list[dict] = []
    base_by_query = base.set_index("query_id", verify_integrity=True)
    identity_query_counts = base["truth_candidate_id"].astype(str).value_counts()
    for query_id, group in candidates.groupby("query_id", sort=False):
        query_id = str(query_id)
        if query_id not in base_by_query.index:
            raise RuntimeError(f"B22 candidate query absent from B17 state: {query_id}")
        state = base_by_query.loc[query_id]
        base_candidate = str(state["final_candidate_id"])
        support: dict[str, set[str]] = {str(candidate): set() for candidate in group["candidate_id"]}
        support.setdefault(base_candidate, set()).add("B17")
        for action, (feature, direction) in ACTION_RULES.items():
            values = group[feature].to_numpy(float)
            optimum = float(np.max(values) if direction == "max" else np.min(values))
            top = group.loc[np.isclose(group[feature], optimum, rtol=0, atol=1e-12)]
            if len(top) == 1:
                support.setdefault(str(top["candidate_id"].iloc[0]), set()).add(action)
        proposed = {candidate for candidate, actions in support.items() if actions}
        local = group.loc[group["candidate_id"].astype(str).isin(proposed)].copy()
        if base_candidate not in set(local["candidate_id"].astype(str)):
            raise RuntimeError(f"B22 B17 base candidate absent: {query_id}")
        base_row = local.loc[local["candidate_id"].astype(str).eq(base_candidate)].iloc[0]
        candidate_count = int(len(group))
        context = {
            "query_baseline_gap": float(group["baseline_gap"].iloc[0]),
            "query_log_candidate_count": float(np.log(candidate_count)),
            "query_network_fraction": float(group["network_member"].mean()),
            "query_path_fraction": float((group["known_path_fraction"] > 0).mean()),
            "query_coabundance_fraction": float(
                (group["coabundance_log_neighbours_mean"] > 0).mean()
            ),
        }
        identity = str(state["truth_candidate_id"])
        query_weight = (2.0 if bool(state["baseline_correct"]) else 1.0) / float(
            identity_query_counts[identity]
        )
        for candidate in local.itertuples(index=False):
            candidate_id = str(candidate.candidate_id)
            record = {
                "query_id": query_id,
                "source": str(state["source"]),
                "truth_candidate_id": identity,
                "truth_formula": str(state["truth_formula"]),
                "candidate_id": candidate_id,
                "is_truth": candidate_id == identity,
                "is_B17_base": candidate_id == base_candidate,
                "baseline_correct": bool(state["baseline_correct"]),
                "baseline_candidate_id": str(state["baseline_candidate_id"]),
                "B17_final_candidate_id": base_candidate,
                "query_weight": query_weight / max(1, len(local)),
                **context,
            }
            for feature in CANDIDATE_FEATURES:
                value = float(getattr(candidate, feature))
                record[f"candidate__{feature}"] = value
                record[f"delta_from_B17__{feature}"] = value - float(base_row[feature])
            actions = support[candidate_id]
            for action in ("B17", *ACTION_RULES):
                record[f"support__{action}"] = float(action in actions)
            record["support_count"] = float(len(actions))
            rows.append(record)
    output = pd.DataFrame(rows)
    if output.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("B22 duplicate alternative candidates")
    if set(output.loc[output["is_B17_base"], "query_id"]) != set(base["query_id"]):
        raise RuntimeError("B22 does not retain every B17 base candidate")
    return output


def model_features(frame: pd.DataFrame) -> list[str]:
    prefixes = ("candidate__", "delta_from_B17__", "support__")
    explicit = {
        "is_B17_base", "query_baseline_gap", "query_log_candidate_count",
        "query_network_fraction", "query_path_fraction",
        "query_coabundance_fraction", "support_count",
    }
    features = sorted(
        column for column in frame
        if column.startswith(prefixes) or column in explicit
    )
    values = frame[features].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("B22 selector features contain non-finite values")
    return features


def fit_selector(
    train: pd.DataFrame,
    features: list[str],
    seed: int,
) -> HistGradientBoostingClassifier:
    if train["is_truth"].nunique() != 2:
        raise RuntimeError("B22 selector training lacks a class")
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=120,
        max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=2.0,
        early_stopping=False, random_state=seed,
    ).fit(
        train[features].to_numpy(float),
        train["is_truth"].to_numpy(bool),
        sample_weight=train["query_weight"].to_numpy(float),
    )


def score_selector(
    model: HistGradientBoostingClassifier,
    alternatives: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    local = alternatives.copy()
    local["selector_probability"] = model.predict_proba(
        local[features].to_numpy(float)
    )[:, 1]
    rows: list[dict] = []
    for query_id, group in local.groupby("query_id", sort=False):
        base = group.loc[group["is_B17_base"].astype(bool)]
        if len(base) != 1:
            raise RuntimeError(f"B22 base candidate is not unique: {query_id}")
        maximum = float(group["selector_probability"].max())
        top = group.loc[np.isclose(
            group["selector_probability"], maximum, rtol=0, atol=1e-12
        )].sort_values("candidate_id", kind="stable")
        rows.append({
            "query_id": str(query_id),
            "source": str(group["source"].iloc[0]),
            "truth_candidate_id": str(group["truth_candidate_id"].iloc[0]),
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_candidate_id": str(group["baseline_candidate_id"].iloc[0]),
            "baseline_correct": bool(group["baseline_correct"].iloc[0]),
            "B17_final_candidate_id": str(base["candidate_id"].iloc[0]),
            "proposed_candidate_id": str(top["candidate_id"].iloc[0]),
            "proposal_unique": len(top) == 1,
            "proposal_probability": maximum,
            "B17_probability": float(base["selector_probability"].iloc[0]),
            "proposal_advantage": maximum - float(base["selector_probability"].iloc[0]),
            "baseline_gap": float(group["query_baseline_gap"].iloc[0]),
        })
    return pd.DataFrame(rows)


def apply_gate(frame: pd.DataFrame, margin: float, advantage: float) -> pd.DataFrame:
    output = frame.copy()
    output["selector_intervene"] = (
        output["proposal_unique"].astype(bool)
        & output["proposed_candidate_id"].astype(str).ne(
            output["B17_final_candidate_id"].astype(str)
        )
        & output["baseline_gap"].astype(float).le(margin + 1e-15)
        & output["proposal_advantage"].astype(float).ge(advantage - 1e-15)
    )
    output["final_candidate_id"] = np.where(
        output["selector_intervene"], output["proposed_candidate_id"],
        output["B17_final_candidate_id"],
    )
    output["intervene"] = output["final_candidate_id"].astype(str).ne(
        output["baseline_candidate_id"].astype(str)
    )
    output["final_correct"] = output["final_candidate_id"].astype(str).eq(
        output["truth_candidate_id"].astype(str)
    )
    output["corrected"] = ~output["baseline_correct"].astype(bool) & output["final_correct"]
    output["introduced"] = output["baseline_correct"].astype(bool) & ~output["final_correct"]
    output["delta"] = output["final_correct"].astype(int) - output["baseline_correct"].astype(int)
    output["gate_margin"] = margin
    output["gate_advantage"] = advantage
    return output


def choose_gate(scored: pd.DataFrame) -> tuple[dict, list[dict]]:
    ledger: list[dict] = []
    for margin, advantage in GATE_GRID:
        result = apply_gate(scored, margin, advantage)
        overall = summarize(result)
        by_domain = {
            source: summarize(result.loc[result["source"].eq(source)])
            for source in sorted(result["source"].unique())
        }
        ledger.append({
            "margin": margin, "advantage": advantage, **overall,
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
    return max(pool, key=lambda item: (
        item["risk_net_lambda2"] / max(1, item["queries"]),
        -item["introduced"] / max(1, item["queries"]),
        item["delta_recall1"], -item["intervention_rate"],
    )), ledger


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
    candidates = prepare(add_reaction_features(candidates))
    for feature in set(CANDIDATE_FEATURES) | {f for f, _ in ACTION_RULES.values()}:
        candidates[feature] = pd.to_numeric(candidates[feature], errors="coerce").fillna(0.0)
    outer_results: list[pd.DataFrame] = []
    base_replays: list[pd.DataFrame] = []
    folds: list[dict] = []
    for outer_index, outer_domain in enumerate(EXPECTED_DOMAINS):
        base_outer, outer_candidates, base_inner, inner_candidates, components = (
            reconstruct_b17_fold(candidates, outer_domain, outer_index, args.seed)
        )
        inner_alternatives = build_alternatives(inner_candidates, base_inner)
        outer_alternatives = build_alternatives(outer_candidates, base_outer)
        features = model_features(inner_alternatives)
        inner_score_parts: list[pd.DataFrame] = []
        inner_fit_reports: list[dict] = []
        for inner_index, inner_domain in enumerate(EXPECTED_DOMAINS):
            if inner_domain == outer_domain:
                continue
            test = inner_alternatives.loc[inner_alternatives["source"].eq(inner_domain)].copy()
            held_ids = set(test["truth_candidate_id"].astype(str))
            held_formulas = set(test["truth_formula"].astype(str))
            train = inner_alternatives.loc[
                ~inner_alternatives["source"].eq(inner_domain)
                & ~inner_alternatives["truth_candidate_id"].astype(str).isin(held_ids)
                & ~inner_alternatives["truth_formula"].astype(str).isin(held_formulas)
            ].copy()
            model = fit_selector(train, features, args.seed + 1000 * outer_index + inner_index)
            inner_score_parts.append(score_selector(model, test, features))
            inner_fit_reports.append({
                "held_domain": inner_domain,
                "train_queries": int(train["query_id"].nunique()),
                "train_identities": int(train["truth_candidate_id"].nunique()),
                "test_queries": int(test["query_id"].nunique()),
            })
        inner_scores = pd.concat(inner_score_parts, ignore_index=True)
        selected, gate_ledger = choose_gate(inner_scores)
        final_model = fit_selector(
            inner_alternatives, features, args.seed + 1000 * outer_index + 99
        )
        outer_scores = score_selector(final_model, outer_alternatives, features)
        result = apply_gate(
            outer_scores, float(selected["margin"]), float(selected["advantage"])
        )
        outer_results.append(result)
        base_replays.append(base_outer)
        folds.append({
            "outer_domain": outer_domain,
            "B17_components": components,
            "selector_features": features,
            "selected_gate": selected,
            "inner_gate_ledger": gate_ledger,
            "inner_fit_reports": inner_fit_reports,
            "outer_B17_replay": summarize(base_outer),
            "outer_result": summarize(result),
        })
        print(f"[B22 {outer_domain}] gate={selected} outer={summarize(result)}", flush=True)

    result = pd.concat(outer_results, ignore_index=True)
    base_replay = pd.concat(base_replays, ignore_index=True)
    b17 = pd.read_csv(args.b17_transitions)
    check = base_replay[["query_id", "final_candidate_id", "corrected", "introduced"]].merge(
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
        raise RuntimeError(f"B22 B17 replay mismatch: {replay_mismatches}")
    if len(result) != 860 or result["query_id"].nunique() != 860:
        raise RuntimeError("B22 outer coverage changed")
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
        "status": "bioaware_b22_crossfit_action_selector_complete",
        "formal": True,
        "protocol": (
            "fixed residual action dictionary; B17 inner-OOF states; third-layer "
            "leave-domain-out selector gate; held identity/formula purge"
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
        "action_rules": ACTION_RULES,
        "candidate_features": list(CANDIDATE_FEATURES),
        "frozen_B17_comparator": {
            "delta_recall1": B17_DELTA, "risk_net_lambda2": B17_RISK_NET,
            "introduced": B17_INTRODUCED,
            "querywise_replay_mismatches": replay_mismatches,
        },
        "folds": folds,
        "gates": gates,
        "strictly_better_action_than_B17": bool(all(gates.values())),
        "contracts": {
            "B19_truth_rank_used_as_model_feature": False,
            "held_outer_outcome_used_for_selection": False,
            "selector_gate_crossfit_inside_outer_train": True,
            "held_truth_identity_and_formula_purged": True,
            "candidate_or_source_identity_as_feature": False,
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
            "Opened nested action-selector test. Passing replaces B17 only as "
            "an action router; it is not blind validation, reaction mechanism, "
            "SOTA, or shared-embedding improvement."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["strictly_better_action_than_B17"]:
        raise RuntimeError(f"B22 did not strictly improve B17: {gates}")


if __name__ == "__main__":
    main()
