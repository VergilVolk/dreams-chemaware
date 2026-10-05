#!/usr/bin/env python
"""Test a bounded nonlinear BioAware switch action across six opened domains.

The model learns only whether candidate A should beat the current DreaMS
baseline candidate B.  Training examples are symmetric candidate-difference
pairs and receive truth-identity-equal mass.  A shallow histogram gradient
booster can express the non-monotone "use catalogue support, but distrust
extreme hubs" relation that B13 could not encode with a fixed veto.

Feature family and intervention gate are selected by inner leave-domain-out
predictions.  The outer domain and every matching truth identity/formula are
removed before fitting.  B12 is a frozen comparison, never a feature or label.
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
    atomic_json,
    cluster_bootstrap,
    sha256,
    summarize,
)
from audit_bioaware_b12_multicohort_catalog_action import (  # noqa: E402
    EXPECTED_DOMAINS,
    build_universe,
    split_domain,
)


FEATURE_FAMILIES = {
    "catalog_pairwise_hgb": [
        "spectral_score", "network_member", "known_log_degree",
        "known_mass_candidate_fraction", "log_reference_spectra",
    ],
    "reaction_context_pairwise_hgb": [
        "spectral_score", "network_member", "known_log_degree",
        "known_mass_candidate_fraction", "log_reference_spectra",
        "known_path_fraction", "known_inverse_depth_mean",
        "known_log_seed_support_mean", "edge0_complete_fraction",
        "edge0_bottleneck_mean",
    ],
}
GATE_GRID = tuple(
    (margin, probability)
    for margin in (0.04, 0.05, 0.08)
    for probability in (0.55, 0.60, 0.65, 0.70, 0.75)
)
RISK_PENALTY = 2
B12_RISK_NET = 42
B12_DELTA = 0.05813953488372092


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    required = set().union(*FEATURE_FAMILIES.values()) | {
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "source", "polarity", "baseline_candidate_id", "baseline_correct",
        "baseline_gap", "is_positive",
    }
    if missing := required - set(output):
        raise RuntimeError(f"B16 candidate table misses: {sorted(missing)}")
    for column in set().union(*FEATURE_FAMILIES.values()):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    if not np.isfinite(output[list(set().union(*FEATURE_FAMILIES.values()))].to_numpy(float)).all():
        raise RuntimeError("B16 feature matrix contains non-finite values")
    return output


def pairwise_training(
    frame: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_meta = frame[["query_id", "truth_candidate_id"]].drop_duplicates()
    identity_counts = query_meta["truth_candidate_id"].astype(str).value_counts()
    x: list[np.ndarray] = []
    y: list[int] = []
    weights: list[float] = []
    for _, group in frame.groupby("query_id", sort=False):
        positive = group.loc[group["is_positive"].astype(bool)]
        negative = group.loc[~group["is_positive"].astype(bool)]
        if len(positive) != 1 or negative.empty:
            raise RuntimeError("B16 requires one truth and at least one wrong candidate")
        difference = (
            positive[features].to_numpy(float)[0]
            - negative[features].to_numpy(float)
        )
        identity = str(group["truth_candidate_id"].iloc[0])
        safety = 2.0 if bool(group["baseline_correct"].iloc[0]) else 1.0
        pair_weight = safety / (float(identity_counts[identity]) * len(negative))
        x.extend((difference, -difference))
        y.extend(([1] * len(difference), [0] * len(difference)))
        weights.extend([pair_weight] * (2 * len(difference)))
    return np.vstack(x), np.concatenate(y), np.asarray(weights, dtype=float)


def fit_model(
    train: pd.DataFrame, features: list[str], seed: int
) -> tuple[HistGradientBoostingClassifier, dict]:
    x, y, weights = pairwise_training(train, features)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    ).fit(x, y, sample_weight=weights)
    return model, {
        "training_pairs_with_symmetric_labels": int(len(y)),
        "training_queries": int(train["query_id"].nunique()),
        "training_identities": int(train["truth_candidate_id"].nunique()),
        "training_formulas": int(train["truth_formula"].nunique()),
    }


def score_queries(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    held_label: str,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    model, fit_report = fit_model(train, features, seed)
    rows: list[dict] = []
    for query_id, group in test.groupby("query_id", sort=False):
        baseline = str(group["baseline_candidate_id"].iloc[0])
        baseline_row = group.loc[group["candidate_id"].astype(str).eq(baseline)]
        if len(baseline_row) != 1:
            raise RuntimeError(f"{query_id}: baseline candidate is not unique")
        baseline_vector = baseline_row[features].to_numpy(float)[0]
        candidate_vectors = group[features].to_numpy(float)
        probabilities = model.predict_proba(candidate_vectors - baseline_vector)[:, 1]
        local = group.copy()
        local["preference_probability"] = probabilities
        maximum = float(local["preference_probability"].max())
        top = local.loc[
            np.isclose(local["preference_probability"], maximum, rtol=0, atol=1e-12)
        ].sort_values("candidate_id", kind="stable")
        proposed = str(top.iloc[0]["candidate_id"])
        rows.append({
            "query_id": str(query_id),
            "held_label": held_label,
            "source": str(group["source"].iloc[0]),
            "truth_candidate_id": str(group["truth_candidate_id"].iloc[0]),
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_candidate_id": baseline,
            "proposed_candidate_id": proposed,
            "baseline_correct": bool(group["baseline_correct"].iloc[0]),
            "proposal_unique": bool(len(top) == 1),
            "proposal_probability": maximum,
            "baseline_gap": float(group["baseline_gap"].iloc[0]),
            "candidate_count": int(len(group)),
        })
    return pd.DataFrame(rows), {"held_label": held_label, **fit_report}


def apply_gate(frame: pd.DataFrame, margin: float, probability: float) -> pd.DataFrame:
    output = frame.copy()
    output["intervene"] = (
        output["proposal_unique"].astype(bool)
        & output["proposed_candidate_id"].astype(str).ne(
            output["baseline_candidate_id"].astype(str)
        )
        & output["baseline_gap"].astype(float).le(margin + 1e-15)
        & output["proposal_probability"].astype(float).ge(probability - 1e-15)
    )
    output["final_candidate_id"] = np.where(
        output["intervene"], output["proposed_candidate_id"], output["baseline_candidate_id"]
    )
    output["final_correct"] = output["final_candidate_id"].astype(str).eq(
        output["truth_candidate_id"].astype(str)
    )
    output["corrected"] = ~output["baseline_correct"].astype(bool) & output["final_correct"]
    output["introduced"] = output["baseline_correct"].astype(bool) & ~output["final_correct"]
    output["delta"] = output["final_correct"].astype(int) - output["baseline_correct"].astype(int)
    output["gate_margin"] = margin
    output["gate_probability"] = probability
    return output


def choose(inner: dict[str, pd.DataFrame]) -> tuple[dict, list[dict]]:
    ledger: list[dict] = []
    for family, scored in inner.items():
        for margin, probability in GATE_GRID:
            result = apply_gate(scored, margin, probability)
            overall = summarize(result)
            by_domain = {
                source: summarize(result.loc[result["source"].eq(source)])
                for source in sorted(result["source"].unique())
            }
            safe = all(item["risk_net_lambda2"] >= 0 for item in by_domain.values())
            ledger.append({
                "feature_family": family,
                "margin": margin,
                "probability": probability,
                **overall,
                "every_inner_domain_risk_nonnegative": safe,
            })
    eligible = [
        item for item in ledger
        if item["every_inner_domain_risk_nonnegative"]
        and item["corrected"] > RISK_PENALTY * item["introduced"]
    ]
    pool = eligible if eligible else ledger
    selected = max(pool, key=lambda item: (
        item["risk_net_lambda2"] / max(1, item["queries"]),
        -item["introduced"] / max(1, item["queries"]),
        item["delta_recall1"],
        -item["intervention_rate"],
    ))
    return {
        **selected,
        "selection_reason": (
            "maximum_safe_inner_oof_risk_net_rate" if eligible
            else "no_safe_configuration__maximum_inner_risk_net_rate"
        ),
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
    outer_tables: list[pd.DataFrame] = []
    folds: list[dict] = []
    for outer_index, outer_domain in enumerate(EXPECTED_DOMAINS):
        outer_train, outer_test = split_domain(candidates, outer_domain)
        outer_scores: dict[str, pd.DataFrame] = {}
        inner_scores: dict[str, pd.DataFrame] = {}
        fits: dict[str, dict] = {}
        for family, features in FEATURE_FAMILIES.items():
            outer_scores[family], outer_fit = score_queries(
                outer_train, outer_test, features, outer_domain,
                args.seed + 100 * outer_index,
            )
            inner_parts: list[pd.DataFrame] = []
            inner_fits: list[dict] = []
            for inner_index, inner_domain in enumerate(EXPECTED_DOMAINS):
                if inner_domain == outer_domain:
                    continue
                inner_train, inner_test = split_domain(outer_train, inner_domain)
                if inner_test["query_id"].nunique() < 10:
                    raise RuntimeError(f"{outer_domain}/{inner_domain}: too few inner queries")
                table, fit = score_queries(
                    inner_train, inner_test, features,
                    f"outer={outer_domain}|inner={inner_domain}",
                    args.seed + 100 * outer_index + inner_index + 1,
                )
                inner_parts.append(table)
                inner_fits.append(fit)
            inner_scores[family] = pd.concat(inner_parts, ignore_index=True)
            fits[family] = {"outer": outer_fit, "inner": inner_fits}
        selected, ledger = choose(inner_scores)
        result = apply_gate(
            outer_scores[selected["feature_family"]],
            float(selected["margin"]), float(selected["probability"]),
        )
        result["selected_feature_family"] = selected["feature_family"]
        outer_tables.append(result)
        folds.append({
            "outer_domain": outer_domain,
            "selected": selected,
            "outer_result": summarize(result),
            "inner_selection_ledger": ledger,
            "fit_reports": fits,
        })
        print(
            f"[B16 {outer_domain}] {selected['feature_family']} "
            f"m={selected['margin']:.2f} p={selected['probability']:.2f} "
            f"{summarize(result)}",
            flush=True,
        )

    result = pd.concat(outer_tables, ignore_index=True)
    if len(result) != 860 or result["query_id"].nunique() != 860:
        raise RuntimeError("B16 outer-domain OOF coverage changed")
    b12 = pd.read_csv(args.b12_transitions)
    comparison = result[["query_id", "baseline_correct"]].merge(
        b12[["query_id", "baseline_correct"]], on="query_id",
        suffixes=("_b16", "_b12"), validate="one_to_one",
    )
    if not comparison["baseline_correct_b16"].astype(bool).equals(
        comparison["baseline_correct_b12"].astype(bool)
    ):
        raise RuntimeError("B16/B12 baseline mismatch")

    overall = summarize(result)
    corrected = result.loc[result["corrected"]]
    introduced = result.loc[result["introduced"]]
    by_domain = {
        source: summarize(result.loc[result["source"].eq(source)])
        for source in EXPECTED_DOMAINS
    }
    identity_ci = cluster_bootstrap(
        result, "truth_candidate_id", args.bootstrap_resamples, args.seed + 1
    )
    formula_ci = cluster_bootstrap(
        result, "truth_formula", args.bootstrap_resamples, args.seed + 2
    )
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
        "risk_net_strictly_beats_b12": overall["risk_net_lambda2"] > B12_RISK_NET,
        "delta_not_below_b12": overall["delta_recall1"] >= B12_DELTA - 1e-15,
        "introduced_no_more_than_b12": overall["introduced"] <= 8,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transition_path = args.output_dir / "nested_domain_loso_transitions.csv.gz"
    result.to_csv(transition_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b16_pairwise_nonlinear_action_complete",
        "formal": True,
        "protocol": (
            "six opened development domains; symmetric pairwise shallow-HGB; "
            "nested leave-domain-out feature-family/gate selection; held truth "
            "identity and formula purged"
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
        "frozen_B12_comparator": {
            "delta_recall1": B12_DELTA,
            "risk_net_lambda2": B12_RISK_NET,
            "introduced": 8,
        },
        "feature_families": FEATURE_FAMILIES,
        "model": {
            "class": "HistGradientBoostingClassifier",
            "learning_rate": 0.05,
            "max_iter": 120,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 20,
            "l2_regularization": 2.0,
            "early_stopping": False,
            "training_representation": "symmetric candidate-minus-baseline differences",
        },
        "folds": folds,
        "gates": gates,
        "strictly_better_action_than_B12": bool(all(gates.values())),
        "contracts": {
            "held_domain_outcomes_used_for_selection": False,
            "held_truth_identity_and_formula_purged": True,
            "truth_candidate_identity_as_feature": False,
            "candidate_identity_as_feature": False,
            "B12_score_or_decision_as_feature": False,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            **provenance["provenance"],
            "B12_transitions_sha256": sha256(args.b12_transitions),
            "transitions_sha256": sha256(transition_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Opened nonlinear action discovery. Passing would justify replacing "
            "B12 for action construction only; it would not establish blind "
            "generalization, reaction mechanism, SOTA, or shared-embedding gain."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["strictly_better_action_than_B12"]:
        raise RuntimeError(f"B16 did not strictly improve B12: {gates}")


if __name__ == "__main__":
    main()
