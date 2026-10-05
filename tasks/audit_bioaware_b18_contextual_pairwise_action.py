#!/usr/bin/env python
"""Mine a query-contextual BioAware pairwise action across six opened domains.

B16 learned a candidate-versus-baseline preference from candidate differences,
but its gate received only the DreaMS Top-1 margin.  B18 tests the narrower,
pre-registered hypothesis that catalogue and reaction evidence are reliable in
different *candidate-set contexts*.  The pair model therefore receives:

* signed candidate differences (which candidate is favoured);
* absolute differences and pair midpoints (how exceptional the comparison is);
* label-free query-set summaries (ambiguity and network-evidence density).

Every model/gate choice is made from inner leave-domain-out predictions.  The
outer domain and all matching truth identities/formulae are purged before fit.
This is opened action discovery, not blind validation or embedding training.
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
from audit_bioaware_b14_reaction_context_action import (  # noqa: E402
    add_reaction_features,
)


CATALOGUE_FEATURES = (
    "spectral_score",
    "network_member",
    "known_log_degree",
    "known_mass_candidate_fraction",
    "log_reference_spectra",
)
REACTION_FEATURES = (
    *CATALOGUE_FEATURES,
    "known_path_present",
    "edge0_present",
    "known_path_per_degree",
    "known_inverse_depth_mean",
    "known_seed_per_degree",
    "edge0_complete_fraction",
    "edge0_bottleneck_mean",
    "edge0_reliability",
)
FEATURE_FAMILIES = {
    "catalogue_context_hgb": CATALOGUE_FEATURES,
    "reaction_context_hgb": REACTION_FEATURES,
}
QUERY_CONTEXT = (
    "ctx_baseline_gap",
    "ctx_log_candidate_count",
    "ctx_spectral_spread",
    "ctx_spectral_std",
    "ctx_network_member_fraction",
    "ctx_path_present_fraction",
    "ctx_edge0_present_fraction",
    "ctx_degree_mean",
    "ctx_degree_max",
)
GATE_GRID = tuple(
    (margin, probability)
    for margin in (0.04, 0.05, 0.08)
    for probability in (0.55, 0.60, 0.65, 0.70, 0.75)
)
RISK_PENALTY = 2
B17_RISK_NET = 43
B17_DELTA = 50 / 860
B17_INTRODUCED = 7


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    output = add_reaction_features(frame)
    required = {
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "source", "polarity", "baseline_candidate_id", "baseline_correct",
        "baseline_gap", "is_positive", *REACTION_FEATURES,
    }
    if missing := required - set(output):
        raise RuntimeError(f"B18 candidate table misses: {sorted(missing)}")
    for column in REACTION_FEATURES:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    if output.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("B18 has duplicate candidate identities within a query")

    grouped = output.groupby("query_id", sort=False)
    count = grouped["candidate_id"].transform("count").astype(float)
    spectral_max = grouped["spectral_score"].transform("max")
    spectral_min = grouped["spectral_score"].transform("min")
    output["ctx_baseline_gap"] = pd.to_numeric(
        output["baseline_gap"], errors="raise"
    ).astype(float)
    output["ctx_log_candidate_count"] = np.log(count)
    output["ctx_spectral_spread"] = spectral_max - spectral_min
    output["ctx_spectral_std"] = grouped["spectral_score"].transform(
        lambda values: float(np.std(values.to_numpy(float), ddof=0))
    )
    output["ctx_network_member_fraction"] = grouped["network_member"].transform("mean")
    output["ctx_path_present_fraction"] = grouped["known_path_present"].transform("mean")
    output["ctx_edge0_present_fraction"] = grouped["edge0_present"].transform("mean")
    output["ctx_degree_mean"] = grouped["known_log_degree"].transform("mean")
    output["ctx_degree_max"] = grouped["known_log_degree"].transform("max")
    numerical = list(REACTION_FEATURES) + list(QUERY_CONTEXT)
    if not np.isfinite(output[numerical].to_numpy(float)).all():
        raise RuntimeError("B18 feature matrix contains non-finite values")
    positive_count = output.groupby("query_id", sort=False)["is_positive"].sum()
    if not positive_count.eq(1).all():
        raise RuntimeError("B18 requires exactly one positive candidate per query")
    return output


def pair_vector(
    left: np.ndarray,
    right: np.ndarray,
    context: np.ndarray,
) -> np.ndarray:
    difference = left - right
    return np.concatenate((difference, np.abs(difference), 0.5 * (left + right), context))


def pairwise_training(
    frame: pd.DataFrame,
    features: tuple[str, ...],
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
            raise RuntimeError("B18 needs one truth and at least one wrong candidate")
        pos = positive[list(features)].to_numpy(float)[0]
        context = positive[list(QUERY_CONTEXT)].to_numpy(float)[0]
        identity = str(positive["truth_candidate_id"].iloc[0])
        safety = 2.0 if bool(positive["baseline_correct"].iloc[0]) else 1.0
        pair_weight = safety / (float(identity_counts[identity]) * len(negative))
        for neg in negative[list(features)].to_numpy(float):
            x.append(pair_vector(pos, neg, context))
            x.append(pair_vector(neg, pos, context))
            y.extend((1, 0))
            weights.extend((pair_weight, pair_weight))
    return np.stack(x), np.asarray(y, dtype=np.int8), np.asarray(weights, dtype=float)


def fit_model(
    train: pd.DataFrame,
    features: tuple[str, ...],
    seed: int,
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
        "symmetric_training_rows": int(len(y)),
        "input_dimension": int(x.shape[1]),
        "training_queries": int(train["query_id"].nunique()),
        "training_identities": int(train["truth_candidate_id"].nunique()),
        "training_formulas": int(train["truth_formula"].nunique()),
    }


def score_queries(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
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
        base = baseline_row[list(features)].to_numpy(float)[0]
        context = baseline_row[list(QUERY_CONTEXT)].to_numpy(float)[0]
        vectors = np.stack([
            pair_vector(candidate, base, context)
            for candidate in group[list(features)].to_numpy(float)
        ])
        probabilities = model.predict_proba(vectors)[:, 1]
        local = group.copy()
        local["preference_probability"] = probabilities
        maximum = float(local["preference_probability"].max())
        top = local.loc[np.isclose(
            local["preference_probability"], maximum, rtol=0, atol=1e-12
        )].sort_values("candidate_id", kind="stable")
        rows.append({
            "query_id": str(query_id),
            "held_label": held_label,
            "source": str(group["source"].iloc[0]),
            "truth_candidate_id": str(group["truth_candidate_id"].iloc[0]),
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_candidate_id": baseline,
            "proposed_candidate_id": str(top["candidate_id"].iloc[0]),
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
    output["gate_margin"] = float(margin)
    output["gate_probability"] = float(probability)
    return output


def choose(inner: dict[str, pd.DataFrame]) -> tuple[dict, list[dict]]:
    ledger: list[dict] = []
    for family, scored in inner.items():
        for margin, probability in GATE_GRID:
            result = apply_gate(scored, margin, probability)
            overall = summarize(result)
            per_domain = {
                source: summarize(result.loc[result["source"].eq(source)])
                for source in sorted(result["source"].unique())
            }
            ledger.append({
                "feature_family": family,
                "margin": margin,
                "probability": probability,
                **overall,
                "every_inner_domain_risk_nonnegative": all(
                    item["risk_net_lambda2"] >= 0 for item in per_domain.values()
                ),
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
    outer_tables: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    for outer_index, outer_domain in enumerate(EXPECTED_DOMAINS):
        outer_train, outer_test = split_domain(candidates, outer_domain)
        outer_scores: dict[str, pd.DataFrame] = {}
        inner_scores: dict[str, pd.DataFrame] = {}
        fit_reports: dict[str, dict] = {}
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
                train, test = split_domain(outer_train, inner_domain)
                scored, fit = score_queries(
                    train, test, features,
                    f"outer={outer_domain}|inner={inner_domain}",
                    args.seed + 100 * outer_index + inner_index + 1,
                )
                inner_parts.append(scored)
                inner_fits.append(fit)
            inner_scores[family] = pd.concat(inner_parts, ignore_index=True)
            fit_reports[family] = {"outer": outer_fit, "inner": inner_fits}
        selected, ledger = choose(inner_scores)
        result = apply_gate(
            outer_scores[selected["feature_family"]],
            float(selected["margin"]), float(selected["probability"]),
        )
        result["selected_feature_family"] = selected["feature_family"]
        outer_tables.append(result)
        fold_reports.append({
            "outer_domain": outer_domain,
            "selected": selected,
            "outer_result": summarize(result),
            "inner_selection_ledger": ledger,
            "fit_reports": fit_reports,
        })
        print(
            f"[B18 {outer_domain}] {selected['feature_family']} "
            f"m={selected['margin']:.2f} p={selected['probability']:.2f} "
            f"{summarize(result)}", flush=True,
        )

    result = pd.concat(outer_tables, ignore_index=True)
    if len(result) != 860 or result["query_id"].nunique() != 860:
        raise RuntimeError("B18 outer-domain OOF coverage changed")
    b17 = pd.read_csv(args.b17_transitions)
    replay = result[["query_id", "baseline_correct"]].merge(
        b17[["query_id", "baseline_correct"]], on="query_id",
        suffixes=("_b18", "_b17"), validate="one_to_one",
    )
    if not replay["baseline_correct_b18"].astype(bool).equals(
        replay["baseline_correct_b17"].astype(bool)
    ):
        raise RuntimeError("B18/B17 baseline mismatch")

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
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transition_path = args.output_dir / "nested_domain_loso_transitions.csv.gz"
    result.to_csv(transition_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b18_contextual_pairwise_action_complete",
        "formal": True,
        "protocol": (
            "six opened domains; contextual symmetric pairwise HGB; nested "
            "leave-domain-out feature-family and gate selection; held truth "
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
        "frozen_B17_comparator": {
            "delta_recall1": B17_DELTA,
            "risk_net_lambda2": B17_RISK_NET,
            "introduced": B17_INTRODUCED,
        },
        "feature_families": {
            key: list(value) for key, value in FEATURE_FAMILIES.items()
        },
        "query_context": list(QUERY_CONTEXT),
        "model": {
            "class": "HistGradientBoostingClassifier",
            "learning_rate": 0.05,
            "max_iter": 120,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 20,
            "l2_regularization": 2.0,
            "training_representation": (
                "signed difference + absolute difference + midpoint + query context"
            ),
        },
        "folds": fold_reports,
        "gates": gates,
        "strictly_better_action_than_B17": bool(all(gates.values())),
        "contracts": {
            "held_domain_outcomes_used_for_selection": False,
            "held_truth_identity_and_formula_purged": True,
            "source_or_candidate_identity_as_feature": False,
            "B12_B16_or_B17_decision_as_feature": False,
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
            "Opened query-context action discovery. Passing would replace B17 "
            "only as an action router; it is not blind validation, reaction "
            "mechanism, SOTA, or shared-embedding improvement."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["strictly_better_action_than_B17"]:
        raise RuntimeError(f"B18 did not strictly improve B17: {gates}")


if __name__ == "__main__":
    main()
