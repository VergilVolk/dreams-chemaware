#!/usr/bin/env python
"""Nested risk router for the frozen B17 BioAware intervention.

B29 does not search for another candidate and cannot create a new switch.  It
asks whether each of the 109 already-open B17 switches should be executed or
reverted to the official DreaMS Top-1.  Features are deployment-visible
candidate/context differences plus exact candidate-own spectrum evidence.

Every outer source and every matching truth identity/formula is untouched
during model and threshold selection.  Hyperparameters are selected from
formula-group OOF predictions inside that already-purged development pool.
Physical duplicate spectra receive one unit of mass.  Introduced errors have
twice the fitting and selection cost of missed corrections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b11_catalog_interaction_action import (  # noqa: E402
    atomic_json, cluster_bootstrap, sha256, summarize,
)
from audit_bioaware_b12_multicohort_catalog_action import (  # noqa: E402
    EXPECTED_DOMAINS, build_universe,
)
from audit_bioaware_b14_reaction_context_action import add_reaction_features  # noqa: E402
from audit_bioaware_b16_pairwise_nonlinear_action import prepare  # noqa: E402
from audit_bioaware_b22_crossfit_action_selector import CANDIDATE_FEATURES  # noqa: E402
from audit_bioaware_b27_candidate_spectral_veto import (  # noqa: E402
    add_candidate_spectral_evidence, load_b20,
)


B17_ROW_CORRECTED = 57
B17_ROW_INTRODUCED = 7
B17_ROW_RISK = 43
B17_ROW_DELTA = 50 / 860
B17_PHYSICAL_CORRECTED = 53
B17_PHYSICAL_INTRODUCED = 7
B17_PHYSICAL_RISK = 39
RISK_PENALTY = 2.0
MODEL_GRID = (
    ("core", 0.01), ("core", 0.10), ("core", 1.00),
    ("context", 0.01), ("context", 0.10), ("context", 1.00),
    ("spectral", 0.01), ("spectral", 0.10), ("spectral", 1.00),
    ("full", 0.01), ("full", 0.10), ("full", 1.00),
)
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)

CORE_FEATURES = (
    "baseline_gap", "log_candidate_count",
    "b12_intervene", "b16_intervene", "branches_agree",
    "delta__spectral_score", "delta__network_member",
    "delta__known_log_degree", "delta__known_mass_candidate_fraction",
    "delta__log_reference_spectra",
)
CONTEXT_FEATURES = CORE_FEATURES + (
    "delta__known_path_fraction", "delta__known_inverse_depth_mean",
    "delta__known_log_seed_support_mean", "delta__edge0_complete_fraction",
    "delta__edge0_bottleneck_mean", "delta__edge0_reliability",
    "delta__edge1_complete_fraction", "delta__edge1_bottleneck_mean",
    "delta__predicted_edge_increment",
    "delta__coabundance_log_neighbours_mean",
    "delta__coabundance_multiwitness_fraction",
    "delta__coabundance_actual_sign_stability_top3_mean",
)
SPECTRAL_FEATURES = CORE_FEATURES + (
    "b27_truncated_direct_advantage", "b27_neutral_loss_advantage",
    "b27_modified_cosine_advantage", "b27_dual_view_advantage",
    "b27_support_votes", "b27_mean_advantage",
)
FEATURE_FAMILIES = {
    "core": CORE_FEATURES,
    "context": CONTEXT_FEATURES,
    "spectral": SPECTRAL_FEATURES,
    "full": tuple(dict.fromkeys((*CONTEXT_FEATURES, *SPECTRAL_FEATURES))),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b20-dir", type=Path, required=True)
    parser.add_argument("--internal-candidates", type=Path, default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz")
    parser.add_argument("--st-candidates", type=Path, default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz")
    parser.add_argument("--st-queries", type=Path, default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz")
    parser.add_argument("--kgmn-candidates", type=Path, default=ROOT / "data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz")
    parser.add_argument("--kgmn-seeds", type=Path, default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def physical_key(frame: pd.DataFrame) -> pd.Series:
    return frame["source"].astype(str) + "::" + frame["physical_query_id"].astype(str)


def collapse_physical(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per physical spectrum, after proving duplicate consistency."""
    local = frame.copy()
    local["physical_key"] = physical_key(local)
    immutable = (
        "source", "physical_query_id", "truth_candidate_id", "truth_formula",
        "baseline_candidate_id", "final_candidate_id", "baseline_correct",
        "final_correct", "corrected", "introduced",
    )
    for column in immutable:
        maximum = local.groupby("physical_key")[column].nunique(dropna=False).max()
        if int(maximum) != 1:
            raise RuntimeError(f"B29 physical duplicates disagree on {column}")
    feature_columns = sorted(set().union(*FEATURE_FAMILIES.values()))
    spans = local.groupby("physical_key")[feature_columns].agg(lambda x: float(np.max(x) - np.min(x)))
    if float(spans.to_numpy(float).max(initial=0.0)) > 1e-8:
        raise RuntimeError("B29 physical duplicate features disagree")
    return local.drop_duplicates("physical_key", keep="first").reset_index(drop=True)


def materialise_features(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frame, body, b20_report = load_b20(args.b20_dir)
    frame = add_candidate_spectral_evidence(frame, body)
    candidates, provenance = build_universe(args)
    candidates = prepare(add_reaction_features(candidates))
    for feature in CANDIDATE_FEATURES:
        candidates[feature] = pd.to_numeric(candidates[feature], errors="coerce").fillna(0.0)
    if candidates.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("B29 candidate universe has duplicate query/candidate rows")
    lookup = candidates[["query_id", "candidate_id", "baseline_gap", *CANDIDATE_FEATURES]].copy()
    counts = candidates.groupby("query_id")["candidate_id"].nunique()
    frame["log_candidate_count"] = np.log(
        frame["query_id"].map(counts).fillna(0).to_numpy(float)
    )
    if not np.isfinite(frame["log_candidate_count"]).all():
        raise RuntimeError("B29 missing candidate count")

    baseline = lookup.rename(columns={
        "candidate_id": "baseline_candidate_id",
        "baseline_gap": "candidate_table_baseline_gap",
        **{feature: f"baseline__{feature}" for feature in CANDIDATE_FEATURES},
    })
    final = lookup.rename(columns={
        "candidate_id": "final_candidate_id",
        "baseline_gap": "candidate_table_final_gap",
        **{feature: f"final__{feature}" for feature in CANDIDATE_FEATURES},
    })
    frame = frame.merge(
        baseline, on=["query_id", "baseline_candidate_id"], how="left",
        validate="many_to_one",
    ).merge(
        final, on=["query_id", "final_candidate_id"], how="left",
        validate="many_to_one",
    )
    if frame["candidate_table_baseline_gap"].isna().any() or frame["candidate_table_final_gap"].isna().any():
        raise RuntimeError("B29 failed to recover baseline/final candidate rows")
    frame["baseline_gap"] = frame["candidate_table_baseline_gap"].astype(float)
    for feature in CANDIDATE_FEATURES:
        frame[f"delta__{feature}"] = (
            frame[f"final__{feature}"].astype(float)
            - frame[f"baseline__{feature}"].astype(float)
        )
    frame["b12_intervene"] = frame["b12_intervene"].astype(float)
    frame["b16_intervene"] = frame["b16_intervene"].astype(float)
    frame["branches_agree"] = frame["b12_final_candidate_id"].astype(str).eq(
        frame["b16_final_candidate_id"].astype(str)
    ).astype(float)
    features = sorted(set().union(*FEATURE_FAMILIES.values()))
    if not np.isfinite(frame[features].to_numpy(float)).all():
        raise RuntimeError("B29 non-finite risk-router features")
    if int(frame["intervene"].sum()) != 109:
        raise RuntimeError("B29 B17 intervention count changed")
    if int(frame["corrected"].sum()) != B17_ROW_CORRECTED or int(frame["introduced"].sum()) != B17_ROW_INTRODUCED:
        raise RuntimeError("B29 B17 outcome count changed")
    physical = collapse_physical(frame)
    return frame, physical, {
        "B20_report_sha256": sha256(args.b20_dir / "report.json"),
        "B20_actions_sha256": sha256(args.b20_dir / "direct_actions.csv.gz"),
        "B20_manifest_sha256": sha256(args.b20_dir / "direct_action_manifest.npz"),
        **provenance["provenance"],
    }


def purge_train(train: pd.DataFrame, held: pd.DataFrame) -> pd.DataFrame:
    identities = set(held["truth_candidate_id"].astype(str))
    formulas = set(held["truth_formula"].astype(str))
    output = train.loc[
        ~train["truth_candidate_id"].astype(str).isin(identities)
        & ~train["truth_formula"].astype(str).isin(formulas)
    ].copy()
    if set(output["truth_candidate_id"].astype(str)) & identities:
        raise RuntimeError("B29 held identity leaked into fit")
    if set(output["truth_formula"].astype(str)) & formulas:
        raise RuntimeError("B29 held formula leaked into fit")
    return output


def fit_model(
    train: pd.DataFrame, features: tuple[str, ...], c_value: float, seed: int,
) -> tuple[StandardScaler, LogisticRegression, dict]:
    action = train.loc[train["intervene"].astype(bool)].copy()
    y = action["corrected"].astype(int).to_numpy()
    if len(action) < 12 or len(np.unique(y)) != 2:
        raise RuntimeError("B29 fitting pool lacks action coverage/classes")
    identity_count = action["truth_candidate_id"].astype(str).value_counts()
    weights = 1.0 / action["truth_candidate_id"].astype(str).map(identity_count).to_numpy(float)
    weights *= np.where(action["introduced"].to_numpy(bool), RISK_PENALTY, 1.0)
    weights /= weights.mean()
    x = action[list(features)].to_numpy(float)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=float(c_value), solver="liblinear", max_iter=2000,
        random_state=seed,
    ).fit(scaler.transform(x), y, sample_weight=weights)
    return scaler, model, {
        "rows": int(len(action)),
        "identities": int(action["truth_candidate_id"].nunique()),
        "formulas": int(action["truth_formula"].nunique()),
        "corrective": int(action["corrected"].sum()),
        "harmful": int(action["introduced"].sum()),
    }


def predict(
    scaler: StandardScaler, model: LogisticRegression,
    frame: pd.DataFrame, features: tuple[str, ...],
) -> np.ndarray:
    return model.predict_proba(scaler.transform(frame[list(features)].to_numpy(float)))[:, 1]


def apply_router(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> pd.DataFrame:
    output = frame.copy()
    output["b29_probability"] = np.asarray(probability, dtype=float)
    output["b29_threshold"] = float(threshold)
    output["B17_final_candidate_id"] = output["final_candidate_id"].astype(str)
    output["B17_corrected"] = output["corrected"].astype(bool)
    output["B17_introduced"] = output["introduced"].astype(bool)
    output["b29_execute"] = (
        output["intervene"].astype(bool)
        & output["b29_probability"].ge(float(threshold))
    )
    output["final_candidate_id"] = np.where(
        output["b29_execute"], output["B17_final_candidate_id"],
        output["baseline_candidate_id"],
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
    return output


def compact_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    corrected = int(frame["corrected"].sum())
    introduced = int(frame["introduced"].sum())
    return {
        "queries": int(len(frame)),
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "delta_recall1": float(frame["delta"].mean()),
        "interventions": int(frame["intervene"].sum()),
    }


def safe_auc(y: np.ndarray, score: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(y)) != 2:
        return None, None
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def choose_configuration(oof: pd.DataFrame) -> tuple[dict, list[dict]]:
    ledger: list[dict] = []
    for family, c_value in MODEL_GRID:
        probability = oof[f"probability__{family}__C{c_value:g}"].to_numpy(float)
        for threshold in THRESHOLDS:
            routed = apply_router(oof, probability, threshold)
            overall = compact_summary(routed)
            by_formula_fold = {
                str(fold): compact_summary(routed.loc[routed["b29_formula_fold"].eq(fold)])
                for fold in sorted(routed["b29_formula_fold"].unique())
            }
            ledger.append({
                "feature_family": family, "C": c_value,
                "threshold": threshold, **overall,
                "every_inner_formula_fold_risk_nonnegative": all(
                    item["risk_net_lambda2"] >= 0 for item in by_formula_fold.values()
                ),
            })
    eligible = [
        row for row in ledger
        if row["every_inner_formula_fold_risk_nonnegative"]
        and row["corrected"] > 2 * row["introduced"]
    ]
    # B17 is the fail-safe configuration: all actions execute.
    b17 = {
        "feature_family": "B17", "C": 0.0, "threshold": 0.0,
        **compact_summary(oof), "every_inner_formula_fold_risk_nonnegative": True,
    }
    pool = [b17, *eligible]
    selected = max(pool, key=lambda row: (
        row["risk_net_lambda2"] / max(1, row["queries"]),
        row["delta_recall1"], -row["introduced"] / max(1, row["queries"]),
        row["feature_family"] == "B17",
    ))
    return selected, ledger


def formula_folds(frame: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Deterministic outcome-blind, query-count-balanced formula folds."""
    counts = frame["truth_formula"].astype(str).value_counts()
    ordered = sorted(
        counts.items(),
        key=lambda item: (
            -int(item[1]),
            hashlib.sha256(f"{seed}|{item[0]}".encode("utf-8")).hexdigest(),
        ),
    )
    loads = [0] * n_folds
    assignment: dict[str, int] = {}
    for formula, count in ordered:
        fold = min(range(n_folds), key=lambda index: (loads[index], index))
        assignment[str(formula)] = fold
        loads[fold] += int(count)
    result = frame["truth_formula"].astype(str).map(assignment)
    if result.isna().any() or result.nunique() != n_folds:
        raise RuntimeError("B29 could not create complete formula folds")
    return result.astype(int)


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    full, physical, provenance = materialise_features(args)
    action_physical = physical.loc[physical["intervene"].astype(bool)].copy()
    outer_results: list[pd.DataFrame] = []
    folds: list[dict] = []
    for outer_index, outer_source in enumerate(EXPECTED_DOMAINS):
        outer = physical.loc[physical["source"].eq(outer_source)].copy()
        development = physical.loc[~physical["source"].eq(outer_source)].copy()
        development = purge_train(development, outer)
        development_actions = development.loc[development["intervene"].astype(bool)].copy()
        development_actions["b29_formula_fold"] = formula_folds(
            development_actions, 3, args.seed + outer_index
        )
        inner_predictions = development_actions.copy()
        fit_reports: list[dict] = []
        for family, c_value in MODEL_GRID:
            column = f"probability__{family}__C{c_value:g}"
            inner_predictions[column] = np.nan
        for inner_index in sorted(development_actions["b29_formula_fold"].unique()):
            inner_test = development_actions.loc[
                development_actions["b29_formula_fold"].eq(inner_index)
            ].copy()
            inner_train = development_actions.loc[
                ~development_actions["b29_formula_fold"].eq(inner_index)
            ].copy()
            if set(inner_train["truth_formula"].astype(str)) & set(inner_test["truth_formula"].astype(str)):
                raise RuntimeError("B29 inner formula fold leaked")
            for family, c_value in MODEL_GRID:
                features = FEATURE_FAMILIES[family]
                scaler, model, fit_report = fit_model(
                    inner_train, features, c_value,
                    args.seed + outer_index * 1000 + int(inner_index) * 100 + int(c_value * 10),
                )
                column = f"probability__{family}__C{c_value:g}"
                inner_predictions.loc[inner_test.index, column] = predict(
                    scaler, model, inner_test, features
                )
                fit_reports.append({
                    "inner_formula_fold": int(inner_index), "feature_family": family,
                    "C": c_value, **fit_report,
                })
        probability_columns = [column for column in inner_predictions if column.startswith("probability__")]
        if inner_predictions[probability_columns].isna().any().any():
            raise RuntimeError(f"B29 incomplete inner predictions for {outer_source}")
        selected, ledger = choose_configuration(inner_predictions)
        outer_probability = np.zeros(len(outer), dtype=float)
        outer_action = outer["intervene"].to_numpy(bool)
        model_report = None
        coefficients = None
        if selected["feature_family"] == "B17":
            outer_probability[outer_action] = 1.0
        else:
            family = str(selected["feature_family"])
            features = FEATURE_FAMILIES[family]
            scaler, model, model_report = fit_model(
                development, features, float(selected["C"]),
                args.seed + outer_index * 1000 + 999,
            )
            outer_probability[outer_action] = predict(
                scaler, model, outer.loc[outer_action], features
            )
            coefficients = {
                feature: float(value)
                for feature, value in zip(features, model.coef_[0], strict=True)
            }
        threshold = float(selected["threshold"])
        result = apply_router(outer, outer_probability, threshold)
        outer_results.append(result)
        outer_action_result = result.loc[result["B17_final_candidate_id"].astype(str).ne(
            result["baseline_candidate_id"].astype(str)
        )]
        y = outer_action_result["B17_corrected"].astype(int).to_numpy()
        auc, auprc = safe_auc(y, outer_action_result["b29_probability"].to_numpy(float))
        folds.append({
            "outer_source": outer_source,
            "development_physical_queries": int(len(development)),
            "development_action_queries": int(len(development_actions)),
            "selected": selected,
            "selection_ledger": ledger,
            "inner_fit_reports": fit_reports,
            "outer": compact_summary(result),
            "outer_action_auc": auc, "outer_action_auprc": auprc,
            "final_fit": model_report, "coefficients": coefficients,
        })
        print(f"[B29 {outer_source}] selected={selected} outer={compact_summary(result)}", flush=True)

    nested_physical = pd.concat(outer_results, ignore_index=True)
    if len(nested_physical) != len(physical) or nested_physical["physical_key"].nunique() != len(physical):
        raise RuntimeError("B29 outer physical coverage changed")
    decision = nested_physical.set_index("physical_key", verify_integrity=True)[
        ["b29_probability", "b29_threshold", "b29_execute"]
    ]
    full["physical_key"] = physical_key(full)
    full = full.join(decision, on="physical_key", validate="many_to_one")
    # Replay the physical decision on every original B17 evaluation row.
    b17_candidate = full["final_candidate_id"].astype(str).copy()
    full["B17_final_candidate_id"] = b17_candidate
    full["final_candidate_id"] = np.where(
        full["b29_execute"].astype(bool), b17_candidate,
        full["baseline_candidate_id"].astype(str),
    )
    full["intervene"] = full["final_candidate_id"].astype(str).ne(full["baseline_candidate_id"].astype(str))
    full["final_correct"] = full["final_candidate_id"].astype(str).eq(full["truth_candidate_id"].astype(str))
    full["corrected"] = ~full["baseline_correct"].astype(bool) & full["final_correct"]
    full["introduced"] = full["baseline_correct"].astype(bool) & ~full["final_correct"]
    full["delta"] = full["final_correct"].astype(int) - full["baseline_correct"].astype(int)
    if full[["b29_probability", "b29_threshold"]].isna().any().any():
        raise RuntimeError("B29 failed to replay physical decisions")

    row_summary = summarize(full)
    physical_summary = compact_summary(nested_physical)
    corrected = full.loc[full["corrected"]]
    introduced = full.loc[full["introduced"]]
    identity_ci = cluster_bootstrap(full, "truth_candidate_id", args.bootstrap_resamples, args.seed + 1)
    formula_ci = cluster_bootstrap(full, "truth_formula", args.bootstrap_resamples, args.seed + 2)
    by_source = {
        source: summarize(full.loc[full["source"].eq(source)])
        for source in EXPECTED_DOMAINS
    }
    gates = {
        "all_860_rows_replayed": len(full) == 860 and full["query_id"].nunique() == 860,
        "all_753_physical_queries_replayed": len(nested_physical) == 753,
        "row_risk_strictly_beats_B17": row_summary["risk_net_lambda2"] > B17_ROW_RISK,
        "row_delta_not_below_B17": row_summary["delta_recall1"] >= B17_ROW_DELTA - 1e-15,
        "row_introduced_below_B17": row_summary["introduced"] < B17_ROW_INTRODUCED,
        "physical_risk_strictly_beats_B17": physical_summary["risk_net_lambda2"] > B17_PHYSICAL_RISK,
        "identity_ci_low_positive": identity_ci["ci_low"] > 0,
        "formula_ci_low_positive": formula_ci["ci_low"] > 0,
        "corrected_identities_ge_25": corrected["truth_candidate_id"].nunique() >= 25,
        "corrected_formulas_ge_25": corrected["truth_formula"].nunique() >= 25,
        "every_outer_source_nonnegative": all(item["delta_recall1"] >= 0 for item in by_source.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    transition_path = args.output_dir / "nested_action_risk_transitions.csv.gz"
    full.to_csv(transition_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b29_action_risk_router_complete",
        "formal": True,
        "protocol": "outer leave-source-out action execution router; held identity/formula purge; inner formula-group OOF selection; physical-spectrum training mass; lambda=2 harm cost",
        "nested_row_oof": {
            **row_summary,
            "corrected_identities": int(corrected["truth_candidate_id"].nunique()),
            "corrected_formulas": int(corrected["truth_formula"].nunique()),
            "introduced_identities": int(introduced["truth_candidate_id"].nunique()),
            "introduced_formulas": int(introduced["truth_formula"].nunique()),
            "identity_cluster_bootstrap": identity_ci,
            "formula_cluster_bootstrap": formula_ci,
            "by_source": by_source,
        },
        "nested_physical_oof": physical_summary,
        "frozen_B17_comparator": {
            "rows": {"corrected": B17_ROW_CORRECTED, "introduced": B17_ROW_INTRODUCED, "risk_net_lambda2": B17_ROW_RISK, "delta_recall1": B17_ROW_DELTA},
            "physical": {"corrected": B17_PHYSICAL_CORRECTED, "introduced": B17_PHYSICAL_INTRODUCED, "risk_net_lambda2": B17_PHYSICAL_RISK},
        },
        "feature_families": {key: list(value) for key, value in FEATURE_FAMILIES.items()},
        "model_grid": [{"feature_family": family, "C": c_value} for family, c_value in MODEL_GRID],
        "thresholds": list(THRESHOLDS),
        "folds": folds,
        "gates": gates,
        "strictly_better_action_than_B17": bool(all(gates.values())),
        "contracts": {
            "router_can_only_execute_or_revert_B17": True,
            "new_candidate_search": False,
            "outer_outcome_used_for_model_or_threshold_selection": False,
            "inner_predictions_crossfit_by_formula": True,
            "held_truth_identity_and_formula_purged": True,
            "physical_duplicates_have_unit_training_mass": True,
            "truth_or_outcome_used_as_feature": False,
            "candidate_identity_or_source_used_as_feature": False,
            "P2b_used": False, "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            **provenance,
            "transitions_sha256": sha256(transition_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "Opened nested action-risk audit. Passing improves the development action router only; it is not blind validation, SOTA, biological mechanism, or shared-embedding improvement.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["strictly_better_action_than_B17"]:
        raise RuntimeError(f"B29 did not strictly improve B17: {gates}")


if __name__ == "__main__":
    main()
