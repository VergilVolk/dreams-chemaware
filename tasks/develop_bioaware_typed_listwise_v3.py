#!/usr/bin/env python
"""Nested formula-OOF BioAware v3 typed listwise reranker.

The model learns within-query candidate order.  DreaMS remains the immutable
unary base; the learned score can only supply a bounded residual and a frozen
inner-OOF gate may abstain.  Every model and policy choice is made without the
outer formula fold.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import binomtest


BASE_FEATURES = [
    "decoder_score_scaled",
    "rule_jaccard_idf",
    "sparse_rule_overlap",
    "known_edge_best_bottleneck",
    "predicted_edge_best_bottleneck",
    "smn_best_bottleneck",
    "rt_score",
]
TYPED_FEATURES = [
    "typed_dependency_corrected_network_support_mean",
    "typed_dependency_corrected_network_support_max",
    "typed_candidate_specific_network_support_mean",
    "typed_candidate_specific_network_support_max",
    "typed_complete_network_support_positive_fraction",
    "typed_direction_supported_network_support_positive_fraction",
    "typed_curated_direction_conflicted_path_count_mean",
    "typed_mean_source_side_completeness_mean",
    "typed_mean_candidate_specificity_mean",
    "typed_unique_seed_compounds_mean",
    "typed_unique_reactions_mean",
    "typed_excluded_identity_noop_path_count_mean",
]
FEATURES = ["spectral_score", *BASE_FEATURES, *TYPED_FEATURES]
FAMILIES = {
    "reaction": [
        "typed_candidate_specific_network_support_mean",
        "typed_candidate_specific_network_support_max",
        "typed_dependency_corrected_network_support_mean",
    ],
    "reaction_complete": ["typed_complete_network_support_positive_fraction"],
    "reaction_direction": ["typed_direction_supported_network_support_positive_fraction"],
    "decoder": ["decoder_score_scaled"],
    "rules": ["rule_jaccard_idf", "sparse_rule_overlap"],
    "data_network": [
        "known_edge_best_bottleneck",
        "predicted_edge_best_bottleneck",
        "smn_best_bottleneck",
    ],
    "retention": ["rt_score"],
}
MODEL_CONFIGS = [
    {"name": "shallow", "num_leaves": 3, "min_child_samples": 20,
     "learning_rate": 0.03, "n_estimators": 120, "reg_lambda": 8.0},
    {"name": "balanced", "num_leaves": 7, "min_child_samples": 15,
     "learning_rate": 0.025, "n_estimators": 200, "reg_lambda": 10.0},
    {"name": "interaction", "num_leaves": 7, "min_child_samples": 10,
     "learning_rate": 0.02, "n_estimators": 280, "reg_lambda": 15.0},
]
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def fold_for(value: str, count: int, salt: str) -> int:
    token = hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()
    return int(token[:16], 16) % count


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["decoder_score_scaled"] = (result["decoder_score"].clip(-1, 1) + 1) / 2
    missing = set(FEATURES) - set(result)
    if missing:
        raise RuntimeError(f"ledger missing model features: {sorted(missing)}")
    if result[FEATURES].isna().any().any():
        raise RuntimeError("model feature matrix contains missing values")
    if result.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("candidate key is not unique")
    truth_counts = result.assign(
        _truth=result["candidate_id"].astype(str) == result["truth_candidate_id"].astype(str)
    ).groupby("query_id")["_truth"].sum()
    if not (truth_counts == 1).all():
        raise RuntimeError("every query needs exactly one truth candidate")
    return result


def fit_ranker(frame: pd.DataFrame, config: dict, seed: int) -> lgb.LGBMRanker:
    ordered = frame.sort_values(["query_id", "candidate_id"]).copy()
    group = ordered.groupby("query_id", sort=False).size().to_numpy()
    label = (
        ordered["candidate_id"].astype(str)
        == ordered["truth_candidate_id"].astype(str)
    ).astype(int).to_numpy()
    weights = np.repeat(1.0 / group, group)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        random_state=seed,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        max_depth=-1,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        num_leaves=config["num_leaves"],
        min_child_samples=config["min_child_samples"],
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
        reg_lambda=config["reg_lambda"],
        reg_alpha=0.5,
    )
    model.fit(ordered[FEATURES], label, group=group, sample_weight=weights)
    return model


def add_model_scores(model: lgb.LGBMRanker, frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["model_score"] = model.predict(scored[FEATURES])
    normalized: list[pd.Series] = []
    for _, group in scored.groupby("query_id", sort=False):
        values = group["model_score"].to_numpy(float)
        scale = float(np.std(values))
        if scale <= EPS:
            z = np.zeros(len(group), dtype=float)
        else:
            z = (values - float(np.mean(values))) / scale
        normalized.append(pd.Series(z, index=group.index))
    scored["model_score_z"] = pd.concat(normalized).sort_index()
    return scored


def top_row(group: pd.DataFrame, column: str) -> tuple[pd.Series, bool]:
    maximum = float(group[column].max())
    tied = group[np.isclose(group[column], maximum, rtol=0, atol=EPS)]
    row = tied.sort_values("candidate_id").iloc[0]
    return row, len(tied) == 1


def decisions(scored: pd.DataFrame, policy: tuple[float, float, float, int]) -> pd.DataFrame:
    alpha, maximum_margin, minimum_model_advantage, minimum_support = policy
    frame = scored.copy()
    frame["fusion_score"] = frame["spectral_score"] + alpha * frame["model_score_z"]
    rows: list[dict] = []
    for query_id, group in frame.groupby("query_id", sort=False):
        truth = str(group["truth_candidate_id"].iloc[0])
        baseline, base_unique = top_row(group, "spectral_score")
        proposed, proposed_unique = top_row(group, "fusion_score")
        ordered = group.sort_values(["spectral_score", "candidate_id"], ascending=[False, True])
        margin = float(ordered.iloc[0].spectral_score - ordered.iloc[1].spectral_score)
        support_names: list[str] = []
        if str(proposed.candidate_id) != str(baseline.candidate_id):
            for family, columns in FAMILIES.items():
                if max(float(proposed[column] - baseline[column]) for column in columns) > EPS:
                    support_names.append(family)
        model_advantage = float(proposed.model_score_z - baseline.model_score_z)
        intervene = bool(
            str(proposed.candidate_id) != str(baseline.candidate_id)
            and proposed_unique
            and margin <= maximum_margin + EPS
            and model_advantage >= minimum_model_advantage - EPS
            and len(support_names) >= minimum_support
        )
        final = str(proposed.candidate_id) if intervene else str(baseline.candidate_id)
        baseline_correct = bool(base_unique and str(baseline.candidate_id) == truth)
        final_correct = final == truth
        rows.append(
            {
                "query_id": str(query_id),
                "truth_candidate_id": truth,
                "truth_formula": str(group["truth_formula"].iloc[0]),
                "baseline_candidate_id": str(baseline.candidate_id),
                "proposed_candidate_id": str(proposed.candidate_id),
                "final_candidate_id": final,
                "baseline_correct": baseline_correct,
                "final_correct": final_correct,
                "intervene": intervene,
                "corrected": (not baseline_correct) and final_correct,
                "introduced": baseline_correct and (not final_correct),
                "spectral_margin": margin,
                "model_advantage": model_advantage,
                "support_count": len(support_names),
                "support_families": ";".join(support_names),
            }
        )
    return pd.DataFrame(rows)


def policy_grid() -> list[tuple[float, float, float, int]]:
    return list(
        itertools.product(
            [0.01, 0.025, 0.05, 0.10, 0.20],
            [0.01, 0.025, 0.05, 0.10, 0.20, 1.0],
            [0.0, 0.25, 0.50, 1.0],
            [1, 2],
        )
    )


def summarize(frame: pd.DataFrame) -> tuple[int, int, int, int]:
    corrected = int(frame["corrected"].sum())
    introduced = int(frame["introduced"].sum())
    interventions = int(frame["intervene"].sum())
    return corrected - 2 * introduced, corrected, -introduced, -interventions


def inner_oof(
    train: pd.DataFrame,
    config: dict,
    outer_fold: int,
    inner_folds: int,
    seed: int,
) -> pd.DataFrame:
    formulas = train[["truth_formula"]].drop_duplicates().copy()
    formulas["inner_fold"] = formulas["truth_formula"].map(
        lambda value: fold_for(str(value), inner_folds, f"inner-{seed}-{outer_fold}")
    )
    pieces: list[pd.DataFrame] = []
    for inner_fold in range(inner_folds):
        validation_formulas = set(
            formulas.loc[formulas["inner_fold"] == inner_fold, "truth_formula"]
        )
        fit = train[~train["truth_formula"].isin(validation_formulas)]
        validation = train[train["truth_formula"].isin(validation_formulas)]
        if validation.empty:
            raise RuntimeError("empty inner fold")
        model = fit_ranker(fit, config, seed + 100 * outer_fold + inner_fold)
        pieces.append(add_model_scores(model, validation))
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(["query_id", "candidate_id"]).any() or len(result) != len(train):
        raise RuntimeError("inner OOF candidate coverage mismatch")
    return result


def select_recipe(
    train: pd.DataFrame,
    outer_fold: int,
    inner_folds: int,
    seed: int,
) -> tuple[dict, tuple[float, float, float, int], dict]:
    candidates: list[tuple[tuple[int, int, int, int], int, tuple, dict]] = []
    diagnostics: list[dict] = []
    for config_index, config in enumerate(MODEL_CONFIGS):
        scored = inner_oof(train, config, outer_fold, inner_folds, seed)
        for policy in policy_grid():
            evaluated = decisions(scored, policy)
            key = summarize(evaluated)
            diagnostics.append(
                {
                    "model": config["name"],
                    "policy": list(policy),
                    "risk_net": key[0],
                    "corrected": key[1],
                    "introduced": -key[2],
                    "interventions": -key[3],
                }
            )
            candidates.append((key, -config_index, policy, config))
    # No-intervention is an explicit comparator, not an error.  A selected
    # intervention recipe must beat its risk-weighted score of zero.
    best = max(candidates, key=lambda item: (item[0], item[1]))
    if best[0][0] <= 0 or best[0][1] <= 0:
        return MODEL_CONFIGS[0], (0.0, 0.0, float("inf"), 99), {
            "selected_no_intervention": True,
            "best_observed": max(diagnostics, key=lambda row: (
                row["risk_net"], row["corrected"], -row["introduced"]
            )),
        }
    return best[3], best[2], {
        "selected_no_intervention": False,
        "inner_risk_net": best[0][0],
        "inner_corrected": best[0][1],
        "inner_introduced": -best[0][2],
        "inner_interventions": -best[0][3],
    }


def formula_bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    grouped = {key: value for key, value in frame.groupby("truth_formula", sort=True)}
    formulas = sorted(grouped)
    rng = np.random.default_rng(seed)
    delta = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sampled = rng.choice(formulas, size=len(formulas), replace=True)
        sample = pd.concat([grouped[str(key)] for key in sampled], ignore_index=True)
        delta[index] = float(sample["final_correct"].mean() - sample["baseline_correct"].mean())
    return {
        "mean": float(frame["final_correct"].mean() - frame["baseline_correct"].mean()),
        "ci_low": float(np.quantile(delta, 0.025)),
        "ci_high": float(np.quantile(delta, 0.975)),
        "clusters": len(formulas),
        "resamples": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger", type=Path,
        default=Path("data/validation/bioaware_typed_candidate_ledger_v1/candidate_evidence_typed.csv.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_typed_listwise_v3"),
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if not args.ledger.exists():
        raise FileNotFoundError(args.ledger)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = prepare(pd.read_csv(args.ledger))
    formulas = ledger[["truth_formula"]].drop_duplicates().copy()
    formulas["outer_fold"] = formulas["truth_formula"].map(
        lambda value: fold_for(str(value), args.outer_folds, f"outer-{args.seed}")
    )
    ledger = ledger.merge(formulas, on="truth_formula", validate="many_to_one")

    candidate_parts: list[pd.DataFrame] = []
    decision_parts: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    for outer_fold in range(args.outer_folds):
        train = ledger[ledger["outer_fold"] != outer_fold].copy()
        test = ledger[ledger["outer_fold"] == outer_fold].copy()
        config, policy, inner = select_recipe(
            train, outer_fold, args.inner_folds, args.seed
        )
        model = fit_ranker(train, config, args.seed + 1000 + outer_fold)
        scored = add_model_scores(model, test)
        evaluated = decisions(scored, policy)
        scored["outer_fold"] = outer_fold
        evaluated["outer_fold"] = outer_fold
        candidate_parts.append(scored)
        decision_parts.append(evaluated)
        fold_reports.append(
            {
                "outer_fold": outer_fold,
                "queries": int(evaluated["query_id"].nunique()),
                "formulas": int(evaluated["truth_formula"].nunique()),
                "model": config,
                "policy": {
                    "alpha": policy[0],
                    "maximum_spectral_margin": policy[1],
                    "minimum_model_advantage": (
                        policy[2] if np.isfinite(policy[2]) else None
                    ),
                    "minimum_support_families": policy[3],
                },
                "inner_selection": inner,
                "corrected": int(evaluated["corrected"].sum()),
                "introduced": int(evaluated["introduced"].sum()),
                "interventions": int(evaluated["intervene"].sum()),
            }
        )
        print(
            f"[outer {outer_fold}] model={config['name']} policy={policy} "
            f"C/I={fold_reports[-1]['corrected']}/{fold_reports[-1]['introduced']}",
            flush=True,
        )

    candidate_oof = pd.concat(candidate_parts, ignore_index=True)
    result = pd.concat(decision_parts, ignore_index=True)
    if result["query_id"].duplicated().any() or result["query_id"].nunique() != 117:
        raise RuntimeError("outer OOF query coverage mismatch")
    corrected = int(result["corrected"].sum())
    introduced = int(result["introduced"].sum())
    baseline = float(result["baseline_correct"].mean())
    final = float(result["final_correct"].mean())
    bootstrap = formula_bootstrap(result, args.bootstrap_resamples, args.seed)
    discordant = corrected + introduced
    mcnemar = float(
        binomtest(min(corrected, introduced), discordant, 0.5, alternative="two-sided").pvalue
    ) if discordant else 1.0
    candidate_path = args.output_dir / "candidate_oof_scores.csv.gz"
    transition_path = args.output_dir / "query_oof_transitions.csv.gz"
    candidate_oof.to_csv(candidate_path, index=False, compression="gzip")
    result.to_csv(transition_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_typed_listwise_v3_development_complete",
        "formal": True,
        "protocol": "nested formula-OOF LightGBM LambdaRank; bounded DreaMS residual; risk gate and abstention",
        "queries": int(len(result)),
        "identities": int(result["truth_candidate_id"].nunique()),
        "formulas": int(result["truth_formula"].nunique()),
        "baseline_recall1": baseline,
        "recall1": final,
        "delta_recall1": final - baseline,
        "corrected": corrected,
        "introduced": introduced,
        "risk_weighted_net": corrected - 2 * introduced,
        "interventions": int(result["intervene"].sum()),
        "formula_cluster_bootstrap": bootstrap,
        "mcnemar_exact_p": mcnemar,
        "folds": fold_reports,
        "gates": {
            "corrected_gt_introduced": corrected > introduced,
            "risk_weighted_net_positive": corrected - 2 * introduced > 0,
            "formula_cluster_ci_positive": bootstrap["ci_low"] > 0,
            "mcnemar_p_lt_0_05": mcnemar < 0.05,
        },
        "contracts": {
            "P2b": "forbidden",
            "phenotype": "forbidden",
            "DreaMS": "immutable unary base",
            "model_selection": "inner formula OOF only",
            "evaluation": "outer formula OOF only",
            "reaction_noops": "excluded upstream",
            "RP": "not opened",
        },
        "provenance": {
            "ledger": sha256(args.ledger),
            "candidate_oof": sha256(candidate_path),
            "transitions": sha256(transition_path),
        },
        "claim_limit": (
            "Consumed development result. A positive OOF result freezes a recipe; "
            "only an untouched external protocol can support an SOTA claim."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
