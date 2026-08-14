"""Train a formula-isolated directional hard-candidate reranker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_frozen_panel_reranker import metrics, paired_formula_bootstrap


def fold_for_formula(formula: str, folds: int) -> int:
    return int.from_bytes(hashlib.blake2b(formula.encode(), digest_size=8).digest(), "little") % folds


def hard_pool(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    positives = frame.loc[frame["label"] == 1]
    negatives = frame.loc[frame["label"] == 0].sort_values(
        ["query", "dreams_similarity"], ascending=[True, False]
    ).groupby("query", sort=False).head(k)
    return pd.concat([positives, negatives], ignore_index=True)


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby(["formula", "label"])["query"].transform("size").to_numpy(float)
    formula_count = frame["formula"].nunique()
    values = 1.0 / (2.0 * formula_count * counts)
    return values * len(values) / values.sum()


def fit(frame: pd.DataFrame, features: list[str], c_value: float):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=5000, random_state=20260813),
    )
    model.fit(frame[features], frame["label"], logisticregression__sample_weight=sample_weights(frame))
    return model


def retrieval(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    rows = []
    for query, group in frame.groupby("query", sort=False):
        positive = group.loc[group["label"] == 1, score_column]
        negative = group.loc[group["label"] == 0]
        if positive.empty or negative.empty:
            continue
        positive_score = float(positive.max())
        molecule_scores = negative.groupby("candidate_ik14")[score_column].max().to_numpy(float)
        rank = 1 + int(np.sum(molecule_scores >= positive_score))
        first = group.iloc[0]
        rows.append({
            "query_index": int(query), "ik14": first.query_ik14,
            "formula": first.formula, "positive_score": positive_score,
            "best_negative_score": float(molecule_scores.max()),
            "margin": positive_score - float(molecule_scores.max()),
            "top1": bool(rank == 1), "rank": rank, "mrr": 1.0 / rank,
            "pairwise_accuracy": float(np.mean(positive_score > molecule_scores)),
        })
    return pd.DataFrame(rows)


def cv_select(
    discovery: pd.DataFrame, features: list[str], c_values: list[float],
    folds: int, hard_k: int,
) -> tuple[float, pd.DataFrame]:
    formula_folds = {formula: fold_for_formula(formula, folds) for formula in discovery["formula"].unique()}
    records = []
    for c_value in c_values:
        scored_parts = []
        for fold in range(folds):
            held_formulas = {formula for formula, value in formula_folds.items() if value == fold}
            train = hard_pool(discovery.loc[~discovery["formula"].isin(held_formulas)], hard_k)
            held = discovery.loc[discovery["formula"].isin(held_formulas)].copy()
            model = fit(train, features, c_value)
            held["score"] = model.decision_function(held[features])
            scored_parts.append(held)
        query = retrieval(pd.concat(scored_parts), "score")
        records.append({"C": c_value} | metrics(query))
    table = pd.DataFrame(records).sort_values(["top1", "mrr", "C"], ascending=[False, False, True])
    return float(table.iloc[0]["C"]), table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=Path("data/validation/directional_panel_candidate_features"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/directional_hard_reranker"))
    parser.add_argument("--hard-k", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    discovery = pd.read_csv(args.feature_dir / "discovery_directional_features.csv")
    confirmation = pd.read_csv(args.feature_dir / "confirmation_directional_features.csv")
    directional = [column for column in discovery if column.startswith("dir_")]
    burden = [column for column in directional if column.endswith(
        ("matched_burden_of_query", "matched_burden_over_all_query", "query_unmatched_intensity")
    )]
    # Query-only prevalence is constant across all candidates and cannot change ranking.
    full = [column for column in directional if not column.endswith("query_intensity")]
    feature_sets = {
        "dreams_only": ["dreams_similarity"],
        "dreams_plus_burden": ["dreams_similarity"] + burden,
        "dreams_plus_directional": ["dreams_similarity"] + full,
    }
    discovery["baseline_score"] = discovery["dreams_similarity"]
    confirmation["baseline_score"] = confirmation["dreams_similarity"]
    baseline_discovery = retrieval(discovery, "baseline_score")
    baseline_confirmation = retrieval(confirmation, "baseline_score")
    report = {
        "status": "directional_hard_candidate_reranker", "hard_negatives_per_query": args.hard_k,
        "training": "all identity positives plus top-k DreaMS negatives per query; formula-balanced logistic regression",
        "selection": f"{args.folds}-fold formula-isolated discovery CV",
        "baseline": {"discovery": metrics(baseline_discovery), "confirmation": metrics(baseline_confirmation)},
        "models": {},
    }
    coefficients = []
    for index, (name, features) in enumerate(feature_sets.items()):
        best_c, table = cv_select(discovery, features, args.c_values, args.folds, args.hard_k)
        table.to_csv(args.output_dir / f"{name}_cv.csv", index=False)
        model = fit(hard_pool(discovery, args.hard_k), features, best_c)
        scored = confirmation.copy()
        scored["score"] = model.decision_function(scored[features])
        query = retrieval(scored, "score")
        query.to_csv(args.output_dir / f"{name}_confirmation_queries.csv", index=False)
        result = metrics(query)
        result.update({
            "selected_C": best_c,
            "top1_minus_dreams": result["top1"] - report["baseline"]["confirmation"]["top1"],
            "top1_formula_bootstrap_ci95": paired_formula_bootstrap(
                baseline_confirmation, query, "top1", args.bootstrap, 20260813 + index,
            ),
            "mrr_minus_dreams": result["mrr"] - report["baseline"]["confirmation"]["mrr"],
            "mrr_formula_bootstrap_ci95": paired_formula_bootstrap(
                baseline_confirmation, query, "mrr", args.bootstrap, 20260913 + index,
            ),
        })
        report["models"][name] = result
        scaler, classifier = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
        for feature, coefficient in zip(features, classifier.coef_[0]):
            coefficients.append({"model": name, "feature": feature, "standardized_coefficient": coefficient})
    pd.DataFrame(coefficients).to_csv(args.output_dir / "coefficients.csv", index=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
