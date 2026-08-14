"""Pairwise ranking audit for frozen DreaMS peak evidence.

The learner sees within-query positive-minus-hard-negative feature differences,
not global pair labels.  This directly optimizes the retrieval ordering while
keeping the DreaMS backbone and frozen chemical panel unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from train_frozen_panel_reranker import RAW_FEATURES, metrics, paired_formula_bootstrap
from train_directional_hard_reranker import retrieval


def fold_for_formula(formula: str, folds: int) -> int:
    return int.from_bytes(hashlib.blake2b(formula.encode(), digest_size=8).digest(), "little") % folds


def augment_features(
    directional: pd.DataFrame, pair_dir: Path, token_dir: Path, split: str,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    frame = directional.copy()
    frame["left"] = np.minimum(frame["query"], frame["candidate"])
    frame["right"] = np.maximum(frame["query"], frame["candidate"])
    raw = pd.read_csv(pair_dir / f"{split}_pair_features.csv")
    token = pd.read_csv(token_dir / f"{split}_token_pair_features.csv")
    token_columns = [column for column in token if column.startswith("token_")]
    frame = frame.merge(
        raw[["left", "right"] + RAW_FEATURES], on=["left", "right"], how="left", validate="many_to_one"
    ).merge(
        token[["left", "right"] + token_columns], on=["left", "right"], how="left", validate="many_to_one"
    )
    directional_columns = [column for column in frame if column.startswith("dir_")]
    burden = [column for column in directional_columns if column.endswith(
        ("matched_burden_of_query", "matched_burden_over_all_query", "query_unmatched_intensity")
    )]
    return frame, burden, directional_columns, token_columns


def ranking_examples(frame: pd.DataFrame, features: list[str], hard_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    differences, formulas = [], []
    for _, group in frame.groupby("query", sort=False):
        positives = group.loc[group["label"] == 1]
        negatives = group.loc[group["label"] == 0]
        if positives.empty or negatives.empty:
            continue
        positive = positives.loc[positives["dreams_similarity"].idxmax()]
        molecule_best = negatives.sort_values("dreams_similarity", ascending=False).drop_duplicates("candidate_ik14")
        hard = molecule_best.head(hard_k)
        delta = positive[features].to_numpy(float)[None, :] - hard[features].to_numpy(float)
        differences.append(delta)
        formulas.extend([positive.formula] * len(delta))
    x_positive = np.concatenate(differences)
    formula = np.asarray(formulas, object)
    # Reverse every comparison so the model cannot absorb an intercept shortcut.
    x = np.vstack([x_positive, -x_positive])
    y = np.r_[np.ones(len(x_positive), dtype=int), np.zeros(len(x_positive), dtype=int)]
    return x, y, np.r_[formula, formula]


def fit_ranker(frame: pd.DataFrame, features: list[str], hard_k: int, c_value: float):
    x, y, formula = ranking_examples(frame, features, hard_k)
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    counts = pd.Series(formula).map(pd.Series(formula).value_counts()).to_numpy(float)
    sample_weight = 1.0 / counts
    sample_weight *= len(sample_weight) / sample_weight.sum()
    model = LogisticRegression(
        C=c_value, fit_intercept=False, max_iter=5000, random_state=20260813
    ).fit(x_scaled, y, sample_weight=sample_weight)
    return scaler, model, len(x) // 2


def score(frame: pd.DataFrame, features: list[str], scaler: StandardScaler, model: LogisticRegression) -> pd.DataFrame:
    output = frame.copy()
    output["score"] = model.decision_function(scaler.transform(output[features]))
    return output


def cv_select(
    frame: pd.DataFrame, features: list[str], hard_k: int,
    c_values: list[float], folds: int,
) -> tuple[float, pd.DataFrame]:
    assignments = {formula: fold_for_formula(formula, folds) for formula in frame["formula"].unique()}
    rows = []
    for c_value in c_values:
        scored = []
        for fold in range(folds):
            held_formulas = {formula for formula, value in assignments.items() if value == fold}
            train = frame.loc[~frame["formula"].isin(held_formulas)]
            held = frame.loc[frame["formula"].isin(held_formulas)]
            scaler, model, _ = fit_ranker(train, features, hard_k, c_value)
            scored.append(score(held, features, scaler, model))
        query = retrieval(pd.concat(scored), "score")
        rows.append({"C": c_value} | metrics(query))
    table = pd.DataFrame(rows).sort_values(["top1", "mrr", "C"], ascending=[False, False, True])
    return float(table.iloc[0]["C"]), table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directional-dir", type=Path, default=Path("data/validation/directional_panel_candidate_features"))
    parser.add_argument("--pair-dir", type=Path, default=Path("data/validation/large_observability_residual_audit"))
    parser.add_argument("--token-dir", type=Path, default=Path("data/validation/peak_token_pair_features"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/pairwise_delta_reranker"))
    parser.add_argument("--hard-k", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.0001, 0.001, 0.01])
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    discovery_raw = pd.read_csv(args.directional_dir / "discovery_directional_features.csv")
    confirmation_raw = pd.read_csv(args.directional_dir / "confirmation_directional_features.csv")
    discovery, burden, directional, token = augment_features(discovery_raw, args.pair_dir, args.token_dir, "discovery")
    confirmation, _, _, _ = augment_features(confirmation_raw, args.pair_dir, args.token_dir, "confirmation")
    baseline_discovery = retrieval(discovery.assign(score=discovery["dreams_similarity"]), "score")
    baseline_confirmation = retrieval(confirmation.assign(score=confirmation["dreams_similarity"]), "score")
    feature_sets = {
        "dreams_only": ["dreams_similarity"],
        "dreams_plus_raw": ["dreams_similarity"] + RAW_FEATURES,
        "dreams_plus_burden": ["dreams_similarity"] + burden,
        "dreams_plus_token": ["dreams_similarity"] + token,
        "dreams_plus_raw_burden": ["dreams_similarity"] + RAW_FEATURES + burden,
        "dreams_plus_raw_burden_token": ["dreams_similarity"] + RAW_FEATURES + burden + token,
    }
    report = {
        "status": "pairwise_delta_reranker", "hard_negatives_per_query": args.hard_k,
        "objective": "within-query positive score > top DreaMS negative molecule scores",
        "selection": f"{args.folds}-fold formula-isolated discovery CV",
        "baseline": {"discovery": metrics(baseline_discovery), "confirmation": metrics(baseline_confirmation)},
        "models": {},
    }
    coefficients = []
    for index, (name, features) in enumerate(feature_sets.items()):
        best_c, cv = cv_select(discovery, features, args.hard_k, args.c_values, args.folds)
        cv.to_csv(args.output_dir / f"{name}_cv.csv", index=False)
        scaler, model, examples = fit_ranker(discovery, features, args.hard_k, best_c)
        query = retrieval(score(confirmation, features, scaler, model), "score")
        query.to_csv(args.output_dir / f"{name}_confirmation_queries.csv", index=False)
        result = metrics(query)
        result.update({
            "selected_C": best_c, "training_comparisons": examples,
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
        for feature, coefficient in zip(features, model.coef_[0]):
            coefficients.append({"model": name, "feature": feature, "standardized_coefficient": coefficient})
    pd.DataFrame(coefficients).to_csv(args.output_dir / "coefficients.csv", index=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
