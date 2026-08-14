"""Train and audit a shallow candidate reranker with the frozen peak panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RAW_FEATURES = [
    "sqrt_cosine", "linear_cosine", "entropy_similarity",
    "intensity_coverage_min", "intensity_coverage_mean",
    "matched_peak_fraction_min", "top10_match_fraction",
    "neutral_loss_sqrt_cosine", "neutral_loss_coverage_min",
    "neutral_loss_coverage_mean", "peak_count_ratio",
]


def fold_for_formula(formula: str, folds: int) -> int:
    digest = hashlib.blake2b(formula.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % folds


def weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby(["formula", "label"])["left"].transform("size").to_numpy(float)
    formula_count = frame["formula"].nunique()
    values = 1.0 / (2.0 * formula_count * counts)
    return values * len(values) / values.sum()


def fit_model(frame: pd.DataFrame, features: list[str], c_value: float):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=5000, random_state=20260813),
    )
    model.fit(frame[features], frame["label"], logisticregression__sample_weight=weights(frame))
    return model


def retrieval(manifest: pd.DataFrame, pairs: pd.DataFrame, score_column: str) -> pd.DataFrame:
    lookup = {
        (int(row.left), int(row.right)): float(getattr(row, score_column))
        for row in pairs[["left", "right", score_column]].itertuples(index=False)
    }
    adjacency: dict[int, list[int]] = {index: [] for index in range(len(manifest))}
    for row in pairs[["left", "right"]].itertuples(index=False):
        adjacency[int(row.left)].append(int(row.right))
        adjacency[int(row.right)].append(int(row.left))
    rows = []
    for query, candidates in adjacency.items():
        if not candidates:
            continue
        query_ik = manifest.at[query, "ik14"]
        positives = [candidate for candidate in candidates if manifest.at[candidate, "ik14"] == query_ik]
        negatives = [candidate for candidate in candidates if manifest.at[candidate, "ik14"] != query_ik]
        if not positives or not negatives:
            continue
        def value(candidate: int) -> float:
            return lookup[(min(query, candidate), max(query, candidate))]
        positive_score = max(value(candidate) for candidate in positives)
        molecule_best: dict[str, float] = {}
        for candidate in negatives:
            key, score = manifest.at[candidate, "ik14"], value(candidate)
            molecule_best[key] = max(score, molecule_best.get(key, -np.inf))
        negative_scores = np.asarray(list(molecule_best.values()), float)
        rank = 1 + int(np.sum(negative_scores >= positive_score))
        rows.append({
            "query_index": query, "ik14": query_ik,
            "formula": manifest.at[query, "formula"], "ring_class": manifest.at[query, "ring_class"],
            "positive_score": positive_score, "best_negative_score": float(negative_scores.max()),
            "margin": positive_score - float(negative_scores.max()),
            "top1": bool(rank == 1), "rank": rank, "mrr": 1.0 / rank,
            "pairwise_accuracy": float(np.mean(positive_score > negative_scores)),
            "negative_molecules": len(negative_scores),
        })
    return pd.DataFrame(rows)


def metrics(query: pd.DataFrame) -> dict[str, float]:
    labels = np.r_[np.ones(len(query)), np.zeros(len(query))]
    scores = np.r_[query["positive_score"], query["best_negative_score"]]
    return {
        "queries": len(query), "molecules": int(query["ik14"].nunique()),
        "formulas": int(query["formula"].nunique()),
        "top1": float(query["top1"].mean()), "mrr": float(query["mrr"].mean()),
        "pairwise_accuracy": float(query["pairwise_accuracy"].mean()),
        "hard_negative_roc_auc": float(roc_auc_score(labels, scores)),
        "mean_margin": float(query["margin"].mean()),
    }


def paired_formula_bootstrap(
    baseline: pd.DataFrame, model: pd.DataFrame, column: str,
    iterations: int, seed: int,
) -> list[float]:
    merged = baseline[["query_index", "formula", column]].merge(
        model[["query_index", column]], on="query_index", suffixes=("_baseline", "_model"),
        validate="one_to_one",
    )
    merged["difference"] = merged[f"{column}_model"].astype(float) - merged[f"{column}_baseline"].astype(float)
    by_formula = merged.groupby("formula")["difference"].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations)
    for i in range(iterations):
        draws[i] = rng.choice(by_formula, len(by_formula), replace=True).mean()
    return np.quantile(draws, [0.025, 0.975]).tolist()


def prepare(pair_dir: Path, panel_dir: Path, split: str, token_dir: Path | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    base = pd.read_csv(pair_dir / f"{split}_pair_features.csv")
    panel = pd.read_csv(panel_dir / f"{split}_panel_pair_features.csv")
    if not np.array_equal(base[["left", "right"]].to_numpy(), panel[["left", "right"]].to_numpy()):
        raise RuntimeError(f"Pair alignment failure: {split}")
    panel_columns = [column for column in panel.columns if column.startswith("panel_")]
    pieces = [base.reset_index(drop=True), panel[panel_columns].reset_index(drop=True)]
    token_columns: list[str] = []
    if token_dir is not None:
        token = pd.read_csv(token_dir / f"{split}_token_pair_features.csv")
        if not np.array_equal(base[["left", "right"]].to_numpy(), token[["left", "right"]].to_numpy()):
            raise RuntimeError(f"Token pair alignment failure: {split}")
        token_columns = [column for column in token if column.startswith("token_")]
        pieces.append(token[token_columns].reset_index(drop=True))
    return pd.concat(pieces, axis=1), panel_columns, token_columns


def cross_validate(
    frame: pd.DataFrame, manifest: pd.DataFrame, features: list[str],
    c_values: list[float], folds: int,
) -> tuple[float, pd.DataFrame]:
    formula_fold = {formula: fold_for_formula(formula, folds) for formula in frame["formula"].unique()}
    records = []
    for c_value in c_values:
        scored_parts = []
        for fold in range(folds):
            held_formulas = {formula for formula, value in formula_fold.items() if value == fold}
            train = frame.loc[~frame["formula"].isin(held_formulas)]
            held = frame.loc[frame["formula"].isin(held_formulas)].copy()
            model = fit_model(train, features, c_value)
            held["reranker_score"] = model.decision_function(held[features])
            scored_parts.append(held)
        scored = pd.concat(scored_parts).sort_index()
        query = retrieval(manifest, scored, "reranker_score")
        records.append({"C": c_value} | metrics(query))
    table = pd.DataFrame(records).sort_values(["top1", "mrr", "C"], ascending=[False, False, True])
    return float(table.iloc[0]["C"]), table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dir", type=Path, default=Path("data/validation/large_observability_residual_audit"))
    parser.add_argument("--panel-dir", type=Path, default=Path("data/validation/frozen_panel_pair_features"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/validation"))
    parser.add_argument("--token-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/frozen_panel_reranker"))
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    discovery, panel_columns, token_columns = prepare(args.pair_dir, args.panel_dir, "discovery", args.token_dir)
    confirmation, _, _ = prepare(args.pair_dir, args.panel_dir, "confirmation", args.token_dir)
    discovery_manifest = pd.read_csv(args.embedding_root / "large_observability_embeddings_discovery" / "manifest.csv")
    confirmation_manifest = pd.read_csv(args.embedding_root / "large_observability_embeddings_confirmation" / "manifest.csv")

    feature_sets = {
        "dreams_only_logistic": ["dreams_similarity"],
        "dreams_plus_raw": ["dreams_similarity"] + RAW_FEATURES,
        "dreams_plus_panel": ["dreams_similarity"] + panel_columns,
        "dreams_plus_raw_plus_panel": ["dreams_similarity"] + RAW_FEATURES + panel_columns,
    }
    if token_columns:
        feature_sets.update({
            "dreams_plus_token": ["dreams_similarity"] + token_columns,
            "dreams_plus_panel_plus_token": ["dreams_similarity"] + panel_columns + token_columns,
            "dreams_plus_raw_plus_panel_plus_token": ["dreams_similarity"] + RAW_FEATURES + panel_columns + token_columns,
        })
    discovery_baseline_pairs = discovery.copy()
    discovery_baseline_pairs["score"] = discovery_baseline_pairs["dreams_similarity"]
    confirmation_baseline_pairs = confirmation.copy()
    confirmation_baseline_pairs["score"] = confirmation_baseline_pairs["dreams_similarity"]
    discovery_baseline = retrieval(discovery_manifest, discovery_baseline_pairs, "score")
    confirmation_baseline = retrieval(confirmation_manifest, confirmation_baseline_pairs, "score")
    discovery_baseline.to_csv(args.output_dir / "discovery_dreams_queries.csv", index=False)
    confirmation_baseline.to_csv(args.output_dir / "confirmation_dreams_queries.csv", index=False)

    report = {
        "status": "frozen_panel_shallow_reranker",
        "training": "formula-balanced logistic regression on discovery formulas only",
        "model_selection": f"{args.folds}-fold formula-isolated discovery CV; confirmation untouched",
        "baseline": {"discovery": metrics(discovery_baseline), "confirmation": metrics(confirmation_baseline)},
        "models": {},
    }
    coefficient_rows = []
    for model_index, (name, features) in enumerate(feature_sets.items()):
        best_c, cv = cross_validate(discovery, discovery_manifest, features, args.c_values, args.folds)
        cv.to_csv(args.output_dir / f"{name}_discovery_cv.csv", index=False)
        model = fit_model(discovery, features, best_c)
        confirmation_scored = confirmation.copy()
        confirmation_scored["reranker_score"] = model.decision_function(confirmation[features])
        query = retrieval(confirmation_manifest, confirmation_scored, "reranker_score")
        query.to_csv(args.output_dir / f"{name}_confirmation_queries.csv", index=False)
        model_metrics = metrics(query)
        model_metrics.update({
            "selected_C": best_c,
            "top1_minus_dreams": model_metrics["top1"] - report["baseline"]["confirmation"]["top1"],
            "top1_formula_bootstrap_ci95": paired_formula_bootstrap(
                confirmation_baseline, query, "top1", args.bootstrap, 20260813 + model_index,
            ),
            "mrr_minus_dreams": model_metrics["mrr"] - report["baseline"]["confirmation"]["mrr"],
            "mrr_formula_bootstrap_ci95": paired_formula_bootstrap(
                confirmation_baseline, query, "mrr", args.bootstrap, 20260913 + model_index,
            ),
        })
        report["models"][name] = model_metrics
        scaler = model.named_steps["standardscaler"]
        classifier = model.named_steps["logisticregression"]
        for feature, coefficient, mean, scale in zip(
            features, classifier.coef_[0], scaler.mean_, scaler.scale_,
        ):
            coefficient_rows.append({
                "model": name, "feature": feature,
                "standardized_coefficient": coefficient,
                "training_mean": mean, "training_scale": scale,
            })
    pd.DataFrame(coefficient_rows).to_csv(args.output_dir / "model_coefficients.csv", index=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
