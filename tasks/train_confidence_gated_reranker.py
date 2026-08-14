"""Select a DreaMS-confidence gate for the pairwise candidate reranker.

All gate thresholds are selected from formula-isolated out-of-fold discovery
predictions.  Confirmation is evaluated only after the gate is frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_frozen_panel_reranker import RAW_FEATURES, metrics, paired_formula_bootstrap
from train_pairwise_delta_reranker import (
    augment_features, fit_ranker, fold_for_formula, score,
)


def query_choices(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    rows = []
    for query, group in frame.groupby("query", sort=False):
        positive = group.loc[group["label"] == 1]
        negative = group.loc[group["label"] == 0]
        if positive.empty or negative.empty:
            continue
        positive_score = float(positive[score_column].max())
        molecule = group.sort_values(score_column, ascending=False).drop_duplicates("candidate_ik14")
        best = molecule.iloc[0]
        top_scores = molecule[score_column].to_numpy(float)
        # This is the deployable confidence signal: the score gap between the
        # first- and second-ranked candidate molecules.  It never uses the
        # identity of the true positive molecule.
        confidence_margin = float(top_scores[0] - top_scores[1])
        negative_scores = negative.groupby("candidate_ik14")[score_column].max().to_numpy(float)
        rank = 1 + int(np.sum(negative_scores >= positive_score))
        rows.append({
            "query_index": int(query), "ik14": group.iloc[0].query_ik14,
            "formula": group.iloc[0].formula,
            "chosen_ik14": best.candidate_ik14,
            "positive_score": positive_score,
            "best_negative_score": float(negative_scores.max()),
            "margin": positive_score - float(negative_scores.max()),
            "confidence_margin": confidence_margin,
            "top1": bool(rank == 1), "rank": rank, "mrr": 1.0 / rank,
            "pairwise_accuracy": float(np.mean(positive_score > negative_scores)),
        })
    return pd.DataFrame(rows)


def oof_score(
    discovery: pd.DataFrame, features: list[str], folds: int,
    hard_k: int, c_value: float,
) -> pd.DataFrame:
    assignment = {formula: fold_for_formula(formula, folds) for formula in discovery["formula"].unique()}
    parts = []
    for fold in range(folds):
        held_formulas = {formula for formula, value in assignment.items() if value == fold}
        train = discovery.loc[~discovery["formula"].isin(held_formulas)]
        held = discovery.loc[discovery["formula"].isin(held_formulas)]
        scaler, model, _ = fit_ranker(train, features, hard_k, c_value)
        parts.append(score(held, features, scaler, model))
    return pd.concat(parts, ignore_index=True)


def gated_queries(baseline: pd.DataFrame, reranked: pd.DataFrame, threshold: float, require_disagreement: bool) -> pd.DataFrame:
    merged = baseline.merge(
        reranked, on=["query_index", "ik14", "formula"], suffixes=("_baseline", "_reranker"),
        validate="one_to_one",
    )
    # Gate only on information available for an unknown query.
    use = merged["confidence_margin_baseline"] <= threshold
    if require_disagreement:
        use &= merged["chosen_ik14_baseline"] != merged["chosen_ik14_reranker"]
    output = pd.DataFrame({
        "query_index": merged["query_index"], "ik14": merged["ik14"], "formula": merged["formula"],
        "gate_used": use,
    })
    for column in ("chosen_ik14", "positive_score", "best_negative_score", "margin", "confidence_margin", "top1", "rank", "mrr", "pairwise_accuracy"):
        output[column] = np.where(use, merged[f"{column}_reranker"], merged[f"{column}_baseline"])
    return output


def select_gate(
    baseline: pd.DataFrame, reranked: pd.DataFrame,
    quantiles: list[float], require_options: list[bool],
) -> tuple[dict, pd.DataFrame]:
    thresholds = sorted(set(float(baseline["confidence_margin"].quantile(q)) for q in quantiles)) + [float("inf")]
    rows = []
    for require in require_options:
        for threshold in thresholds:
            query = gated_queries(baseline, reranked, threshold, require)
            result = metrics(query)
            rows.append({
                "threshold": threshold, "require_disagreement": require,
                "gate_fraction": float(query["gate_used"].mean()),
            } | result)
    table = pd.DataFrame(rows).sort_values(
        ["top1", "mrr", "gate_fraction", "require_disagreement"],
        ascending=[False, False, True, False],
    )
    return table.iloc[0].to_dict(), table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directional-dir", type=Path, default=Path("data/validation/directional_panel_candidate_features"))
    parser.add_argument("--pair-dir", type=Path, default=Path("data/validation/large_observability_residual_audit"))
    parser.add_argument("--token-dir", type=Path, default=Path("data/validation/peak_token_pair_features"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/confidence_gated_reranker"))
    parser.add_argument("--hard-k", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c-value", type=float, default=0.01)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    discovery_raw = pd.read_csv(args.directional_dir / "discovery_directional_features.csv")
    confirmation_raw = pd.read_csv(args.directional_dir / "confirmation_directional_features.csv")
    discovery, burden, _, token = augment_features(discovery_raw, args.pair_dir, args.token_dir, "discovery")
    confirmation, _, _, _ = augment_features(confirmation_raw, args.pair_dir, args.token_dir, "confirmation")
    feature_sets = {
        "raw": ["dreams_similarity"] + RAW_FEATURES,
        "raw_panel": ["dreams_similarity"] + RAW_FEATURES + burden,
        "raw_panel_token": ["dreams_similarity"] + RAW_FEATURES + burden + token,
    }
    discovery["baseline_score"] = discovery["dreams_similarity"]
    confirmation["baseline_score"] = confirmation["dreams_similarity"]
    baseline_oof = query_choices(discovery, "baseline_score")
    baseline_confirmation = query_choices(confirmation, "baseline_score")
    report = {
        "status": "confidence_gated_pairwise_reranker",
        "selection": "formula-isolated discovery OOF predictions only",
        "gate_signal": "DreaMS top1-minus-top2 candidate-molecule score gap (label-free)",
        "baseline_confirmation": metrics(baseline_confirmation), "models": {},
    }
    for index, (name, features) in enumerate(feature_sets.items()):
        oof_pairs = oof_score(discovery, features, args.folds, args.hard_k, args.c_value)
        oof_query = query_choices(oof_pairs, "score")
        selected, sweep = select_gate(baseline_oof, oof_query, args.quantiles, [False, True])
        sweep.to_csv(args.output_dir / f"{name}_discovery_gate_sweep.csv", index=False)
        scaler, model, _ = fit_ranker(discovery, features, args.hard_k, args.c_value)
        confirmation_query = query_choices(score(confirmation, features, scaler, model), "score")
        gated = gated_queries(
            baseline_confirmation, confirmation_query,
            float(selected["threshold"]), bool(selected["require_disagreement"]),
        )
        gated.to_csv(args.output_dir / f"{name}_confirmation_gated_queries.csv", index=False)
        result = metrics(gated)
        result.update({
            "selected_threshold": float(selected["threshold"]),
            "selected_require_disagreement": bool(selected["require_disagreement"]),
            "discovery_selected_gate_fraction": float(selected["gate_fraction"]),
            "confirmation_gate_fraction": float(gated["gate_used"].mean()),
            "top1_minus_dreams": result["top1"] - report["baseline_confirmation"]["top1"],
            "top1_formula_bootstrap_ci95": paired_formula_bootstrap(
                baseline_confirmation, gated, "top1", args.bootstrap, 20260813 + index,
            ),
            "mrr_minus_dreams": result["mrr"] - report["baseline_confirmation"]["mrr"],
            "mrr_formula_bootstrap_ci95": paired_formula_bootstrap(
                baseline_confirmation, gated, "mrr", args.bootstrap, 20260913 + index,
            ),
            "wrong_to_correct": int(((~baseline_confirmation["top1"]) & gated["top1"]).sum()),
            "correct_to_wrong": int((baseline_confirmation["top1"] & (~gated["top1"])).sum()),
        })
        report["models"][name] = result
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
