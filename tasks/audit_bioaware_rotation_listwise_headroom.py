#!/usr/bin/env python
"""Formula-OOF headroom audit for rotation-level BioAware listwise learning.

Unlike the aggregated v3 ledger, this audit retains every held-out seed
rotation during training.  Predictions are aggregated back to one score per
real query/candidate before any retrieval metric is computed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


BASE = [
    "spectral_score", "decoder_score_scaled", "rule_jaccard_idf",
    "sparse_rule_overlap", "known_edge_best_bottleneck",
    "predicted_edge_best_bottleneck", "smn_best_bottleneck", "rt_score",
]
CONTEXT = [
    "raw_network_support", "dependency_corrected_network_support",
    "candidate_specific_network_support", "complete_network_support",
    "direction_supported_network_support", "raw_path_count",
    "complete_path_count", "unique_seed_compounds", "unique_reactions",
    "mean_source_side_completeness", "mean_candidate_specificity",
    "curated_direction_supported_path_count",
    "curated_direction_conflicted_path_count",
    "excluded_identity_noop_path_count",
]
FEATURES = [*BASE, *CONTEXT]
CONFIGS = [
    {"name": "shallow", "num_leaves": 3, "min_child_samples": 30,
     "learning_rate": 0.03, "n_estimators": 120, "reg_lambda": 10.0},
    {"name": "balanced", "num_leaves": 7, "min_child_samples": 20,
     "learning_rate": 0.02, "n_estimators": 220, "reg_lambda": 15.0},
]
EPS = 1e-12


def formula_fold(value: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"rotation-oof-{seed}|{value}".encode()).hexdigest()
    return int(digest[:16], 16) % folds


def fit_ranker(frame: pd.DataFrame, config: dict, seed: int) -> lgb.LGBMRanker:
    ordered = frame.sort_values(["rotation_unit", "candidate_id"]).copy()
    group = ordered.groupby("rotation_unit", sort=False).size().to_numpy()
    label = (
        ordered["candidate_id"].astype(str)
        == ordered["truth_candidate_id"].astype(str)
    ).astype(int).to_numpy()
    rotations_per_query = ordered.groupby("query_id")["rotation_fold"].transform("nunique")
    group_size = ordered.groupby("rotation_unit")["candidate_id"].transform("size")
    sample_weight = 1.0 / (rotations_per_query.to_numpy(float) * group_size.to_numpy(float))
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", random_state=seed,
        verbosity=-1, deterministic=True, force_col_wise=True,
        feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0,
        num_leaves=config["num_leaves"],
        min_child_samples=config["min_child_samples"],
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
        reg_lambda=config["reg_lambda"], reg_alpha=0.5,
    )
    model.fit(ordered[FEATURES], label, group=group, sample_weight=sample_weight)
    return model


def aggregate_predictions(model: lgb.LGBMRanker, frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["rotation_model_score"] = model.predict(scored[FEATURES])
    normalized: list[pd.Series] = []
    for _, group in scored.groupby("rotation_unit", sort=False):
        values = group["rotation_model_score"].to_numpy(float)
        scale = float(np.std(values))
        z = np.zeros(len(group)) if scale <= EPS else (values - np.mean(values)) / scale
        normalized.append(pd.Series(z, index=group.index))
    scored["rotation_model_z"] = pd.concat(normalized).sort_index()
    aggregate = scored.groupby(["query_id", "candidate_id"], sort=False).agg(
        model_z_mean=("rotation_model_z", "mean"),
        model_z_median=("rotation_model_z", "median"),
        context_support_fraction=("raw_network_support", lambda x: float((x > 0).mean())),
        direction_support_fraction=("direction_supported_network_support", lambda x: float((x > 0).mean())),
    ).reset_index()
    static_columns = [
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "spectral_score", "decoder_score_scaled", "rule_jaccard_idf",
        "sparse_rule_overlap", "known_edge_best_bottleneck",
        "predicted_edge_best_bottleneck", "smn_best_bottleneck", "rt_score",
    ]
    static = scored[static_columns].drop_duplicates(["query_id", "candidate_id"])
    return static.merge(aggregate, on=["query_id", "candidate_id"], validate="one_to_one")


def top(group: pd.DataFrame, column: str) -> tuple[str, bool]:
    maximum = float(group[column].max())
    tied = group[np.isclose(group[column], maximum, rtol=0, atol=EPS)]
    return str(tied.sort_values("candidate_id").iloc[0].candidate_id), len(tied) == 1


def evaluate(frame: pd.DataFrame, alpha: float, max_margin: float, min_support: int) -> dict:
    scored = frame.copy()
    scored["fusion"] = scored["spectral_score"] + alpha * scored["model_z_mean"]
    rows: list[dict] = []
    for query_id, group in scored.groupby("query_id", sort=False):
        truth = str(group["truth_candidate_id"].iloc[0])
        baseline, base_unique = top(group, "spectral_score")
        proposed, prop_unique = top(group, "fusion")
        ordered = group.sort_values(["spectral_score", "candidate_id"], ascending=[False, True])
        margin = float(ordered.iloc[0].spectral_score - ordered.iloc[1].spectral_score)
        proposed_row = group[group["candidate_id"].astype(str) == proposed].iloc[0]
        baseline_row = group[group["candidate_id"].astype(str) == baseline].iloc[0]
        support = sum(
            [
                proposed_row.context_support_fraction > baseline_row.context_support_fraction + EPS,
                proposed_row.direction_support_fraction > baseline_row.direction_support_fraction + EPS,
                proposed_row.known_edge_best_bottleneck > baseline_row.known_edge_best_bottleneck + EPS,
                proposed_row.predicted_edge_best_bottleneck > baseline_row.predicted_edge_best_bottleneck + EPS,
                proposed_row.decoder_score_scaled > baseline_row.decoder_score_scaled + EPS,
                proposed_row.rule_jaccard_idf > baseline_row.rule_jaccard_idf + EPS,
                proposed_row.smn_best_bottleneck > baseline_row.smn_best_bottleneck + EPS,
                proposed_row.rt_score > baseline_row.rt_score + EPS,
            ]
        )
        intervene = bool(
            proposed != baseline and prop_unique and margin <= max_margin + EPS
            and support >= min_support
        )
        final = proposed if intervene else baseline
        baseline_correct = bool(base_unique and baseline == truth)
        final_correct = final == truth
        rows.append(
            {
                "query_id": query_id, "truth_formula": group["truth_formula"].iloc[0],
                "baseline_correct": baseline_correct, "final_correct": final_correct,
                "corrected": (not baseline_correct) and final_correct,
                "introduced": baseline_correct and not final_correct,
                "intervene": intervene,
            }
        )
    decisions = pd.DataFrame(rows)
    corrected = int(decisions.corrected.sum())
    introduced = int(decisions.introduced.sum())
    return {
        "alpha": alpha, "maximum_spectral_margin": max_margin,
        "minimum_support_families": min_support,
        "baseline_recall1": float(decisions.baseline_correct.mean()),
        "recall1": float(decisions.final_correct.mean()),
        "delta_recall1": float(decisions.final_correct.mean() - decisions.baseline_correct.mean()),
        "corrected": corrected, "introduced": introduced,
        "risk_weighted_net": corrected - 2 * introduced,
        "interventions": int(decisions.intervene.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger", type=Path,
        default=Path("data/validation/bioaware_typed_candidate_ledger_v1_20260830/candidate_evidence_typed.csv.gz"),
    )
    parser.add_argument(
        "--rotations", type=Path,
        default=Path("data/validation/bioaware_typed_candidate_ledger_v1_20260830/rotation_typed_features.csv.gz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/validation/bioaware_rotation_listwise_headroom_20260830.json"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"fail-closed: output exists: {args.output}")
    ledger = pd.read_csv(args.ledger)
    ledger["decoder_score_scaled"] = (ledger["decoder_score"].clip(-1, 1) + 1) / 2
    rotations = pd.read_csv(args.rotations)
    static = ledger[[
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "spectral_score", "decoder_score_scaled", "rule_jaccard_idf",
        "sparse_rule_overlap", "known_edge_best_bottleneck",
        "predicted_edge_best_bottleneck", "smn_best_bottleneck", "rt_score",
    ]]
    frame = rotations.merge(static, on=["query_id", "candidate_id"], validate="many_to_one")
    frame["rotation_unit"] = frame["rotation_fold"].astype(str) + "|" + frame["query_id"].astype(str)
    if frame[FEATURES].isna().any().any():
        raise RuntimeError("rotation feature matrix contains missing values")
    formula = frame[["truth_formula"]].drop_duplicates().copy()
    formula["outer_fold"] = formula["truth_formula"].map(
        lambda value: formula_fold(str(value), args.folds, args.seed)
    )
    frame = frame.merge(formula, on="truth_formula", validate="many_to_one")
    reports: dict[str, list[dict]] = {}
    for config_index, config in enumerate(CONFIGS):
        parts: list[pd.DataFrame] = []
        for outer_fold in range(args.folds):
            train = frame[frame.outer_fold != outer_fold]
            test = frame[frame.outer_fold == outer_fold]
            model = fit_ranker(train, config, args.seed + 100 * config_index + outer_fold)
            predicted = aggregate_predictions(model, test)
            predicted["outer_fold"] = outer_fold
            parts.append(predicted)
        oof = pd.concat(parts, ignore_index=True)
        if oof.duplicated(["query_id", "candidate_id"]).any() or oof.query_id.nunique() != 117:
            raise RuntimeError("rotation OOF aggregation coverage mismatch")
        reports[config["name"]] = [
            evaluate(oof, alpha, margin, support)
            for alpha, margin, support in itertools.product(
                [0.01, 0.025, 0.05, 0.10, 0.20],
                [0.025, 0.05, 0.10, 0.20, 1.0],
                [1, 2],
            )
        ]
    flat = [dict(model=model, **row) for model, rows in reports.items() for row in rows]
    best = max(flat, key=lambda row: (
        row["risk_weighted_net"], row["corrected"], -row["introduced"],
        row["delta_recall1"], -row["interventions"],
    ))
    payload = {
        "status": "bioaware_rotation_listwise_headroom_complete",
        "formal": False,
        "protocol": "formula-OOF rotation-level listwise training; query-level score aggregation; fixed audit matrix",
        "queries": 117,
        "rotation_units": int(frame.rotation_unit.nunique()),
        "candidate_rotation_rows": int(len(frame)),
        "configurations_reported": len(flat),
        "best_development_cell": best,
        "all_cells": flat,
        "contracts": {
            "P2b": "forbidden", "phenotype": "forbidden",
            "formula_OOF": True, "selection_status": "headroom only; nested selection still required",
        },
        "claim_limit": "Fixed development matrix; the best cell is post-outcome and not deployable.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "all_cells"}, indent=2))


if __name__ == "__main__":
    main()
