"""Refit clean-visible action value on exact current-geometry replay outcomes.

For each source, compare full clean+cell features against cell-only and
formula-fold-preserving permuted-clean controls.  Raw ranks and transition
outcomes remain in the replay artifact and are not copied into the training
ledger.  Harm risk uses the conservative maximum of full and cell-only OOF
probabilities.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from build_noise_final_dynamic_direct_ledger import clean_query_features, fit_crossfit, sha256_file
from noise_final_dynamic_direct_core import WeightConfig, build_action_weights


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    validation = ROOT / "data/validation"
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=validation / "g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--embedding-cache", type=Path, default=validation / "g8r_p2_official_embeddings.npz")
    parser.add_argument("--token-dir", type=Path, default=validation / "g8r_noise_final_f1_full_tokens")
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--embedding-projection-dim", type=int, default=32)
    parser.add_argument("--token-projection-dim", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def formula_delta_ci(values: np.ndarray, formulas: np.ndarray, repeats: int, seed: int) -> dict[str, float]:
    frame = pd.DataFrame({"value": np.asarray(values, float), "formula": np.asarray(formulas, str)})
    grouped = frame.groupby("formula", sort=True)["value"].agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        chosen = rng.integers(0, len(sums), len(sums))
        draws[index] = sums[chosen].sum() / counts[chosen].sum()
    return {"mean": float(np.mean(values)), "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975))}


def permute_query_features(features: np.ndarray, folds: np.ndarray, seed: int) -> np.ndarray:
    output = np.empty_like(features)
    rng = np.random.default_rng(seed)
    for fold in sorted(set(map(int, folds))):
        indices = np.flatnonzero(folds == fold)
        output[indices] = features[rng.permutation(indices)]
    return output


def strict_bool(series: pd.Series, name: str) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    values = series.astype(str).str.strip().str.lower()
    if not values.isin({"true", "false", "1", "0"}).all():
        raise RuntimeError(f"{name} is not a strict boolean outcome")
    return values.isin({"true", "1"}).to_numpy(bool)


def main() -> None:
    args = arguments()
    if args.outer_fold not in range(5) or args.bootstrap_resamples < 1000:
        raise ValueError("invalid M2 crossfit parameters")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite M2 crossfit: {args.output_dir}")
    required = [
        args.ledger_dir / "report.json", args.ledger_dir / "training_actions.csv.gz",
        args.replay_dir / "report.json", args.replay_dir / "current_geometry_outcomes.csv.gz",
        args.replay_dir / "current_geometry_embeddings.npz",
        args.graph, args.embedding_cache, args.token_dir / "report.json",
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    ledger_report = json.loads(required[0].read_text(encoding="utf-8"))
    replay_report = json.loads((args.replay_dir / "report.json").read_text(encoding="utf-8"))
    if (ledger_report.get("formal") is not True or replay_report.get("formal") is not True
            or int(ledger_report.get("outer_formula_fold", -1)) != args.outer_fold
            or replay_report.get("pass_to_current_geometry_crossfit") is not True):
        raise RuntimeError("M2 requires matching formal ledger and replay")
    actions = pd.read_csv(args.ledger_dir / "training_actions.csv.gz", low_memory=False)
    outcomes = pd.read_csv(args.replay_dir / "current_geometry_outcomes.csv.gz", low_memory=False)
    if set(actions["action_id"].astype(str)) != set(outcomes["action_id"].astype(str)):
        raise RuntimeError("M2 action/replay membership differs")
    outcome_columns = ["action_id", "paired_advantage", "positive", "harmful"]
    frame = actions.drop(columns=["p_clean", "risk", "lagged_advantage"], errors="ignore").merge(
        outcomes[outcome_columns], on="action_id", how="left", validate="one_to_one",
    )
    if frame[outcome_columns[1:]].isna().any().any():
        raise RuntimeError("M2 current-geometry outcome join is incomplete")

    with np.load(args.graph, allow_pickle=True) as body:
        query_rows = np.asarray(body["query_row"], dtype=np.int64)
        query_formula = np.asarray(body["query_formula"], dtype=str)
    # The predictor must see the same mature E4 geometry whose action labels it
    # models.  Contextual peak tokens remain the frozen label-free cache, while
    # the global clean embedding component is replaced by this replay cache.
    original_embedding_cache = args.embedding_cache
    args.embedding_cache = args.replay_dir / "current_geometry_embeddings.npz"
    clean = clean_query_features(args, query_rows)
    args.embedding_cache = original_embedding_cache
    formula_fold = np.full(len(query_rows), -1, dtype=np.int8)
    for query, fold in frame[["query_index", "formula_fold"]].drop_duplicates().itertuples(index=False):
        formula_fold[int(query)] = int(fold)
    if np.any(formula_fold[frame["query_index"].unique()] < 0):
        raise RuntimeError("M2 query fold mapping is incomplete")
    permuted = permute_query_features(clean, np.where(formula_fold < 0, args.outer_fold, formula_fold), args.seed)

    frame["current_p_clean"] = np.nan
    frame["current_risk"] = np.nan
    branch_reports: dict[str, object] = {}
    for source_index, (source, block) in enumerate(frame.groupby("source", sort=True)):
        index = block.index.to_numpy(np.int64)
        cells = sorted(block["cell_id"].astype(str).unique())
        cell_index = {cell: position for position, cell in enumerate(cells)}
        descriptor = np.zeros((len(block), len(cells)), dtype=np.float32)
        descriptor[np.arange(len(block)), [cell_index[value] for value in block["cell_id"].astype(str)]] = 1.0
        query = block["query_index"].to_numpy(np.int64)
        folds = block["formula_fold"].to_numpy(np.int8)
        formulas = block["formula"].to_numpy(str)
        gain = block["paired_advantage"].to_numpy(np.float32)
        positive = strict_bool(block["positive"], "positive")
        harmful = strict_bool(block["harmful"], "harmful")
        feature_sets = {
            "full": np.concatenate([clean[query], descriptor], axis=1),
            "cell_only": descriptor,
            "permuted_clean": np.concatenate([permuted[query], descriptor], axis=1),
        }
        predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]] = {}
        for offset, (name, values) in enumerate(feature_sets.items()):
            predictions[name] = fit_crossfit(
                values, formulas, folds, gain, positive, harmful,
                args.outer_fold, args.seed + 100 * source_index + offset,
            )
        full, cell, shuffled = predictions["full"], predictions["cell_only"], predictions["permuted_clean"]
        frame.loc[index, "current_p_clean"] = full[1]
        frame.loc[index, "current_risk"] = np.maximum(full[2], cell[2])
        positive_float = positive.astype(float)
        full_brier = (full[1] - positive_float) ** 2
        cell_brier = (cell[1] - positive_float) ** 2
        shuffled_brier = (shuffled[1] - positive_float) ** 2
        branch_reports[str(source)] = {
            "actions": int(len(block)), "queries": int(block["query_index"].nunique()),
            "identities": int(block["identity"].nunique()), "formulas": int(block["formula"].nunique()),
            "positive_harmful_overlap": int(np.sum(positive & harmful)),
            "models": {name: result[3] for name, result in predictions.items()},
            "positive_brier_gain_full_vs_cell_only": formula_delta_ci(
                cell_brier - full_brier, formulas, args.bootstrap_resamples, args.seed + 1000 + source_index,
            ),
            "positive_brier_gain_full_vs_permuted": formula_delta_ci(
                shuffled_brier - full_brier, formulas, args.bootstrap_resamples, args.seed + 2000 + source_index,
            ),
            "risk_probability": "maximum(full clean+cell OOF, cell-only OOF)",
        }
    if frame[["current_p_clean", "current_risk"]].isna().any().any():
        raise RuntimeError("M2 OOF predictions are incomplete")
    training = frame.drop(columns=["paired_advantage", "positive", "harmful"])
    training["p_clean"] = frame["current_p_clean"].to_numpy(np.float32)
    training["risk"] = frame["current_risk"].to_numpy(np.float32)
    training["lagged_advantage"] = outcomes.set_index("action_id").loc[
        training["action_id"], "paired_advantage"
    ].to_numpy(np.float32)
    dynamic, dynamic_report = build_action_weights(training, "dynamic", WeightConfig())
    static, static_report = build_action_weights(training, "static", WeightConfig())
    training["dynamic_weight"] = dynamic["weight"].to_numpy(np.float32)
    training["static_weight"] = static["weight"].to_numpy(np.float32)
    training["dynamic_no_op_weight"] = dynamic["no_op_weight"].to_numpy(np.float32)
    training["static_no_op_weight"] = static["no_op_weight"].to_numpy(np.float32)
    gates = {
        "all_actions_refit": len(training) == len(actions),
        "all_sources_refit": set(branch_reports) == {"N", "P_intensity", "P_transfer"},
        "raw_rank_and_transition_outcomes_not_published": not bool(
            {"clean_rank", "target_rank", "control_rank", "corrected", "introduced", "positive", "harmful"}
            & set(training.columns)
        ),
        "positive_no_op_every_query": bool(
            training.groupby("query_index")["dynamic_no_op_weight"].first().gt(0).all()
        ),
        "P2b_forbidden": True, "P3_not_consumed": True,
    }
    report = {
        "status": "noise_final_dynamic_direct_m2_current_crossfit_complete", "formal": True,
        "outer_formula_fold": args.outer_fold, "actions": int(len(training)),
        "queries": int(training["query_index"].nunique()), "cells": int(training["cell_id"].nunique()),
        "source_crossfit": branch_reports, "dynamic_weight": dynamic_report,
        "static_weight": static_report, "gates": gates,
        "contracts": {
            "current_geometry_labels_only": True, "formula_crossfit": True,
            "cell_only_and_permuted_controls": True, "risk_uses_conservative_upper_model": True,
            "no_op_explicit": True, "P2b": "forbidden", "P3_consumed": False,
        },
        "provenance": {
            "ledger": sha256_file(args.ledger_dir / "training_actions.csv.gz"),
            "replay": sha256_file(args.replay_dir / "current_geometry_outcomes.csv.gz"),
            "current_geometry_embeddings": sha256_file(
                args.replay_dir / "current_geometry_embeddings.npz"
            ),
            "script": sha256_file(Path(__file__)),
        },
        "pass_to_schedule": bool(all(gates.values())),
        "claim_limit": "Current-geometry formula-OOF action weighting; not a trained embedding result.",
    }
    if not report["pass_to_schedule"]:
        raise RuntimeError(f"M2 crossfit gates failed: {gates}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dynamic_direct_m2_", dir=args.output_dir.parent))
    try:
        training.to_csv(staging / "training_actions.csv.gz", index=False, compression="gzip")
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
