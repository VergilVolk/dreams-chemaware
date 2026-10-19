#!/usr/bin/env python
"""Adjudicate unresolved-chemistry increment on paired query outcomes.

Every query must be evaluated in all evidence worlds.  Bootstrap resampling is
performed over predeclared biological clusters.  In every draw the full world
is compared with the strongest control in that draw, preventing post-hoc
selection of a weak comparator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FULL_WORLD = "all"
CONTROL_WORLDS = ("known_only", "measurement_matched_known", "unresolved_shuffled_ms2")
REQUIRED_WORLDS = (FULL_WORLD, *CONTROL_WORLDS)


def parse_bool(series: pd.Series, label: str) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not values.isin(mapping).all():
        raise RuntimeError(f"{label} contains invalid booleans")
    return values.map(mapping).astype(bool)


def validate_outcomes(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "query_id", "cluster_id", "source_holdout", "instrument_holdout",
        "world", "truth_type", "answered", "correct", "rank",
    }
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"outcomes missing columns: {sorted(missing)}")
    frame = raw.copy()
    frame["answered"] = parse_bool(frame["answered"], "answered")
    frame["correct"] = parse_bool(frame["correct"], "correct")
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    if not set(frame["world"]).issubset(REQUIRED_WORLDS):
        raise RuntimeError("unrecognized evidence world")
    if set(frame["world"]) != set(REQUIRED_WORLDS):
        raise RuntimeError("all four evidence worlds are required")
    if not set(frame["truth_type"]).issubset({"match", "no_match"}):
        raise RuntimeError("truth_type must be match or no_match")
    if frame.duplicated(["query_id", "world"]).any():
        raise RuntimeError("duplicate query-world outcome")
    world_counts = frame.groupby("query_id")["world"].nunique()
    if (world_counts != len(REQUIRED_WORLDS)).any():
        raise RuntimeError("every query must occur in every evidence world")
    invariant = ["cluster_id", "source_holdout", "instrument_holdout", "truth_type"]
    for column in invariant:
        if (frame.groupby("query_id")[column].nunique() != 1).any():
            raise RuntimeError(f"{column} changes across evidence worlds")
    if (frame["correct"] & ~frame["answered"] & frame["truth_type"].eq("match")).any():
        raise RuntimeError("a match query cannot be correct while abstaining")
    match = frame["truth_type"].eq("match")
    if (match & frame["answered"] & (frame["rank"].isna() | (frame["rank"] < 1))).any():
        raise RuntimeError("answered match queries require rank >= 1")
    if (match & frame["correct"] & (frame["rank"] != 1)).any():
        raise RuntimeError("correct match query must have rank 1")
    if ((~match) & frame["answered"] & frame["correct"]).any():
        raise RuntimeError("answered no-match query cannot be correct")
    expected_no_match_correct = (~match) & (~frame["answered"])
    if not frame.loc[~match, "correct"].equals(expected_no_match_correct.loc[~match]):
        raise RuntimeError("no-match correctness must equal abstention")
    return frame


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    match = frame["truth_type"].eq("match")
    no_match = ~match
    match_n = int(match.sum())
    no_match_n = int(no_match.sum())
    recall1 = float(frame.loc[match, "correct"].mean()) if match_n else float("nan")
    reciprocal = np.where(
        match & frame["answered"] & frame["rank"].notna(),
        1.0 / frame["rank"].fillna(np.inf).to_numpy(float),
        0.0,
    )
    mrr = float(reciprocal[match.to_numpy()].mean()) if match_n else float("nan")
    no_match_accuracy = float(frame.loc[no_match, "correct"].mean()) if no_match_n else float("nan")
    if match_n and no_match_n:
        balanced = (recall1 + no_match_accuracy) / 2
    elif match_n:
        balanced = recall1
    else:
        balanced = no_match_accuracy
    incorrect_answer = frame["answered"] & ~frame["correct"]
    return {
        "queries": int(len(frame)),
        "match_queries": match_n,
        "no_match_queries": no_match_n,
        "coverage": float(frame["answered"].mean()),
        "match_recall1": recall1,
        "match_mrr": mrr,
        "no_match_accuracy": no_match_accuracy,
        "balanced_endpoint": float(balanced),
        "incorrect_answer_rate": float(incorrect_answer.mean()),
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    draws: int,
    seed: int,
    endpoint: str,
) -> dict[str, object]:
    clusters = frame[["cluster_id"]].drop_duplicates()["cluster_id"].tolist()
    if len(clusters) < 3:
        return {"status": "insufficient_clusters", "clusters": len(clusters)}
    by_cluster = {cluster: frame.loc[frame["cluster_id"] == cluster].copy() for cluster in clusters}
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws, dtype=float)
    error_deltas = np.empty(draws, dtype=float)
    winner_counts = {world: 0 for world in CONTROL_WORLDS}
    for index in range(draws):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = []
        for replicate, cluster in enumerate(sampled):
            piece = by_cluster[cluster].copy()
            piece["_bootstrap_cluster"] = replicate
            pieces.append(piece)
        boot = pd.concat(pieces, ignore_index=True)
        world_metrics = {world: metrics(boot.loc[boot["world"] == world]) for world in REQUIRED_WORLDS}
        control = max(CONTROL_WORLDS, key=lambda world: float(world_metrics[world][endpoint]))
        winner_counts[control] += 1
        deltas[index] = float(world_metrics[FULL_WORLD][endpoint]) - float(world_metrics[control][endpoint])
        error_deltas[index] = (
            float(world_metrics[FULL_WORLD]["incorrect_answer_rate"])
            - float(world_metrics[control]["incorrect_answer_rate"])
        )
    return {
        "status": "complete",
        "clusters": len(clusters),
        "draws": draws,
        "delta_vs_drawwise_strongest_control": {
            "median": float(np.median(deltas)),
            "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
            "fraction_positive": float(np.mean(deltas > 0)),
        },
        "incorrect_answer_rate_delta_vs_selected_control": {
            "median": float(np.median(error_deltas)),
            "ci95": [float(np.quantile(error_deltas, 0.025)), float(np.quantile(error_deltas, 0.975))],
        },
        "drawwise_strongest_control_counts": winner_counts,
    }


def adjudicate(
    raw: pd.DataFrame,
    draws: int = 5000,
    seed: int = 20261013,
    endpoint: str = "balanced_endpoint",
) -> dict[str, object]:
    frame = validate_outcomes(raw)
    overall_metrics = {world: metrics(frame.loc[frame["world"] == world]) for world in REQUIRED_WORLDS}
    overall_bootstrap = cluster_bootstrap(frame, draws, seed, endpoint)
    source_results = {}
    for index, source in enumerate(sorted(frame["source_holdout"].unique())):
        subset = frame.loc[frame["source_holdout"] == source]
        source_results[str(source)] = {
            "metrics": {world: metrics(subset.loc[subset["world"] == world]) for world in REQUIRED_WORLDS},
            "bootstrap": cluster_bootstrap(subset, draws, seed + 1000 + index, endpoint),
        }
    instrument_results = {}
    for index, instrument in enumerate(sorted(frame["instrument_holdout"].unique())):
        subset = frame.loc[frame["instrument_holdout"] == instrument]
        instrument_results[str(instrument)] = {
            "metrics": {world: metrics(subset.loc[subset["world"] == world]) for world in REQUIRED_WORLDS},
            "bootstrap": cluster_bootstrap(subset, draws, seed + 2000 + index, endpoint),
        }

    def strict_positive(result: dict[str, object]) -> bool:
        return (
            result.get("status") == "complete"
            and float(result["delta_vs_drawwise_strongest_control"]["ci95"][0]) > 0  # type: ignore[index]
            and float(result["incorrect_answer_rate_delta_vs_selected_control"]["ci95"][1]) <= 0  # type: ignore[index]
        )

    overall_pass = strict_positive(overall_bootstrap)
    source_pass = bool(source_results) and all(strict_positive(value["bootstrap"]) for value in source_results.values())
    instrument_pass = bool(instrument_results) and all(strict_positive(value["bootstrap"]) for value in instrument_results.values())
    gates = {
        "overall_delta_ci_low_gt_zero_at_no_higher_error": overall_pass,
        "every_source_holdout_passes": source_pass,
        "every_instrument_holdout_passes": instrument_pass,
    }
    return {
        "status": "CONTRAST_ATLAS_UNKNOWN_INCREMENT_ADJUDICATED",
        "formal": True,
        "endpoint": endpoint,
        "worlds": list(REQUIRED_WORLDS),
        "query_denominator": int(frame["query_id"].nunique()),
        "cluster_denominator": int(frame["cluster_id"].nunique()),
        "overall_metrics": overall_metrics,
        "overall_cluster_bootstrap": overall_bootstrap,
        "source_holdouts": source_results,
        "instrument_holdouts": instrument_results,
        "gates": gates,
        "pass_unknown_increment": all(gates.values()),
        "claim_limit": "Paired-world retrieval increment only; biological mechanism and molecular identity remain outside this endpoint.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--endpoint", default="balanced_endpoint", choices=["balanced_endpoint", "match_recall1", "match_mrr"])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    outcomes = pd.read_csv(args.query_outcomes)
    report = adjudicate(outcomes, args.draws, args.seed, args.endpoint)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "query_denominator": report["query_denominator"],
        "cluster_denominator": report["cluster_denominator"],
        "gates": report["gates"],
        "pass_unknown_increment": report["pass_unknown_increment"],
    }, indent=2))


if __name__ == "__main__":
    main()
