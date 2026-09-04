#!/usr/bin/env python
"""Formula-isolated BioAware fusion using within-query evidence consensus.

Version 1 demonstrated that adding heterogeneous raw evidence values to the
DreaMS cosine is ill-scaled: the fitted residual was too small to alter a
candidate ordering.  This version gives every independent evidence family one
normalised vote inside the candidate set, learns only non-negative family
weights, and retains nested formula-group gate selection with abstention.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


RAW_FAMILIES = {
    "decoder": ["decoder_score"],
    "rules": ["rule_jaccard_idf", "sparse_rule_overlap"],
    "known_reaction": ["known_edge_best_bottleneck"],
    "predicted_reaction": ["predicted_edge_best_bottleneck"],
    "structure_network": ["smn_best_bottleneck"],
    "retention_time": ["rt_score"],
}
FAMILY_FEATURES = [f"family_{name}" for name in RAW_FAMILIES]
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def formula_fold(formula: str, folds: int, salt: str) -> int:
    token = hashlib.sha256(f"{salt}|{formula}".encode()).hexdigest()
    return int(token[:16], 16) % folds


def unit_interval(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.zeros(len(values), dtype=float)
    if not finite.any():
        return result
    low = float(np.min(values[finite]))
    high = float(np.max(values[finite]))
    if high - low <= EPS:
        return result
    result[finite] = (values[finite] - low) / (high - low)
    return result


def add_family_features(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("query_id", sort=False):
        group = group.copy()
        for family, columns in RAW_FAMILIES.items():
            normalised = np.stack(
                [unit_interval(group[column].to_numpy(float)) for column in columns], axis=1
            )
            # Multiple measurements of one mechanism remain one vote.
            group[f"family_{family}"] = np.max(normalised, axis=1)
        parts.append(group)
    result = pd.concat(parts, ignore_index=True)
    if result[FAMILY_FEATURES].isna().any().any():
        raise RuntimeError("family normalisation produced missing values")
    return result


def candidate_top(group: pd.DataFrame, column: str) -> tuple[str, bool]:
    maximum = float(group[column].max())
    tied = group[np.isclose(group[column], maximum, rtol=0, atol=1e-12)]
    predicted = str(tied.sort_values("candidate_id").iloc[0].candidate_id)
    return predicted, bool(len(tied) == 1)


def pairwise_training_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectral_delta: list[float] = []
    feature_delta: list[np.ndarray] = []
    weights: list[float] = []
    for _, group in frame.groupby("query_id", sort=False):
        truth_id = str(group["truth_candidate_id"].iloc[0])
        truth = group[group["candidate_id"].astype(str) == truth_id]
        wrong = group[group["candidate_id"].astype(str) != truth_id]
        if len(truth) != 1 or wrong.empty:
            raise RuntimeError("query needs one truth and at least one wrong candidate")
        truth_row = truth.iloc[0]
        query_weight = 1.0 / len(wrong)
        for row in wrong.itertuples(index=False):
            spectral_delta.append(float(truth_row.spectral_score - row.spectral_score))
            feature_delta.append(
                truth_row[FAMILY_FEATURES].to_numpy(float)
                - np.asarray([getattr(row, feature) for feature in FAMILY_FEATURES], dtype=float)
            )
            weights.append(query_weight)
    return np.asarray(spectral_delta), np.stack(feature_delta), np.asarray(weights)


def fit_family_weights(
    frame: pd.DataFrame, temperature: float, l2: float, maximum_weight: float
) -> np.ndarray:
    spectral, delta, sample_weight = pairwise_training_arrays(frame)

    def objective(weight: np.ndarray) -> tuple[float, np.ndarray]:
        margin = spectral + delta @ weight
        loss = np.logaddexp(0.0, -margin / temperature)
        sigmoid = 1.0 / (1.0 + np.exp(np.clip(margin / temperature, -50, 50)))
        value = float(
            np.sum(sample_weight * loss) / np.sum(sample_weight) + l2 * np.sum(weight**2)
        )
        gradient = (
            np.sum(
                sample_weight[:, None] * (-sigmoid[:, None] / temperature) * delta,
                axis=0,
            )
            / np.sum(sample_weight)
            + 2.0 * l2 * weight
        )
        return value, gradient

    result = minimize(
        lambda weight: objective(weight)[0],
        np.zeros(len(FAMILY_FEATURES), dtype=float),
        jac=lambda weight: objective(weight)[1],
        method="L-BFGS-B",
        bounds=[(0.0, maximum_weight)] * len(FAMILY_FEATURES),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"family-weight optimizer failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def score_queries(frame: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    scored = frame.copy()
    scored["fusion_score"] = (
        scored["spectral_score"] + scored[FAMILY_FEATURES].to_numpy(float) @ weights
    )
    rows: list[dict] = []
    for query_id, group in scored.groupby("query_id", sort=False):
        truth_id = str(group["truth_candidate_id"].iloc[0])
        baseline_id, baseline_unique = candidate_top(group, "spectral_score")
        proposed_id, proposed_unique = candidate_top(group, "fusion_score")
        ordered = group.sort_values(["spectral_score", "candidate_id"], ascending=[False, True])
        baseline = group[group["candidate_id"].astype(str) == baseline_id].iloc[0]
        proposed = group[group["candidate_id"].astype(str) == proposed_id].iloc[0]
        support_names = [
            family
            for family in RAW_FAMILIES
            if float(proposed[f"family_{family}"] - baseline[f"family_{family}"]) > EPS
        ] if proposed_id != baseline_id else []
        rows.append({
            "query_id": str(query_id),
            "truth_candidate_id": truth_id,
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_candidate_id": baseline_id,
            "proposed_candidate_id": proposed_id,
            "baseline_correct": bool(baseline_unique and baseline_id == truth_id),
            "proposed_correct": bool(proposed_unique and proposed_id == truth_id),
            "proposed_unique": bool(proposed_unique),
            "spectral_margin": float(ordered.iloc[0].spectral_score - ordered.iloc[1].spectral_score),
            "fusion_advantage": float(proposed.fusion_score - baseline.fusion_score),
            "support_count": int(len(support_names)),
            "support_families": ";".join(support_names),
            "changes_top1": bool(proposed_id != baseline_id),
        })
    return pd.DataFrame(rows)


def apply_gate(predictions: pd.DataFrame, gate: tuple[float, float, int]) -> pd.DataFrame:
    max_margin, min_advantage, min_support = gate
    result = predictions.copy()
    result["intervene"] = (
        result["changes_top1"]
        & result["proposed_unique"]
        & (result["spectral_margin"] <= max_margin + EPS)
        & (result["fusion_advantage"] >= min_advantage - EPS)
        & (result["support_count"] >= min_support)
    )
    result["final_candidate_id"] = np.where(
        result["intervene"], result["proposed_candidate_id"], result["baseline_candidate_id"]
    )
    # Exact fallback is semantic, not merely a displayed candidate ID.  A
    # lexicographically selected truth inside a spectral tie remains incorrect
    # under the strict protocol; otherwise abstention would create free wins.
    result["final_correct"] = np.where(
        result["intervene"], result["proposed_correct"], result["baseline_correct"]
    ).astype(bool)
    result["corrected"] = ~result["baseline_correct"] & result["final_correct"]
    result["introduced"] = result["baseline_correct"] & ~result["final_correct"]
    return result


def select_gate(predictions: pd.DataFrame) -> tuple[float, float, int]:
    configurations = itertools.product(
        [0.025, 0.05, 0.10, 0.20], [0.0, 0.01, 0.025, 0.05], [2, 3, 4]
    )
    candidates: list[tuple[tuple[int, int, int, int], tuple[float, float, int]]] = []
    for gate in configurations:
        evaluated = apply_gate(predictions, gate)
        corrected = int(evaluated["corrected"].sum())
        introduced = int(evaluated["introduced"].sum())
        coverage = int(evaluated["intervene"].sum())
        risk_net = corrected - 2 * introduced
        if corrected > introduced and risk_net > 0:
            candidates.append(((risk_net, corrected, -introduced, -coverage), gate))
    if not candidates:
        return (0.0, float("inf"), len(RAW_FAMILIES) + 1)
    return max(candidates, key=lambda item: item[0])[1]


def inner_oof_predictions(
    train: pd.DataFrame, outer_fold: int, args: argparse.Namespace
) -> pd.DataFrame:
    formulas = train[["truth_formula"]].drop_duplicates().copy()
    formulas["inner_fold"] = formulas["truth_formula"].map(
        lambda value: formula_fold(str(value), 4, f"inner-rank-{args.seed}-{outer_fold}")
    )
    parts: list[pd.DataFrame] = []
    for inner_fold in range(4):
        validation_formulas = set(formulas.loc[formulas["inner_fold"] == inner_fold, "truth_formula"])
        fit = train[~train["truth_formula"].isin(validation_formulas)]
        validation = train[train["truth_formula"].isin(validation_formulas)]
        if validation.empty or fit["query_id"].nunique() < 20:
            continue
        weights = fit_family_weights(fit, args.temperature, args.l2, args.maximum_family_weight)
        parts.append(score_queries(validation, weights))
    if not parts:
        raise RuntimeError(f"outer fold {outer_fold} produced no inner OOF predictions")
    result = pd.concat(parts, ignore_index=True)
    if result["query_id"].duplicated().any() or result["query_id"].nunique() != train["query_id"].nunique():
        raise RuntimeError(f"outer fold {outer_fold} inner OOF coverage mismatch")
    return result


def formula_cluster_bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    grouped = {formula: group for formula, group in frame.groupby("truth_formula", sort=True)}
    formulas = sorted(grouped)
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sampled = rng.choice(formulas, size=len(formulas), replace=True)
        bootstrap = pd.concat([grouped[str(formula)] for formula in sampled], ignore_index=True)
        deltas[index] = float(
            bootstrap["final_correct"].mean() - bootstrap["baseline_correct"].mean()
        )
    return {
        "mean": float(frame["final_correct"].mean() - frame["baseline_correct"].mean()),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "clusters": int(len(formulas)),
        "resamples": int(repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=Path(
        "data/validation/bioaware_candidate_evidence_ledger_v1/candidate_evidence.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/validation/bioaware_rank_consensus_fusion_v2"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--maximum-family-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if not args.ledger.exists():
        raise FileNotFoundError(args.ledger)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ledger = add_family_features(pd.read_csv(args.ledger))
    formula_table = ledger[["truth_formula"]].drop_duplicates().copy()
    formula_table["outer_fold"] = formula_table["truth_formula"].map(
        lambda value: formula_fold(str(value), args.outer_folds, f"outer-rank-{args.seed}")
    )
    ledger = ledger.merge(formula_table, on="truth_formula", how="left", validate="many_to_one")

    outer_predictions: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    for outer_fold in range(args.outer_folds):
        train = ledger[ledger["outer_fold"] != outer_fold].copy()
        test = ledger[ledger["outer_fold"] == outer_fold].copy()
        inner_predictions = inner_oof_predictions(train, outer_fold, args)
        gate = select_gate(inner_predictions)
        weights = fit_family_weights(train, args.temperature, args.l2, args.maximum_family_weight)
        predictions = apply_gate(score_queries(test, weights), gate)
        predictions["outer_fold"] = outer_fold
        outer_predictions.append(predictions)
        fold_reports.append({
            "outer_fold": outer_fold,
            "queries": int(len(predictions)),
            "formulas": int(predictions["truth_formula"].nunique()),
            "inner_oof_changed": int(inner_predictions["changes_top1"].sum()),
            "gate": {
                "maximum_spectral_margin": gate[0],
                "minimum_fusion_advantage": gate[1] if np.isfinite(gate[1]) else None,
                "minimum_support_families": gate[2],
            },
            "weights": dict(zip(FAMILY_FEATURES, map(float, weights), strict=True)),
            "corrected": int(predictions["corrected"].sum()),
            "introduced": int(predictions["introduced"].sum()),
            "interventions": int(predictions["intervene"].sum()),
        })
    result = pd.concat(outer_predictions, ignore_index=True)
    if result["query_id"].duplicated().any() or result["query_id"].nunique() != ledger["query_id"].nunique():
        raise RuntimeError("outer OOF predictions do not cover every query exactly once")
    corrected = int(result["corrected"].sum())
    introduced = int(result["introduced"].sum())
    baseline = float(result["baseline_correct"].mean())
    final = float(result["final_correct"].mean())
    bootstrap = formula_cluster_bootstrap(result, args.bootstrap_resamples, args.seed)
    transitions = output / "query_oof_transitions.csv.gz"
    result.to_csv(transitions, index=False)
    payload = {
        "status": "bioaware_rank_consensus_fusion_development_complete",
        "formal": True,
        "protocol": "nested formula-group OOF; within-query family normalisation; bounded monotone family weights; consensus gate; abstention",
        "queries": int(len(result)),
        "identities": int(result["truth_candidate_id"].nunique()),
        "formulas": int(result["truth_formula"].nunique()),
        "baseline_recall1": baseline,
        "final_recall1": final,
        "delta_recall1": final - baseline,
        "corrected": corrected,
        "introduced": introduced,
        "risk_weighted_net": corrected - 2 * introduced,
        "interventions": int(result["intervene"].sum()),
        "formula_cluster_bootstrap": bootstrap,
        "folds": fold_reports,
        "gates": {
            "corrected_gt_introduced": corrected > introduced,
            "risk_weighted_net_positive": corrected - 2 * introduced > 0,
            "formula_cluster_ci_positive": bootstrap["ci_low"] > 0,
            "ten_point_gain": final - baseline >= 0.10,
        },
        "contracts": {
            "P2b": "forbidden",
            "phenotype": "forbidden",
            "evidence_family": "one normalised vote per independent mechanism",
            "gate_selection": "inner formula OOF only",
            "RP": "sealed",
        },
        "provenance": {
            "ledger_sha256": sha256(args.ledger),
            "transitions_sha256": sha256(transitions),
        },
        "claim_limit": "Consumed-development nested OOF; RP and external validation remain required for a performance claim.",
    }
    atomic_json(output / "report.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
