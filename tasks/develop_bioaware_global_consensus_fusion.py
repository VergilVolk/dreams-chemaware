#!/usr/bin/env python
"""Nested formula-OOF BioAware fusion with leave-query-out global consensus.

Repeated measurements of one molecular formula are useful deployment context,
but a formula may contain multiple isomers.  Therefore peer evidence is a soft,
candidate-specific residual computed from *other* queries only, never a hard
same-identity or one-candidate-per-formula constraint.
"""
from __future__ import annotations

import argparse
import json
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.develop_bioaware_rank_consensus_fusion import (
    EPS,
    FAMILY_FEATURES,
    RAW_FAMILIES,
    add_family_features,
    atomic_json,
    candidate_top,
    formula_cluster_bootstrap,
    formula_fold,
    sha256,
)


PEER_FEATURES = [feature.replace("family_", "peer_family_") for feature in FAMILY_FEATURES]
ALL_FEATURES = FAMILY_FEATURES + PEER_FEATURES


def add_leave_query_out_peer_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for feature in PEER_FEATURES:
        result[feature] = 0.0
    for _, formula_group in result.groupby("truth_formula", sort=False):
        query_count = formula_group["query_id"].nunique()
        if query_count < 2:
            continue
        # Candidate sets can differ slightly.  Missing peer candidates contribute
        # no evidence rather than an imputed negative vote.
        for query_id, query_group in formula_group.groupby("query_id", sort=False):
            peers = formula_group[formula_group["query_id"] != query_id]
            peer_means = peers.groupby("candidate_id", sort=False)[FAMILY_FEATURES].mean()
            for row_index, candidate_id in zip(query_group.index, query_group["candidate_id"], strict=True):
                if candidate_id not in peer_means.index:
                    continue
                result.loc[row_index, PEER_FEATURES] = peer_means.loc[candidate_id].to_numpy(float)
    if result[ALL_FEATURES].isna().any().any():
        raise RuntimeError("peer consensus produced missing values")
    return result


def pairwise_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectral_delta: list[float] = []
    feature_delta: list[np.ndarray] = []
    sample_weights: list[float] = []
    for _, group in frame.groupby("query_id", sort=False):
        truth_id = str(group["truth_candidate_id"].iloc[0])
        truth = group[group["candidate_id"].astype(str) == truth_id]
        wrong = group[group["candidate_id"].astype(str) != truth_id]
        if len(truth) != 1 or wrong.empty:
            raise RuntimeError("query needs one truth and at least one wrong candidate")
        truth_row = truth.iloc[0]
        for row in wrong.itertuples(index=False):
            spectral_delta.append(float(truth_row.spectral_score - row.spectral_score))
            feature_delta.append(
                truth_row[ALL_FEATURES].to_numpy(float)
                - np.asarray([getattr(row, feature) for feature in ALL_FEATURES], dtype=float)
            )
            sample_weights.append(1.0 / len(wrong))
    return np.asarray(spectral_delta), np.stack(feature_delta), np.asarray(sample_weights)


def fit_weights(
    frame: pd.DataFrame,
    temperature: float,
    l2: float,
    maximum_local_weight: float,
    maximum_peer_weight: float,
) -> np.ndarray:
    spectral, delta, sample_weight = pairwise_arrays(frame)

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

    bounds = [(0.0, maximum_local_weight)] * len(FAMILY_FEATURES) + [
        (0.0, maximum_peer_weight)
    ] * len(PEER_FEATURES)
    result = minimize(
        lambda weight: objective(weight)[0],
        np.zeros(len(ALL_FEATURES), dtype=float),
        jac=lambda weight: objective(weight)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"global-consensus optimizer failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def score_queries(frame: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    scored = frame.copy()
    scored["fusion_score"] = scored["spectral_score"] + scored[ALL_FEATURES].to_numpy(float) @ weights
    rows: list[dict] = []
    for query_id, group in scored.groupby("query_id", sort=False):
        truth_id = str(group["truth_candidate_id"].iloc[0])
        baseline_id, baseline_unique = candidate_top(group, "spectral_score")
        proposed_id, proposed_unique = candidate_top(group, "fusion_score")
        ordered = group.sort_values(["spectral_score", "candidate_id"], ascending=[False, True])
        baseline = group[group["candidate_id"].astype(str) == baseline_id].iloc[0]
        proposed = group[group["candidate_id"].astype(str) == proposed_id].iloc[0]
        local_support: list[str] = []
        peer_support: list[str] = []
        if proposed_id != baseline_id:
            for family in RAW_FAMILIES:
                local = f"family_{family}"
                peer = f"peer_family_{family}"
                if float(proposed[local] - baseline[local]) > EPS:
                    local_support.append(family)
                if float(proposed[peer] - baseline[peer]) > EPS:
                    peer_support.append(family)
        independent = sorted(set(local_support) | set(peer_support))
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
            "support_count": int(len(independent)),
            "peer_support_count": int(len(set(peer_support))),
            "support_families": ";".join(independent),
            "peer_support_families": ";".join(sorted(set(peer_support))),
            "changes_top1": bool(proposed_id != baseline_id),
        })
    return pd.DataFrame(rows)


def apply_gate(predictions: pd.DataFrame, gate: tuple[float, float, int, int]) -> pd.DataFrame:
    max_margin, min_advantage, min_support, min_peer_support = gate
    result = predictions.copy()
    result["intervene"] = (
        result["changes_top1"]
        & result["proposed_unique"]
        & (result["spectral_margin"] <= max_margin + EPS)
        & (result["fusion_advantage"] >= min_advantage - EPS)
        & (result["support_count"] >= min_support)
        & (result["peer_support_count"] >= min_peer_support)
    )
    result["final_candidate_id"] = np.where(
        result["intervene"], result["proposed_candidate_id"], result["baseline_candidate_id"]
    )
    result["final_correct"] = result["final_candidate_id"] == result["truth_candidate_id"]
    result["corrected"] = ~result["baseline_correct"] & result["final_correct"]
    result["introduced"] = result["baseline_correct"] & ~result["final_correct"]
    return result


def select_gate(predictions: pd.DataFrame) -> tuple[float, float, int, int]:
    configurations = itertools.product(
        [0.025, 0.05, 0.10, 0.20],
        [0.0, 0.01, 0.025, 0.05],
        [2, 3, 4],
        [0, 1, 2],
    )
    candidates: list[tuple[tuple[int, int, int, int], tuple[float, float, int, int]]] = []
    for gate in configurations:
        evaluated = apply_gate(predictions, gate)
        corrected = int(evaluated["corrected"].sum())
        introduced = int(evaluated["introduced"].sum())
        coverage = int(evaluated["intervene"].sum())
        risk_net = corrected - 2 * introduced
        if corrected > introduced and risk_net > 0:
            candidates.append(((risk_net, corrected, -introduced, -coverage), gate))
    if not candidates:
        return (0.0, float("inf"), len(RAW_FAMILIES) + 1, len(RAW_FAMILIES) + 1)
    return max(candidates, key=lambda item: item[0])[1]


def inner_oof(train: pd.DataFrame, outer_fold: int, args: argparse.Namespace) -> pd.DataFrame:
    formulas = train[["truth_formula"]].drop_duplicates().copy()
    formulas["inner_fold"] = formulas["truth_formula"].map(
        lambda value: formula_fold(str(value), 4, f"inner-global-{args.seed}-{outer_fold}")
    )
    parts: list[pd.DataFrame] = []
    for inner_fold in range(4):
        heldout = set(formulas.loc[formulas["inner_fold"] == inner_fold, "truth_formula"])
        fit = train[~train["truth_formula"].isin(heldout)]
        validation = train[train["truth_formula"].isin(heldout)]
        if validation.empty or fit["query_id"].nunique() < 20:
            continue
        weights = fit_weights(
            fit,
            args.temperature,
            args.l2,
            args.maximum_local_weight,
            args.maximum_peer_weight,
        )
        parts.append(score_queries(validation, weights))
    if not parts:
        raise RuntimeError(f"outer fold {outer_fold} produced no inner OOF predictions")
    result = pd.concat(parts, ignore_index=True)
    if result["query_id"].duplicated().any() or result["query_id"].nunique() != train["query_id"].nunique():
        raise RuntimeError(f"outer fold {outer_fold} inner OOF coverage mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=Path(
        "data/validation/bioaware_candidate_evidence_ledger_v1/candidate_evidence.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/validation/bioaware_global_consensus_fusion_v3"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--maximum-local-weight", type=float, default=1.0)
    parser.add_argument("--maximum-peer-weight", type=float, default=1.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if not args.ledger.exists():
        raise FileNotFoundError(args.ledger)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ledger = add_leave_query_out_peer_features(add_family_features(pd.read_csv(args.ledger)))
    formula_table = ledger[["truth_formula"]].drop_duplicates().copy()
    formula_table["outer_fold"] = formula_table["truth_formula"].map(
        lambda value: formula_fold(str(value), args.outer_folds, f"outer-global-{args.seed}")
    )
    ledger = ledger.merge(formula_table, on="truth_formula", how="left", validate="many_to_one")

    all_predictions: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    for outer_fold in range(args.outer_folds):
        train = ledger[ledger["outer_fold"] != outer_fold].copy()
        test = ledger[ledger["outer_fold"] == outer_fold].copy()
        inner = inner_oof(train, outer_fold, args)
        gate = select_gate(inner)
        weights = fit_weights(
            train,
            args.temperature,
            args.l2,
            args.maximum_local_weight,
            args.maximum_peer_weight,
        )
        predictions = apply_gate(score_queries(test, weights), gate)
        predictions["outer_fold"] = outer_fold
        all_predictions.append(predictions)
        fold_reports.append({
            "outer_fold": outer_fold,
            "queries": int(len(predictions)),
            "formulas": int(predictions["truth_formula"].nunique()),
            "gate": {
                "maximum_spectral_margin": gate[0],
                "minimum_fusion_advantage": gate[1] if np.isfinite(gate[1]) else None,
                "minimum_support_families": gate[2],
                "minimum_peer_support_families": gate[3],
            },
            "weights": dict(zip(ALL_FEATURES, map(float, weights), strict=True)),
            "corrected": int(predictions["corrected"].sum()),
            "introduced": int(predictions["introduced"].sum()),
            "interventions": int(predictions["intervene"].sum()),
        })
    result = pd.concat(all_predictions, ignore_index=True)
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
        "status": "bioaware_global_consensus_fusion_development_complete",
        "formal": True,
        "protocol": "nested formula-group OOF; local family evidence plus leave-query-out peer consensus; risk gate; abstention",
        "queries": int(len(result)),
        "identities": int(result["truth_candidate_id"].nunique()),
        "formulas": int(result["truth_formula"].nunique()),
        "multi_query_formulas": int(
            (ledger[["query_id", "truth_formula"]].drop_duplicates().groupby("truth_formula").size() > 1).sum()
        ),
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
            "peer_context": "same-formula other queries only; current query excluded",
            "hard_identity_or_uniqueness_constraint": False,
            "gate_selection": "inner formula OOF only",
            "RP": "sealed",
        },
        "provenance": {
            "ledger_sha256": sha256(args.ledger),
            "transitions_sha256": sha256(transitions),
        },
        "claim_limit": "Transductive consumed-development OOF. Peer context requires multiple query features and is not a single-spectrum identity model.",
    }
    atomic_json(output / "report.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
