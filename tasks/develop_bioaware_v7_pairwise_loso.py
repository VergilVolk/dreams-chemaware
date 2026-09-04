#!/usr/bin/env python
"""Nonlinear pairwise BioAware reranking under nested leave-study-out.

The model learns candidate A versus candidate B from deployable spectral,
reaction-path, structure-network, rule, and RT evidence.  Every outer study is
excluded from model and gate selection.  Evidence is normalised only within a
query, which avoids transferring study-specific score scales.  A conservative
gate can always fall back exactly to the strict official DreaMS decision.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier

from develop_bioaware_rank_consensus_fusion import (
    FAMILY_FEATURES,
    RAW_FAMILIES,
    add_family_features,
    unit_interval,
)


UNITS = (
    "BV2cell__hilic", "BV2cell__rplc", "Mouse_brain__hilic",
    "Mouse_brain__rplc", "Mouse_liver__hilic", "Mouse_liver__rplc",
    "NIST_plasma__hilic", "NIST_plasma__rplc",
)
STUDY = {unit: unit.split("__", 1)[0] for unit in UNITS}
RAW_EVIDENCE = tuple(dict.fromkeys(
    column for columns in RAW_FAMILIES.values() for column in columns
))
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_top(group: pd.DataFrame, score: str) -> tuple[str, bool, float]:
    values = group[score].to_numpy(float)
    maximum = float(np.max(values))
    tied = group[np.isclose(values, maximum, rtol=0, atol=1e-12)]
    selected = str(tied.sort_values("candidate_id").iloc[0].candidate_id)
    ordered = np.sort(values)[::-1]
    margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float("inf")
    return selected, len(tied) == 1, margin


def load_ledgers(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    parts: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for unit in UNITS:
        path = root / unit / "ledger" / "candidate_evidence.csv.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {
            "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
            "spectral_score", *RAW_EVIDENCE,
        }
        if missing := required - set(frame.columns):
            raise RuntimeError(f"{path} lacks {sorted(missing)}")
        frame = frame.copy()
        frame["source_query_id"] = frame.query_id.astype(str)
        frame["query_id"] = unit + "|" + frame.source_query_id
        frame["unit_id"] = unit
        frame["study_id"] = STUDY[unit]
        frame["is_hilic"] = float(unit.endswith("__hilic"))
        parts.append(frame)
        hashes[unit] = sha256(path)
    ledger = pd.concat(parts, ignore_index=True)
    if ledger[["query_id", "candidate_id"]].duplicated().any():
        raise RuntimeError("candidate rows overlap after namespacing")
    ledger = add_family_features(ledger)
    normalised: list[pd.DataFrame] = []
    for _, group in ledger.groupby("query_id", sort=False):
        group = group.copy()
        for column in RAW_EVIDENCE:
            group[f"norm_{column}"] = unit_interval(group[column].to_numpy(float))
        group["norm_reference_spectra"] = unit_interval(
            np.log1p(group.get("reference_spectra", pd.Series(np.ones(len(group)))).to_numpy(float))
        )
        normalised.append(group)
    ledger = pd.concat(normalised, ignore_index=True)
    return ledger, hashes


DELTA_SOURCE = (
    "spectral_score", *(f"norm_{column}" for column in RAW_EVIDENCE),
    "norm_reference_spectra", *FAMILY_FEATURES,
)
PAIR_FEATURES = (
    *(f"delta_{column}" for column in DELTA_SOURCE),
    "absolute_spectral_delta", "is_hilic",
)


def pair_vector(first: pd.Series, second: pd.Series, is_hilic: float) -> np.ndarray:
    delta = [float(first[column] - second[column]) for column in DELTA_SOURCE]
    return np.asarray([*delta, abs(delta[0]), float(is_hilic)], dtype=np.float32)


def training_pairs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for _, group in frame.groupby("query_id", sort=False):
        truth_id = str(group.truth_candidate_id.iloc[0])
        truth = group[group.candidate_id.astype(str) == truth_id]
        wrong = group[group.candidate_id.astype(str) != truth_id]
        if len(truth) != 1 or wrong.empty:
            raise RuntimeError("each query needs one truth and at least one wrong candidate")
        truth_row = truth.iloc[0]
        weight = 0.5 / len(wrong)
        for _, wrong_row in wrong.iterrows():
            vector = pair_vector(truth_row, wrong_row, float(group.is_hilic.iloc[0]))
            vectors.extend((vector, -vector.copy()))
            # Context features do not change when reversing pair orientation.
            vectors[-1][-2] = vector[-2]
            vectors[-1][-1] = vector[-1]
            labels.extend((1, 0))
            weights.extend((weight, weight))
    return np.stack(vectors), np.asarray(labels), np.asarray(weights, dtype=float)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    max_leaf_nodes: int
    min_samples_leaf: int
    l2: float


CONFIGS = (
    ModelConfig("stump_safe", 3, 30, 10.0),
    ModelConfig("shallow_safe", 7, 30, 10.0),
    ModelConfig("shallow_flexible", 7, 20, 3.0),
    ModelConfig("moderate_safe", 15, 30, 10.0),
)


def fit_model(frame: pd.DataFrame, config: ModelConfig, seed: int) -> HistGradientBoostingClassifier:
    x, y, weight = training_pairs(frame)
    monotonic = np.asarray([1] * len(DELTA_SOURCE) + [0, 0], dtype=int)
    model = HistGradientBoostingClassifier(
        learning_rate=.05,
        max_iter=120,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2,
        monotonic_cst=monotonic,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(x, y, sample_weight=weight)
    return model


def score_queries(frame: pd.DataFrame, model: HistGradientBoostingClassifier) -> pd.DataFrame:
    rows: list[dict] = []
    for query_id, group in frame.groupby("query_id", sort=False):
        group = group.reset_index(drop=True).copy()
        n = len(group)
        tournament = np.zeros(n, dtype=float)
        for first, second in itertools.combinations(range(n), 2):
            forward = pair_vector(group.iloc[first], group.iloc[second], float(group.is_hilic.iloc[0]))
            reverse = -forward.copy()
            reverse[-2] = forward[-2]
            reverse[-1] = forward[-1]
            p_forward = float(model.predict_proba(forward[None, :])[0, 1])
            p_reverse = float(model.predict_proba(reverse[None, :])[0, 1])
            probability = .5 * (p_forward + (1.0 - p_reverse))
            probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
            edge = float(logit(probability))
            tournament[first] += edge
            tournament[second] -= edge
        group["pairwise_score"] = tournament / max(n - 1, 1)
        truth_id = str(group.truth_candidate_id.iloc[0])
        baseline_id, baseline_unique, spectral_margin = strict_top(group, "spectral_score")
        proposed_id, proposed_unique, proposed_margin = strict_top(group, "pairwise_score")
        baseline = group[group.candidate_id.astype(str) == baseline_id].iloc[0]
        proposed = group[group.candidate_id.astype(str) == proposed_id].iloc[0]
        support = sum(
            float(proposed[feature] - baseline[feature]) > EPS for feature in FAMILY_FEATURES
        ) if proposed_id != baseline_id else 0
        rows.append({
            "query_id": str(query_id),
            "truth_candidate_id": truth_id,
            "truth_formula": str(group.truth_formula.iloc[0]),
            "baseline_candidate_id": baseline_id,
            "proposed_candidate_id": proposed_id,
            "baseline_correct": bool(baseline_unique and baseline_id == truth_id),
            "proposed_correct": bool(proposed_unique and proposed_id == truth_id),
            "proposed_unique": bool(proposed_unique),
            "spectral_margin": spectral_margin,
            "pairwise_margin": proposed_margin,
            "support_count": int(support),
            "changes_top1": bool(proposed_id != baseline_id),
        })
    return pd.DataFrame(rows)


Gate = tuple[float, float, int]


def apply_gate(predictions: pd.DataFrame, gate: Gate) -> pd.DataFrame:
    max_spectral_margin, min_pairwise_margin, min_support = gate
    result = predictions.copy()
    result["intervene"] = (
        result.changes_top1 & result.proposed_unique
        & (result.spectral_margin <= max_spectral_margin + EPS)
        & (result.pairwise_margin >= min_pairwise_margin - EPS)
        & (result.support_count >= min_support)
    )
    result["final_candidate_id"] = np.where(
        result.intervene, result.proposed_candidate_id, result.baseline_candidate_id
    )
    result["final_correct"] = np.where(
        result.intervene, result.proposed_correct, result.baseline_correct
    ).astype(bool)
    result["corrected"] = ~result.baseline_correct & result.final_correct
    result["introduced"] = result.baseline_correct & ~result.final_correct
    return result


def select_gate(predictions: pd.DataFrame) -> Gate:
    gates = itertools.product(
        (0.025, 0.05, 0.10, 0.20), (0.0, 0.25, 0.50, 1.0), (1, 2, 3)
    )
    candidates: list[tuple[tuple[int, int, int, int], Gate]] = []
    for gate in gates:
        result = apply_gate(predictions, gate)
        corrected = int(result.corrected.sum())
        introduced = int(result.introduced.sum())
        intervention = int(result.intervene.sum())
        risk = corrected - 2 * introduced
        if corrected > introduced and risk > 0:
            candidates.append(((risk, corrected, -introduced, -intervention), gate))
    if not candidates:
        return 0.0, float("inf"), len(FAMILY_FEATURES) + 1
    return max(candidates, key=lambda item: item[0])[1]


def inner_oof(train: pd.DataFrame, config: ModelConfig, seed: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    studies = sorted(train.study_id.unique())
    if len(studies) != 3:
        raise RuntimeError(f"outer training must contain three studies, found {studies}")
    for index, heldout in enumerate(studies):
        fit = train[train.study_id != heldout]
        validation = train[train.study_id == heldout]
        model = fit_model(fit, config, seed + index)
        prediction = score_queries(validation, model)
        prediction["inner_study"] = heldout
        parts.append(prediction)
    result = pd.concat(parts, ignore_index=True)
    if result.query_id.duplicated().any() or set(result.query_id) != set(train.query_id.unique()):
        raise RuntimeError("inner OOF coverage mismatch")
    return result


def metrics(frame: pd.DataFrame) -> dict:
    corrected = int(frame.corrected.sum())
    introduced = int(frame.introduced.sum())
    discordant = corrected + introduced
    return {
        "queries": int(len(frame)),
        "formulas": int(frame.truth_formula.nunique()),
        "baseline_recall1": float(frame.baseline_correct.mean()),
        "recall1": float(frame.final_correct.mean()),
        "delta_recall1": float(frame.final_correct.mean() - frame.baseline_correct.mean()),
        "corrected": corrected,
        "introduced": introduced,
        "risk_weighted_net_lambda2": corrected - 2 * introduced,
        "interventions": int(frame.intervene.sum()),
        "mcnemar_exact_p": float(binomtest(min(corrected, introduced), discordant, .5).pvalue) if discordant else 1.0,
    }


def cluster_bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    groups = {str(k): g for k, g in frame.groupby("truth_formula", sort=True)}
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        draw = rng.choice(keys, len(keys), replace=True)
        sample = pd.concat([groups[str(key)] for key in draw], ignore_index=True)
        values[index] = float(sample.final_correct.mean() - sample.baseline_correct.mean())
    return {
        "mean": float(frame.final_correct.mean() - frame.baseline_correct.mean()),
        "ci_low": float(np.quantile(values, .025)),
        "ci_high": float(np.quantile(values, .975)),
        "clusters": len(keys),
        "resamples": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/validation/bioaware_metdna3_external_v3_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_v7_pairwise_loso_v1"))
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")

    ledger, hashes = load_ledgers(args.root)
    predictions: list[pd.DataFrame] = []
    fold_reports: dict[str, dict] = {}
    for outer_index, outer_study in enumerate(sorted(ledger.study_id.unique())):
        train = ledger[ledger.study_id != outer_study].copy()
        test = ledger[ledger.study_id == outer_study].copy()
        trials = []
        for config in CONFIGS:
            oof = inner_oof(train, config, args.seed + outer_index * 100)
            gate = select_gate(oof)
            evaluated = apply_gate(oof, gate)
            result = metrics(evaluated)
            trials.append((
                (result["risk_weighted_net_lambda2"], result["corrected"], -result["introduced"]),
                config, gate, result,
            ))
        _, config, gate, inner_result = max(trials, key=lambda item: item[0])
        model = fit_model(train, config, args.seed + outer_index * 1000)
        scored = apply_gate(score_queries(test, model), gate)
        scored["outer_study"] = outer_study
        seen_formula = set(train.truth_formula.astype(str))
        scored["formula_seen_in_training_studies"] = scored.truth_formula.astype(str).isin(seen_formula)
        predictions.append(scored)
        fold_reports[outer_study] = {
            **metrics(scored),
            "selected_config": config.__dict__,
            "selected_gate": {
                "maximum_spectral_margin": gate[0],
                "minimum_pairwise_margin": None if not np.isfinite(gate[1]) else gate[1],
                "minimum_support_families": gate[2],
            },
            "inner_oof_selected": inner_result,
        }

    pooled = pd.concat(predictions, ignore_index=True)
    if pooled.query_id.duplicated().any() or set(pooled.query_id) != set(ledger.query_id.unique()):
        raise RuntimeError("outer-study OOF coverage mismatch")
    overall = metrics(pooled)
    bootstrap = cluster_bootstrap(pooled, args.bootstrap_resamples, args.seed)
    unseen = pooled[~pooled.formula_seen_in_training_studies]
    gates = {
        "all_four_studies_present": len(fold_reports) == 4,
        "global_formula_cluster_ci_positive": bootstrap["ci_low"] > 0,
        "corrected_gt_introduced": overall["corrected"] > overall["introduced"],
        "risk_weighted_net_positive": overall["risk_weighted_net_lambda2"] > 0,
        "every_outer_study_nonnegative": all(v["delta_recall1"] >= 0 for v in fold_reports.values()),
        "unseen_formula_nonnegative": unseen.empty or metrics(unseen)["delta_recall1"] >= 0,
    }
    args.output_dir.mkdir(parents=True)
    transitions = args.output_dir / "query_oof_transitions.csv.gz"
    pooled.to_csv(transitions, index=False)
    report = {
        "status": "bioaware_v7_pairwise_loso_complete",
        "formal": True,
        "protocol": "nested leave-study-out nonlinear pairwise candidate ranking with exact DreaMS fallback",
        "overall": overall,
        "global_formula_cluster_bootstrap": bootstrap,
        "unseen_formula": metrics(unseen) if not unseen.empty else None,
        "outer_studies": fold_reports,
        "gates": gates,
        "pass": bool(all(gates.values())),
        "contracts": {
            "outer_study_used_for_model_or_gate": False,
            "phenotype": "forbidden",
            "P2b": "forbidden",
            "strict_ties": "count against truth",
            "fallback": "exact official DreaMS correctness state",
            "claim_type": "cross-study OOF development, not final blind test",
        },
        "provenance": {
            "ledgers": hashes,
            "transitions_sha256": sha256(transitions),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "Positive LOSO transfer is not an untouched blind or matched-author SOTA result.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
