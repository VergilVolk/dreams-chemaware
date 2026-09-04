#!/usr/bin/env python
"""Cross-unit development of a risk-controlled negative-ion BioAware ranker.

This is deliberately a low-capacity pairwise model.  It combines the official
DreaMS cosine with outcome-blind, identity-heldout MetDNA/KGMN-style path
features and raw-MS2 validation of those paths.  Each of the eight MetDNA3
external units is held out in turn.  The primary gate is frozen in code and is
not selected from held-out outcomes.

The eight units have already been opened during protocol repair, so this is a
development/transfer audit, not a new external SOTA test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "spectral_score",
    "known_mass_candidate_fraction",
    "known_path_fraction",
    "known_inverse_depth_mean",
    "known_log_seed_support_mean",
    "known_log_degree",
    "edge0_complete_fraction",
    "edge0_bottleneck_mean",
    "edge1_complete_fraction",
    "edge1_bottleneck_mean",
    "predicted_edge_increment",
]
PRIMARY_C = 0.1
PRIMARY_BASELINE_MARGIN_MAX = 0.05
PRIMARY_PROPOSAL_PROBABILITY_MIN = 0.75
RISK_WEIGHT_BASELINE_CORRECT = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_known(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    available = frame["path_available"].astype(bool).to_numpy()
    depth = pd.to_numeric(frame["minimum_depth"], errors="coerce").to_numpy(float)
    frame["inverse_depth"] = np.where(available & np.isfinite(depth), 1.0 / depth, 0.0)
    frame["log_seed_support"] = np.log1p(frame["shortest_seed_count"].astype(float))
    frame["log_degree"] = np.log1p(frame["candidate_degree"].astype(float))
    return frame.groupby(["query_id", "candidate_id"], sort=False).agg(
        known_mass_candidate_fraction=("mass_candidate", "mean"),
        known_path_fraction=("path_available", "mean"),
        known_inverse_depth_mean=("inverse_depth", "mean"),
        known_log_seed_support_mean=("log_seed_support", "mean"),
        known_log_degree=("log_degree", "mean"),
    ).reset_index()


def aggregate_edge(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["maximum_depth"].eq(2)].copy()
    complete = frame["complete_ms2_paths"].astype(float) > 0
    bottleneck = pd.to_numeric(frame["best_bottleneck"], errors="coerce")
    frame["complete"] = complete.astype(float)
    frame["bottleneck_zero"] = bottleneck.fillna(0.0).astype(float)
    return frame.groupby(["query_id", "candidate_id"], sort=False).agg(**{
        f"{prefix}_complete_fraction": ("complete", "mean"),
        f"{prefix}_bottleneck_mean": ("bottleneck_zero", "mean"),
    }).reset_index()


def build_candidate_table(
    benchmark_dir: Path, path_root: Path, edge0_root: Path, edge1_root: Path
) -> tuple[pd.DataFrame, dict[str, str]]:
    scores_path = benchmark_dir / "candidate_scores.csv.gz"
    queries_path = benchmark_dir / "queries.csv.gz"
    transitions_path = benchmark_dir / "transitions.csv.gz"
    scores = pd.read_csv(scores_path)
    # Candidate scores already carry the frozen truth identity/formula.  The
    # query table contributes only the unit key; retaining duplicate truth
    # columns would silently create pandas ``_x/_y`` aliases.
    queries = pd.read_csv(queries_path)[["query_id", "unit_id"]]
    transitions = pd.read_csv(transitions_path)[
        ["query_id", "baseline_correct", "top_candidate_id"]
    ]
    frames = []
    provenance = {
        "scores": sha256(scores_path), "queries": sha256(queries_path),
        "transitions": sha256(transitions_path),
    }
    for unit in sorted(queries["unit_id"].unique()):
        known_path = path_root / str(unit) / "candidate_paths.csv.gz"
        edge0_path = edge0_root / str(unit) / "candidate_edge_evidence.csv.gz"
        edge1_path = edge1_root / str(unit) / "candidate_edge_evidence.csv.gz"
        for path in (known_path, edge0_path, edge1_path):
            if not path.exists():
                raise FileNotFoundError(path)
        local_scores = scores[scores["unit_id"].eq(unit)].copy()
        local = local_scores.merge(
            aggregate_known(known_path), on=["query_id", "candidate_id"], how="left",
            validate="one_to_one",
        ).merge(
            aggregate_edge(edge0_path, "edge0"), on=["query_id", "candidate_id"],
            how="left", validate="one_to_one",
        ).merge(
            aggregate_edge(edge1_path, "edge1"), on=["query_id", "candidate_id"],
            how="left", validate="one_to_one",
        )
        provenance[f"{unit}_known"] = sha256(known_path)
        provenance[f"{unit}_edge0"] = sha256(edge0_path)
        provenance[f"{unit}_edge1"] = sha256(edge1_path)
        frames.append(local)
    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates.merge(queries, on=["query_id", "unit_id"], validate="many_to_one")
    candidates = candidates.merge(transitions, on="query_id", validate="many_to_one")
    candidates["predicted_edge_increment"] = (
        candidates["edge1_bottleneck_mean"] - candidates["edge0_bottleneck_mean"]
    )
    for feature in FEATURES:
        candidates[feature] = pd.to_numeric(candidates[feature], errors="coerce").fillna(0.0)
    candidates["is_positive"] = candidates["candidate_id"].astype(str).eq(
        candidates["truth_candidate_id"].astype(str)
    )
    if candidates.groupby("query_id")["is_positive"].sum().ne(1).any():
        raise RuntimeError("each query must contain exactly one truth candidate")
    return candidates, provenance


def pairwise_training_rows(
    frame: pd.DataFrame, features: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = FEATURES if features is None else features
    rows: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for _, group in frame.groupby("query_id", sort=False):
        positive = group[group["is_positive"]]
        negative = group[~group["is_positive"]]
        if len(positive) != 1 or negative.empty:
            raise RuntimeError("invalid candidate group")
        delta = positive[features].to_numpy(float)[0] - negative[features].to_numpy(float)
        query_weight = (
            RISK_WEIGHT_BASELINE_CORRECT if bool(group["baseline_correct"].iloc[0]) else 1.0
        ) / len(negative)
        rows.extend([delta, -delta])
        labels.extend([1] * len(delta) + [0] * len(delta))
        weights.extend([query_weight] * (2 * len(delta)))
    return np.vstack(rows), np.asarray(labels), np.asarray(weights)


def strict_top(group: pd.DataFrame, column: str) -> tuple[str, bool]:
    maximum = float(group[column].max())
    top = group[np.isclose(group[column], maximum, rtol=0, atol=1e-12)]
    candidate = str(top.sort_values("candidate_id").iloc[0]["candidate_id"])
    return candidate, len(top) == 1


def spectral_top_gap(group: pd.DataFrame) -> float:
    values = np.sort(group["spectral_score"].to_numpy(float))[::-1]
    if len(values) < 2:
        raise RuntimeError("query lacks a competing candidate")
    return float(values[0] - values[1])


def evaluate_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    heldout_unit: str,
    features: list[str] | None = None,
    require_raw_step0_edge: bool = True,
    maximum_baseline_margin: float = PRIMARY_BASELINE_MARGIN_MAX,
    minimum_proposal_probability: float = PRIMARY_PROPOSAL_PROBABILITY_MIN,
) -> tuple[pd.DataFrame, dict]:
    features = FEATURES if features is None else features
    x, y, weights = pairwise_training_rows(train, features)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=PRIMARY_C, fit_intercept=False, solver="lbfgs", max_iter=2000,
        random_state=20260901,
    ).fit(scaler.transform(x), y, sample_weight=weights)
    local = test.copy()
    local["model_score"] = model.decision_function(
        scaler.transform(local[features].to_numpy(float))
    )
    rows = []
    train_truth = set(train["truth_candidate_id"].astype(str))
    train_formula = set(train["truth_formula"].astype(str))
    for query_id, group in local.groupby("query_id", sort=False):
        truth = str(group["truth_candidate_id"].iloc[0])
        baseline = str(group["top_candidate_id"].iloc[0])
        proposed, unique = strict_top(group, "model_score")
        baseline_correct = bool(group["baseline_correct"].iloc[0])
        # Deployment-visible confidence only.  The frozen benchmark's `margin`
        # is truth-minus-hardest-wrong and is therefore forbidden here.
        baseline_margin = spectral_top_gap(group)
        proposed_row = group[group["candidate_id"].astype(str).eq(proposed)].iloc[0]
        baseline_row = group[group["candidate_id"].astype(str).eq(baseline)].iloc[0]
        probability = float(expit(float(proposed_row.model_score - baseline_row.model_score)))
        raw_edge_validated = bool(
            proposed_row.edge0_complete_fraction > 0
            and proposed_row.edge0_bottleneck_mean > 0
        )
        intervene = bool(
            unique and proposed != baseline
            and baseline_margin <= maximum_baseline_margin
            and probability >= minimum_proposal_probability
            and (raw_edge_validated or not require_raw_step0_edge)
        )
        final = proposed if intervene else baseline
        final_correct = final == truth
        rows.append({
            "query_id": query_id, "unit_id": heldout_unit,
            "truth_candidate_id": truth, "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_candidate_id": baseline, "proposed_candidate_id": proposed,
            "final_candidate_id": final, "baseline_correct": baseline_correct,
            "final_correct": final_correct,
            "corrected": (not baseline_correct) and final_correct,
            "introduced": baseline_correct and (not final_correct),
            "delta": int(final_correct) - int(baseline_correct),
            "intervene": intervene, "proposal_unique": unique,
            "proposal_probability": probability,
            "baseline_margin": baseline_margin,
            "raw_edge_validated": raw_edge_validated,
            "truth_identity_unseen_in_training_units": truth not in train_truth,
            "truth_formula_unseen_in_training_units": str(group["truth_formula"].iloc[0]) not in train_formula,
        })
    result = pd.DataFrame(rows)
    return result, {
        "heldout_unit": heldout_unit, "train_queries": int(train["query_id"].nunique()),
        "test_queries": int(result.shape[0]),
        "coefficients": {feature: float(value) for feature, value in zip(features, model.coef_[0])},
    }


def summarize(frame: pd.DataFrame) -> dict:
    corrected = int(frame["corrected"].sum())
    introduced = int(frame["introduced"].sum())
    discordant = corrected + introduced
    return {
        "queries": int(len(frame)),
        "baseline_recall1": float(frame["baseline_correct"].mean()),
        "final_recall1": float(frame["final_correct"].mean()),
        "delta_recall1": float(frame["delta"].mean()),
        "corrected": corrected, "introduced": introduced,
        "risk_weighted_net_lambda2": corrected - 2 * introduced,
        "interventions": int(frame["intervene"].sum()),
        "mcnemar_exact_p": float(binomtest(min(corrected, introduced), discordant, 0.5).pvalue) if discordant else 1.0,
    }


def formula_bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    grouped = frame.groupby("truth_formula", sort=False)["delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, float)
    for index in range(repeats):
        draw = rng.integers(0, len(grouped), len(grouped))
        values[index] = sums[draw].sum() / counts[draw].sum()
    return {
        "mean": float(frame["delta"].mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "formulas": int(len(grouped)), "resamples": int(repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/validation/bioaware_metdna3_external_negative_dreams_v1"))
    parser.add_argument("--path-root", type=Path, default=Path("data/validation/bioaware_metdna3_external_negative_paths_v1"))
    parser.add_argument("--edge0-root", type=Path, default=Path("data/validation/bioaware_metdna3_external_negative_edge_step0_v1"))
    parser.add_argument("--edge1-root", type=Path, default=Path("data/validation/bioaware_metdna3_external_negative_edge_step1_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_metdna3_external_negative_loso_ranker_v3_identity_purged"))
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    candidates, provenance = build_candidate_table(
        args.benchmark_dir, args.path_root, args.edge0_root, args.edge1_root
    )
    results = []
    fold_reports = []
    for unit in sorted(candidates["unit_id"].unique()):
        test = candidates[candidates["unit_id"].eq(unit)]
        test_truth_identities = set(test["truth_candidate_id"].astype(str))
        train = candidates[
            (~candidates["unit_id"].eq(unit))
            & (~candidates["truth_candidate_id"].astype(str).isin(test_truth_identities))
        ]
        if set(train["truth_candidate_id"].astype(str)) & test_truth_identities:
            raise RuntimeError(f"truth-identity purge failed for {unit}")
        result, fold = evaluate_fold(train, test, str(unit))
        fold["test_truth_identities"] = int(len(test_truth_identities))
        fold["training_truth_identity_overlap"] = 0
        fold["result"] = summarize(result)
        results.append(result)
        fold_reports.append(fold)
        print(f"[negative LOSO] {unit}: {fold['result']}", flush=True)
    transitions = pd.concat(results, ignore_index=True)
    pooled = summarize(transitions)
    novel_identity = transitions[transitions["truth_identity_unseen_in_training_units"]]
    novel_formula = transitions[transitions["truth_formula_unseen_in_training_units"]]
    report = {
        "status": "bioaware_metdna3_negative_loso_ranker_complete",
        "formal": True,
        "protocol": "eight-unit LOSO with all test truth IK14 purged from training; identity-heldout rotation features; fixed high-precision gate",
        "features": FEATURES,
        "configuration": {
            "C": PRIMARY_C,
            "baseline_correct_training_weight": RISK_WEIGHT_BASELINE_CORRECT,
            "maximum_baseline_margin": PRIMARY_BASELINE_MARGIN_MAX,
            "minimum_pairwise_proposal_probability": PRIMARY_PROPOSAL_PROBABILITY_MIN,
            "requires_raw_step0_edge_validation": True,
        },
        "pooled": pooled,
        "formula_cluster_bootstrap": formula_bootstrap(
            transitions, args.bootstrap_resamples, 20260901
        ),
        "folds": fold_reports,
        "unseen_truth_identity_subgroup": summarize(novel_identity) if len(novel_identity) else None,
        "unseen_truth_formula_subgroup": summarize(novel_formula) if len(novel_formula) else None,
        "gates": {
            "corrected_gt_introduced": pooled["corrected"] > pooled["introduced"],
            "risk_weighted_net_positive": pooled["risk_weighted_net_lambda2"] > 0,
            "formula_cluster_ci_positive": False,
            "all_units_nonnegative": all(fold["result"]["delta_recall1"] >= 0 for fold in fold_reports),
        },
        "contracts": {
            "test_unit_fit_or_threshold_tuning": False,
            "test_truth_identity_seen_as_training_positive": False,
            "deployment_gate_uses_top1_minus_top2_not_truth_margin": True,
            "candidate_identity_as_feature": False,
            "truth_or_formula_as_feature": False,
            "phenotype": "forbidden", "P2b": "forbidden",
            "shared_embedding_changed": False,
        },
        "provenance": provenance,
        "claim_limit": "Cross-unit development result on the already-opened external units. A new identity-disjoint external cohort is required for SOTA or confirmatory claims.",
    }
    report["gates"]["formula_cluster_ci_positive"] = report["formula_cluster_bootstrap"]["ci_low"] > 0
    report["pass_to_new_external_validation"] = all(report["gates"].values())
    args.output_dir.mkdir(parents=True)
    candidate_path = args.output_dir / "candidate_features.csv.gz"
    transition_path = args.output_dir / "query_transitions.csv.gz"
    candidates.to_csv(candidate_path, index=False, compression="gzip")
    transitions.to_csv(transition_path, index=False, compression="gzip")
    report["provenance"]["candidate_features"] = sha256(candidate_path)
    report["provenance"]["query_transitions"] = sha256(transition_path)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
