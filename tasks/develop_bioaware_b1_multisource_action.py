#!/usr/bin/env python
"""Test whether BioAware reaction evidence adds value beyond catalog opportunity.

This is an opened-development, action-discovery experiment.  It reuses the
already-built MetDNA3 candidate graphs for four biological sources and both
ion polarities.  No spectra are re-encoded.  The primary comparison is a
reaction-aware low-capacity ranker versus a catalog/opportunity control under
full biological-source leave-one-out validation.

Every held source's truth identities *and formulas* are purged from training.
Candidate identity, truth identity, formula, source, phenotype and P2b scores
are never model inputs.  Ties count against the truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from develop_bioaware_metdna3_negative_loso_ranker import (  # noqa: E402
    aggregate_edge,
    aggregate_known,
)


EXPECTED_SOURCES = ("BV2cell", "Mouse_brain", "Mouse_liver", "NIST_plasma")
EXPECTED_UNITS = tuple(
    f"{source}__{separation}"
    for source in EXPECTED_SOURCES
    for separation in ("hilic", "rplc")
)

# Nested, frozen recipes.  The opportunity model is the primary comparator:
# it can exploit library/network coverage but no reaction-path assignment.
RECIPES: dict[str, list[str]] = {
    "spectral_only": ["spectral_score"],
    "catalog_opportunity": [
        "spectral_score",
        "log_reference_spectra",
        "network_member",
        "known_log_degree",
        "known_mass_candidate_fraction",
    ],
    "reaction_availability": [
        "spectral_score",
        "log_reference_spectra",
        "network_member",
        "known_log_degree",
        "known_mass_candidate_fraction",
        "known_path_present",
        "edge0_present",
    ],
    "reaction_strength": [
        "spectral_score",
        "log_reference_spectra",
        "network_member",
        "known_log_degree",
        "known_mass_candidate_fraction",
        "known_path_present",
        "edge0_present",
        "known_path_per_degree",
        "known_inverse_depth_mean",
        "known_seed_per_degree",
        "edge0_complete_fraction",
        "edge0_bottleneck_mean",
        "edge0_reliability",
    ],
}
REACTION_BLOCK = [
    "known_path_present",
    "edge0_present",
    "known_path_per_degree",
    "known_inverse_depth_mean",
    "known_seed_per_degree",
    "edge0_complete_fraction",
    "edge0_bottleneck_mean",
    "edge0_reliability",
]

PRIMARY_C = 0.1
BASELINE_CORRECT_RISK = 2.0
BASELINE_MARGIN_MAX = 0.05
PROPOSAL_PROBABILITY_MIN = 0.75


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(body, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def source_from_unit(unit: str) -> str:
    source = str(unit).split("__", 1)[0]
    if source not in EXPECTED_SOURCES:
        raise RuntimeError(f"unknown biological source in unit {unit!r}")
    return source


def strict_spectral_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for query_id, group in frame.groupby("query_id", sort=False):
        truth = str(group["truth_candidate_id"].iloc[0])
        positive = group.loc[group["candidate_id"].astype(str).eq(truth)]
        wrong = group.loc[~group["candidate_id"].astype(str).eq(truth)]
        if len(positive) != 1 or wrong.empty:
            raise RuntimeError(f"{query_id}: expected one truth and at least one wrong candidate")
        scores = group["spectral_score"].to_numpy(float)
        maximum = float(np.max(scores))
        top = group.loc[np.isclose(scores, maximum, rtol=0, atol=1e-12)].copy()
        top = top.sort_values("candidate_id", kind="stable")
        ordered = np.sort(scores)[::-1]
        truth_score = float(positive["spectral_score"].iloc[0])
        hardest_wrong = float(wrong["spectral_score"].max())
        rows.append(
            {
                "query_id": str(query_id),
                "baseline_candidate_id": str(top["candidate_id"].iloc[0]),
                "baseline_unique": bool(len(top) == 1),
                "baseline_correct": bool(len(top) == 1 and str(top["candidate_id"].iloc[0]) == truth),
                "baseline_gap": float(ordered[0] - ordered[1]),
                "truth_margin": truth_score - hardest_wrong,
            }
        )
    return pd.DataFrame(rows)


def validate_candidate_frame(frame: pd.DataFrame, label: str) -> None:
    required = {
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "spectral_score", "reference_spectra", "unit_id", "source", "polarity",
        "known_mass_candidate_fraction", "known_path_fraction",
        "known_inverse_depth_mean", "known_log_seed_support_mean",
        "known_log_degree", "edge0_complete_fraction", "edge0_bottleneck_mean",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label}: missing columns {sorted(missing)}")
    if frame.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError(f"{label}: duplicate query/candidate rows")
    if frame["query_id"].astype(str).str.len().eq(0).any():
        raise RuntimeError(f"{label}: empty query identity")
    if frame["candidate_id"].astype(str).str.len().ne(14).any():
        raise RuntimeError(f"{label}: candidate IDs are not IK14")
    for query_id, group in frame.groupby("query_id", sort=False):
        if group["truth_candidate_id"].astype(str).nunique() != 1:
            raise RuntimeError(f"{label}/{query_id}: inconsistent truth identity")
        if group["truth_formula"].astype(str).nunique() != 1:
            raise RuntimeError(f"{label}/{query_id}: inconsistent truth formula")
        truth = str(group["truth_candidate_id"].iloc[0])
        if int(group["candidate_id"].astype(str).eq(truth).sum()) != 1:
            raise RuntimeError(f"{label}/{query_id}: truth candidate multiplicity is not one")
        if len(group) < 2:
            raise RuntimeError(f"{label}/{query_id}: no competing candidate")
    numerical = sorted(set().union(*RECIPES.values()) | set(REACTION_BLOCK))
    for column in numerical:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{label}: {column} contains non-finite values")


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    # Negative-ion frozen tables carry historical baseline columns.  Recompute
    # them for *both* polarities from the harmonised score table so pandas does
    # not create _x/_y aliases and so one strict tie rule governs the study.
    historical_baseline = [
        "baseline_candidate_id", "baseline_unique", "baseline_correct",
        "baseline_gap", "truth_margin", "top_candidate_id",
    ]
    output = output.drop(
        columns=[name for name in historical_baseline if name in output.columns]
    )
    numeric = [
        "spectral_score", "reference_spectra", "known_mass_candidate_fraction",
        "known_path_fraction", "known_inverse_depth_mean",
        "known_log_seed_support_mean", "known_log_degree",
        "edge0_complete_fraction", "edge0_bottleneck_mean",
    ]
    for column in numeric:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    output["log_reference_spectra"] = np.log1p(output["reference_spectra"].clip(lower=0))
    output["network_member"] = (output["known_log_degree"] > 0).astype(float)
    output["known_path_present"] = (output["known_path_fraction"] > 0).astype(float)
    output["edge0_present"] = (output["edge0_complete_fraction"] > 0).astype(float)
    degree_scale = 1.0 + output["known_log_degree"].clip(lower=0)
    output["known_path_per_degree"] = output["known_path_fraction"].clip(lower=0) / degree_scale
    output["known_seed_per_degree"] = (
        output["known_log_seed_support_mean"].clip(lower=0) / degree_scale
    )
    output["edge0_reliability"] = (
        output["edge0_complete_fraction"].clip(lower=0)
        * output["edge0_bottleneck_mean"].clip(lower=0)
        / degree_scale
    )
    output["is_positive"] = output["candidate_id"].astype(str).eq(
        output["truth_candidate_id"].astype(str)
    )
    baseline = strict_spectral_metadata(output)
    output = output.merge(baseline, on="query_id", validate="many_to_one")
    return output


def load_positive(root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    provenance: dict[str, str] = {}
    for unit in EXPECTED_UNITS:
        unit_root = root / unit
        ledger_path = unit_root / "ledger" / "candidate_evidence.csv.gz"
        known_path = unit_root / "paths" / "candidate_paths.csv.gz"
        edge_path = unit_root / "edge_step0" / "candidate_edge_evidence.csv.gz"
        for path in (ledger_path, known_path, edge_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        base = pd.read_csv(ledger_path)[
            ["query_id", "candidate_id", "truth_candidate_id", "truth_formula",
             "spectral_score", "reference_spectra"]
        ].copy()
        known = aggregate_known(known_path)
        edge = aggregate_edge(edge_path, "edge0")
        before = len(base)
        local = base.merge(
            known, on=["query_id", "candidate_id"], how="left", validate="one_to_one"
        ).merge(
            edge, on=["query_id", "candidate_id"], how="left", validate="one_to_one"
        )
        if len(local) != before:
            raise RuntimeError(f"{unit}: feature merge changed candidate count")
        needed = [
            "known_mass_candidate_fraction", "known_path_fraction",
            "known_inverse_depth_mean", "known_log_seed_support_mean",
            "known_log_degree", "edge0_complete_fraction", "edge0_bottleneck_mean",
        ]
        if local[needed].isna().any().any():
            raise RuntimeError(f"{unit}: positive harmonisation produced missing evidence")
        local["unit_id"] = unit
        local["source"] = source_from_unit(unit)
        local["polarity"] = "positive"
        frames.append(local)
        provenance[f"{unit}/ledger"] = sha256(ledger_path)
        provenance[f"{unit}/paths"] = sha256(known_path)
        provenance[f"{unit}/edge_step0"] = sha256(edge_path)
    result = pd.concat(frames, ignore_index=True)
    if result["query_id"].nunique() != 878 or len(result) != 3381:
        raise RuntimeError(
            f"positive protocol changed: rows={len(result)} queries={result.query_id.nunique()}"
        )
    return result, provenance


def load_negative(
    frozen_path: Path,
    path_root: Path,
    edge_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not frozen_path.is_file() or frozen_path.stat().st_size == 0:
        raise FileNotFoundError(frozen_path)
    frozen = pd.read_csv(frozen_path).copy()
    if len(frozen) != 2003 or frozen["query_id"].nunique() != 548:
        raise RuntimeError("negative frozen candidate protocol changed")
    checks: list[pd.DataFrame] = []
    provenance: dict[str, object] = {"frozen_candidates": sha256(frozen_path)}
    for unit in EXPECTED_UNITS:
        known_path = path_root / unit / "candidate_paths.csv.gz"
        edge_path = edge_root / unit / "candidate_edge_evidence.csv.gz"
        for path in (known_path, edge_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        known = aggregate_known(known_path)
        edge = aggregate_edge(edge_path, "edge0")
        checks.append(known.merge(edge, on=["query_id", "candidate_id"], validate="one_to_one"))
        provenance[f"{unit}/paths"] = sha256(known_path)
        provenance[f"{unit}/edge_step0"] = sha256(edge_path)
    replay = pd.concat(checks, ignore_index=True)
    columns = [
        "known_mass_candidate_fraction", "known_path_fraction",
        "known_inverse_depth_mean", "known_log_seed_support_mean", "known_log_degree",
        "edge0_complete_fraction", "edge0_bottleneck_mean",
    ]
    merged = frozen[["query_id", "candidate_id", *columns]].merge(
        replay[["query_id", "candidate_id", *columns]],
        on=["query_id", "candidate_id"], suffixes=("_frozen", "_replay"),
        validate="one_to_one",
    )
    if len(merged) != len(frozen):
        raise RuntimeError("negative raw-evidence replay did not cover frozen candidates")
    maximum_errors: dict[str, float] = {}
    for column in columns:
        error = float(np.max(np.abs(
            merged[f"{column}_frozen"].to_numpy(float)
            - merged[f"{column}_replay"].to_numpy(float)
        )))
        maximum_errors[column] = error
        if error > 1e-12:
            raise RuntimeError(f"negative evidence replay mismatch for {column}: {error}")
    frozen["source"] = frozen["unit_id"].map(source_from_unit)
    frozen["polarity"] = "negative"
    provenance["maximum_replay_errors"] = maximum_errors
    return frozen, provenance


def pairwise_training_rows(
    frame: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    identity_counts = frame[["query_id", "truth_candidate_id"]].drop_duplicates()[
        "truth_candidate_id"
    ].value_counts()
    rows: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    for _, group in frame.groupby("query_id", sort=False):
        positive = group.loc[group["is_positive"]]
        wrong = group.loc[~group["is_positive"]]
        if len(positive) != 1 or wrong.empty:
            raise RuntimeError("invalid pairwise training group")
        delta = positive[features].to_numpy(float)[0] - wrong[features].to_numpy(float)
        identity = str(group["truth_candidate_id"].iloc[0])
        risk = BASELINE_CORRECT_RISK if bool(group["baseline_correct"].iloc[0]) else 1.0
        query_weight = risk / (float(identity_counts[identity]) * len(wrong))
        rows.extend([delta, -delta])
        labels.extend([1] * len(delta) + [0] * len(delta))
        weights.extend([query_weight] * (2 * len(delta)))
    return np.vstack(rows), np.asarray(labels), np.asarray(weights)


def fit_ranker(train: pd.DataFrame, features: list[str]) -> tuple[StandardScaler, LogisticRegression]:
    x, y, weights = pairwise_training_rows(train, features)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=PRIMARY_C,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=2000,
        random_state=20260906,
    ).fit(scaler.transform(x), y, sample_weight=weights)
    return scaler, model


def evaluate_model(
    test: pd.DataFrame,
    scaler: StandardScaler,
    model: LogisticRegression,
    features: list[str],
    held_source: str,
    recipe: str,
) -> pd.DataFrame:
    local = test.copy()
    local["model_score"] = model.decision_function(
        scaler.transform(local[features].to_numpy(float))
    )
    rows: list[dict] = []
    for query_id, group in local.groupby("query_id", sort=False):
        truth = str(group["truth_candidate_id"].iloc[0])
        baseline_candidate = str(group["baseline_candidate_id"].iloc[0])
        baseline_correct = bool(group["baseline_correct"].iloc[0])
        maximum = float(group["model_score"].max())
        proposed_rows = group.loc[
            np.isclose(group["model_score"], maximum, rtol=0, atol=1e-12)
        ].sort_values("candidate_id", kind="stable")
        proposed = str(proposed_rows["candidate_id"].iloc[0])
        proposal_unique = len(proposed_rows) == 1
        baseline_model_score = float(
            group.loc[group["candidate_id"].astype(str).eq(baseline_candidate), "model_score"].iloc[0]
        )
        probability = float(expit(maximum - baseline_model_score))
        intervene = bool(
            proposal_unique
            and proposed != baseline_candidate
            and float(group["baseline_gap"].iloc[0]) <= BASELINE_MARGIN_MAX
            and probability >= PROPOSAL_PROBABILITY_MIN
        )
        gated_correct = bool(proposed == truth) if intervene else baseline_correct
        direct_correct = bool(proposal_unique and proposed == truth)
        rows.append(
            {
                "query_id": str(query_id),
                "unit_id": str(group["unit_id"].iloc[0]),
                "source": held_source,
                "polarity": str(group["polarity"].iloc[0]),
                "truth_candidate_id": truth,
                "truth_formula": str(group["truth_formula"].iloc[0]),
                "recipe": recipe,
                "baseline_candidate_id": baseline_candidate,
                "proposed_candidate_id": proposed,
                "baseline_correct": baseline_correct,
                "direct_correct": direct_correct,
                "gated_correct": gated_correct,
                "corrected": (not baseline_correct) and gated_correct,
                "introduced": baseline_correct and (not gated_correct),
                "intervene": intervene,
                "proposal_unique": proposal_unique,
                "proposal_probability": probability,
                "baseline_gap": float(group["baseline_gap"].iloc[0]),
                "truth_margin": float(group["truth_margin"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def run_source_loso(
    candidates: pd.DataFrame, recipe: str, features: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    outputs: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    for held_source in EXPECTED_SOURCES:
        test = candidates.loc[candidates["source"].eq(held_source)].copy()
        held_identities = set(test["truth_candidate_id"].astype(str))
        held_formulas = set(test["truth_formula"].astype(str))
        train = candidates.loc[
            ~candidates["source"].eq(held_source)
            & ~candidates["truth_candidate_id"].astype(str).isin(held_identities)
            & ~candidates["truth_formula"].astype(str).isin(held_formulas)
        ].copy()
        train_queries = train[["query_id", "truth_candidate_id", "truth_formula"]].drop_duplicates()
        if train_queries["query_id"].nunique() < 150:
            raise RuntimeError(f"{held_source}: fewer than 150 purged training queries")
        if set(train_queries["truth_candidate_id"].astype(str)) & held_identities:
            raise RuntimeError(f"{held_source}: truth-identity leakage")
        if set(train_queries["truth_formula"].astype(str)) & held_formulas:
            raise RuntimeError(f"{held_source}: truth-formula leakage")
        scaler, model = fit_ranker(train, features)
        result = evaluate_model(test, scaler, model, features, held_source, recipe)
        outputs.append(result)
        fold_reports.append(
            {
                "held_source": held_source,
                "train_queries_after_purge": int(train_queries.query_id.nunique()),
                "train_identities_after_purge": int(train_queries.truth_candidate_id.nunique()),
                "train_formulas_after_purge": int(train_queries.truth_formula.nunique()),
                "test_queries": int(result.query_id.nunique()),
                "identity_overlap": 0,
                "formula_overlap": 0,
                "coefficients": {
                    name: float(value)
                    for name, value in zip(features, model.coef_[0], strict=True)
                },
            }
        )
    combined = pd.concat(outputs, ignore_index=True)
    if len(combined) != candidates["query_id"].nunique():
        raise RuntimeError("source-LOSO changed query coverage")
    return combined, fold_reports


def summarize(frame: pd.DataFrame) -> dict:
    corrected = int(frame["corrected"].sum())
    introduced = int(frame["introduced"].sum())
    changed = frame.loc[frame["corrected"] | frame["introduced"]]
    denominator = len(frame)
    discordant = corrected + introduced
    return {
        "queries": denominator,
        "identities": int(frame["truth_candidate_id"].nunique()),
        "formulas": int(frame["truth_formula"].nunique()),
        "baseline_recall1": float(frame["baseline_correct"].mean()),
        "direct_recall1": float(frame["direct_correct"].mean()),
        "gated_recall1": float(frame["gated_correct"].mean()),
        "gated_delta_recall1": float((frame["gated_correct"].astype(int) - frame["baseline_correct"].astype(int)).mean()),
        "corrected": corrected,
        "introduced": introduced,
        "risk_weighted_net_lambda2": corrected - 2 * introduced,
        "interventions": int(frame["intervene"].sum()),
        "intervention_rate": float(frame["intervene"].mean()),
        "corrected_identities": int(frame.loc[frame.corrected, "truth_candidate_id"].nunique()),
        "corrected_formulas": int(frame.loc[frame.corrected, "truth_formula"].nunique()),
        "introduced_identities": int(frame.loc[frame.introduced, "truth_candidate_id"].nunique()),
        "introduced_formulas": int(frame.loc[frame.introduced, "truth_formula"].nunique()),
        "changed_identities": int(changed["truth_candidate_id"].nunique()),
        "mcnemar_exact_p": float(
            binomtest(min(corrected, introduced), discordant, 0.5).pvalue
        ) if discordant else 1.0,
    }


def paired_cluster_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cluster: str,
    repeats: int,
    seed: int,
) -> dict:
    keys = ["query_id", "truth_candidate_id", "truth_formula", "source", "polarity"]
    paired = left[keys + ["gated_correct"]].merge(
        right[keys + ["gated_correct"]], on=keys, suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    paired["difference"] = (
        paired["gated_correct_left"].astype(int) - paired["gated_correct_right"].astype(int)
    )
    grouped = paired.groupby(cluster, sort=False)["difference"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        draw = rng.integers(0, len(grouped), len(grouped))
        values[index] = sums[draw].sum() / counts[draw].sum()
    return {
        "mean": float(paired["difference"].mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "clusters": int(len(grouped)),
        "resamples": int(repeats),
    }


def paired_transition(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    paired = left[["query_id", "gated_correct"]].merge(
        right[["query_id", "gated_correct"]], on="query_id",
        suffixes=("_left", "_right"), validate="one_to_one",
    )
    corrected = int((~paired.gated_correct_right & paired.gated_correct_left).sum())
    introduced = int((paired.gated_correct_right & ~paired.gated_correct_left).sum())
    return {
        "corrected_vs_right": corrected,
        "introduced_vs_right": introduced,
        "net": corrected - introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_v3_v1",
    )
    parser.add_argument(
        "--negative-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_loso_ranker_v4_chemically_filtered/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--negative-path-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_paths_v1",
    )
    parser.add_argument(
        "--negative-edge0-root", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_edge_step0_v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    positive, positive_provenance = load_positive(args.positive_root)
    negative, negative_provenance = load_negative(
        args.negative_candidates, args.negative_path_root, args.negative_edge0_root
    )
    if set(positive.query_id.astype(str)) & set(negative.query_id.astype(str)):
        raise RuntimeError("positive and negative query IDs collide")
    candidates = add_derived_features(pd.concat([positive, negative], ignore_index=True))
    validate_candidate_frame(candidates, "combined")
    queries = candidates[[
        "query_id", "truth_candidate_id", "truth_formula", "unit_id", "source",
        "polarity", "baseline_correct",
    ]].drop_duplicates()
    if len(queries) != 1426:
        raise RuntimeError(f"combined query protocol changed: {len(queries)}")
    if tuple(sorted(queries.source.unique())) != tuple(sorted(EXPECTED_SOURCES)):
        raise RuntimeError("combined source set changed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "harmonized_candidates.csv.gz"
    candidates.to_csv(candidate_path, index=False, compression="gzip")

    transitions: dict[str, pd.DataFrame] = {}
    fold_reports: dict[str, list[dict]] = {}
    recipe_reports: dict[str, dict] = {}
    for offset, (recipe, features) in enumerate(RECIPES.items()):
        print(f"[B1] source-LOSO recipe={recipe} features={len(features)}", flush=True)
        result, folds = run_source_loso(candidates, recipe, features)
        transitions[recipe] = result
        fold_reports[recipe] = folds
        result_path = args.output_dir / f"{recipe}__transitions.csv.gz"
        result.to_csv(result_path, index=False, compression="gzip")
        recipe_reports[recipe] = {
            "features": features,
            "pooled": summarize(result),
            "by_source": {
                source: summarize(result.loc[result.source.eq(source)])
                for source in EXPECTED_SOURCES
            },
            "by_polarity": {
                polarity: summarize(result.loc[result.polarity.eq(polarity)])
                for polarity in ("negative", "positive")
            },
            "vs_spectral_formula_cluster": paired_cluster_bootstrap(
                result, transitions["spectral_only"], "truth_formula",
                args.bootstrap_resamples, args.seed + offset,
            ),
            "transition_vs_spectral": paired_transition(result, transitions["spectral_only"]),
            "folds": folds,
            "transitions_sha256": sha256(result_path),
        }

    full = transitions["reaction_strength"]
    opportunity = transitions["catalog_opportunity"]
    availability = transitions["reaction_availability"]
    primary_formula = paired_cluster_bootstrap(
        full, opportunity, "truth_formula", args.bootstrap_resamples, args.seed + 100
    )
    primary_identity = paired_cluster_bootstrap(
        full, opportunity, "truth_candidate_id", args.bootstrap_resamples, args.seed + 101
    )
    strength_formula = paired_cluster_bootstrap(
        full, availability, "truth_formula", args.bootstrap_resamples, args.seed + 102
    )
    primary_transition = paired_transition(full, opportunity)
    by_source_primary = {}
    for source in EXPECTED_SOURCES:
        left = full.loc[full.source.eq(source)]
        right = opportunity.loc[opportunity.source.eq(source)]
        by_source_primary[source] = {
            **paired_transition(left, right),
            "delta": float(left.gated_correct.mean() - right.gated_correct.mean()),
        }
    by_polarity_primary = {}
    for polarity in ("negative", "positive"):
        left = full.loc[full.polarity.eq(polarity)]
        right = opportunity.loc[opportunity.polarity.eq(polarity)]
        by_polarity_primary[polarity] = {
            **paired_transition(left, right),
            "delta": float(left.gated_correct.mean() - right.gated_correct.mean()),
            "direct_delta": float(left.direct_correct.mean() - right.direct_correct.mean()),
        }

    gates = {
        "reaction_increment_ge_3pp": primary_formula["mean"] >= 0.03,
        "reaction_increment_formula_ci_low_positive": primary_formula["ci_low"] > 0,
        "reaction_increment_identity_ci_low_positive": primary_identity["ci_low"] > 0,
        "reaction_corrected_gt_2x_introduced": (
            primary_transition["corrected_vs_right"]
            > 2 * primary_transition["introduced_vs_right"]
        ),
        "reaction_corrected_identities_ge_25": int(
            full.loc[
                full.gated_correct
                & ~opportunity.set_index("query_id").loc[full.query_id, "gated_correct"].to_numpy(),
                "truth_candidate_id",
            ].nunique()
        ) >= 25,
        "at_least_3_of_4_sources_nonnegative": sum(
            value["delta"] >= 0 for value in by_source_primary.values()
        ) >= 3,
        "worst_source_above_minus_1pp": min(
            value["delta"] for value in by_source_primary.values()
        ) >= -0.01,
    }
    report = {
        "schema_version": "bioaware_b1_multisource_action_v1",
        "status": "bioaware_b1_multisource_action_complete",
        "formal": True,
        "opened_development_data": True,
        "model_type": "low-capacity pairwise logistic ranker",
        "candidate_protocol": {
            "queries": int(queries.query_id.nunique()),
            "candidate_rows": int(len(candidates)),
            "truth_identities": int(queries.truth_candidate_id.nunique()),
            "truth_formulas": int(queries.truth_formula.nunique()),
            "sources": int(queries.source.nunique()),
            "units": int(queries.unit_id.nunique()),
            "polarity_queries": {
                key: int(value) for key, value in queries.polarity.value_counts().to_dict().items()
            },
            "baseline_recall1": float(queries.baseline_correct.mean()),
        },
        "feature_semantics": {
            "catalog_opportunity": "DreaMS score plus reference-count and network coverage/degree controls",
            "reaction_availability": "adds whether an identity-heldout seed path and raw-MS2-complete path exist",
            "reaction_strength": "adds degree-normalised path strength, depth, seed support and raw-MS2 bottleneck",
            "negative_replay_maximum_error": negative_provenance["maximum_replay_errors"],
        },
        "configuration": {
            "C": PRIMARY_C,
            "baseline_correct_risk_weight": BASELINE_CORRECT_RISK,
            "maximum_baseline_gap_for_action": BASELINE_MARGIN_MAX,
            "minimum_pairwise_proposal_probability": PROPOSAL_PROBABILITY_MIN,
            "identity_equal_training_mass": True,
        },
        "recipes": recipe_reports,
        "primary_reaction_strength_vs_catalog_opportunity": {
            "formula_cluster_bootstrap": primary_formula,
            "identity_cluster_bootstrap": primary_identity,
            "transition": primary_transition,
            "by_source": by_source_primary,
            "by_polarity": by_polarity_primary,
        },
        "secondary_strength_vs_availability": {
            "formula_cluster_bootstrap": strength_formula,
            "transition": paired_transition(full, availability),
        },
        "gates": gates,
        "pass_to_falsification": bool(all(gates.values())),
        "contracts": {
            "biological_source_held_out": True,
            "held_truth_identity_and_formula_purged": True,
            "truth_candidate_or_formula_as_feature": False,
            "outcome_used_for_feature_construction": False,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "positive": positive_provenance,
            "negative": negative_provenance,
            "harmonized_candidates_sha256": sha256(candidate_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Opened-development source-LOSO action test. A pass identifies a reaction-specific "
            "candidate-ranking action worth conditioned-null falsification; it is not external "
            "SOTA evidence and does not establish a shared-embedding improvement."
        ),
    }
    report["scientific_pass"] = report["pass_to_falsification"]
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
