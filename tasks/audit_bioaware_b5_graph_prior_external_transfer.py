#!/usr/bin/env python
"""Freeze the B4 graph-prior action and audit transfer on opened external panels.

This is a transfer-development audit, not a blind confirmation.  A single
four-coordinate ranker is fitted once on Full16 after purging the union of all
external truth identities and formulas.  The serialized numeric artifact is
then reloaded and used without sklearn fitting on any external candidate row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binomtest
import sklearn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from develop_bioaware_b1_multisource_action import (  # noqa: E402
    BASELINE_CORRECT_RISK,
    BASELINE_MARGIN_MAX,
    PRIMARY_C,
    PROPOSAL_PROBABILITY_MIN,
    fit_ranker,
    strict_spectral_metadata,
)


FEATURES = [
    "spectral_score",
    "network_member",
    "known_log_degree",
    "known_mass_candidate_fraction",
]
EXPECTED_PANEL_QUERIES = {
    "st001154_author_candidates": 161,
    "st001154_same_formula_10ppm": 150,
    "kgmn200std_hidden_seed": 162,
}


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


def prepare_external_candidates(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    panel: str,
    repeat_column: str | None = None,
) -> pd.DataFrame:
    required_candidates = {
        "query_id", "candidate_id", "spectral_score", "known_log_degree",
        "known_mass_candidate_fraction",
    }
    required_truth = {"query_id", "truth_candidate_id", "truth_formula"}
    missing = required_candidates - set(candidates.columns)
    if missing:
        raise RuntimeError(f"{panel}: candidate columns missing {sorted(missing)}")
    missing = required_truth - set(truth.columns)
    if missing:
        raise RuntimeError(f"{panel}: truth columns missing {sorted(missing)}")
    truth = truth[list(required_truth)].drop_duplicates("query_id")
    local = candidates.merge(truth, on="query_id", how="inner", validate="many_to_one")
    if repeat_column is not None:
        if repeat_column not in local.columns:
            raise RuntimeError(f"{panel}: missing repeat column {repeat_column}")
        local["base_query_id"] = local["query_id"].astype(str)
        local["query_id"] = (
            local["query_id"].astype(str)
            + "::repeat="
            + local[repeat_column].astype(int).astype(str)
        )
    if local.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError(f"{panel}: duplicate query/candidate rows")
    local["network_member"] = (local["known_log_degree"].astype(float) > 0).astype(float)
    local["is_positive"] = local["candidate_id"].astype(str).eq(
        local["truth_candidate_id"].astype(str)
    )
    for query_id, group in local.groupby("query_id", sort=False):
        if len(group) < 2:
            raise RuntimeError(f"{panel}/{query_id}: fewer than two candidates")
        if int(group["is_positive"].sum()) != 1:
            raise RuntimeError(f"{panel}/{query_id}: truth multiplicity is not one")
    local = local.merge(
        strict_spectral_metadata(local), on="query_id", validate="many_to_one"
    )
    local["unit_id"] = panel
    local["source"] = panel
    local["polarity"] = "negative"
    if local[FEATURES].isna().any().any():
        raise RuntimeError(f"{panel}: non-finite graph-prior feature")
    return local


def load_st_panel(candidate_path: Path, query_path: Path, panel: str) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_path)
    truth = pd.read_csv(query_path).rename(columns={"truth_candidate_id": "truth_candidate_id"})
    return prepare_external_candidates(candidates, truth, panel)


def load_kgmn_panel(candidate_path: Path, seed_path: Path, panel: str) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_path)
    truth = pd.read_csv(seed_path).rename(
        columns={"feature_name": "query_id", "ik14": "truth_candidate_id"}
    )
    return prepare_external_candidates(candidates, truth, panel, repeat_column="repeat")


def freeze_artifact(train: pd.DataFrame, path: Path, training_sha256: str) -> dict:
    scaler, model = fit_ranker(train, FEATURES)
    artifact = {
        "status": "bioaware_b5_graph_prior_frozen",
        "features": FEATURES,
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "model_coef": [float(value) for value in model.coef_[0]],
        "model_intercept": 0.0,
        "fit": {
            "C": PRIMARY_C,
            "baseline_correct_risk": BASELINE_CORRECT_RISK,
            "training_queries": int(train.query_id.nunique()),
            "training_identities": int(train.truth_candidate_id.nunique()),
            "training_formulas": int(train.truth_formula.nunique()),
            "sklearn_version": sklearn.__version__,
        },
        "gate": {
            "baseline_margin_max": BASELINE_MARGIN_MAX,
            "proposal_probability_min": PROPOSAL_PROBABILITY_MIN,
        },
        "training_candidates_sha256": training_sha256,
    }
    atomic_json(path, artifact)
    return artifact


def frozen_candidate_scores(frame: pd.DataFrame, artifact: dict) -> np.ndarray:
    if artifact["features"] != FEATURES:
        raise RuntimeError("frozen artifact feature order changed")
    values = frame[FEATURES].to_numpy(float)
    mean = np.asarray(artifact["scaler_mean"], dtype=float)
    scale = np.asarray(artifact["scaler_scale"], dtype=float)
    coef = np.asarray(artifact["model_coef"], dtype=float)
    if values.shape[1] != len(mean) or np.any(scale <= 0):
        raise RuntimeError("invalid frozen scaler")
    return ((values - mean) / scale) @ coef + float(artifact["model_intercept"])


def score_panel(frame: pd.DataFrame, artifact: dict, panel: str) -> pd.DataFrame:
    local = frame.copy()
    local["model_score"] = frozen_candidate_scores(local, artifact)
    rows: list[dict] = []
    gate = artifact["gate"]
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
        baseline_score = float(group.loc[
            group["candidate_id"].astype(str).eq(baseline_candidate), "model_score"
        ].iloc[0])
        probability = float(expit(maximum - baseline_score))
        intervene = bool(
            proposal_unique
            and proposed != baseline_candidate
            and float(group["baseline_gap"].iloc[0]) <= gate["baseline_margin_max"]
            and probability >= gate["proposal_probability_min"]
        )
        direct_correct = bool(proposal_unique and proposed == truth)
        gated_correct = direct_correct if intervene else baseline_correct
        rows.append({
            "panel": panel,
            "query_id": str(query_id),
            "truth_candidate_id": truth,
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_candidate_id": baseline_candidate,
            "proposed_candidate_id": proposed,
            "baseline_correct": baseline_correct,
            "direct_correct": direct_correct,
            "gated_correct": gated_correct,
            "direct_corrected": (not baseline_correct) and direct_correct,
            "direct_introduced": baseline_correct and (not direct_correct),
            "gated_corrected": (not baseline_correct) and gated_correct,
            "gated_introduced": baseline_correct and (not gated_correct),
            "intervene": intervene,
            "proposal_unique": proposal_unique,
            "proposal_probability": probability,
            "baseline_gap": float(group["baseline_gap"].iloc[0]),
            "candidate_count": int(len(group)),
        })
    return pd.DataFrame(rows)


def cluster_bootstrap(
    result: pd.DataFrame,
    outcome: str,
    cluster: str,
    repeats: int,
    seed: int,
) -> dict:
    difference = result[outcome].astype(int) - result["baseline_correct"].astype(int)
    work = result[[cluster]].copy()
    work["difference"] = difference
    grouped = work.groupby(cluster, sort=False)["difference"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    for index in range(repeats):
        selection = rng.integers(0, len(grouped), len(grouped))
        draws[index] = sums[selection].sum() / counts[selection].sum()
    return {
        "mean": float(difference.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(grouped)),
        "resamples": int(repeats),
    }


def transition_summary(
    result: pd.DataFrame,
    outcome: str,
    corrected_column: str,
    introduced_column: str,
    repeats: int,
    seed: int,
) -> dict:
    corrected = int(result[corrected_column].sum())
    introduced = int(result[introduced_column].sum())
    discordant = corrected + introduced
    return {
        "recall1": float(result[outcome].mean()),
        "delta_recall1": float(
            result[outcome].astype(int).sub(result["baseline_correct"].astype(int)).mean()
        ),
        "corrected": corrected,
        "introduced": introduced,
        "risk_weighted_net_lambda2": corrected - 2 * introduced,
        "mcnemar_exact_p": float(
            binomtest(min(corrected, introduced), discordant, 0.5).pvalue
        ) if discordant else 1.0,
        "identity_cluster_bootstrap": cluster_bootstrap(
            result, outcome, "truth_candidate_id", repeats, seed
        ),
        "formula_cluster_bootstrap": cluster_bootstrap(
            result, outcome, "truth_formula", repeats, seed + 1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-author-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_frozen_evaluation_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-author-queries", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_frozen_evaluation_v1/per_query.csv.gz",
    )
    parser.add_argument(
        "--st-formula-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-formula-queries", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz",
    )
    parser.add_argument(
        "--kgmn-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--kgmn-seeds", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    inputs = [
        args.train_candidates, args.st_author_candidates, args.st_author_queries,
        args.st_formula_candidates, args.st_formula_queries,
        args.kgmn_candidates, args.kgmn_seeds,
    ]
    for path in inputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels = {
        "st001154_author_candidates": load_st_panel(
            args.st_author_candidates, args.st_author_queries,
            "st001154_author_candidates",
        ),
        "st001154_same_formula_10ppm": load_st_panel(
            args.st_formula_candidates, args.st_formula_queries,
            "st001154_same_formula_10ppm",
        ),
        "kgmn200std_hidden_seed": load_kgmn_panel(
            args.kgmn_candidates, args.kgmn_seeds, "kgmn200std_hidden_seed"
        ),
    }
    for name, frame in panels.items():
        observed = int(frame.query_id.nunique())
        if observed != EXPECTED_PANEL_QUERIES[name]:
            raise RuntimeError(f"{name}: expected {EXPECTED_PANEL_QUERIES[name]} queries, got {observed}")

    external_identities: set[str] = set()
    external_formulas: set[str] = set()
    for frame in panels.values():
        external_identities.update(frame.truth_candidate_id.astype(str))
        external_formulas.update(frame.truth_formula.astype(str))

    training = pd.read_csv(args.train_candidates)
    required_train = set(FEATURES) | {
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "is_positive", "baseline_correct",
    }
    missing = required_train - set(training.columns)
    if missing:
        raise RuntimeError(f"training candidate columns missing {sorted(missing)}")
    training = training.loc[
        ~training.truth_candidate_id.astype(str).isin(external_identities)
        & ~training.truth_formula.astype(str).isin(external_formulas)
    ].copy()
    train_queries = training[
        ["query_id", "truth_candidate_id", "truth_formula"]
    ].drop_duplicates("query_id")
    identity_overlap = set(train_queries.truth_candidate_id.astype(str)) & external_identities
    formula_overlap = set(train_queries.truth_formula.astype(str)) & external_formulas
    if identity_overlap or formula_overlap:
        raise RuntimeError("external truth purge failed")
    if train_queries.query_id.nunique() < 1000:
        raise RuntimeError("fewer than 1000 Full16 queries remain after external purge")

    artifact_path = args.output_dir / "graph_prior_artifact.json"
    artifact = freeze_artifact(training, artifact_path, sha256(args.train_candidates))
    # Reload from disk before touching any outcome.  This makes the freeze
    # boundary explicit and catches serialization/order errors.
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    panel_reports: dict[str, dict] = {}
    for offset, (name, frame) in enumerate(panels.items()):
        result = score_panel(frame, artifact, name)
        result_path = args.output_dir / f"{name}__transitions.csv.gz"
        result.to_csv(result_path, index=False, compression="gzip")
        panel_reports[name] = {
            "queries": int(len(result)),
            "identities": int(result.truth_candidate_id.nunique()),
            "formulas": int(result.truth_formula.nunique()),
            "baseline_recall1": float(result.baseline_correct.mean()),
            "direct": transition_summary(
                result, "direct_correct", "direct_corrected", "direct_introduced",
                args.bootstrap_resamples, args.seed + 10 * offset,
            ),
            "frozen_gate": transition_summary(
                result, "gated_correct", "gated_corrected", "gated_introduced",
                args.bootstrap_resamples, args.seed + 10 * offset + 2,
            ),
            "gate_interventions": int(result.intervene.sum()),
            "gate_intervention_rate": float(result.intervene.mean()),
            "candidate_feature_shift": {
                feature: {
                    "mean": float(frame[feature].mean()),
                    "std": float(frame[feature].std(ddof=0)),
                }
                for feature in FEATURES
            },
            "transitions_sha256": sha256(result_path),
        }

    direct_large_panels = sum(
        report["direct"]["delta_recall1"] >= 0.03
        for report in panel_reports.values()
    )
    direct_corrected = sum(report["direct"]["corrected"] for report in panel_reports.values())
    direct_introduced = sum(report["direct"]["introduced"] for report in panel_reports.values())
    gates = {
        "external_truth_identity_overlap_zero": len(identity_overlap) == 0,
        "external_truth_formula_overlap_zero": len(formula_overlap) == 0,
        "at_least_two_direct_panels_gain_ge_3pp": direct_large_panels >= 2,
        "pooled_direct_corrected_gt_2x_introduced": direct_corrected > 2 * direct_introduced,
        "frozen_gate_all_panels_nonnegative": all(
            report["frozen_gate"]["delta_recall1"] >= 0
            for report in panel_reports.values()
        ),
        "frozen_gate_any_panel_positive": any(
            report["frozen_gate"]["delta_recall1"] > 0
            for report in panel_reports.values()
        ),
    }
    report = {
        "status": "bioaware_b5_graph_prior_external_transfer_complete",
        "formal": True,
        "opened_transfer_development_only": True,
        "model_fitted_once_before_external_scoring": True,
        "training_after_union_purge": {
            "queries": int(train_queries.query_id.nunique()),
            "identities": int(train_queries.truth_candidate_id.nunique()),
            "formulas": int(train_queries.truth_formula.nunique()),
            "external_identity_overlap": len(identity_overlap),
            "external_formula_overlap": len(formula_overlap),
        },
        "panels": panel_reports,
        "gates": gates,
        "pass_to_calibration_research": bool(
            gates["external_truth_identity_overlap_zero"]
            and gates["external_truth_formula_overlap_zero"]
            and gates["at_least_two_direct_panels_gain_ge_3pp"]
            and gates["pooled_direct_corrected_gt_2x_introduced"]
            and gates["frozen_gate_all_panels_nonnegative"]
        ),
        "frozen_gate_is_validated": bool(gates["frozen_gate_any_panel_positive"]),
        "contracts": {
            "feature_recipe_frozen_from_B4": True,
            "external_outcomes_used_for_model_fit": False,
            "external_truth_used_only_for_training_purge_and_evaluation": True,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "artifact": sha256(artifact_path),
            "script": sha256(Path(__file__)),
            **{path.name: sha256(path) for path in inputs},
        },
        "claim_limit": (
            "Opened external transfer-development audit. Direct ranking headroom and "
            "frozen-gate performance are separate endpoints. It cannot establish blind "
            "generalization, reaction mechanism, shared-embedding improvement, or SOTA."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
