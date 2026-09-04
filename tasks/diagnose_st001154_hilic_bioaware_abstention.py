#!/usr/bin/env python
"""Postmortem for the opened ST001154 HILIC-negative BioAware evaluation.

This script is diagnostic only.  It replays the already-frozen artifact, then
decomposes proposal quality, individual gates, and feature-distribution drift.
It never fits a model or selects a new threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation.bioaware_negative_expert import FrozenNegativeBioAwareExpert  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_top(group: pd.DataFrame, column: str) -> tuple[str, bool]:
    maximum = float(group[column].max())
    top = group[np.isclose(group[column], maximum, atol=1e-12, rtol=0)]
    return str(top.sort_values("candidate_id").iloc[0]["candidate_id"]), len(top) == 1


def top_gap(group: pd.DataFrame) -> float:
    scores = np.sort(group["spectral_score"].to_numpy(float))[::-1]
    if len(scores) < 2:
        raise RuntimeError("each diagnostic query must contain at least two candidates")
    return float(scores[0] - scores[1])


def stable_sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0))))


def build_proposals(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    expert: FrozenNegativeBioAwareExpert,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"query_id", "candidate_id", "spectral_score", *expert.feature_names}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise RuntimeError(f"candidate features missing columns: {missing}")
    if candidates.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("duplicate query/candidate rows")
    truth = truth[["query_id", "truth_candidate_id"]].drop_duplicates()
    if truth.duplicated("query_id").any():
        raise RuntimeError("truth table has conflicting query rows")
    frame = candidates.copy()
    frame["network_score"] = expert.score(frame[list(expert.feature_names)].to_numpy(float))
    truth_lookup = dict(zip(truth["query_id"].astype(str), truth["truth_candidate_id"].astype(str), strict=True))
    rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    for query_id, group in frame.groupby("query_id", sort=False):
        query_id = str(query_id)
        if query_id not in truth_lookup:
            raise RuntimeError(f"missing truth for query {query_id}")
        truth_id = truth_lookup[query_id]
        baseline, baseline_unique = unique_top(group, "spectral_score")
        proposal, proposal_unique = unique_top(group, "network_score")
        if truth_id not in set(group["candidate_id"].astype(str)):
            raise RuntimeError(f"truth candidate absent for query {query_id}")
        baseline_row = group[group["candidate_id"].astype(str).eq(baseline)].iloc[0]
        proposal_row = group[group["candidate_id"].astype(str).eq(proposal)].iloc[0]
        truth_row = group[group["candidate_id"].astype(str).eq(truth_id)].iloc[0]
        advantage = float(proposal_row.network_score - baseline_row.network_score)
        raw_valid = bool(
            float(proposal_row.edge0_complete_fraction) > 0
            and float(proposal_row.edge0_bottleneck_mean) > 0
        )
        rows.append({
            "query_id": query_id,
            "truth_candidate_id": truth_id,
            "baseline_candidate_id": baseline,
            "proposal_candidate_id": proposal,
            "baseline_unique": bool(baseline_unique),
            "proposal_unique": bool(proposal_unique),
            "proposal_differs": proposal != baseline,
            "baseline_correct": baseline_unique and baseline == truth_id,
            "proposal_correct": proposal_unique and proposal == truth_id,
            "dreams_gap": top_gap(group),
            "network_score_advantage": advantage,
            "proposal_probability": stable_sigmoid(advantage),
            "proposal_raw_step0_valid": raw_valid,
            "baseline_raw_step0_valid": bool(
                float(baseline_row.edge0_complete_fraction) > 0
                and float(baseline_row.edge0_bottleneck_mean) > 0
            ),
            "truth_raw_step0_valid": bool(
                float(truth_row.edge0_complete_fraction) > 0
                and float(truth_row.edge0_bottleneck_mean) > 0
            ),
        })
        for feature, coefficient, scale in zip(
            expert.feature_names, expert.model_coef, expert.scaler_scale, strict=True
        ):
            difference = float(proposal_row[feature] - baseline_row[feature])
            contribution_rows.append({
                "query_id": query_id,
                "feature": feature,
                "proposal_minus_baseline": difference,
                "network_score_contribution": difference * float(coefficient) / float(scale),
            })
    proposals = pd.DataFrame(rows)
    if len(proposals) != candidates["query_id"].nunique():
        raise RuntimeError("proposal reconstruction lost queries")
    return proposals, pd.DataFrame(contribution_rows)


def gate_summary(proposals: pd.DataFrame, mask: pd.Series) -> dict[str, float | int]:
    mask = mask.astype(bool)
    baseline = proposals["baseline_correct"].astype(bool)
    proposal = proposals["proposal_correct"].astype(bool)
    final = baseline.where(~mask, proposal)
    corrected = int((mask & ~baseline & proposal).sum())
    introduced = int((mask & baseline & ~proposal).sum())
    return {
        "interventions": int(mask.sum()),
        "corrected": corrected,
        "introduced": introduced,
        "net": corrected - introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "recall1": float(final.mean()),
        "delta_recall1": float(final.mean() - baseline.mean()),
    }


def gate_decomposition(
    proposals: pd.DataFrame,
    expert: FrozenNegativeBioAwareExpert,
) -> dict[str, dict[str, float | int]]:
    eligible = (
        proposals["baseline_unique"].astype(bool)
        & proposals["proposal_unique"].astype(bool)
        & proposals["proposal_differs"].astype(bool)
    )
    gap = proposals["dreams_gap"] <= expert.maximum_dreams_top1_top2_gap
    probability = (
        proposals["proposal_probability"] >= expert.minimum_pairwise_proposal_probability
    )
    raw = proposals["proposal_raw_step0_valid"].astype(bool)
    masks = {
        "proposal_only": eligible,
        "proposal_plus_dreams_gap": eligible & gap,
        "proposal_plus_probability": eligible & probability,
        "proposal_plus_raw_step0": eligible & raw,
        "sequential_gap_then_probability": eligible & gap & probability,
        "sequential_gap_then_probability_then_raw": eligible & gap & probability & raw,
    }
    return {name: gate_summary(proposals, mask) for name, mask in masks.items()}


def feature_drift(
    development: pd.DataFrame,
    external: pd.DataFrame,
    features: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for feature in features:
        left = development[feature].to_numpy(float)
        right = external[feature].to_numpy(float)
        pooled = float(np.std(left, ddof=0))
        output[feature] = {
            "development_mean": float(np.mean(left)),
            "external_mean": float(np.mean(right)),
            "standardized_mean_shift": float((np.mean(right) - np.mean(left)) / pooled) if pooled > 0 else 0.0,
            "development_zero_fraction": float(np.mean(np.isclose(left, 0.0))),
            "external_zero_fraction": float(np.mean(np.isclose(right, 0.0))),
            "ks_statistic": float(ks_2samp(left, right).statistic),
        }
    return output


def within_query_variation(frame: pd.DataFrame, features: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    grouped = frame.groupby("query_id", sort=False)
    for feature in features:
        ranges = grouped[feature].agg(lambda values: float(values.max() - values.min()))
        output[feature] = {
            "queries": int(len(ranges)),
            "queries_with_nonzero_range": int((ranges > 1e-12).sum()),
            "fraction_with_nonzero_range": float((ranges > 1e-12).mean()),
            "median_range": float(ranges.median()),
            "p90_range": float(ranges.quantile(0.90)),
        }
    return output


def contribution_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    if frame.empty:
        return {}
    grouped = frame.groupby("feature", sort=False)["network_score_contribution"]
    return {
        str(feature): {
            "mean_signed": float(values.mean()),
            "mean_absolute": float(values.abs().mean()),
            "median_signed": float(values.median()),
        }
        for feature, values in grouped
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-dir", type=Path,
        default=Path("data/validation/bioaware_st001154_hilic_frozen_evaluation_v1"),
    )
    parser.add_argument(
        "--development-features", type=Path,
        default=Path(
            "data/validation/bioaware_metdna3_external_negative_loso_ranker_v4_chemically_filtered/"
            "candidate_features.csv.gz"
        ),
    )
    parser.add_argument(
        "--artifact", type=Path,
        default=Path(
            "data/validation/bioaware_metdna3_negative_network_expert_v2_chemically_filtered/"
            "artifact.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_st001154_hilic_abstention_diagnostic_v1"),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    external_features_path = args.external_dir / "candidate_features.csv.gz"
    external_query_path = args.external_dir / "per_query.csv.gz"
    external_report_path = args.external_dir / "report.json"
    for path in (
        external_features_path, external_query_path, external_report_path,
        args.development_features, args.artifact,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    external_report = json.loads(external_report_path.read_text(encoding="utf-8"))
    if external_report.get("status") != "bioaware_st001154_hilic_frozen_evaluation_complete":
        raise RuntimeError("unexpected external evaluation report")
    for name, path in (
        ("candidate_features", external_features_path),
        ("per_query", external_query_path),
    ):
        if sha256(path) != external_report["provenance"][f"{name}_sha256"]:
            raise RuntimeError(f"external {name} hash mismatch")
    artifact_payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    expected_development = artifact_payload["provenance"]["candidate_features_sha256"]
    if sha256(args.development_features) != expected_development:
        raise RuntimeError("development features do not match the frozen artifact")
    expert = FrozenNegativeBioAwareExpert.load(args.artifact)
    development = pd.read_csv(args.development_features)
    external = pd.read_csv(external_features_path)
    external_truth = pd.read_csv(external_query_path)[["query_id", "truth_candidate_id"]]
    development_truth = development[["query_id", "truth_candidate_id"]]
    external_proposals, external_contributions = build_proposals(
        external, external_truth, expert
    )
    development_proposals, development_contributions = build_proposals(
        development, development_truth, expert
    )
    disagreements = external_proposals["proposal_differs"].astype(bool)
    report = {
        "status": "bioaware_st001154_hilic_abstention_diagnostic_complete",
        "formal": False,
        "opened_external_diagnostic_only": True,
        "external_queries": int(len(external_proposals)),
        "external_candidate_rows": int(len(external)),
        "development_queries": int(len(development_proposals)),
        "development_candidate_rows": int(len(development)),
        "external_baseline_recall1": float(external_proposals["baseline_correct"].mean()),
        "proposal_quality_without_safety_gates": gate_decomposition(external_proposals, expert),
        "external_proposal_counts": {
            "different_top1": int(disagreements.sum()),
            "proposal_correct_among_differences": int(
                (disagreements & external_proposals["proposal_correct"].astype(bool)).sum()
            ),
            "baseline_correct_among_differences": int(
                (disagreements & external_proposals["baseline_correct"].astype(bool)).sum()
            ),
            "proposal_raw_step0_valid": int(
                external_proposals["proposal_raw_step0_valid"].sum()
            ),
            "truth_raw_step0_valid": int(external_proposals["truth_raw_step0_valid"].sum()),
        },
        "development_proposal_reference": gate_decomposition(development_proposals, expert),
        "candidate_level_feature_drift": feature_drift(
            development, external, expert.feature_names
        ),
        "within_query_feature_variation": {
            "development": within_query_variation(development, expert.feature_names),
            "external": within_query_variation(external, expert.feature_names),
        },
        "proposal_vs_baseline_score_contributions": {
            "development": contribution_summary(
                development_contributions.loc[
                    development_contributions["query_id"].isin(
                        set(development_proposals.loc[
                            development_proposals["proposal_differs"], "query_id"
                        ])
                    )
                ]
            ),
            "external": contribution_summary(
                external_contributions.loc[
                    external_contributions["query_id"].isin(
                        set(external_proposals.loc[disagreements, "query_id"])
                    )
                ]
            ),
        },
        "protocol_findings": {
            "known_mass_candidate_fraction_external_constant": bool(
                external["known_mass_candidate_fraction"].nunique() == 1
            ),
            "known_mass_candidate_fraction_semantics": (
                "development records whether a candidate was recovered by the sample MS1 mass layer; "
                "the external candidate graph was itself built from strict same-mass candidates, so the "
                "feature is structurally constant and contributes no within-query discrimination"
            ),
            "frozen_artifact_refit": False,
            "thresholds_changed": False,
        },
        "provenance": {
            "artifact_sha256": sha256(args.artifact),
            "development_features_sha256": sha256(args.development_features),
            "external_features_sha256": sha256(external_features_path),
            "external_per_query_sha256": sha256(external_query_path),
            "external_report_sha256": sha256(external_report_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Post-hoc diagnostic on an opened external cohort. No model fitting or threshold selection "
            "is performed, and these outcomes cannot be used as a new confirmatory result."
        ),
    }
    args.output_dir.mkdir(parents=True)
    external_proposals.to_csv(
        args.output_dir / "external_proposals.csv.gz", index=False, compression="gzip"
    )
    external_contributions.to_csv(
        args.output_dir / "external_score_contributions.csv.gz", index=False, compression="gzip"
    )
    report["provenance"]["external_proposals_sha256"] = sha256(
        args.output_dir / "external_proposals.csv.gz"
    )
    report["provenance"]["external_score_contributions_sha256"] = sha256(
        args.output_dir / "external_score_contributions.csv.gz"
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
