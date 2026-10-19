#!/usr/bin/env python
"""Open frozen truth and build query-level Contrast Atlas outcomes.

Candidate scores and calibrated associations must already be frozen.  The truth
ledger is used only here.  Every candidate requires an explicit truth row;
unlisted candidates are never silently converted to negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ASSOCIATIONS = {"similar_response", "opposite_response"}
TRUTH_STATES = {*ASSOCIATIONS, "irrelevant"}
TRUTH_BASES = {
    "independent_intervention_replication",
    "independent_disease_replication",
    "designed_reversal_or_rescue",
    "orthogonal_functional_replication",
    "prespecified_mechanistic_decoy",
    "prespecified_no_match",
    "synthetic_smoke",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(
    candidates_raw: pd.DataFrame,
    registry_raw: pd.DataFrame,
    truth_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_required = {
        "world", "query_study_id", "query_contrast_id", "target_study_id", "target_contrast_id",
        "calibration_status", "calibrated_association", "q_similar", "q_opposite", "observed_score",
    }
    registry_required = {
        "query_id", "query_study_id", "query_contrast_id", "cluster_id",
        "source_holdout", "instrument_holdout",
    }
    truth_required = {
        "query_id", "target_study_id", "target_contrast_id", "expected_association",
        "truth_basis", "truth_data_role", "truth_frozen_before_scoring",
        "selection_blind_to_atlas_score",
    }
    for label, frame, required in (
        ("candidates", candidates_raw, candidate_required),
        ("query registry", registry_raw, registry_required),
        ("truth ledger", truth_raw, truth_required),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{label} missing columns: {sorted(missing)}")
    candidates = candidates_raw.copy()
    registry = registry_raw.copy()
    truth = truth_raw.copy()
    for column in ("truth_frozen_before_scoring", "selection_blind_to_atlas_score"):
        values = truth[column].astype(str).str.strip().str.lower()
        mapping = {"true": True, "false": False, "1": True, "0": False}
        if not values.isin(mapping).all():
            raise RuntimeError(f"truth ledger has invalid boolean: {column}")
        truth[column] = values.map(mapping).astype(bool)
    if registry.duplicated(["query_id"]).any() or registry.duplicated(["query_study_id", "query_contrast_id"]).any():
        raise RuntimeError("query registry keys must be unique")
    if truth.duplicated(["query_id", "target_study_id", "target_contrast_id"]).any():
        raise RuntimeError("duplicate query-target truth")
    if not set(truth["expected_association"]).issubset(TRUTH_STATES):
        raise RuntimeError("unrecognized expected_association")
    if not set(truth["truth_basis"]).issubset(TRUTH_BASES):
        raise RuntimeError("unrecognized truth_basis")
    if not truth["truth_frozen_before_scoring"].all():
        raise RuntimeError("truth ledger was not frozen before scoring")
    if not truth["selection_blind_to_atlas_score"].all():
        raise RuntimeError("truth selection used Atlas scores")
    candidates = candidates.merge(
        registry,
        on=["query_study_id", "query_contrast_id"],
        how="left",
        validate="many_to_one",
    )
    if candidates["query_id"].isna().any():
        raise RuntimeError("candidate query absent from registry")
    candidate_keys = candidates[["query_id", "target_study_id", "target_contrast_id"]].drop_duplicates()
    checked = candidate_keys.merge(
        truth,
        on=["query_id", "target_study_id", "target_contrast_id"],
        how="left",
        validate="one_to_one",
    )
    if checked["expected_association"].isna().any():
        examples = checked.loc[checked["expected_association"].isna()].head(3).to_dict("records")
        raise RuntimeError(f"candidate truth is incomplete: {examples}")
    candidates = candidates.merge(
        truth,
        on=["query_id", "target_study_id", "target_contrast_id"],
        how="left",
        validate="many_to_one",
    )
    return candidates, registry, truth


def build_outcomes(
    candidates_raw: pd.DataFrame,
    registry_raw: pd.DataFrame,
    truth_raw: pd.DataFrame,
    fdr_threshold: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates, registry, truth = validate_inputs(candidates_raw, registry_raw, truth_raw)
    candidates["direction_q"] = np.where(
        candidates["calibrated_association"].eq("similar_response"), candidates["q_similar"],
        np.where(candidates["calibrated_association"].eq("opposite_response"), candidates["q_opposite"], np.nan),
    )
    candidates["accepted"] = (
        candidates["calibration_status"].eq("calibrated_within_target_pattern_permutation")
        & candidates["calibrated_association"].isin(ASSOCIATIONS)
        & candidates["direction_q"].notna()
        & (candidates["direction_q"] <= fdr_threshold)
    )
    candidates["direction_correct"] = (
        candidates["accepted"]
        & candidates["expected_association"].isin(ASSOCIATIONS)
        & candidates["calibrated_association"].eq(candidates["expected_association"])
    )
    candidates["evidence_strength"] = np.where(
        candidates["accepted"],
        -np.log10(np.maximum(candidates["direction_q"].astype(float), np.finfo(float).tiny)),
        -np.inf,
    )
    candidates["absolute_score"] = candidates["observed_score"].abs()

    rows = []
    ranked_rows = []
    worlds = sorted(candidates["world"].unique())
    for query in registry.itertuples(index=False):
        query_truth = truth.loc[truth["query_id"] == query.query_id]
        truth_type = "match" if query_truth["expected_association"].isin(ASSOCIATIONS).any() else "no_match"
        for world in worlds:
            subset = candidates.loc[(candidates["query_id"] == query.query_id) & (candidates["world"] == world)].copy()
            if subset.empty:
                raise RuntimeError(f"missing candidate world for query {query.query_id}: {world}")
            accepted = subset.loc[subset["accepted"]].sort_values(
                ["evidence_strength", "absolute_score", "target_study_id", "target_contrast_id"],
                ascending=[False, False, True, True],
                kind="mergesort",
            )
            accepted["accepted_rank"] = np.arange(1, len(accepted) + 1)
            ranked_rows.append(accepted)
            answered = len(accepted) > 0
            if truth_type == "match":
                correct_rows = accepted.loc[accepted["direction_correct"]]
                rank = float(correct_rows["accepted_rank"].iloc[0]) if len(correct_rows) else float("nan")
                correct = bool(answered and rank == 1)
            else:
                rank = float("nan")
                correct = not answered
            rows.append({
                "query_id": query.query_id,
                "cluster_id": query.cluster_id,
                "source_holdout": query.source_holdout,
                "instrument_holdout": query.instrument_holdout,
                "world": world,
                "truth_type": truth_type,
                "answered": bool(answered),
                "correct": bool(correct),
                "rank": rank,
                "accepted_candidates": int(len(accepted)),
                "candidate_denominator": int(len(subset)),
            })
    return pd.DataFrame(rows), pd.concat(ranked_rows, ignore_index=True) if ranked_rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrated-candidates", type=Path, required=True)
    parser.add_argument("--query-registry", type=Path, required=True)
    parser.add_argument("--truth-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fdr-threshold", type=float, default=0.10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    candidates = pd.read_csv(args.calibrated_candidates)
    registry = pd.read_csv(args.query_registry)
    truth = pd.read_csv(args.truth_ledger)
    outcomes, accepted = build_outcomes(candidates, registry, truth, args.fdr_threshold)
    args.output_dir.mkdir(parents=True)
    outcomes.to_csv(args.output_dir / "query_outcomes.csv", index=False)
    accepted.to_csv(args.output_dir / "accepted_candidate_ranking.csv", index=False)
    report = {
        "status": "CONTRAST_ATLAS_QUERY_OUTCOMES_COMPLETE",
        "formal": True,
        "truth_opened_only_at_evaluation": True,
        "fdr_threshold": args.fdr_threshold,
        "queries": int(outcomes["query_id"].nunique()),
        "worlds": sorted(outcomes["world"].unique()),
        "truth_basis_counts": truth["truth_basis"].value_counts().sort_index().to_dict(),
        "truth_data_role_counts": truth["truth_data_role"].value_counts().sort_index().to_dict(),
        "contains_synthetic_truth": bool(truth["truth_basis"].eq("synthetic_smoke").any()),
        "input_sha256": {
            "calibrated_candidates": sha256(args.calibrated_candidates),
            "query_registry": sha256(args.query_registry),
            "truth_ledger": sha256(args.truth_ledger),
        },
        "claim_limit": "Query-level frozen retrieval outcomes only; use paired-world adjudication for unknown increment.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
