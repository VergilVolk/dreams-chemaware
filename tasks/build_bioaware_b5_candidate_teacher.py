#!/usr/bin/env python
"""Freeze candidate-level, nested-OOF BioAware teacher scores for B5.

Unlike B4, this ledger retains the exact candidate logits and intervention
chosen by the BioAware teacher.  The outer formula fold is never scored and
can therefore be used only for later evaluation, never as a teacher target.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_metdna3_negative_loso_ablation import ABLATIONS  # noqa: E402
from build_bioaware_b4_direct_manifest import sha256_file, stable_fold  # noqa: E402
from develop_bioaware_metdna3_negative_loso_ranker import (  # noqa: E402
    PRIMARY_C,
    evaluate_fold,
    pairwise_training_rows,
    strict_top,
)


ARMS = {
    "full_bioaware_safe": ("full_bioaware", "safe"),
    "full_no_edge_high_recall": ("full_no_edge_gate", "recall"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-features", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_loso_ranker_v4_chemically_filtered/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--b4-manifest", type=Path,
        default=ROOT / "data/validation/bioaware_b4_direct_shared_embedding_v1_20260905/manifest/manifest.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    return parser.parse_args()


def score_candidates(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str],
) -> pd.DataFrame:
    """Fit the exact frozen router and return every held candidate logit."""
    x, y, weights = pairwise_training_rows(train, features)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=PRIMARY_C, fit_intercept=False, solver="lbfgs", max_iter=2000,
        random_state=20260901,
    ).fit(scaler.transform(x), y, sample_weight=weights)
    scored = test.copy()
    scored["teacher_model_score"] = model.decision_function(
        scaler.transform(scored[features].to_numpy(float))
    )
    if not np.isfinite(scored["teacher_model_score"]).all():
        raise RuntimeError("non-finite candidate-level teacher score")
    return scored


def assert_candidate_replay(scored: pd.DataFrame, transition: pd.DataFrame) -> None:
    expected = transition.set_index("query_id")
    if scored["query_id"].nunique() != len(expected):
        raise RuntimeError("candidate ledger query coverage differs from transition")
    for query_id, group in scored.groupby("query_id", sort=False):
        row = expected.loc[str(query_id)]
        proposed, unique = strict_top(group, "teacher_model_score")
        if proposed != str(row["proposed_candidate_id"]) or unique != bool(row["proposal_unique"]):
            raise RuntimeError(f"{query_id}: candidate-level proposed action replay failed")
        baseline = str(row["baseline_candidate_id"])
        proposed_score = float(group.loc[
            group["candidate_id"].astype(str).eq(proposed), "teacher_model_score"
        ].iloc[0])
        baseline_score = float(group.loc[
            group["candidate_id"].astype(str).eq(baseline), "teacher_model_score"
        ].iloc[0])
        probability = float(expit(proposed_score - baseline_score))
        if abs(probability - float(row["proposal_probability"])) > 1e-10:
            raise RuntimeError(f"{query_id}: teacher probability replay failed")


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    for path in (args.candidate_features, args.b4_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    candidates = pd.read_csv(args.candidate_features)
    with np.load(args.b4_manifest, allow_pickle=False) as loaded:
        manifest = {key: loaded[key] for key in loaded.files}
    query_ids = list(map(str, manifest["query_id"]))
    query_set = set(query_ids)
    candidates = candidates[candidates["query_id"].astype(str).isin(query_set)].copy()
    if candidates["query_id"].nunique() != len(query_ids):
        raise RuntimeError("candidate feature table does not cover the B4 query universe")
    query_position = {value: index for index, value in enumerate(query_ids)}
    fold_by_query = {
        query_id: stable_fold(str(formula), args.folds, args.fold_seed)
        for query_id, formula in zip(
            query_ids, manifest["query_formula"].astype(str), strict=True,
        )
    }
    candidates["formula_fold"] = candidates["query_id"].astype(str).map(fold_by_query)

    candidate_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    reports: dict[str, dict] = {}
    for arm, (ablation_name, route_prefix) in ARMS.items():
        spec = ABLATIONS[ablation_name]
        arm_reports = {}
        for outer in range(args.folds):
            held_outer = candidates["formula_fold"].eq(outer)
            outer_candidates = []
            outer_actions = []
            for inner in range(args.folds):
                if inner == outer:
                    continue
                test = candidates[candidates["formula_fold"].eq(inner)].copy()
                inner_truth = set(test["truth_candidate_id"].astype(str))
                train = candidates[
                    (~held_outer)
                    & (~candidates["formula_fold"].eq(inner))
                    & (~candidates["truth_candidate_id"].astype(str).isin(inner_truth))
                ].copy()
                transition, _ = evaluate_fold(
                    train, test, f"formula_outer_{outer}_inner_{inner}",
                    features=spec["features"],
                    require_raw_step0_edge=spec["require_raw_step0_edge"],
                )
                scored = score_candidates(train, test, spec["features"])
                assert_candidate_replay(scored, transition)
                scored["arm"] = arm
                scored["outer_fold"] = outer
                scored["inner_fold"] = inner
                transition["arm"] = arm
                transition["outer_fold"] = outer
                transition["inner_fold"] = inner
                outer_candidates.append(scored)
                outer_actions.append(transition)
            scored_outer = pd.concat(outer_candidates, ignore_index=True)
            actions_outer = pd.concat(outer_actions, ignore_index=True)
            expected_queries = set(np.asarray(query_ids)[manifest["formula_fold"] != outer])
            if (actions_outer["query_id"].duplicated().any()
                    or set(actions_outer["query_id"].astype(str)) != expected_queries):
                raise RuntimeError(f"{arm}/outer={outer}: nested OOF action coverage failed")
            corrected_key = f"{route_prefix}_corrected_by_outer"
            introduced_key = f"{route_prefix}_introduced_by_outer"
            for action in actions_outer.itertuples(index=False):
                position = query_position[str(action.query_id)]
                if bool(action.corrected) != bool(manifest[corrected_key][outer, position]):
                    raise RuntimeError(f"{arm}/outer={outer}: corrected replay mismatch")
                if bool(action.introduced) != bool(manifest[introduced_key][outer, position]):
                    raise RuntimeError(f"{arm}/outer={outer}: introduced replay mismatch")
            arm_reports[str(outer)] = {
                "queries": int(len(actions_outer)),
                "candidate_rows": int(len(scored_outer)),
                "corrected": int(actions_outer["corrected"].sum()),
                "introduced": int(actions_outer["introduced"].sum()),
                "corrected_identities": int(actions_outer.loc[
                    actions_outer["corrected"], "truth_candidate_id"
                ].nunique()),
                "corrected_formulas": int(actions_outer.loc[
                    actions_outer["corrected"], "truth_formula"
                ].nunique()),
            }
            candidate_parts.append(scored_outer)
            action_parts.append(actions_outer)
        reports[arm] = arm_reports

    candidate_ledger = pd.concat(candidate_parts, ignore_index=True)
    action_ledger = pd.concat(action_parts, ignore_index=True)
    if not candidate_ledger.groupby(["arm", "outer_fold", "query_id"]).size().ge(2).all():
        raise RuntimeError("teacher ledger contains a query with fewer than two candidates")
    args.output_dir.mkdir(parents=True)
    candidate_path = args.output_dir / "candidate_teacher_scores.csv.gz"
    action_path = args.output_dir / "teacher_actions.csv.gz"
    candidate_ledger.to_csv(candidate_path, index=False, compression="gzip")
    action_ledger.to_csv(action_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b5_candidate_teacher_frozen",
        "formal": False,
        "protocol": "nested truth-formula OOF candidate-level BioAware logits; outer fold never scored",
        "queries": int(len(query_ids)),
        "candidate_score_rows": int(len(candidate_ledger)),
        "action_rows": int(len(action_ledger)),
        "arms": reports,
        "contracts": {
            "candidate_level_teacher_scores_retained": True,
            "exact_proposed_candidate_retained": True,
            "outer_fold_never_teacher_scored": True,
            "teacher_action_replayed_exactly": True,
            "identity_truth_used_only_by_supervised_router_training": True,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "provenance": {
            "candidate_features_sha256": sha256_file(args.candidate_features),
            "b4_manifest_sha256": sha256_file(args.b4_manifest),
            "candidate_ledger_sha256": sha256_file(candidate_path),
            "action_ledger_sha256": sha256_file(action_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": "Candidate-level teacher ledger only; no embedding improvement or transfer claim.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
