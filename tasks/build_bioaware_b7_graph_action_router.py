#!/usr/bin/env python
"""Freeze graph-prior hard-example routes for direct shared-encoder training.

BioAware is used only to find useful retrieval boundaries.  The resulting
ledger stores whether a nested formula-OOF graph action corrected or harmed an
official DreaMS decision.  No graph score is used as a student target: B7
training uses the true molecular identity and a direct listwise ranking loss.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from build_bioaware_b4_direct_manifest import sha256_file, stable_fold  # noqa: E402
from develop_bioaware_b1_multisource_action import (  # noqa: E402
    evaluate_model,
    fit_ranker,
)


FEATURES = [
    "spectral_score",
    "network_member",
    "known_log_degree",
    "known_mass_candidate_fraction",
]
ARM = "graph_prior_direct"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-features", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--b4-manifest", type=Path,
        default=ROOT / "data/validation/bioaware_b4_direct_shared_embedding_v1_20260905/manifest/manifest.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    return parser.parse_args()


def strict_proposed(group: pd.DataFrame) -> tuple[str, bool]:
    maximum = float(group["graph_action_score"].max())
    top = group.loc[
        np.isclose(group["graph_action_score"], maximum, rtol=0, atol=1e-12)
    ].sort_values("candidate_id", kind="stable")
    return str(top["candidate_id"].iloc[0]), len(top) == 1


def main() -> None:
    args = arguments()
    if args.folds != 5:
        raise ValueError("B7 is preregistered for exactly five formula folds")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    for path in (args.candidate_features, args.b4_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    candidates = pd.read_csv(args.candidate_features)
    required = set(FEATURES) | {
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "source", "polarity", "baseline_correct", "baseline_candidate_id",
        "baseline_gap", "truth_margin", "is_positive", "unit_id",
    }
    if missing := required - set(candidates):
        raise RuntimeError(f"candidate feature table lacks {sorted(missing)}")
    for column in ("query_id", "candidate_id", "truth_candidate_id", "truth_formula"):
        candidates[column] = candidates[column].astype(str)
    if not np.isfinite(candidates[FEATURES].to_numpy(float)).all():
        raise RuntimeError("graph-prior features contain non-finite values")

    with np.load(args.b4_manifest, allow_pickle=False) as loaded:
        manifest = {key: loaded[key] for key in loaded.files}
    for key in ("query_id", "query_formula", "formula_fold", "query_ptr", "molecule_id"):
        if key not in manifest:
            raise RuntimeError(f"B4 manifest lacks {key}")
    query_ids = list(map(str, manifest["query_id"]))
    query_set = set(query_ids)
    if len(query_ids) != 548 or len(query_set) != len(query_ids):
        raise RuntimeError("B7 expects the frozen 548-query negative-ion manifest")
    selected = candidates[candidates["query_id"].isin(query_set)].copy()
    if set(selected["query_id"]) != query_set or selected["query_id"].nunique() != 548:
        raise RuntimeError("Full16 candidate cache does not cover the B4 query universe")
    if not selected["polarity"].eq("negative").all():
        raise RuntimeError("B7 action universe must be negative ion only")

    manifest_formula = dict(zip(
        query_ids, map(str, manifest["query_formula"]), strict=True,
    ))
    manifest_fold = dict(zip(
        query_ids, map(int, manifest["formula_fold"]), strict=True,
    ))
    for query_id, group in selected.groupby("query_id", sort=False):
        if group["truth_formula"].nunique() != 1:
            raise RuntimeError(f"{query_id}: non-unique truth formula")
        formula = str(group["truth_formula"].iloc[0])
        if formula != manifest_formula[query_id]:
            raise RuntimeError(f"{query_id}: candidate/manifest formula mismatch")
        if stable_fold(formula, args.folds, args.fold_seed) != manifest_fold[query_id]:
            raise RuntimeError(f"{query_id}: formula-fold replay mismatch")
        position = query_ids.index(query_id)
        left, right = map(int, manifest["query_ptr"][position:position + 2])
        manifest_ids = set(map(str, manifest["molecule_id"][left:right]))
        feature_ids = set(group["candidate_id"])
        if manifest_ids != feature_ids:
            raise RuntimeError(
                f"{query_id}: candidate universe mismatch "
                f"manifest_only={sorted(manifest_ids-feature_ids)[:5]} "
                f"features_only={sorted(feature_ids-manifest_ids)[:5]}"
            )

    candidates["formula_fold"] = candidates["truth_formula"].map(
        lambda value: stable_fold(str(value), args.folds, args.fold_seed)
    )
    selected["formula_fold"] = selected["query_id"].map(manifest_fold)
    candidate_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    fold_reports: dict[str, dict] = {}

    for outer in range(args.folds):
        outer_candidates: list[pd.DataFrame] = []
        outer_actions: list[pd.DataFrame] = []
        for inner in range(args.folds):
            if inner == outer:
                continue
            test = selected[selected["formula_fold"].eq(inner)].copy()
            test_identities = set(test["truth_candidate_id"])
            test_formulas = set(test["truth_formula"])
            train = candidates[
                ~candidates["formula_fold"].isin((outer, inner))
                & ~candidates["truth_candidate_id"].isin(test_identities)
                & ~candidates["truth_formula"].isin(test_formulas)
            ].copy()
            train_queries = train[["query_id", "truth_candidate_id", "truth_formula"]].drop_duplicates()
            if train_queries["query_id"].nunique() < 500:
                raise RuntimeError(f"outer={outer} inner={inner}: insufficient purged training data")
            if set(train_queries["truth_candidate_id"]) & test_identities:
                raise RuntimeError("truth-identity leakage into graph router fit")
            if set(train_queries["truth_formula"]) & test_formulas:
                raise RuntimeError("truth-formula leakage into graph router fit")

            scaler, model = fit_ranker(train, FEATURES)
            action = evaluate_model(
                test, scaler, model, FEATURES,
                f"outer_{outer}_inner_{inner}", ARM,
            )
            scored = test.copy()
            scored["graph_action_score"] = model.decision_function(
                scaler.transform(scored[FEATURES].to_numpy(float))
            )
            action_by_query = action.set_index("query_id")
            for query_id, group in scored.groupby("query_id", sort=False):
                proposed, unique = strict_proposed(group)
                expected = action_by_query.loc[str(query_id)]
                if proposed != str(expected["proposed_candidate_id"]) or unique != bool(expected["proposal_unique"]):
                    raise RuntimeError(f"{query_id}: graph action replay failed")
            # Direct training uses only the truth-labelled outcome of the OOF
            # action; the model score itself never becomes a student target.
            action["corrected"] = (~action["baseline_correct"]) & action["direct_correct"]
            action["introduced"] = action["baseline_correct"] & (~action["direct_correct"])
            action["intervene"] = action["proposed_candidate_id"].astype(str).ne(
                action["baseline_candidate_id"].astype(str)
            )
            action["arm"] = ARM
            action["outer_fold"] = outer
            action["inner_fold"] = inner
            scored["arm"] = ARM
            scored["outer_fold"] = outer
            scored["inner_fold"] = inner
            outer_actions.append(action)
            outer_candidates.append(scored)

        actions_outer = pd.concat(outer_actions, ignore_index=True)
        scores_outer = pd.concat(outer_candidates, ignore_index=True)
        expected_queries = {
            query_id for query_id in query_ids if manifest_fold[query_id] != outer
        }
        if actions_outer["query_id"].astype(str).duplicated().any():
            raise RuntimeError(f"outer={outer}: duplicate nested action")
        if set(actions_outer["query_id"].astype(str)) != expected_queries:
            raise RuntimeError(f"outer={outer}: nested action coverage mismatch")
        delta = actions_outer["direct_correct"].astype(int) - actions_outer["baseline_correct"].astype(int)
        fold_reports[str(outer)] = {
            "queries": int(len(actions_outer)),
            "candidate_rows": int(len(scores_outer)),
            "baseline_recall1": float(actions_outer["baseline_correct"].mean()),
            "action_recall1": float(actions_outer["direct_correct"].mean()),
            "delta_recall1": float(delta.mean()),
            "corrected": int(actions_outer["corrected"].sum()),
            "introduced": int(actions_outer["introduced"].sum()),
            "risk_net_lambda2": int(
                actions_outer["corrected"].sum() - 2 * actions_outer["introduced"].sum()
            ),
            "corrected_identities": int(actions_outer.loc[
                actions_outer["corrected"], "truth_candidate_id"
            ].nunique()),
            "corrected_formulas": int(actions_outer.loc[
                actions_outer["corrected"], "truth_formula"
            ].nunique()),
        }
        action_parts.append(actions_outer)
        candidate_parts.append(scores_outer)

    action_ledger = pd.concat(action_parts, ignore_index=True)
    candidate_ledger = pd.concat(candidate_parts, ignore_index=True)
    corrected = int(action_ledger["corrected"].sum())
    introduced = int(action_ledger["introduced"].sum())
    overall = {
        "action_rows": int(len(action_ledger)),
        "candidate_score_rows": int(len(candidate_ledger)),
        "baseline_recall1": float(action_ledger["baseline_correct"].mean()),
        "action_recall1": float(action_ledger["direct_correct"].mean()),
        "delta_recall1": float(
            (action_ledger["direct_correct"].astype(int)
             - action_ledger["baseline_correct"].astype(int)).mean()
        ),
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "corrected_identities": int(action_ledger.loc[
            action_ledger["corrected"], "truth_candidate_id"
        ].nunique()),
        "corrected_formulas": int(action_ledger.loc[
            action_ledger["corrected"], "truth_formula"
        ].nunique()),
    }
    gates = {
        "direct_action_gain_ge_3pp": overall["delta_recall1"] >= 0.03,
        "corrected_gt_introduced": corrected > introduced,
        "risk_net_lambda2_positive": overall["risk_net_lambda2"] > 0,
        "corrected_identities_ge_25": overall["corrected_identities"] >= 25,
        "every_outer_fold_nonnegative": all(
            value["delta_recall1"] >= 0 for value in fold_reports.values()
        ),
    }
    args.output_dir.mkdir(parents=True)
    action_path = args.output_dir / "action_routes.csv.gz"
    candidate_path = args.output_dir / "candidate_action_scores.csv.gz"
    action_ledger.to_csv(action_path, index=False, compression="gzip")
    candidate_ledger.to_csv(candidate_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b7_graph_action_router_frozen",
        "formal": True,
        "arm": ARM,
        "features": FEATURES,
        "protocol": "nested formula-OOF graph-prior action mining; direct truth-labelled encoder training",
        "overall_repeated_outer_views": overall,
        "outer_folds": fold_reports,
        "gates": gates,
        "pass_to_direct_gradient_gate": bool(all(gates.values())),
        "contracts": {
            "graph_score_is_training_router_only": True,
            "graph_score_is_not_student_target": True,
            "student_target_is_true_molecular_identity": True,
            "outer_formula_fold_never_router_scored": True,
            "inner_truth_identity_and_formula_purged": True,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "provenance": {
            "candidate_features_sha256": sha256_file(args.candidate_features),
            "b4_manifest_sha256": sha256_file(args.b4_manifest),
            "actions_sha256": sha256_file(action_path),
            "candidate_scores_sha256": sha256_file(candidate_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": "Training-space action headroom only; no shared-embedding gain until held-fold direct fine-tuning succeeds.",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if not report["pass_to_direct_gradient_gate"]:
        raise RuntimeError(f"B7 graph action headroom gate failed: {gates}")


if __name__ == "__main__":
    main()
