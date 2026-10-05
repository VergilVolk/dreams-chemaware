#!/usr/bin/env python
"""Nested spectral arbitration of the 46 B12/B16 branch disagreements.

Unlike B27, B28 never vetoes a B17 action merely because a conventional raw
similarity is low.  It operates only when the two independently trained
BioAware branches propose different candidate identities and asks which
candidate's own exact reference spectrum has stronger label-free peak support.
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

from audit_bioaware_b11_catalog_interaction_action import (  # noqa: E402
    atomic_json, cluster_bootstrap, sha256, summarize,
)
from audit_bioaware_b12_multicohort_catalog_action import EXPECTED_DOMAINS  # noqa: E402
from audit_bioaware_b15_action_spectrum_support import (  # noqa: E402
    candidate_reference_map, load_mona_reference_tensors,
)
from audit_bioaware_b9_reaction_spectral_specificity import reaction_spectral_views  # noqa: E402


B17_DELTA = 50 / 860
B17_RISK = 43
B17_INTRODUCED = 7
VIEWS = ("truncated_direct", "neutral_loss", "modified_cosine", "dual_view")
POLICIES = (
    "B17", "direct", "neutral_loss", "modified_cosine", "dual_view",
    "three_view_majority", "three_view_mean",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b20-dir", type=Path, required=True)
    parser.add_argument("--internal-candidates", type=Path, default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz")
    parser.add_argument("--st-manifest-dir", type=Path, default=ROOT / "data/validation/bioaware_st001154_hilic_extension_manifest_v1")
    parser.add_argument("--st-candidate-scores", type=Path, default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz")
    parser.add_argument("--st-evaluation-dir", type=Path, default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1")
    parser.add_argument("--kgmn-manifest-dir", type=Path, default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2")
    parser.add_argument("--mona-mgf", type=Path, default=ROOT / "data/models/mona_neg_full.mgf")
    parser.add_argument("--mona-manifest", type=Path, default=ROOT / "data/models/mona_neg_dreams_emb/manifest.csv")
    parser.add_argument("--mona-embeddings", type=Path, default=ROOT / "data/models/mona_neg_dreams_emb/embeddings.npy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def materialise_branch_evidence(args: argparse.Namespace) -> pd.DataFrame:
    table_path = args.b20_dir / "direct_actions.csv.gz"
    manifest_path = args.b20_dir / "direct_action_manifest.npz"
    report_path = args.b20_dir / "report.json"
    for path in (table_path, manifest_path, report_path, args.mona_mgf,
                 args.mona_manifest, args.mona_embeddings):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("status") != "bioaware_b20_direct_action_manifest_complete"
            or report.get("pass_to_direct_gradient_canary") is not True):
        raise RuntimeError("B20 did not pass")
    frame = pd.read_csv(table_path)
    with np.load(manifest_path, allow_pickle=False) as handle:
        query_id = handle["query_id"].astype(str)
        query_tensor = np.asarray(handle["query_tensor"], dtype=np.float32)
    if len(frame) != 860 or not frame["query_id"].astype(str).equals(pd.Series(query_id)):
        raise RuntimeError("B20 table/tensor alignment changed")
    disagreement = frame["b12_final_candidate_id"].astype(str).ne(
        frame["b16_final_candidate_id"].astype(str)
    )
    if int(disagreement.sum()) != 46:
        raise RuntimeError("B25 branch-disagreement count changed")

    mapping = candidate_reference_map(args)
    branch_rows: dict[tuple[int, str], int] = {}
    missing = []
    for index, row in frame.loc[disagreement].iterrows():
        for branch in ("b12", "b16"):
            candidate = str(row[f"{branch}_final_candidate_id"])
            key = (str(row["source"]), str(row["source_query_id"]), candidate)
            reference = mapping.get(key)
            if reference is None:
                missing.append(key)
            else:
                branch_rows[(int(index), branch)] = int(reference)
    if missing:
        raise RuntimeError(f"B28 missing {len(missing)} exact branch references: {missing[:5]}")
    selected_rows = np.asarray(sorted(set(branch_rows.values())), dtype=np.int64)
    tensors = load_mona_reference_tensors(args.mona_mgf, args.mona_manifest, selected_rows)
    position = {int(row): local for local, row in enumerate(selected_rows)}

    for branch in ("b12", "b16"):
        frame[f"b28_{branch}_reference_row"] = -1
        for view in VIEWS:
            frame[f"b28_{branch}_{view}"] = 0.0
    for index in frame.index[disagreement]:
        for branch in ("b12", "b16"):
            row = branch_rows[(int(index), branch)]
            views = reaction_spectral_views(query_tensor[index], tensors[position[row]])
            frame.at[index, f"b28_{branch}_reference_row"] = row
            for view in VIEWS:
                frame.at[index, f"b28_{branch}_{view}"] = float(views[view])
    for view in VIEWS:
        frame[f"b28_{view}_advantage_b12_minus_b16"] = (
            frame[f"b28_b12_{view}"] - frame[f"b28_b16_{view}"]
        )
    frame["b28_branch_disagreement"] = disagreement
    advantages = frame.loc[
        disagreement,
        [f"b28_{view}_advantage_b12_minus_b16" for view in VIEWS],
    ].to_numpy(float)
    if not np.isfinite(advantages).all():
        raise RuntimeError("B28 non-finite branch spectral evidence")
    return frame


def apply_policy(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy not in POLICIES:
        raise ValueError(policy)
    output = frame.copy()
    disagree = output["b28_branch_disagreement"].to_numpy(bool)
    b17 = output["final_candidate_id"].astype(str).to_numpy(copy=True)
    b12 = output["b12_final_candidate_id"].astype(str).to_numpy()
    b16 = output["b16_final_candidate_id"].astype(str).to_numpy()
    if policy == "B17":
        proposal = b17.copy()
    else:
        if policy == "direct":
            advantage = output["b28_truncated_direct_advantage_b12_minus_b16"].to_numpy(float)
        elif policy == "neutral_loss":
            advantage = output["b28_neutral_loss_advantage_b12_minus_b16"].to_numpy(float)
        elif policy == "modified_cosine":
            advantage = output["b28_modified_cosine_advantage_b12_minus_b16"].to_numpy(float)
        elif policy == "dual_view":
            advantage = output["b28_dual_view_advantage_b12_minus_b16"].to_numpy(float)
        elif policy == "three_view_mean":
            advantage = output[[
                "b28_truncated_direct_advantage_b12_minus_b16",
                "b28_neutral_loss_advantage_b12_minus_b16",
                "b28_modified_cosine_advantage_b12_minus_b16",
            ]].to_numpy(float).mean(axis=1)
        else:
            local = output[[
                "b28_truncated_direct_advantage_b12_minus_b16",
                "b28_neutral_loss_advantage_b12_minus_b16",
                "b28_modified_cosine_advantage_b12_minus_b16",
            ]].to_numpy(float)
            advantage = np.sum(local > 0, axis=1) - np.sum(local < 0, axis=1)
        proposal = b17.copy()
        proposal[disagree & (advantage > 0)] = b12[disagree & (advantage > 0)]
        proposal[disagree & (advantage < 0)] = b16[disagree & (advantage < 0)]
        # Exact ties retain the frozen B17 decision.
    output["B17_final_candidate_id"] = b17
    output["b28_policy"] = policy
    output["b28_changed"] = proposal != b17
    output["final_candidate_id"] = proposal
    output["intervene"] = output["final_candidate_id"].astype(str).ne(
        output["baseline_candidate_id"].astype(str)
    )
    output["final_correct"] = output["final_candidate_id"].astype(str).eq(
        output["truth_candidate_id"].astype(str)
    )
    output["corrected"] = ~output["baseline_correct"].astype(bool) & output["final_correct"]
    output["introduced"] = output["baseline_correct"].astype(bool) & ~output["final_correct"]
    output["delta"] = output["final_correct"].astype(int) - output["baseline_correct"].astype(int)
    return output


def physical_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    local = frame.drop_duplicates(["source", "physical_query_id"])
    corrected = int(local["corrected"].sum())
    introduced = int(local["introduced"].sum())
    return {
        "physical_queries": int(len(local)),
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "delta_recall1": float(local["delta"].mean()),
    }


def choose_policy(inner: pd.DataFrame) -> tuple[str, list[dict]]:
    ledger = []
    for policy in POLICIES:
        result = apply_policy(inner, policy)
        overall = physical_summary(result)
        by_domain = {
            domain: physical_summary(result.loc[result["source"].eq(domain)])
            for domain in sorted(result["source"].unique())
        }
        ledger.append({
            "policy": policy, **overall,
            "changed": int(result["b28_changed"].sum()),
            "every_inner_domain_risk_nonnegative": all(
                value["risk_net_lambda2"] >= 0 for value in by_domain.values()
            ),
        })
    eligible = [
        item for item in ledger
        if item["every_inner_domain_risk_nonnegative"]
        and item["corrected"] > 2 * item["introduced"]
    ]
    pool = eligible if eligible else ledger
    selected = max(pool, key=lambda item: (
        item["risk_net_lambda2"] / max(1, item["physical_queries"]),
        item["delta_recall1"],
        -item["introduced"] / max(1, item["physical_queries"]),
        item["policy"] == "B17",
    ))
    return str(selected["policy"]), ledger


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    frame = materialise_branch_evidence(args)
    results, folds = [], []
    for domain in EXPECTED_DOMAINS:
        inner = frame.loc[~frame["source"].eq(domain)].copy()
        outer = frame.loc[frame["source"].eq(domain)].copy()
        policy, ledger = choose_policy(inner)
        result = apply_policy(outer, policy)
        results.append(result)
        folds.append({
            "outer_domain": domain,
            "selected_policy": policy,
            "inner_ledger": ledger,
            "outer": summarize(result),
            "outer_physical": physical_summary(result),
            "outer_changed": int(result["b28_changed"].sum()),
        })
        print(f"[B28 {domain}] policy={policy} {summarize(result)}", flush=True)
    result = pd.concat(results, ignore_index=True)
    if len(result) != 860 or result["query_id"].nunique() != 860:
        raise RuntimeError("B28 nested outer coverage changed")
    overall = summarize(result)
    corrected = result.loc[result["corrected"]]
    introduced = result.loc[result["introduced"]]
    identity_ci = cluster_bootstrap(result, "truth_candidate_id", args.bootstrap_resamples, args.seed + 1)
    formula_ci = cluster_bootstrap(result, "truth_formula", args.bootstrap_resamples, args.seed + 2)
    by_domain = {domain: summarize(result.loc[result["source"].eq(domain)]) for domain in EXPECTED_DOMAINS}
    full_pool = []
    for policy in POLICIES:
        local = apply_policy(frame, policy)
        full_pool.append({
            "policy": policy, **summarize(local),
            "physical": physical_summary(local),
            "changed": int(local["b28_changed"].sum()),
        })
    gates = {
        "gain_ge_5pp": overall["delta_recall1"] >= 0.05,
        "identity_ci_low_positive": identity_ci["ci_low"] > 0,
        "formula_ci_low_positive": formula_ci["ci_low"] > 0,
        "corrected_gt_2x_introduced": overall["corrected"] > 2 * overall["introduced"],
        "corrected_identities_ge_25": corrected["truth_candidate_id"].nunique() >= 25,
        "corrected_formulas_ge_25": corrected["truth_formula"].nunique() >= 25,
        "every_outer_domain_nonnegative": all(value["delta_recall1"] >= 0 for value in by_domain.values()),
        "risk_net_strictly_beats_B17": overall["risk_net_lambda2"] > B17_RISK,
        "delta_not_below_B17": overall["delta_recall1"] >= B17_DELTA - 1e-15,
        "introduced_no_more_than_B17": overall["introduced"] <= B17_INTRODUCED,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transition_path = args.output_dir / "nested_domain_loso_transitions.csv.gz"
    result.to_csv(transition_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b28_branch_spectral_arbitration_complete",
        "formal": True,
        "protocol": "B12/B16 disagreements only; exact candidate-reference peak evidence; inner-source policy selection",
        "nested_oof": {
            **overall,
            "physical": physical_summary(result),
            "branch_disagreements": int(result["b28_branch_disagreement"].sum()),
            "changed": int(result["b28_changed"].sum()),
            "corrected_identities": int(corrected["truth_candidate_id"].nunique()),
            "corrected_formulas": int(corrected["truth_formula"].nunique()),
            "introduced_identities": int(introduced["truth_candidate_id"].nunique()),
            "introduced_formulas": int(introduced["truth_formula"].nunique()),
            "identity_cluster_bootstrap": identity_ci,
            "formula_cluster_bootstrap": formula_ci,
            "by_domain": by_domain,
        },
        "frozen_B17_comparator": {"delta_recall1": B17_DELTA, "risk_net_lambda2": B17_RISK, "introduced": B17_INTRODUCED},
        "folds": folds,
        "opened_full_pool_diagnostic_only": full_pool,
        "gates": gates,
        "strictly_better_action_than_B17": bool(all(gates.values())),
        "contracts": {
            "only_B12_B16_disagreements_changed": True,
            "candidate_own_reference_spectrum_used": True,
            "outer_outcome_used_for_policy_selection": False,
            "truth_used_as_action_feature": False,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "B20_report_sha256": sha256(args.b20_dir / "report.json"),
            "B20_manifest_sha256": sha256(args.b20_dir / "direct_action_manifest.npz"),
            "transitions_sha256": sha256(transition_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "Opened nested branch arbitration; not blind, shared-embedding, mechanism, or SOTA evidence.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["strictly_better_action_than_B17"]:
        raise RuntimeError(f"B28 did not strictly improve B17: {gates}")


if __name__ == "__main__":
    main()
