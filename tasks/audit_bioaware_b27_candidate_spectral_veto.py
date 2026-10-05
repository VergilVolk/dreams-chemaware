#!/usr/bin/env python
"""Nested candidate-spectrum veto for the frozen B17 BioAware action.

B9 tested query-to-reaction-neighbour spectral evidence.  B27 asks a different
question: when B17 proposes replacing the official DreaMS Top-1, does the
proposed candidate's own exact reference spectrum support that switch more than
the baseline candidate's reference spectrum?

Seven fixed, label-free veto policies are selected on inner source domains and
then applied once to the outer source.  A veto can only revert to DreaMS; it
cannot introduce a truth-selected candidate or create new action coverage.
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
from audit_bioaware_b9_reaction_spectral_specificity import reaction_spectral_views  # noqa: E402


B17_DELTA = 57 / 860 - 7 / 860
B17_RISK = 57 - 2 * 7
B17_INTRODUCED = 7
VIEW_NAMES = ("truncated_direct", "neutral_loss", "modified_cosine")
POLICIES = (
    "B17",
    "veto_unanimous_baseline",
    "require_any_support",
    "require_majority_support",
    "require_unanimous_support",
    "require_mean_advantage_gt_0",
    "require_mean_advantage_ge_0_01",
    "require_dual_view_advantage_gt_0",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b20-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def load_b20(directory: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    paths = (
        directory / "report.json",
        directory / "direct_actions.csv.gz",
        directory / "direct_action_manifest.npz",
    )
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(paths[0].read_text(encoding="utf-8"))
    if (report.get("status") != "bioaware_b20_direct_action_manifest_complete"
            or report.get("pass_to_direct_gradient_canary") is not True):
        raise RuntimeError("B20 is not a passing exact action manifest")
    if report["provenance"].get("manifest_sha256") != sha256(paths[2]):
        raise RuntimeError("B20 manifest hash mismatch")
    frame = pd.read_csv(paths[1])
    with np.load(paths[2], allow_pickle=False) as handle:
        body = {name: handle[name] for name in handle.files}
    if len(frame) != 860 or len(body.get("query_id", [])) != 860:
        raise RuntimeError("B20 coverage changed")
    if not frame["query_id"].astype(str).equals(pd.Series(body["query_id"].astype(str))):
        raise RuntimeError("B20 table/tensor order mismatch")
    return frame, body, report


def add_candidate_spectral_evidence(
    frame: pd.DataFrame, body: dict[str, np.ndarray],
) -> pd.DataFrame:
    output = frame.copy()
    query_tensor = np.asarray(body["query_tensor"], dtype=np.float32)
    reference_tensor = np.asarray(body["reference_tensor"], dtype=np.float32)
    baseline_position = np.asarray(body["baseline_reference_position"], dtype=np.int64)
    final_position = np.asarray(body["final_reference_position"], dtype=np.int64)
    cache: dict[tuple[int, int], dict[str, float]] = {}
    values = {f"b27_{name}_baseline": [] for name in (*VIEW_NAMES, "dual_view")}
    values |= {f"b27_{name}_final": [] for name in (*VIEW_NAMES, "dual_view")}
    for index in range(len(output)):
        baseline_key = (index, int(baseline_position[index]))
        final_key = (index, int(final_position[index]))
        if baseline_key not in cache:
            cache[baseline_key] = reaction_spectral_views(
                query_tensor[index], reference_tensor[baseline_key[1]],
            )
        if final_key not in cache:
            cache[final_key] = reaction_spectral_views(
                query_tensor[index], reference_tensor[final_key[1]],
            )
        for name in (*VIEW_NAMES, "dual_view"):
            values[f"b27_{name}_baseline"].append(cache[baseline_key][name])
            values[f"b27_{name}_final"].append(cache[final_key][name])
    for name, column in values.items():
        output[name] = np.asarray(column, dtype=np.float64)
    for name in (*VIEW_NAMES, "dual_view"):
        output[f"b27_{name}_advantage"] = (
            output[f"b27_{name}_final"] - output[f"b27_{name}_baseline"]
        )
    advantage = output[[f"b27_{name}_advantage" for name in VIEW_NAMES]].to_numpy(float)
    if not np.isfinite(advantage).all():
        raise RuntimeError("B27 non-finite candidate spectral evidence")
    output["b27_support_votes"] = np.sum(advantage > 0, axis=1).astype(np.int8)
    output["b27_mean_advantage"] = np.mean(advantage, axis=1)
    return output


def policy_keep(frame: pd.DataFrame, policy: str) -> np.ndarray:
    if policy not in POLICIES:
        raise ValueError(policy)
    advantage = frame[[f"b27_{name}_advantage" for name in VIEW_NAMES]].to_numpy(float)
    votes = frame["b27_support_votes"].to_numpy(int)
    if policy == "B17":
        keep = np.ones(len(frame), dtype=bool)
    elif policy == "veto_unanimous_baseline":
        keep = ~np.all(advantage < 0, axis=1)
    elif policy == "require_any_support":
        keep = votes >= 1
    elif policy == "require_majority_support":
        keep = votes >= 2
    elif policy == "require_unanimous_support":
        keep = votes >= 3
    elif policy == "require_mean_advantage_gt_0":
        keep = frame["b27_mean_advantage"].to_numpy(float) > 0
    elif policy == "require_mean_advantage_ge_0_01":
        keep = frame["b27_mean_advantage"].to_numpy(float) >= 0.01
    else:
        keep = frame["b27_dual_view_advantage"].to_numpy(float) > 0
    # Rows on which B17 never switched are immutable under every veto.
    return keep | ~frame["intervene"].to_numpy(bool)


def apply_policy(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    output = frame.copy()
    keep = policy_keep(output, policy)
    output["B17_final_candidate_id"] = output["final_candidate_id"].astype(str)
    output["b27_policy"] = policy
    output["b27_veto"] = output["intervene"].astype(bool) & ~keep
    output["final_candidate_id"] = np.where(
        keep,
        output["B17_final_candidate_id"].astype(str),
        output["baseline_candidate_id"].astype(str),
    )
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
    physical = frame.drop_duplicates(["source", "physical_query_id"])
    corrected = int(physical["corrected"].sum())
    introduced = int(physical["introduced"].sum())
    return {
        "physical_queries": int(len(physical)),
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "delta_recall1": float(physical["delta"].mean()),
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
            "policy": policy,
            **overall,
            "vetoed_rows": int(result["b27_veto"].sum()),
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
    frame, body, b20_report = load_b20(args.b20_dir)
    frame = add_candidate_spectral_evidence(frame, body)

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
            "outer_vetoed_rows": int(result["b27_veto"].sum()),
        })
        print(f"[B27 {domain}] policy={policy} {summarize(result)}", flush=True)

    result = pd.concat(results, ignore_index=True)
    if len(result) != 860 or result["query_id"].nunique() != 860:
        raise RuntimeError("B27 nested outer coverage changed")
    b17 = apply_policy(frame, "B17")
    replay = result[["query_id"]].merge(
        b17[["query_id", "baseline_candidate_id", "truth_candidate_id"]],
        on="query_id", validate="one_to_one",
    )
    if len(replay) != 860:
        raise RuntimeError("B27 B17 query replay mismatch")

    overall = summarize(result)
    physical = physical_summary(result)
    corrected = result.loc[result["corrected"]]
    introduced = result.loc[result["introduced"]]
    identity_ci = cluster_bootstrap(
        result, "truth_candidate_id", args.bootstrap_resamples, args.seed + 1,
    )
    formula_ci = cluster_bootstrap(
        result, "truth_formula", args.bootstrap_resamples, args.seed + 2,
    )
    by_domain = {
        domain: summarize(result.loc[result["source"].eq(domain)])
        for domain in EXPECTED_DOMAINS
    }
    full_pool = []
    for policy in POLICIES:
        local = apply_policy(frame, policy)
        full_pool.append({
            "policy": policy,
            **summarize(local),
            "physical": physical_summary(local),
            "vetoed_rows": int(local["b27_veto"].sum()),
        })
    gates = {
        "gain_ge_5pp": overall["delta_recall1"] >= 0.05,
        "identity_ci_low_positive": identity_ci["ci_low"] > 0,
        "formula_ci_low_positive": formula_ci["ci_low"] > 0,
        "corrected_gt_2x_introduced": overall["corrected"] > 2 * overall["introduced"],
        "corrected_identities_ge_25": corrected["truth_candidate_id"].nunique() >= 25,
        "corrected_formulas_ge_25": corrected["truth_formula"].nunique() >= 25,
        "every_outer_domain_nonnegative": all(
            value["delta_recall1"] >= 0 for value in by_domain.values()
        ),
        "risk_net_strictly_beats_B17": overall["risk_net_lambda2"] > B17_RISK,
        "delta_not_below_B17": overall["delta_recall1"] >= B17_DELTA - 1e-15,
        "introduced_no_more_than_B17": overall["introduced"] <= B17_INTRODUCED,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transition_path = args.output_dir / "nested_domain_loso_transitions.csv.gz"
    result.to_csv(transition_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b27_candidate_spectral_veto_complete",
        "formal": True,
        "protocol": (
            "candidate-own-spectrum evidence; fixed veto selected on inner source domains; "
            "outer source evaluated once"
        ),
        "nested_oof": {
            **overall,
            "physical": physical,
            "corrected_identities": int(corrected["truth_candidate_id"].nunique()),
            "corrected_formulas": int(corrected["truth_formula"].nunique()),
            "introduced_identities": int(introduced["truth_candidate_id"].nunique()),
            "introduced_formulas": int(introduced["truth_formula"].nunique()),
            "identity_cluster_bootstrap": identity_ci,
            "formula_cluster_bootstrap": formula_ci,
            "by_domain": by_domain,
        },
        "frozen_B17_comparator": {
            "delta_recall1": B17_DELTA,
            "risk_net_lambda2": B17_RISK,
            "introduced": B17_INTRODUCED,
        },
        "folds": folds,
        "opened_full_pool_diagnostic_only": full_pool,
        "gates": gates,
        "strictly_better_action_than_B17": bool(all(gates.values())),
        "contracts": {
            "candidate_own_reference_spectrum_used": True,
            "veto_only_reverts_to_DreaMS": True,
            "outer_outcome_used_for_policy_selection": False,
            "truth_used_as_action_feature": False,
            "reaction_neighbour_spectrum_used": False,
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
        "claim_limit": (
            "Opened nested action validation. It is not blind validation, a reaction "
            "mechanism, shared-embedding improvement, or a SOTA result."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["strictly_better_action_than_B17"]:
        raise RuntimeError(f"B27 did not strictly improve B17: {gates}")


if __name__ == "__main__":
    main()
