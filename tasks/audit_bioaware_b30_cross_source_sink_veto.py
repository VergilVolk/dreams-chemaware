#!/usr/bin/env python
"""Cross-source annotation-sink veto for the frozen B17 BioAware action.

B17 can repeatedly route unrelated queries into the same catalogue-supported
candidate.  B30 estimates each proposed candidate's action reliability using
other biological sources only.  A recurring candidate is vetoed when it was
observed in at least two development sources, received at least three actions,
and its physical-spectrum risk net (corrected - 2 * introduced) is nonpositive.

The rule is fixed and can only revert a B17 action to official DreaMS.  It does
not learn an unseen-candidate rule; candidates without cross-source history
fall back to B17.  This is therefore a bounded known-candidate safety action.
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
from audit_bioaware_b27_candidate_spectral_veto import load_b20  # noqa: E402


B17_ROW_DELTA = 50 / 860
B17_ROW_RISK = 43
B17_ROW_INTRODUCED = 7
B17_PHYSICAL_RISK = 39
MINIMUM_DEVELOPMENT_SOURCES = 2
MINIMUM_DEVELOPMENT_ACTIONS = 3
RISK_PENALTY = 2


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b20-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def physical(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    local["physical_key"] = (
        local["source"].astype(str) + "::" + local["physical_query_id"].astype(str)
    )
    immutable = (
        "truth_candidate_id", "truth_formula", "baseline_candidate_id",
        "final_candidate_id", "baseline_correct", "final_correct",
        "corrected", "introduced", "intervene",
    )
    for column in immutable:
        if int(local.groupby("physical_key")[column].nunique(dropna=False).max()) != 1:
            raise RuntimeError(f"B30 physical duplicate mismatch: {column}")
    return local.drop_duplicates("physical_key", keep="first").reset_index(drop=True)


def candidate_history(actions: pd.DataFrame) -> pd.DataFrame:
    history = actions.groupby("final_candidate_id", sort=False).agg(
        development_actions=("physical_key", "size"),
        development_sources=("source", "nunique"),
        development_corrected=("corrected", "sum"),
        development_introduced=("introduced", "sum"),
    ).reset_index()
    history["development_neutral"] = (
        history["development_actions"]
        - history["development_corrected"]
        - history["development_introduced"]
    )
    history["development_risk_net_lambda2"] = (
        history["development_corrected"]
        - RISK_PENALTY * history["development_introduced"]
    )
    history["eligible_history"] = (
        history["development_sources"].ge(MINIMUM_DEVELOPMENT_SOURCES)
        & history["development_actions"].ge(MINIMUM_DEVELOPMENT_ACTIONS)
    )
    history["candidate_sink"] = (
        history["eligible_history"]
        & history["development_risk_net_lambda2"].le(0)
    )
    return history


def apply_veto(frame: pd.DataFrame, sinks: set[str]) -> pd.DataFrame:
    output = frame.copy()
    output["B17_final_candidate_id"] = output["final_candidate_id"].astype(str)
    output["b30_candidate_sink"] = output["B17_final_candidate_id"].isin(sinks)
    output["b30_veto"] = output["intervene"].astype(bool) & output["b30_candidate_sink"]
    output["final_candidate_id"] = np.where(
        output["b30_veto"], output["baseline_candidate_id"],
        output["B17_final_candidate_id"],
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


def compact(frame: pd.DataFrame) -> dict[str, float | int]:
    corrected = int(frame["corrected"].sum())
    introduced = int(frame["introduced"].sum())
    return {
        "queries": int(len(frame)), "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - RISK_PENALTY * introduced,
        "delta_recall1": float(frame["delta"].mean()),
        "interventions": int(frame["intervene"].sum()),
        "vetoes": int(frame["b30_veto"].sum()),
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    frame, _, b20_report = load_b20(args.b20_dir)
    physical_frame = physical(frame)
    if len(physical_frame) != 753 or int(physical_frame["intervene"].sum()) != 103:
        raise RuntimeError("B30 physical B17 coverage changed")

    physical_results: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    folds: list[dict] = []
    for outer_source in EXPECTED_DOMAINS:
        development_actions = physical_frame.loc[
            ~physical_frame["source"].eq(outer_source)
            & physical_frame["intervene"].astype(bool)
        ].copy()
        history = candidate_history(development_actions)
        history["outer_source"] = outer_source
        histories.append(history)
        sinks = set(history.loc[history["candidate_sink"], "final_candidate_id"].astype(str))
        outer = apply_veto(
            physical_frame.loc[physical_frame["source"].eq(outer_source)].copy(), sinks
        )
        physical_results.append(outer)
        folds.append({
            "outer_source": outer_source,
            "development_physical_actions": int(len(development_actions)),
            "development_sources": int(development_actions["source"].nunique()),
            "identified_candidate_sinks": sorted(sinks),
            "outer": compact(outer),
        })
        print(f"[B30 {outer_source}] sinks={sorted(sinks)} outer={compact(outer)}", flush=True)

    nested_physical = pd.concat(physical_results, ignore_index=True)
    if len(nested_physical) != 753 or nested_physical["physical_key"].nunique() != 753:
        raise RuntimeError("B30 physical outer coverage changed")
    decisions = nested_physical.set_index("physical_key", verify_integrity=True)[
        ["b30_candidate_sink", "b30_veto"]
    ]
    frame["physical_key"] = (
        frame["source"].astype(str) + "::" + frame["physical_query_id"].astype(str)
    )
    frame = frame.join(decisions, on="physical_key", validate="many_to_one")
    frame["B17_final_candidate_id"] = frame["final_candidate_id"].astype(str)
    frame["final_candidate_id"] = np.where(
        frame["b30_veto"], frame["baseline_candidate_id"], frame["B17_final_candidate_id"]
    )
    frame["intervene"] = frame["final_candidate_id"].astype(str).ne(frame["baseline_candidate_id"].astype(str))
    frame["final_correct"] = frame["final_candidate_id"].astype(str).eq(frame["truth_candidate_id"].astype(str))
    frame["corrected"] = ~frame["baseline_correct"].astype(bool) & frame["final_correct"]
    frame["introduced"] = frame["baseline_correct"].astype(bool) & ~frame["final_correct"]
    frame["delta"] = frame["final_correct"].astype(int) - frame["baseline_correct"].astype(int)
    if frame[["b30_candidate_sink", "b30_veto"]].isna().any().any():
        raise RuntimeError("B30 failed to replay physical decisions")

    overall = summarize(frame)
    physical_overall = compact(nested_physical)
    corrected = frame.loc[frame["corrected"]]
    introduced = frame.loc[frame["introduced"]]
    identity_ci = cluster_bootstrap(frame, "truth_candidate_id", args.bootstrap_resamples, args.seed + 1)
    formula_ci = cluster_bootstrap(frame, "truth_formula", args.bootstrap_resamples, args.seed + 2)
    by_source = {
        source: summarize(frame.loc[frame["source"].eq(source)])
        for source in EXPECTED_DOMAINS
    }
    gates = {
        "row_risk_strictly_beats_B17": overall["risk_net_lambda2"] > B17_ROW_RISK,
        "row_delta_strictly_beats_B17": overall["delta_recall1"] > B17_ROW_DELTA,
        "row_introduced_below_B17": overall["introduced"] < B17_ROW_INTRODUCED,
        "physical_risk_strictly_beats_B17": physical_overall["risk_net_lambda2"] > B17_PHYSICAL_RISK,
        "identity_ci_low_positive": identity_ci["ci_low"] > 0,
        "formula_ci_low_positive": formula_ci["ci_low"] > 0,
        "corrected_identities_ge_25": corrected["truth_candidate_id"].nunique() >= 25,
        "corrected_formulas_ge_25": corrected["truth_formula"].nunique() >= 25,
        "every_outer_source_nonnegative": all(item["delta_recall1"] >= 0 for item in by_source.values()),
        "all_860_rows_and_753_physical_queries": len(frame) == 860 and len(nested_physical) == 753,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    transitions_path = args.output_dir / "nested_sink_veto_transitions.csv.gz"
    history_path = args.output_dir / "cross_source_candidate_history.csv.gz"
    frame.to_csv(transitions_path, index=False, compression="gzip")
    pd.concat(histories, ignore_index=True).to_csv(history_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b30_cross_source_sink_veto_complete",
        "formal": True,
        "protocol": "leave-one-source-out proposed-candidate reliability; physical-spectrum history; fixed minimum 2 sources/3 actions; veto when corrected-2*introduced <= 0",
        "nested_row_oof": {
            **overall,
            "corrected_identities": int(corrected["truth_candidate_id"].nunique()),
            "corrected_formulas": int(corrected["truth_formula"].nunique()),
            "introduced_identities": int(introduced["truth_candidate_id"].nunique()),
            "introduced_formulas": int(introduced["truth_formula"].nunique()),
            "identity_cluster_bootstrap": identity_ci,
            "formula_cluster_bootstrap": formula_ci,
            "by_source": by_source,
        },
        "nested_physical_oof": physical_overall,
        "frozen_B17_comparator": {
            "row_delta_recall1": B17_ROW_DELTA,
            "row_risk_net_lambda2": B17_ROW_RISK,
            "row_introduced": B17_ROW_INTRODUCED,
            "physical_risk_net_lambda2": B17_PHYSICAL_RISK,
        },
        "folds": folds,
        "gates": gates,
        "strictly_better_action_than_B17": bool(all(gates.values())),
        "contracts": {
            "candidate_sink_history_excludes_outer_source": True,
            "physical_duplicates_have_one_history_vote": True,
            "router_can_only_revert_B17": True,
            "unseen_candidate_falls_back_to_B17": True,
            "outer_outcome_used_for_sink_definition": False,
            "candidate_identity_is_explicit_known_candidate_memory": True,
            "unseen_candidate_generalisation_claimed": False,
            "P2b_used": False, "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "B20_report_sha256": sha256(args.b20_dir / "report.json"),
            "B20_actions_sha256": sha256(args.b20_dir / "direct_actions.csv.gz"),
            "B20_manifest_sha256": sha256(args.b20_dir / "direct_action_manifest.npz"),
            "transitions_sha256": sha256(transitions_path),
            "candidate_history_sha256": sha256(history_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "Opened cross-source known-candidate safety action. It cannot establish unseen-candidate generalisation, blind performance, reaction mechanism, SOTA, or shared-embedding improvement.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["strictly_better_action_than_B17"]:
        raise RuntimeError(f"B30 did not strictly improve B17: {gates}")


if __name__ == "__main__":
    main()
