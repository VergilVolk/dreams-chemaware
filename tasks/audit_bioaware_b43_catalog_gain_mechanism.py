#!/usr/bin/env python
"""Decompose B42 gains into catalogue membership and within-catalogue ranking.

This is a read-only analysis of frozen B42 outer-fold transitions.  It asks
whether the apparent gain survives when the truth and the official DreaMS
top candidate are both represented in the same catalogue.  That stratum is
the minimum evidence needed before interpreting degree/topology as more than
a known-metabolite membership prior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from audit_bioaware_b11_catalog_interaction_action import atomic_json, sha256  # noqa: E402
from audit_bioaware_b36_reaction_specificity_ablation import cluster_ci_values  # noqa: E402
from evaluate_bioaware_b39_m2_fixed_action import atomic_csv_gzip  # noqa: E402


ARM_MEMBER = {
    "strict_kegg_archived_replay": "strict_kegg_member",
    "rhea_independent_topology": "rhea_member",
    "kegg_rhea_consensus": "any_kegg_rhea_member",
}


def summarize(frame: pd.DataFrame, repeats: int, seed: int) -> dict[str, Any]:
    if frame.empty:
        return {
            "queries": 0, "identities": 0, "formulas": 0,
            "delta_recall1": None, "corrected": 0, "introduced": 0,
            "risk_net_lambda2": 0, "identity_cluster_bootstrap": None,
            "formula_cluster_bootstrap": None,
        }
    effect = frame["delta"].to_numpy(float)
    return {
        "queries": int(len(frame)),
        "identities": int(frame["truth_candidate_id"].nunique()),
        "formulas": int(frame["truth_formula"].nunique()),
        "baseline_recall1": float(frame["baseline_correct"].astype(float).mean()),
        "final_recall1": float(frame["final_correct"].astype(float).mean()),
        "delta_recall1": float(effect.mean()),
        "corrected": int(frame["corrected"].sum()),
        "introduced": int(frame["introduced"].sum()),
        "risk_net_lambda2": int(frame["corrected"].sum() - 2 * frame["introduced"].sum()),
        "identity_cluster_bootstrap": cluster_ci_values(
            effect, frame["truth_candidate_id"].astype(str), repeats, seed,
        ),
        "formula_cluster_bootstrap": cluster_ci_values(
            effect, frame["truth_formula"].astype(str), repeats, seed + 1,
        ),
    }


def attach_query_features(
    transitions: pd.DataFrame,
    candidates: pd.DataFrame,
    arm: str,
    member_column: str,
) -> pd.DataFrame:
    local = transitions.loc[transitions["arm"].eq(arm)].copy()
    features = candidates.copy()
    if member_column == "any_kegg_rhea_member":
        features[member_column] = (
            features["strict_kegg_member"].astype(float)
            + features["rhea_member"].astype(float)
        ).gt(0).astype(float)
    lookup = features.set_index(["query_id", "candidate_id"])[member_column]

    def pull(candidate_column: str) -> np.ndarray:
        index = pd.MultiIndex.from_arrays([
            local["query_id"].astype(str), local[candidate_column].astype(str)
        ])
        try:
            return lookup.loc[index].to_numpy(float)
        except KeyError as exc:
            raise RuntimeError(f"{arm}: candidate feature lookup failed for {candidate_column}") from exc

    local["truth_member"] = pull("truth_candidate_id")
    local["baseline_member"] = pull("baseline_candidate_id")
    local["final_member"] = pull("final_candidate_id")
    group_membership = features.groupby("query_id")[member_column]
    all_member = group_membership.min().astype(float)
    any_member = group_membership.max().astype(float)
    local["all_candidates_member"] = local["query_id"].map(all_member).astype(float)
    local["any_candidate_member"] = local["query_id"].map(any_member).astype(float)
    local["membership_stratum"] = np.select(
        [
            local["truth_member"].gt(0.5) & local["baseline_member"].le(0.5),
            local["truth_member"].gt(0.5) & local["baseline_member"].gt(0.5),
            local["truth_member"].le(0.5) & local["baseline_member"].gt(0.5),
        ],
        ["truth_only", "both_truth_and_baseline", "baseline_only"],
        default="neither_truth_nor_baseline",
    )
    local["arm_member_column"] = member_column
    return local


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b42-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b42_independent_catalog_topology_localcheck_20260913_v2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    if args.bootstrap_resamples < 10000:
        raise ValueError("formal B43 requires at least 10,000 bootstrap resamples")
    report_path = args.b42_dir / "report.json"
    transitions_path = args.b42_dir / "cross_catalog_transitions.csv.gz"
    candidates_path = args.b42_dir / "candidate_catalog_features.csv.gz"
    for path in (report_path, transitions_path, candidates_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    b42 = json.loads(report_path.read_text(encoding="utf-8"))
    if b42.get("status") != "bioaware_b42_independent_catalog_topology_complete":
        raise RuntimeError("B43 requires completed B42")
    transitions = pd.read_csv(transitions_path, low_memory=False)
    candidates = pd.read_csv(candidates_path, low_memory=False)
    if candidates.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("B42 candidate features are not query/candidate unique")

    attached_parts: list[pd.DataFrame] = []
    arm_reports: dict[str, Any] = {}
    for arm_index, (arm, member_column) in enumerate(ARM_MEMBER.items()):
        local = attach_query_features(transitions, candidates, arm, member_column)
        attached_parts.append(local)
        strata = {
            name: summarize(group.copy(), args.bootstrap_resamples, args.seed + arm_index * 1000 + index * 10)
            for index, (name, group) in enumerate(local.groupby("membership_stratum", sort=True))
        }
        strata["all_candidates_mapped"] = summarize(
            local.loc[local["all_candidates_member"].gt(0.5)].copy(),
            args.bootstrap_resamples, args.seed + arm_index * 1000 + 500,
        )
        strata["mapped_competition"] = summarize(
            local.loc[local["truth_member"].gt(0.5) & local["baseline_member"].gt(0.5)].copy(),
            args.bootstrap_resamples, args.seed + arm_index * 1000 + 510,
        )
        strata["membership_asymmetry"] = summarize(
            local.loc[local["truth_member"].gt(0.5) & local["baseline_member"].le(0.5)].copy(),
            args.bootstrap_resamples, args.seed + arm_index * 1000 + 520,
        )
        arm_reports[arm] = {
            "member_definition": member_column,
            "overall": summarize(local, args.bootstrap_resamples, args.seed + arm_index * 1000 + 530),
            "strata": strata,
        }
    attached = pd.concat(attached_parts, ignore_index=True)

    mechanism_gates: dict[str, Any] = {}
    for arm, values in arm_reports.items():
        mapped = values["strata"]["mapped_competition"]
        asymmetry = values["strata"]["membership_asymmetry"]
        mechanism_gates[arm] = {
            "mapped_competition_ge_100_queries": bool(mapped["queries"] >= 100),
            "mapped_competition_formula_ci_low_positive": bool(
                mapped["formula_cluster_bootstrap"] is not None
                and mapped["formula_cluster_bootstrap"]["ci_low"] > 0
            ),
            "mapped_competition_corrected_gt_introduced": bool(
                mapped["corrected"] > mapped["introduced"]
            ),
            "membership_asymmetry_risk_net_positive": bool(asymmetry["risk_net_lambda2"] > 0),
        }
        mechanism_gates[arm]["within_catalogue_signal_pass"] = bool(all([
            mechanism_gates[arm]["mapped_competition_ge_100_queries"],
            mechanism_gates[arm]["mapped_competition_formula_ci_low_positive"],
            mechanism_gates[arm]["mapped_competition_corrected_gt_introduced"],
        ]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = args.output_dir / "catalog_gain_mechanism_per_query.csv.gz"
    atomic_csv_gzip(per_query_path, attached)
    report = {
        "status": "bioaware_b43_catalog_gain_mechanism_complete",
        "formal": True,
        "protocol": "read-only decomposition of frozen B42 OOF transitions by truth-versus-baseline catalogue membership",
        "arm_reports": arm_reports,
        "mechanism_gates": mechanism_gates,
        "interpretation_rule": {
            "membership_prior": "gain concentrated in truth-only membership asymmetry",
            "within_catalogue_topology": "mapped-competition formula-cluster CI lower bound above zero with corrected > introduced",
        },
        "pass_to_shared_embedding": False,
        "contracts": {
            "models_refit": False,
            "gates_reselected": False,
            "B42_outcomes_changed": False,
            "sample_context_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            "b42_report": sha256(report_path),
            "b42_transitions": sha256(transitions_path),
            "b42_candidate_features": sha256(candidates_path),
            "per_query": sha256(per_query_path),
            "script": sha256(Path(__file__)),
        },
        "claim_limit": "Opened mechanistic decomposition of a candidate-identity prior. It does not establish reaction context, prospective annotation, biological mechanism, shared-embedding gain or SOTA.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps({
        "status": report["status"],
        "arm_reports": arm_reports,
        "mechanism_gates": mechanism_gates,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
