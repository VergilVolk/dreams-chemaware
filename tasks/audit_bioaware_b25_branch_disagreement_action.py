#!/usr/bin/env python
"""Nested test of one B12/B16 disagreement action beyond BioAware B17.

B17 selects B12/B16 policies at domain level.  On queries where their proposed
candidates differ, B25 tests one fixed candidate-local arbitration: keep the
candidate with larger known catalogue log-degree.  Every outer domain chooses
between frozen B17 and this arbitration using inner-domain OOF results only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b11_catalog_interaction_action import (  # noqa: E402
    atomic_json, cluster_bootstrap, sha256, summarize,
)
from audit_bioaware_b12_multicohort_catalog_action import (  # noqa: E402
    EXPECTED_DOMAINS, build_universe,
)
from audit_bioaware_b14_reaction_context_action import add_reaction_features  # noqa: E402
from audit_bioaware_b16_pairwise_nonlinear_action import prepare  # noqa: E402
from audit_bioaware_b22_crossfit_action_selector import (  # noqa: E402
    B17_DELTA, B17_INTRODUCED, B17_RISK_NET, reconstruct_b17_fold,
)


def attach_branch_degrees(base: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    lookup = candidates[["query_id", "candidate_id", "known_log_degree"]].copy()
    lookup["query_id"] = lookup["query_id"].astype(str)
    lookup["candidate_id"] = lookup["candidate_id"].astype(str)
    if lookup.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("B25 duplicate query/candidate rows")
    left = lookup.rename(columns={
        "candidate_id": "b12_final_candidate_id",
        "known_log_degree": "b12_known_log_degree",
    })
    right = lookup.rename(columns={
        "candidate_id": "b16_final_candidate_id",
        "known_log_degree": "b16_known_log_degree",
    })
    output = base.merge(
        left, on=["query_id", "b12_final_candidate_id"], validate="one_to_one"
    ).merge(
        right, on=["query_id", "b16_final_candidate_id"], validate="one_to_one"
    )
    if len(output) != len(base):
        raise RuntimeError("B25 branch-candidate join changed coverage")
    return output


def apply_arbitration(
    base: pd.DataFrame, candidates: pd.DataFrame, enabled: bool,
) -> pd.DataFrame:
    output = attach_branch_degrees(base, candidates)
    disagree = output["b12_final_candidate_id"].astype(str).ne(
        output["b16_final_candidate_id"].astype(str)
    )
    b12_wins = output["b12_known_log_degree"].astype(float).gt(
        output["b16_known_log_degree"].astype(float)
    )
    b16_wins = output["b16_known_log_degree"].astype(float).gt(
        output["b12_known_log_degree"].astype(float)
    )
    proposal = output["final_candidate_id"].astype(str).copy()
    proposal = proposal.where(~b12_wins, output["b12_final_candidate_id"].astype(str))
    proposal = proposal.where(~b16_wins, output["b16_final_candidate_id"].astype(str))
    use = disagree & (b12_wins | b16_wins) if enabled else pd.Series(False, index=output.index)
    output["B17_final_candidate_id"] = output["final_candidate_id"].astype(str)
    output["branch_disagreement"] = disagree.astype(bool)
    output["branch_degree_arbitration"] = use.astype(bool)
    output["branch_proposal_candidate_id"] = proposal.astype(str)
    output["final_candidate_id"] = np.where(
        use, proposal, output["B17_final_candidate_id"]
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
    output["branch_arbitration_enabled"] = bool(enabled)
    return output


def choose_arbitration(
    base: pd.DataFrame, candidates: pd.DataFrame,
) -> tuple[bool, list[dict]]:
    ledger: list[dict] = []
    for enabled in (False, True):
        result = apply_arbitration(base, candidates, enabled)
        overall = summarize(result)
        by_domain = {
            source: summarize(result.loc[result["source"].eq(source)])
            for source in sorted(result["source"].unique())
        }
        ledger.append({
            "branch_arbitration_enabled": enabled,
            "disagreements": int(result["branch_disagreement"].sum()),
            "changed_by_arbitration": int(
                result["final_candidate_id"].astype(str).ne(
                    result["B17_final_candidate_id"].astype(str)
                ).sum()
            ),
            **overall,
            "every_inner_domain_risk_nonnegative": all(
                item["risk_net_lambda2"] >= 0 for item in by_domain.values()
            ),
        })
    eligible = [
        item for item in ledger
        if item["every_inner_domain_risk_nonnegative"]
        and item["corrected"] > 2 * item["introduced"]
    ]
    pool = eligible if eligible else ledger
    selected = max(pool, key=lambda item: (
        item["risk_net_lambda2"] / max(1, item["queries"]),
        -item["introduced"] / max(1, item["queries"]),
        item["delta_recall1"],
        not item["branch_arbitration_enabled"],
    ))
    return bool(selected["branch_arbitration_enabled"]), ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-candidates", type=Path, default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz")
    parser.add_argument("--st-candidates", type=Path, default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz")
    parser.add_argument("--st-queries", type=Path, default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz")
    parser.add_argument("--kgmn-candidates", type=Path, default=ROOT / "data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz")
    parser.add_argument("--kgmn-seeds", type=Path, default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz")
    parser.add_argument("--b17-transitions", type=Path, default=ROOT / "data/validation/bioaware_b17_nested_union_localcheck_20260907_v2/nested_domain_loso_transitions.csv.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    for path in (args.internal_candidates, args.st_candidates, args.st_queries,
                 args.kgmn_candidates, args.kgmn_seeds, args.b17_transitions):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    candidates, provenance = build_universe(args)
    candidates = prepare(add_reaction_features(candidates))
    candidates["known_log_degree"] = pd.to_numeric(
        candidates["known_log_degree"], errors="coerce"
    ).fillna(0.0)

    results: list[pd.DataFrame] = []
    bases: list[pd.DataFrame] = []
    folds: list[dict] = []
    for outer_index, domain in enumerate(EXPECTED_DOMAINS):
        base_outer, outer_candidates, base_inner, inner_candidates, components = (
            reconstruct_b17_fold(candidates, domain, outer_index, args.seed)
        )
        enabled, ledger = choose_arbitration(base_inner, inner_candidates)
        result = apply_arbitration(base_outer, outer_candidates, enabled)
        results.append(result)
        bases.append(base_outer)
        folds.append({
            "outer_domain": domain,
            "B17_components": components,
            "branch_arbitration_selected": enabled,
            "inner_arbitration_ledger": ledger,
            "outer_B17_replay": summarize(base_outer),
            "outer_result": summarize(result),
            "outer_disagreements": int(result["branch_disagreement"].sum()),
            "outer_changed": int(result["final_candidate_id"].astype(str).ne(
                result["B17_final_candidate_id"].astype(str)
            ).sum()),
        })
        print(f"[B25 {domain}] enabled={enabled} {summarize(result)}", flush=True)

    result = pd.concat(results, ignore_index=True)
    base_replay = pd.concat(bases, ignore_index=True)
    frozen = pd.read_csv(args.b17_transitions)
    check = base_replay[["query_id", "final_candidate_id", "corrected", "introduced"]].merge(
        frozen[["query_id", "final_candidate_id", "corrected", "introduced"]],
        on="query_id", suffixes=("_replay", "_frozen"), validate="one_to_one",
    )
    replay_mismatches = {
        column: int(check[f"{column}_replay"].astype(str).ne(
            check[f"{column}_frozen"].astype(str)
        ).sum())
        for column in ("final_candidate_id", "corrected", "introduced")
    }
    if any(replay_mismatches.values()):
        raise RuntimeError(f"B25 B17 replay mismatch: {replay_mismatches}")
    if len(result) != 860 or result["query_id"].nunique() != 860:
        raise RuntimeError("B25 outer coverage changed")

    overall = summarize(result)
    corrected = result.loc[result["corrected"]]
    introduced = result.loc[result["introduced"]]
    identity_ci = cluster_bootstrap(result, "truth_candidate_id", args.bootstrap_resamples, args.seed + 1)
    formula_ci = cluster_bootstrap(result, "truth_formula", args.bootstrap_resamples, args.seed + 2)
    by_domain = {source: summarize(result.loc[result["source"].eq(source)]) for source in EXPECTED_DOMAINS}
    gates = {
        "gain_ge_5pp": overall["delta_recall1"] >= 0.05,
        "identity_ci_low_positive": identity_ci["ci_low"] > 0,
        "formula_ci_low_positive": formula_ci["ci_low"] > 0,
        "corrected_gt_2x_introduced": overall["corrected"] > 2 * overall["introduced"],
        "corrected_identities_ge_25": corrected["truth_candidate_id"].nunique() >= 25,
        "corrected_formulas_ge_25": corrected["truth_formula"].nunique() >= 25,
        "every_outer_domain_nonnegative": all(item["delta_recall1"] >= 0 for item in by_domain.values()),
        "risk_net_strictly_beats_b17": overall["risk_net_lambda2"] > B17_RISK_NET,
        "delta_not_below_b17": overall["delta_recall1"] >= B17_DELTA - 1e-15,
        "introduced_no_more_than_b17": overall["introduced"] <= B17_INTRODUCED,
        "frozen_B17_querywise_replay_exact": not any(replay_mismatches.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transition_path = args.output_dir / "nested_domain_loso_transitions.csv.gz"
    result.to_csv(transition_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b25_branch_disagreement_action_complete",
        "formal": True,
        "protocol": "B12/B16 disagreement arbitration by known_log_degree; none/action selected on inner domain-OOF rows",
        "nested_oof": {**overall,
            "branch_disagreements": int(result["branch_disagreement"].sum()),
            "changed_by_arbitration": int(result["final_candidate_id"].astype(str).ne(result["B17_final_candidate_id"].astype(str)).sum()),
            "corrected_identities": int(corrected["truth_candidate_id"].nunique()),
            "corrected_formulas": int(corrected["truth_formula"].nunique()),
            "introduced_identities": int(introduced["truth_candidate_id"].nunique()),
            "introduced_formulas": int(introduced["truth_formula"].nunique()),
            "identity_cluster_bootstrap": identity_ci,
            "formula_cluster_bootstrap": formula_ci,
            "by_domain": by_domain,
        },
        "frozen_B17_comparator": {"delta_recall1": B17_DELTA, "risk_net_lambda2": B17_RISK_NET, "introduced": B17_INTRODUCED, "querywise_replay_mismatches": replay_mismatches},
        "folds": folds,
        "gates": gates,
        "strictly_better_action_than_B17": bool(all(gates.values())),
        "contracts": {"outer_outcome_used_for_action_selection": False, "candidate_identity_as_feature": False, "P2b_used": False, "phenotype_used": False, "shared_embedding_changed": False},
        "provenance": {**provenance["provenance"], "B17_transitions_sha256": sha256(args.b17_transitions), "transitions_sha256": sha256(transition_path), "script_sha256": sha256(Path(__file__))},
        "claim_limit": "Opened nested branch-action test; not blind validation, mechanism proof, SOTA, or shared-embedding improvement.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["strictly_better_action_than_B17"]:
        raise RuntimeError(f"B25 did not strictly improve B17: {gates}")


if __name__ == "__main__":
    main()
