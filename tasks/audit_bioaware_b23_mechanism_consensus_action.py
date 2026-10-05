#!/usr/bin/env python
"""Test a nested, typed-mechanism consensus fallback beyond BioAware B17.

B19 showed that many residual errors are reachable by raw evidence actions, but
the same actions can also damage correct queries.  B22 showed that a generic
candidate classifier does not transport beyond B17.  B23 therefore tests the
lower-capacity scientific hypothesis: intervene only when independently typed
catalogue, reaction and co-abundance mechanisms nominate the same candidate.

Every configuration is selected on inner domain-OOF rows.  The held outer
domain is used once, and the fallback can act only where B17 abstains.
"""
from __future__ import annotations

import argparse
import itertools
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
    ACTION_RULES, B17_DELTA, B17_INTRODUCED, B17_RISK_NET,
    reconstruct_b17_fold,
)


MECHANISM_FAMILIES = {
    "catalogue_mass_coverage": "catalogue",
    "catalogue_membership": "catalogue",
    "catalogue_degree": "catalogue",
    "reaction_path": "reaction",
    "reaction_edge0_reliability": "reaction",
    "reaction_edge1_bottleneck": "reaction",
    "predicted_edge_increment": "reaction",
    "coabundance_neighbour_support": "coabundance",
    "coabundance_multiwitness": "coabundance",
}
CONFIGS = (
    None,
    *tuple(
        {
            "minimum_families": families,
            "minimum_votes": votes,
            "maximum_baseline_gap": margin,
            "maximum_spectral_loss": spectral_loss,
        }
        for families, votes, margin, spectral_loss in itertools.product(
            (2, 3), (2, 3, 4), (0.04, 0.05, 0.08), (0.02, 0.05, 0.10)
        )
    ),
)


def consensus_proposals(candidates: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    required = {"query_id", "candidate_id", "spectral_score", "baseline_gap"}
    required |= {feature for feature, _ in ACTION_RULES.values()}
    if missing := required - set(candidates):
        raise RuntimeError(f"B23 candidates miss columns: {sorted(missing)}")
    if base["query_id"].duplicated().any():
        raise RuntimeError("B23 base state has duplicate query IDs")
    base_by_query = base.set_index("query_id")
    rows: list[dict] = []
    for query_id, group in candidates.groupby("query_id", sort=False):
        query_id = str(query_id)
        if query_id not in base_by_query.index:
            raise RuntimeError(f"B23 candidate query absent from base: {query_id}")
        state = base_by_query.loc[query_id]
        base_candidate = str(state["final_candidate_id"])
        support: dict[str, set[str]] = {
            str(candidate): set() for candidate in group["candidate_id"]
        }
        for action, (feature, direction) in ACTION_RULES.items():
            values = group[feature].to_numpy(float)
            optimum = float(np.max(values) if direction == "max" else np.min(values))
            top = group.loc[np.isclose(values, optimum, rtol=0, atol=1e-12)]
            if len(top) == 1:
                support[str(top["candidate_id"].iloc[0])].add(action)
        table: list[dict] = []
        for candidate_id, actions in support.items():
            families = {MECHANISM_FAMILIES[action] for action in actions}
            table.append({
                "candidate_id": candidate_id,
                "votes": len(actions),
                "families": len(families),
                "catalogue_support": int("catalogue" in families),
                "reaction_support": int("reaction" in families),
                "coabundance_support": int("coabundance" in families),
            })
        table.sort(key=lambda item: (item["families"], item["votes"]), reverse=True)
        best = table[0]
        tied = [
            item for item in table
            if (item["families"], item["votes"]) == (best["families"], best["votes"])
        ]
        proposal_unique = len(tied) == 1
        proposal_candidate = str(best["candidate_id"])
        score_by_candidate = group.set_index("candidate_id")["spectral_score"].astype(float)
        if base_candidate not in score_by_candidate.index:
            raise RuntimeError(f"B23 B17 candidate absent from candidates: {query_id}")
        rows.append({
            "query_id": query_id,
            "proposal_candidate_id": proposal_candidate,
            "proposal_unique": proposal_unique,
            "proposal_votes": int(best["votes"]),
            "proposal_families": int(best["families"]),
            "proposal_catalogue_support": int(best["catalogue_support"]),
            "proposal_reaction_support": int(best["reaction_support"]),
            "proposal_coabundance_support": int(best["coabundance_support"]),
            "proposal_spectral_loss": float(
                score_by_candidate.loc[base_candidate]
                - score_by_candidate.loc[proposal_candidate]
            ),
            "consensus_baseline_gap": float(group["baseline_gap"].iloc[0]),
        })
    output = pd.DataFrame(rows)
    if output["query_id"].duplicated().any():
        raise RuntimeError("B23 duplicate consensus proposal")
    return output


def apply_consensus(
    base: pd.DataFrame,
    candidates: pd.DataFrame,
    config: dict | None,
) -> pd.DataFrame:
    output = base.merge(
        consensus_proposals(candidates, base), on="query_id", validate="one_to_one"
    )
    if config is None:
        use = pd.Series(False, index=output.index)
    else:
        use = (
            ~output["intervene"].astype(bool)
            & output["proposal_unique"].astype(bool)
            & output["proposal_candidate_id"].astype(str).ne(
                output["final_candidate_id"].astype(str)
            )
            & output["proposal_families"].astype(int).ge(config["minimum_families"])
            & output["proposal_votes"].astype(int).ge(config["minimum_votes"])
            & output["consensus_baseline_gap"].astype(float).le(
                config["maximum_baseline_gap"] + 1e-15
            )
            & output["proposal_spectral_loss"].astype(float).le(
                config["maximum_spectral_loss"] + 1e-15
            )
        )
    output["B17_intervene"] = output["intervene"].astype(bool)
    output["B17_final_candidate_id"] = output["final_candidate_id"].astype(str)
    output["consensus_intervene"] = use.astype(bool)
    output["final_candidate_id"] = np.where(
        use, output["proposal_candidate_id"], output["B17_final_candidate_id"]
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
    output["consensus_config"] = "none" if config is None else json.dumps(
        config, sort_keys=True, separators=(",", ":")
    )
    return output


def config_label(config: dict | None) -> str:
    return "none" if config is None else json.dumps(
        config, sort_keys=True, separators=(",", ":")
    )


def choose_config(
    base: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[dict | None, list[dict]]:
    ledger: list[dict] = []
    for config in CONFIGS:
        result = apply_consensus(base, candidates, config)
        overall = summarize(result)
        by_domain = {
            source: summarize(result.loc[result["source"].eq(source)])
            for source in sorted(result["source"].unique())
        }
        ledger.append({
            "configuration": config_label(config), **overall,
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
        item["configuration"] == "none",
    ))
    if selected["configuration"] == "none":
        return None, ledger
    return json.loads(selected["configuration"]), ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-queries", type=Path,
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
    parser.add_argument(
        "--b17-transitions", type=Path,
        default=ROOT / "data/validation/bioaware_b17_nested_union_localcheck_20260907_v2/nested_domain_loso_transitions.csv.gz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    for path in (
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds, args.b17_transitions,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    candidates, provenance = build_universe(args)
    candidates = prepare(add_reaction_features(candidates))
    for feature in {"spectral_score", "baseline_gap"} | {
        item[0] for item in ACTION_RULES.values()
    }:
        candidates[feature] = pd.to_numeric(
            candidates[feature], errors="coerce"
        ).fillna(0.0)

    outer_results: list[pd.DataFrame] = []
    base_replays: list[pd.DataFrame] = []
    folds: list[dict] = []
    for outer_index, outer_domain in enumerate(EXPECTED_DOMAINS):
        base_outer, outer_candidates, base_inner, inner_candidates, components = (
            reconstruct_b17_fold(candidates, outer_domain, outer_index, args.seed)
        )
        selected, ledger = choose_config(base_inner, inner_candidates)
        result = apply_consensus(base_outer, outer_candidates, selected)
        outer_results.append(result)
        base_replays.append(base_outer)
        folds.append({
            "outer_domain": outer_domain,
            "B17_components": components,
            "selected_configuration": config_label(selected),
            "inner_configuration_ledger": ledger,
            "outer_B17_replay": summarize(base_outer),
            "outer_result": summarize(result),
        })
        print(
            f"[B23 {outer_domain}] config={config_label(selected)} "
            f"outer={summarize(result)}", flush=True,
        )

    result = pd.concat(outer_results, ignore_index=True)
    base_replay = pd.concat(base_replays, ignore_index=True)
    frozen = pd.read_csv(args.b17_transitions)
    check = base_replay[[
        "query_id", "final_candidate_id", "corrected", "introduced"
    ]].merge(
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
        raise RuntimeError(f"B23 B17 replay mismatch: {replay_mismatches}")
    if len(result) != 860 or result["query_id"].nunique() != 860:
        raise RuntimeError("B23 outer coverage changed")

    overall = summarize(result)
    corrected = result.loc[result["corrected"]]
    introduced = result.loc[result["introduced"]]
    identity_ci = cluster_bootstrap(
        result, "truth_candidate_id", args.bootstrap_resamples, args.seed + 1
    )
    formula_ci = cluster_bootstrap(
        result, "truth_formula", args.bootstrap_resamples, args.seed + 2
    )
    by_domain = {
        source: summarize(result.loc[result["source"].eq(source)])
        for source in EXPECTED_DOMAINS
    }
    gates = {
        "gain_ge_5pp": overall["delta_recall1"] >= 0.05,
        "identity_ci_low_positive": identity_ci["ci_low"] > 0,
        "formula_ci_low_positive": formula_ci["ci_low"] > 0,
        "corrected_gt_2x_introduced": overall["corrected"] > 2 * overall["introduced"],
        "corrected_identities_ge_25": corrected["truth_candidate_id"].nunique() >= 25,
        "corrected_formulas_ge_25": corrected["truth_formula"].nunique() >= 25,
        "every_outer_domain_nonnegative": all(
            item["delta_recall1"] >= 0 for item in by_domain.values()
        ),
        "risk_net_strictly_beats_b17": overall["risk_net_lambda2"] > B17_RISK_NET,
        "delta_not_below_b17": overall["delta_recall1"] >= B17_DELTA - 1e-15,
        "introduced_no_more_than_b17": overall["introduced"] <= B17_INTRODUCED,
        "frozen_B17_querywise_replay_exact": not any(replay_mismatches.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transition_path = args.output_dir / "nested_domain_loso_transitions.csv.gz"
    result.to_csv(transition_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b23_mechanism_consensus_action_complete",
        "formal": True,
        "protocol": (
            "typed catalogue/reaction/coabundance consensus; B17 residual only; "
            "configuration selected on inner domain-OOF rows"
        ),
        "nested_oof": {
            **overall,
            "corrected_identities": int(corrected["truth_candidate_id"].nunique()),
            "corrected_formulas": int(corrected["truth_formula"].nunique()),
            "introduced_identities": int(introduced["truth_candidate_id"].nunique()),
            "introduced_formulas": int(introduced["truth_formula"].nunique()),
            "identity_cluster_bootstrap": identity_ci,
            "formula_cluster_bootstrap": formula_ci,
            "by_domain": by_domain,
        },
        "mechanism_families": MECHANISM_FAMILIES,
        "configurations_preregistered": len(CONFIGS),
        "frozen_B17_comparator": {
            "delta_recall1": B17_DELTA,
            "risk_net_lambda2": B17_RISK_NET,
            "introduced": B17_INTRODUCED,
            "querywise_replay_mismatches": replay_mismatches,
        },
        "folds": folds,
        "gates": gates,
        "strictly_better_action_than_B17": bool(all(gates.values())),
        "contracts": {
            "B19_outcome_used_for_candidate_selection": False,
            "outer_outcome_used_for_configuration_selection": False,
            "fallback_only_when_B17_abstains": True,
            "typed_mechanism_consensus_required": True,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
        },
        "provenance": {
            **provenance["provenance"],
            "B17_transitions_sha256": sha256(args.b17_transitions),
            "transitions_sha256": sha256(transition_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Opened nested consensus-action test. Passing replaces B17 only as "
            "an action router; it is not blind validation, mechanism proof, "
            "SOTA, or shared-embedding improvement."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["strictly_better_action_than_B17"]:
        raise RuntimeError(f"B23 did not strictly improve B17: {gates}")


if __name__ == "__main__":
    main()
