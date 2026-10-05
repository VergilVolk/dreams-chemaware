#!/usr/bin/env python
"""Explain the B36 gain and quantify unused reaction-context headroom.

B37 has two deliberately separate parts.

1. A nested leave-domain-out ablation replays the exact B36 catalogue arm and
   decomposes it into graph-node topology and database-observability features.
2. A model-free, within-query audit asks whether monotone reaction-context
   evidence can distinguish the truth from wrong candidates, both before and
   after the B36 catalogue action.

The second part is an upper-bound/identifiability audit, not a deployable
oracle.  It never calls a reaction feature a successful action merely because
the truth happens to maximize that feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b11_catalog_interaction_action import atomic_json, sha256, summarize  # noqa: E402
from audit_bioaware_b12_multicohort_catalog_action import (  # noqa: E402
    EXPECTED_DOMAINS,
    build_universe,
    split_domain,
)
from audit_bioaware_b36_reaction_specificity_ablation import (  # noqa: E402
    CATALOG_FEATURES,
    REACTION_FEATURES,
    SPECTRAL_FEATURES,
    apply_selected_gate,
    choose_gate,
    paired_contrast,
    prepare_candidates,
    score_arm,
    summarize_arm,
)


EXPECTED_QUERIES = 860
CATALOG_ARMS: dict[str, list[str]] = {
    "spectral_only": SPECTRAL_FEATURES,
    "catalog_network_member": SPECTRAL_FEATURES + ["network_member"],
    "catalog_degree": SPECTRAL_FEATURES + ["known_log_degree"],
    "catalog_mass_coverage": SPECTRAL_FEATURES + ["known_mass_candidate_fraction"],
    "catalog_reference_density": SPECTRAL_FEATURES + ["log_reference_spectra"],
    "catalog_topology": SPECTRAL_FEATURES + ["network_member", "known_log_degree"],
    "catalog_observability": SPECTRAL_FEATURES + [
        "known_mass_candidate_fraction", "log_reference_spectra"
    ],
    "catalog_full": SPECTRAL_FEATURES + CATALOG_FEATURES,
    "catalog_without_network_member": SPECTRAL_FEATURES + [
        feature for feature in CATALOG_FEATURES if feature != "network_member"
    ],
    "catalog_without_degree": SPECTRAL_FEATURES + [
        feature for feature in CATALOG_FEATURES if feature != "known_log_degree"
    ],
    "catalog_without_mass_coverage": SPECTRAL_FEATURES + [
        feature for feature in CATALOG_FEATURES
        if feature != "known_mass_candidate_fraction"
    ],
    "catalog_without_reference_density": SPECTRAL_FEATURES + [
        feature for feature in CATALOG_FEATURES if feature != "log_reference_spectra"
    ],
}


def _as_builtin(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def compare_b36_catalog_replay(
    observed: pd.DataFrame, archived_path: Path
) -> tuple[pd.DataFrame, dict]:
    if not archived_path.is_file() or archived_path.stat().st_size == 0:
        raise FileNotFoundError(archived_path)
    archived = pd.read_csv(archived_path)
    archived = archived.loc[archived["arm"].eq("spectral_plus_catalog")].copy()
    observed = observed.loc[observed["arm"].eq("catalog_full")].copy()
    if len(archived) != EXPECTED_QUERIES or archived["query_id"].nunique() != EXPECTED_QUERIES:
        raise RuntimeError("B36 archive does not contain exactly 860 catalogue queries")
    if len(observed) != EXPECTED_QUERIES or observed["query_id"].nunique() != EXPECTED_QUERIES:
        raise RuntimeError("B37 full-catalog arm does not contain exactly 860 queries")
    columns = [
        "query_id", "source", "truth_candidate_id", "truth_formula",
        "baseline_candidate_id", "proposed_candidate_id", "final_candidate_id",
        "baseline_correct", "proposal_unique", "intervene", "final_correct",
        "corrected", "introduced", "delta", "gate_name",
        "proposal_probability", "gate_margin", "gate_probability",
    ]
    missing = set(columns) - set(archived.columns) | (set(columns) - set(observed.columns))
    if missing:
        raise RuntimeError(f"B36/B37 replay columns missing: {sorted(missing)}")
    joined = archived[columns].merge(
        observed[columns], on="query_id", how="outer", validate="one_to_one",
        suffixes=("_b36", "_b37"), indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise RuntimeError("B36/B37 query sets differ")
    categorical = [column for column in columns if column not in {
        "query_id", "proposal_probability", "gate_margin", "gate_probability"
    }]
    mismatches: dict[str, int] = {}
    for column in categorical:
        left = joined[f"{column}_b36"].astype(str)
        right = joined[f"{column}_b37"].astype(str)
        mismatches[column] = int((left != right).sum())
    numeric_errors: dict[str, float] = {}
    for column in ("proposal_probability", "gate_margin", "gate_probability"):
        left = pd.to_numeric(joined[f"{column}_b36"], errors="raise").to_numpy(float)
        right = pd.to_numeric(joined[f"{column}_b37"], errors="raise").to_numpy(float)
        numeric_errors[column] = float(np.max(np.abs(left - right), initial=0.0))
    passed = not any(mismatches.values()) and all(
        error <= 1e-12 for error in numeric_errors.values()
    )
    report = {
        "queries": int(len(joined)),
        "categorical_mismatches": mismatches,
        "maximum_numeric_errors": numeric_errors,
        "pass": bool(passed),
    }
    if not passed:
        raise RuntimeError(f"B37 failed exact B36 catalogue replay: {report}")
    return archived, report


def run_catalog_ablation(
    candidates: pd.DataFrame, seed: int, bootstrap_resamples: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict], list[dict]]:
    outer_tables: list[pd.DataFrame] = []
    selection_rows: list[dict] = []
    fold_reports: list[dict] = []
    for outer_index, outer_domain in enumerate(EXPECTED_DOMAINS):
        outer_train, outer_test = split_domain(candidates, outer_domain)
        for arm, features in CATALOG_ARMS.items():
            specification = {
                "arm": arm, "features": features,
                "null_method": None, "null_seed": None,
            }
            # Exact B36 common-random-number contract.
            model_seed = seed + 100 * outer_index
            outer_scored, outer_fit, _ = score_arm(
                outer_train, outer_test, specification, outer_domain, model_seed
            )
            inner_parts: list[pd.DataFrame] = []
            inner_fits: list[dict] = []
            for inner_index, inner_domain in enumerate(EXPECTED_DOMAINS):
                if inner_domain == outer_domain:
                    continue
                inner_train, inner_test = split_domain(outer_train, inner_domain)
                scored, fit_report, _ = score_arm(
                    inner_train, inner_test, specification,
                    f"outer={outer_domain}|inner={inner_domain}",
                    model_seed + inner_index + 1,
                )
                inner_parts.append(scored)
                inner_fits.append(fit_report)
            selected, ledger = choose_gate(pd.concat(inner_parts, ignore_index=True))
            for row in ledger:
                selection_rows.append({"outer_domain": outer_domain, "arm": arm, **row})
            result = apply_selected_gate(outer_scored, selected)
            result["arm"] = arm
            result["outer_domain"] = outer_domain
            outer_tables.append(result)
            outer_result = summarize(result)
            fold_reports.append({
                "outer_domain": outer_domain,
                "arm": arm,
                "features": features,
                "selected_gate": selected,
                "outer_fit": outer_fit,
                "inner_fits": inner_fits,
                "outer_result": outer_result,
            })
            print(
                f"[B37 {outer_domain} {arm}] gate={selected['gate_name']} "
                f"dR1={outer_result['delta_recall1']:+.4f} "
                f"C/I={outer_result['corrected']}/{outer_result['introduced']}",
                flush=True,
            )
    results = pd.concat(outer_tables, ignore_index=True)
    for arm in CATALOG_ARMS:
        local = results.loc[results["arm"].eq(arm)]
        if len(local) != EXPECTED_QUERIES or local["query_id"].nunique() != EXPECTED_QUERIES:
            raise RuntimeError(f"{arm}: incomplete B37 outer OOF coverage")
    arm_reports = {
        arm: {
            **summarize_arm(
                results.loc[results["arm"].eq(arm)].copy(),
                bootstrap_resamples,
                seed + 100000 + index * 10,
            ),
            "features": CATALOG_ARMS[arm],
            "by_domain": {
                domain: summarize(results.loc[
                    results["arm"].eq(arm) & results["source"].eq(domain)
                ])
                for domain in EXPECTED_DOMAINS
            },
        }
        for index, arm in enumerate(CATALOG_ARMS)
    }
    return results, pd.DataFrame(selection_rows), arm_reports, fold_reports


def reaction_observability(
    candidates: pd.DataFrame, catalog_transitions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    catalog_map = catalog_transitions.set_index("query_id")[
        ["final_candidate_id", "final_correct"]
    ]
    feature_rows: list[dict] = []
    query_rows: list[dict] = []
    centered_parts: list[pd.DataFrame] = []
    for query_id, group in candidates.groupby("query_id", sort=False):
        if query_id not in catalog_map.index:
            raise RuntimeError(f"{query_id}: missing B36 catalogue decision")
        truth = group.loc[group["is_positive"].astype(bool)]
        wrong = group.loc[~group["is_positive"].astype(bool)]
        if len(truth) != 1 or wrong.empty:
            raise RuntimeError(f"{query_id}: requires one truth and at least one wrong candidate")
        truth_id = str(truth.iloc[0]["candidate_id"])
        baseline_id = str(group["baseline_candidate_id"].iloc[0])
        catalog_id = str(catalog_map.loc[query_id, "final_candidate_id"])
        baseline_correct = bool(group["baseline_correct"].iloc[0])
        catalog_correct = bool(catalog_map.loc[query_id, "final_correct"])
        local_oracle = False
        local_available = False
        local_harm = False
        for feature in REACTION_FEATURES:
            values = group[feature].to_numpy(float)
            truth_value = float(truth.iloc[0][feature])
            wrong_values = wrong[feature].to_numpy(float)
            wrong_max = float(wrong_values.max())
            maximum = float(values.max())
            minimum = float(values.min())
            top_mask = np.isclose(values, maximum, rtol=0, atol=1e-12)
            unique_top = bool(top_mask.sum() == 1)
            proposed_id = str(group.iloc[int(np.flatnonzero(top_mask)[0])]["candidate_id"])
            varies = bool(maximum - minimum > 1e-12)
            nonzero = bool(np.any(np.abs(values) > 1e-12))
            truth_strict = bool(truth_value > wrong_max + 1e-12)
            wrong_strict = bool(wrong_max > truth_value + 1e-12)
            local_oracle = local_oracle or truth_strict
            local_available = local_available or varies
            local_harm = local_harm or wrong_strict
            feature_rows.append({
                "query_id": str(query_id),
                "source": str(group["source"].iloc[0]),
                "truth_candidate_id": truth_id,
                "truth_formula": str(group["truth_formula"].iloc[0]),
                "feature": feature,
                "candidate_count": int(len(group)),
                "nonzero": nonzero,
                "within_query_varies": varies,
                "truth_value": truth_value,
                "maximum_wrong_value": wrong_max,
                "truth_minus_max_wrong": truth_value - wrong_max,
                "truth_strictly_max": truth_strict,
                "wrong_strictly_above_truth": wrong_strict,
                "unique_maximum": unique_top,
                "monotone_proposed_candidate_id": proposed_id,
                "monotone_proposal_correct": bool(unique_top and proposed_id == truth_id),
                "baseline_correct": baseline_correct,
                "catalog_correct": catalog_correct,
                "baseline_error_oracle_recoverable": bool(not baseline_correct and truth_strict),
                "catalog_error_oracle_recoverable": bool(not catalog_correct and truth_strict),
                "baseline_correct_at_risk": bool(baseline_correct and wrong_strict),
                "catalog_correct_at_risk": bool(catalog_correct and wrong_strict),
            })
        query_rows.append({
            "query_id": str(query_id),
            "source": str(group["source"].iloc[0]),
            "truth_candidate_id": truth_id,
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "candidate_count": int(len(group)),
            "baseline_candidate_id": baseline_id,
            "baseline_correct": baseline_correct,
            "catalog_candidate_id": catalog_id,
            "catalog_correct": catalog_correct,
            "any_reaction_feature_varies": local_available,
            "any_reaction_feature_truth_strictly_max": local_oracle,
            "any_reaction_feature_wrong_strictly_above_truth": local_harm,
            "baseline_error_any_feature_oracle_recoverable": bool(
                not baseline_correct and local_oracle
            ),
            "catalog_error_any_feature_oracle_recoverable": bool(
                not catalog_correct and local_oracle
            ),
            "baseline_correct_any_feature_at_risk": bool(baseline_correct and local_harm),
            "catalog_correct_any_feature_at_risk": bool(catalog_correct and local_harm),
        })
        centered = group[[*CATALOG_FEATURES, *REACTION_FEATURES]].astype(float)
        centered = centered - centered.mean(axis=0)
        centered["query_id"] = str(query_id)
        centered_parts.append(centered)

    feature_frame = pd.DataFrame(feature_rows)
    query_frame = pd.DataFrame(query_rows)
    centered = pd.concat(centered_parts, ignore_index=True)
    correlation = centered[[*CATALOG_FEATURES, *REACTION_FEATURES]].corr(method="spearman")
    cross = correlation.loc[REACTION_FEATURES, CATALOG_FEATURES]

    def feature_summary(frame: pd.DataFrame) -> dict[str, dict]:
        summaries: dict[str, dict] = {}
        for feature, local in frame.groupby("feature", sort=False):
            summaries[str(feature)] = {
                "queries": int(len(local)),
                "nonzero_queries": int(local["nonzero"].sum()),
                "varying_queries": int(local["within_query_varies"].sum()),
                "truth_strictly_max": int(local["truth_strictly_max"].sum()),
                "wrong_strictly_above_truth": int(local["wrong_strictly_above_truth"].sum()),
                "baseline_error_oracle_recoverable": int(
                    local["baseline_error_oracle_recoverable"].sum()
                ),
                "catalog_error_oracle_recoverable": int(
                    local["catalog_error_oracle_recoverable"].sum()
                ),
                "baseline_correct_at_risk": int(local["baseline_correct_at_risk"].sum()),
                "catalog_correct_at_risk": int(local["catalog_correct_at_risk"].sum()),
                "ungated_monotone_corrected": int(
                    (~local["baseline_correct"] & local["monotone_proposal_correct"]).sum()
                ),
                "ungated_monotone_introduced": int(
                    (local["baseline_correct"] & ~local["monotone_proposal_correct"]
                     & local["unique_maximum"]).sum()
                ),
            }
        return summaries

    baseline_errors = int((~query_frame["baseline_correct"]).sum())
    catalog_errors = int((~query_frame["catalog_correct"]).sum())
    baseline_recoverable = int(
        query_frame["baseline_error_any_feature_oracle_recoverable"].sum()
    )
    catalog_recoverable = int(
        query_frame["catalog_error_any_feature_oracle_recoverable"].sum()
    )
    report = {
        "queries": int(len(query_frame)),
        "candidate_rows": int(len(candidates)),
        "queries_with_any_varying_reaction_feature": int(
            query_frame["any_reaction_feature_varies"].sum()
        ),
        "baseline_errors": baseline_errors,
        "catalog_errors": catalog_errors,
        "baseline_error_any_feature_oracle_recoverable": baseline_recoverable,
        "catalog_error_any_feature_oracle_recoverable": catalog_recoverable,
        "baseline_oracle_headroom_delta": baseline_recoverable / len(query_frame),
        "catalog_residual_oracle_headroom_delta": catalog_recoverable / len(query_frame),
        "baseline_correct_any_feature_at_risk": int(
            query_frame["baseline_correct_any_feature_at_risk"].sum()
        ),
        "catalog_correct_any_feature_at_risk": int(
            query_frame["catalog_correct_any_feature_at_risk"].sum()
        ),
        "per_feature": feature_summary(feature_frame),
        "per_source": {
            source: {
                "queries": int(len(local)),
                "baseline_errors": int((~local["baseline_correct"]).sum()),
                "catalog_errors": int((~local["catalog_correct"]).sum()),
                "baseline_recoverable": int(
                    local["baseline_error_any_feature_oracle_recoverable"].sum()
                ),
                "catalog_residual_recoverable": int(
                    local["catalog_error_any_feature_oracle_recoverable"].sum()
                ),
                "catalog_residual_oracle_headroom_delta": float(
                    local["catalog_error_any_feature_oracle_recoverable"].mean()
                ),
                "varying_reaction_evidence": int(local["any_reaction_feature_varies"].sum()),
                "varying_reaction_evidence_fraction": float(
                    local["any_reaction_feature_varies"].mean()
                ),
            }
            for source, local in query_frame.groupby("source", sort=True)
        },
        "within_query_spearman_reaction_vs_catalog": {
            reaction: {
                catalog: _as_builtin(cross.loc[reaction, catalog])
                for catalog in CATALOG_FEATURES
            }
            for reaction in REACTION_FEATURES
        },
        "interpretation": (
            "Any-feature oracle uses the truth to choose a feature and is headroom only. "
            "Ungated monotone results fix the direction 'more support is better' but do "
            "not constitute an OOF learned action."
        ),
    }
    return feature_frame, query_frame, report


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
        "--b36-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b36_reaction_specificity_2336381",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap-resamples must be at least 100")
    for path in (
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds,
        args.b36_dir / "nested_domain_loso_arm_transitions.csv.gz",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    candidates, provenance = build_universe(args)
    candidates = prepare_candidates(candidates)
    evaluated_candidates = candidates.loc[candidates["polarity"].eq("negative")].copy()
    # B36 evaluates 860 negative-mode queries but deliberately retains both
    # polarities in each outer/inner training partition.  Filtering before
    # split_domain changes the fitted action and is a protocol violation.
    if evaluated_candidates["query_id"].nunique() != EXPECTED_QUERIES:
        raise RuntimeError("B37 evaluation universe is not the frozen 860-query graph")

    transitions, selection, arms, folds = run_catalog_ablation(
        candidates, args.seed, args.bootstrap_resamples
    )
    archived_catalog, replay = compare_b36_catalog_replay(
        transitions,
        args.b36_dir / "nested_domain_loso_arm_transitions.csv.gz",
    )
    feature_frame, query_frame, reaction_report = reaction_observability(
        evaluated_candidates, archived_catalog
    )
    b36_report = json.loads((args.b36_dir / "report.json").read_text(encoding="utf-8"))

    contrasts = {
        f"catalog_full_vs_{arm}": paired_contrast(
            transitions, "catalog_full", arm, args.bootstrap_resamples,
            args.seed + 200000 + index * 10,
        )
        for index, arm in enumerate(CATALOG_ARMS)
        if arm != "catalog_full"
    }
    full = arms["catalog_full"]
    topology = arms["catalog_topology"]
    observability = arms["catalog_observability"]
    residual_headroom = float(reaction_report["catalog_residual_oracle_headroom_delta"])
    sources_with_residual_headroom_ge_3pp = int(sum(
        values["catalog_residual_oracle_headroom_delta"] >= 0.03
        for values in reaction_report["per_source"].values()
    ))
    sources_with_majority_variation = int(sum(
        values["varying_reaction_evidence_fraction"] >= 0.50
        for values in reaction_report["per_source"].values()
    ))
    gates = {
        "exact_B36_catalog_querywise_replay": bool(replay["pass"]),
        "catalog_gain_ge_3pp": bool(full["delta_recall1"] >= 0.03),
        "catalog_formula_cluster_ci_positive": bool(
            full["formula_cluster_bootstrap"]["ci_low"] > 0
        ),
        "topology_alone_gain_ge_3pp": bool(topology["delta_recall1"] >= 0.03),
        "observability_alone_gain_ge_3pp": bool(observability["delta_recall1"] >= 0.03),
        "reaction_residual_oracle_headroom_ge_3pp": bool(residual_headroom >= 0.03),
        "reaction_evidence_varies_in_majority_of_queries": bool(
            reaction_report["queries_with_any_varying_reaction_feature"]
            >= 0.5 * EXPECTED_QUERIES
        ),
        "reaction_residual_oracle_headroom_ge_3pp_in_every_source": bool(
            sources_with_residual_headroom_ge_3pp == len(EXPECTED_DOMAINS)
        ),
        "reaction_evidence_varies_in_majority_of_every_source": bool(
            sources_with_majority_variation == len(EXPECTED_DOMAINS)
        ),
        "B36_reaction_specificity_established": bool(
            b36_report.get("scientific_pass", False)
        ),
    }
    if gates["topology_alone_gain_ge_3pp"] and not gates["observability_alone_gain_ge_3pp"]:
        mechanism = "graph_node_topology_prior_dominant"
    elif gates["observability_alone_gain_ge_3pp"] and not gates["topology_alone_gain_ge_3pp"]:
        mechanism = "database_observability_prior_dominant"
    elif gates["observability_alone_gain_ge_3pp"] and gates["topology_alone_gain_ge_3pp"]:
        mechanism = "topology_and_observability_both_predictive"
    else:
        mechanism = "distributed_or_interaction_dependent_catalog_signal"
    pass_to_raw_path_diagnostic = bool(
        gates["reaction_residual_oracle_headroom_ge_3pp"]
        and gates["reaction_evidence_varies_in_majority_of_queries"]
    )
    pass_to_shared_embedding = bool(
        pass_to_raw_path_diagnostic
        and gates["reaction_residual_oracle_headroom_ge_3pp_in_every_source"]
        and gates["reaction_evidence_varies_in_majority_of_every_source"]
        and gates["B36_reaction_specificity_established"]
    )
    next_decision = (
        "raw_path_representation_diagnostic_justified__direct_shared_embedding_blocked"
        if pass_to_raw_path_diagnostic and not pass_to_shared_embedding
        else (
            "reaction_context_shared_embedding_preflight_justified"
            if pass_to_shared_embedding
            else "retain_topology_calibrator_and_do_not_inject_current_reaction_scalars"
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transition_path = args.output_dir / "catalog_ablation_transitions.csv.gz"
    selection_path = args.output_dir / "catalog_inner_selection_ledger.csv.gz"
    feature_path = args.output_dir / "reaction_feature_per_query.csv.gz"
    query_path = args.output_dir / "reaction_observability_per_query.csv.gz"
    transitions.to_csv(transition_path, index=False, compression="gzip")
    selection.to_csv(selection_path, index=False, compression="gzip")
    feature_frame.to_csv(feature_path, index=False, compression="gzip")
    query_frame.to_csv(query_path, index=False, compression="gzip")

    report = {
        "status": "bioaware_b37_catalog_mechanism_and_reaction_headroom_complete",
        "formal": True,
        "catalog_mechanism_audit_pass": bool(
            gates["exact_B36_catalog_querywise_replay"]
            and gates["catalog_gain_ge_3pp"]
            and gates["catalog_formula_cluster_ci_positive"]
        ),
        "pass_to_raw_path_diagnostic": pass_to_raw_path_diagnostic,
        "pass_to_shared_embedding": pass_to_shared_embedding,
        "protocol": (
            "exact B36 six-domain nested LOSO replay; one-feature/group/leave-one-out "
            "catalogue ablation; model-free within-query reaction observability audit"
        ),
        "b36_catalog_querywise_replay": replay,
        "catalog_ablation": {
            "arms": arms,
            "full_vs_reduced_contrasts": contrasts,
            "mechanism_classification": mechanism,
        },
        "reaction_observability_and_headroom": reaction_report,
        "reaction_cross_source_headroom": {
            "sources_with_residual_oracle_headroom_ge_3pp": sources_with_residual_headroom_ge_3pp,
            "sources_with_majority_reaction_feature_variation": sources_with_majority_variation,
            "sources_total": len(EXPECTED_DOMAINS),
        },
        "gates": gates,
        "decision": next_decision,
        "folds": folds,
        "contracts": {
            "same_candidate_graph_across_arms": True,
            "same_model_capacity_and_rng_schedule_across_arms": True,
            "held_truth_identity_and_formula_purged": True,
            "outer_outcomes_used_for_selection": False,
            "reaction_oracle_is_not_deployable": True,
            "shared_embedding_changed": False,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "provenance": {
            **provenance["provenance"],
            "b36_report_sha256": sha256(args.b36_dir / "report.json"),
            "b36_transitions_sha256": sha256(
                args.b36_dir / "nested_domain_loso_arm_transitions.csv.gz"
            ),
            "scikit_learn_version": sklearn.__version__,
            "catalog_ablation_transitions_sha256": sha256(transition_path),
            "catalog_inner_selection_ledger_sha256": sha256(selection_path),
            "reaction_feature_per_query_sha256": sha256(feature_path),
            "reaction_observability_per_query_sha256": sha256(query_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "B37 explains an opened-development reranking gain and quantifies reaction "
            "headroom. It is not a blind result, SOTA claim, causal metabolic mechanism, "
            "or shared-embedding improvement. Catalogue priors are candidate/database "
            "attributes and cannot be claimed as spectrum-intrinsic representation gain."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps({
        "status": report["status"],
        "b36_replay": replay,
        "catalog_arms": {
            arm: {
                key: values[key] for key in (
                    "recall1", "delta_recall1", "corrected", "introduced",
                    "risk_net_lambda2",
                )
            }
            for arm, values in arms.items()
        },
        "mechanism_classification": mechanism,
        "reaction_headroom": {
            key: reaction_report[key] for key in (
                "queries_with_any_varying_reaction_feature",
                "baseline_errors", "catalog_errors",
                "baseline_error_any_feature_oracle_recoverable",
                "catalog_error_any_feature_oracle_recoverable",
                "catalog_residual_oracle_headroom_delta",
            )
        },
        "gates": gates,
        "pass_to_raw_path_diagnostic": pass_to_raw_path_diagnostic,
        "pass_to_shared_embedding": pass_to_shared_embedding,
        "decision": next_decision,
        "output_dir": str(args.output_dir),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
