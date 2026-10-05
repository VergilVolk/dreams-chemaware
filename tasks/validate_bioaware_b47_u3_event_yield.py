#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from audit_bioaware_b47_u3_event_yield import sha256_file  # noqa: E402
from bioaware_b47_u3_core import parse_bool  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input.resolve()
    paths = {
        "report": directory / "report.json",
        "atomic_events": directory / "atomic_events.csv.gz",
        "candidate_features": directory / "candidate_event_features.csv.gz",
        "query_opportunities": directory / "query_event_opportunities.csv.gz",
        "null_summary": directory / "structured_null_summary.csv.gz",
        "null_permutation_audit": directory / "null_permutation_audit.csv.gz",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    if (
        report.get("status") != "bioaware_b47_u3_truthblind_event_yield_complete"
        or report.get("protocol_version") != "U3-v3-reaction-signature-20260922"
        or report.get("formal") is not True
        or report.get("truth_blind") is not True
    ):
        raise RuntimeError("invalid U3 report identity")
    provenance = report.get("provenance", {})
    for key, path_key in (
        ("atomic_events_sha256", "atomic_events"),
        ("candidate_features_sha256", "candidate_features"),
        ("query_opportunities_sha256", "query_opportunities"),
        ("null_summary_sha256", "null_summary"),
        ("null_permutation_audit_sha256", "null_permutation_audit"),
    ):
        if provenance.get(key) != sha256_file(paths[path_key]):
            raise RuntimeError(f"U3 hash mismatch: {path_key}")
    events = pd.read_csv(paths["atomic_events"], dtype={"query_id": str, "seed_query_id": str})
    features = pd.read_csv(paths["candidate_features"], dtype={"query_id": str})
    opportunities = pd.read_csv(paths["query_opportunities"], dtype={"query_id": str})
    nulls = pd.read_csv(paths["null_summary"])
    permutation_audit = pd.read_csv(paths["null_permutation_audit"])
    if features.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("U3 candidate features are not unique")
    if opportunities["query_id"].duplicated().any():
        raise RuntimeError("U3 query opportunities are not unique")
    if len(events) and (events["query_id"] == events["seed_query_id"]).any():
        raise RuntimeError("U3 contains self-seed leakage")
    if len(events):
        event_context = events[["query_id", "study", "sample"]].drop_duplicates()
        if event_context["query_id"].duplicated().any():
            raise RuntimeError("one U3 query appears in multiple study/sample contexts")
        expected_context = opportunities[["query_id", "study", "sample"]]
        checked = event_context.merge(
            expected_context,
            on="query_id",
            how="left",
            suffixes=("_event", "_query"),
            validate="one_to_one",
        )
        if checked[["study_query", "sample_query"]].isna().any().any() or not (
            checked["study_event"].astype(str).eq(checked["study_query"].astype(str))
            & checked["sample_event"].astype(str).eq(checked["sample_query"].astype(str))
        ).all():
            raise RuntimeError("U3 sample-local event check failed")
        event_eligible = events["event_eligible"].map(parse_bool)
        ineligible = ~event_eligible
        if (events.loc[ineligible, "contribution"].abs() > 1e-12).any():
            raise RuntimeError("ineligible U3 transforms contribute support")
        expected_eligible = (
            events["supported_adduct_pair"].map(parse_bool)
            & events["identity_mass_contract_within_10ppm"].map(parse_bool)
            & events["reaction_transform_signature_match"].map(parse_bool)
            & events["expected_reaction_delta"].notna()
            & events["transform_within_combined_10ppm"].map(parse_bool)
        )
        if not expected_eligible.equals(event_eligible):
            raise RuntimeError("U3 event eligibility no longer matches the frozen mass contract")
        eligible = events[~ineligible].copy()
        key = ["query_id", "seed_query_id", "seed_compound_id"]
        expected_competition = eligible.groupby(key)["candidate_id"].nunique()
        observed = expected_competition.reindex(
            pd.MultiIndex.from_frame(eligible[key])
        ).to_numpy(dtype=int)
        if not (observed == eligible["competing_query_candidate_count"].to_numpy(int)).all():
            raise RuntimeError("candidate specificity was not computed across all reactions")
        if not (
            events["exclusive_event"].map(parse_bool)
            == (event_eligible
                & events["competing_query_candidate_count"].eq(1))
        ).all():
            raise RuntimeError("exclusive-event semantics changed")
    if opportunities.loc[
        ~opportunities["baseline_top_unique"].map(parse_bool),
        "event_disagrees_with_baseline",
    ].map(parse_bool).any():
        raise RuntimeError("baseline ties were converted into arbitrary disagreements")
    forbidden = ("truth", "phenotype", "disease", "case_control")
    for name, frame in (("events", events), ("features", features), ("opportunities", opportunities)):
        bad = [column for column in frame if any(token in column.casefold() for token in forbidden)]
        if bad:
            raise RuntimeError(f"forbidden columns in U3 {name}: {bad}")
    requested_repeats = int(
        report["structured_nulls"]["requested_repeats_per_family"]
    )
    expected_repeats = int(
        report["structured_nulls"]["completed_repeats_per_family"]
    )
    early_stop = report["structured_nulls"].get("early_stop")
    if expected_repeats < requested_repeats:
        if (
            not isinstance(early_stop, dict)
            or early_stop.get("reason")
            != "STRUCTURAL_NULL_DOMINATES_REAL_CANDIDATE_SPECIFIC_OPPORTUNITY"
        ):
            raise RuntimeError("U3 null repeats are incomplete without a valid fail-fast reason")
        if report.get("pass_to_frozen_event_ranking_evaluation") is not False:
            raise RuntimeError("an early-stopped U3 result cannot authorize outcome evaluation")
    families = (
        "degree_rewired", "seed_context_permuted", "candidate_identity_permuted",
    )
    for family in families:
        if int((nulls["null_family"] == family).sum()) != expected_repeats:
            raise RuntimeError(f"U3 null repeat count mismatch: {family}")
    if expected_repeats < 1:
        raise RuntimeError("U3 contains no completed structural-null repeat")
    null_metric = report["structured_nulls"].get("comparison_metric")
    if null_metric != "candidate_specific_queries" or null_metric not in nulls:
        raise RuntimeError("U3 structural nulls are not evaluated on candidate-specific events")
    real = report["real_event_yield"]
    for source, source_result in report["structured_nulls"].get("by_source", {}).items():
        source_column = f"study__{source}__{null_metric}"
        if source_column not in nulls:
            raise RuntimeError(f"U3 null summary lacks source-stratified metric: {source}")
        source_real = int(real["by_study"][source][null_metric])
        if int(source_result["real"]) != source_real:
            raise RuntimeError(f"U3 source null denominator mismatch: {source}")
    for family in ("seed_context_permuted", "candidate_identity_permuted"):
        selected = permutation_audit[permutation_audit["null_family"].eq(family)]
        if len(selected) != expected_repeats or selected["moved_fraction"].isna().any():
            raise RuntimeError(f"U3 permutation audit mismatch: {family}")
    if len(events) != int(real["atomic_events"]):
        raise RuntimeError("U3 event count mismatch")
    if len(opportunities) != int(real["queries"]):
        raise RuntimeError("U3 query count mismatch")
    expected_actions = (
        opportunities["candidate_specific_opportunity"].map(parse_bool)
        & (
            ~opportunities["baseline_top_unique"].map(parse_bool)
            | opportunities["event_disagrees_with_baseline"].map(parse_bool)
        )
    )
    if not expected_actions.equals(
        opportunities["candidate_specific_intervention_opportunity"].map(parse_bool)
    ):
        raise RuntimeError("U3 intervention opportunity is not decision-changing")
    if int(expected_actions.sum()) != int(real["candidate_specific_intervention_queries"]):
        raise RuntimeError("U3 actionable intervention count mismatch")
    actionable = opportunities[expected_actions]
    if actionable["event_top_candidate"].nunique() != int(
        real["intervention_candidate_identities"]
    ):
        raise RuntimeError("U3 intervention candidate concentration mismatch")
    if actionable["event_top_formula"].nunique() != int(
        real["intervention_candidate_formulas"]
    ):
        raise RuntimeError("U3 intervention formula concentration mismatch")
    required_actions = int(math.ceil(0.03 * len(opportunities)))
    if int(report["structured_nulls"][
        "minimum_interventions_for_three_point_target"
    ]) != required_actions:
        raise RuntimeError("U3 three-point feasibility denominator changed")
    contracts = report.get("contracts", {})
    required_false = (
        "truth_opened", "phenotype_used", "performance_computed", "model_fitted",
        "P2b_used", "EMRN_used_as_exact_event", "catalogue_membership_used_as_outcome",
    )
    if any(contracts.get(key) is not False for key in required_false):
        raise RuntimeError("U3 truth-blind/no-model contract changed")
    required_true = (
        "sample_local_events_only", "same_query_seed_excluded",
        "unchanged_stoichiometric_participants_excluded",
        "independent_seed_before_noisy_or",
        "baseline_ties_excluded_from_disagreement",
        "reference_multiplicity_preserved_in_candidate_null",
    )
    if any(contracts.get(key) is not True for key in required_true):
        raise RuntimeError("U3 locality contract changed")
    expected_pass = all(bool(value) for value in report.get("gates", {}).values())
    if report.get("pass_to_frozen_event_ranking_evaluation") is not expected_pass:
        raise RuntimeError("U3 decision gate is inconsistent")
    print(
        "[validate_bioaware_b47_u3_event_yield] PASS "
        f"events={len(events):,} specific_queries={real['candidate_specific_queries']:,} "
        f"pass_to_ranking={expected_pass}",
        flush=True,
    )


if __name__ == "__main__":
    main()
