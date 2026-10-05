#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from bioaware_b47_u3_core import (  # noqa: E402
    aggregate_candidate_features, materialize_sample_events,
    permute_candidate_identities_within_query_adduct,
    permute_seed_events_across_samples, query_opportunities,
    reaction_pair_table, summarize_opportunity, PROTON_ADDUCT_MASS, formula_mass,
    transformation_signature,
)
from audit_bioaware_b47_u3_event_yield import _scalar_null_record  # noqa: E402


def fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a, b, c, d = (
        "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC", "DDDDDDDDDDDDDD",
    )
    seed_formula, target_formula = "C5H14O", "C6H12O"
    seed_mass, target_mass = formula_mass(seed_formula), formula_mass(target_formula)
    participants = pd.DataFrame([
        {"reaction_id": "R1", "side": "left", "compound_id": a, "is_currency": False,
         "stoichiometry": 1.0, "direction_semantics": "reactome_consensus_lr", "reaction_weight": 1.0,
         "exact_mass": seed_mass, "formula": seed_formula},
        {"reaction_id": "R1", "side": "right", "compound_id": b, "is_currency": False,
         "stoichiometry": 1.0, "direction_semantics": "reactome_consensus_lr", "reaction_weight": 1.0,
         "exact_mass": target_mass, "formula": target_formula},
        {"reaction_id": "R1", "side": "right", "compound_id": c, "is_currency": False,
         "stoichiometry": 1.0, "direction_semantics": "reactome_consensus_lr", "reaction_weight": 1.0,
         "exact_mass": target_mass, "formula": target_formula},
    ])
    queries = pd.DataFrame([
        {"query_id": "q1", "study": "ST001122", "sample": "s1", "feature_id": "f1", "feature_mz": target_mass + PROTON_ADDUCT_MASS},
        {"query_id": "seedq", "study": "ST001122", "sample": "s1", "feature_id": "fs", "feature_mz": seed_mass + PROTON_ADDUCT_MASS},
        {"query_id": "q2", "study": "ST003356", "sample": "s2", "feature_id": "f2", "feature_mz": target_mass + PROTON_ADDUCT_MASS},
    ])
    candidates = pd.DataFrame([
        {"query_id": "q1", "candidate_id": b, "candidate_formula": target_formula, "spectral_score": 0.70,
         "candidate_reference_mz": target_mass + PROTON_ADDUCT_MASS, "best_reference_adduct": "[M+H]+", "reference_spectra": 2},
        {"query_id": "q1", "candidate_id": c, "candidate_formula": target_formula, "spectral_score": 0.69,
         "candidate_reference_mz": target_mass + PROTON_ADDUCT_MASS, "best_reference_adduct": "[M+H]+", "reference_spectra": 3},
        {"query_id": "q1", "candidate_id": d, "candidate_formula": target_formula, "spectral_score": 0.68,
         "candidate_reference_mz": target_mass + PROTON_ADDUCT_MASS, "best_reference_adduct": "[M+H]+", "reference_spectra": 4},
        {"query_id": "seedq", "candidate_id": a, "candidate_formula": seed_formula, "spectral_score": 0.95,
         "candidate_reference_mz": seed_mass + PROTON_ADDUCT_MASS, "best_reference_adduct": "[M+H]+", "reference_spectra": 2},
        {"query_id": "seedq", "candidate_id": d, "candidate_formula": seed_formula, "spectral_score": 0.1,
         "candidate_reference_mz": seed_mass + PROTON_ADDUCT_MASS, "best_reference_adduct": "[M+H]+", "reference_spectra": 1},
        {"query_id": "q2", "candidate_id": b, "candidate_formula": target_formula, "spectral_score": 0.7,
         "candidate_reference_mz": target_mass + PROTON_ADDUCT_MASS, "best_reference_adduct": "[M+H]+", "reference_spectra": 1},
        {"query_id": "q2", "candidate_id": d, "candidate_formula": target_formula, "spectral_score": 0.6,
         "candidate_reference_mz": target_mass + PROTON_ADDUCT_MASS, "best_reference_adduct": "[M+H]+", "reference_spectra": 1},
    ])
    seeds = pd.DataFrame([
        {"study": "ST001122", "sample": "s1", "seed_query_id": "seedq",
         "seed_feature_id": "fs", "seed_compound_id": a, "seed_score": 0.95,
         "seed_margin": 0.3},
    ])
    return participants, queries, candidates, seeds


def test_events_are_sample_local_and_candidate_specificity_is_competition_aware() -> None:
    participants, queries, candidates, seeds = fixtures()
    pairs, audit = reaction_pair_table(
        participants,
        set(seeds["seed_compound_id"]),
        set(candidates["candidate_id"]),
    )
    assert audit["typed_pairs"] == 2
    events = materialize_sample_events(candidates, queries, seeds, pairs)
    assert set(events["query_id"]) == {"q1"}
    assert set(events["candidate_id"]) == {"BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"}
    assert set(events["competing_query_candidate_count"]) == {2}
    assert np.allclose(events["candidate_specificity"], 0.5)
    assert not events["exclusive_event"].any()
    assert events["transform_within_combined_10ppm"].all()
    assert events["reaction_transform_signature_match"].all()


def test_reaction_signature_is_independent_of_reassigned_identity() -> None:
    assert transformation_signature("C6H12O", "C5H14O") == "C:+1"
    # Charged Rhea forms have the same non-hydrogen transformation.
    assert transformation_signature("C6H11O-", "C5H13O-") == "C:+1"
    # A rewired slot cannot redefine the original reaction transformation.
    assert transformation_signature("C5H14O2", "C5H14O") == "O:+1"
    assert transformation_signature("C5H14O2", "C5H14O") != "C:+1"
    # Charge stripping must retain the final element count.
    assert transformation_signature("C5H9O2-", "C5H11NO") == "N:-1|O:+1"
    assert transformation_signature("C21H25N7O17P3-3", "C21H26N7O14P2-") == "O:+3|P:+1"


def test_rewired_identity_keeps_original_reaction_slot_signature() -> None:
    participants, _, candidates, seeds = fixtures()
    rewired = participants.copy()
    rewired.loc[rewired["compound_id"].eq("BBBBBBBBBBBBBB"), "compound_id"] = (
        "DDDDDDDDDDDDDD"
    )
    pairs, _ = reaction_pair_table(
        rewired, set(seeds["seed_compound_id"]), set(candidates["candidate_id"]),
    )
    selected = pairs[pairs["candidate_id"].eq("DDDDDDDDDDDDDD")]
    assert len(selected) == 1
    assert selected.iloc[0]["reaction_candidate_formula"] == "C6H12O"
    assert selected.iloc[0]["reaction_transform_signature"] == "C:+1"


def test_query_opportunity_reports_unsupported_frozen_candidates_without_overclaim() -> None:
    participants, queries, candidates, seeds = fixtures()
    pairs, _ = reaction_pair_table(
        participants, set(seeds["seed_compound_id"]), set(candidates["candidate_id"]),
    )
    events = materialize_sample_events(candidates, queries, seeds, pairs)
    features = aggregate_candidate_features(candidates, events)
    opportunity = query_opportunities(features, queries).set_index("query_id")
    assert int(opportunity.loc["q1", "unsupported_candidates_in_frozen_set"]) == 1
    assert bool(opportunity.loc["q1", "event_available"])
    assert not bool(opportunity.loc["q1", "event_top_unique"])
    assert not bool(opportunity.loc["q1", "candidate_specific_intervention_opportunity"])
    summary = summarize_opportunity(events, opportunity.reset_index())
    assert summary["event_supported_queries"] == 1
    assert summary["candidate_specific_queries"] == 0


def test_seed_context_permutation_moves_intact_payloads_and_replays() -> None:
    seeds = pd.DataFrame({
        "study": ["S"] * 4,
        "sample": ["s1", "s2", "s3", "s4"],
        "seed_query_id": ["q1", "q2", "q3", "q4"],
        "seed_feature_id": ["f1", "f2", "f3", "f4"],
        "seed_compound_id": [
            "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC", "DDDDDDDDDDDDDD",
        ],
        "seed_score": [0.91, 0.92, 0.93, 0.94],
        "seed_margin": [0.2, 0.2, 0.2, 0.2],
    })
    degree = pd.Series({value: 4 for value in seeds["seed_compound_id"]})
    one, audit_one = permute_seed_events_across_samples(seeds, degree, 7)
    two, audit_two = permute_seed_events_across_samples(seeds, degree, 7)
    assert one.equals(two)
    assert audit_one == audit_two
    assert sorted(one["seed_compound_id"]) == sorted(seeds["seed_compound_id"])
    assert sorted(one["seed_query_id"]) == sorted(seeds["seed_query_id"])
    assert audit_one["moved_fraction"] == 1.0


def test_seed_context_null_materializes_with_authentic_seed_references() -> None:
    participants, queries, candidates, seeds = fixtures()
    # Add a second authentic seed event in another sample so the permutation
    # actually moves a query/identity/reference payload.
    extra_seed_query = {
        "query_id": "seedq2", "study": "ST001122", "sample": "s2",
        "feature_id": "fs2", "feature_mz": queries.loc[
            queries["query_id"].eq("seedq"), "feature_mz"
        ].iloc[0],
    }
    queries = pd.concat([queries, pd.DataFrame([extra_seed_query])], ignore_index=True)
    seed_candidate = candidates.loc[
        (candidates["query_id"].eq("seedq"))
        & (candidates["candidate_id"].eq("AAAAAAAAAAAAAA"))
    ].copy()
    seed_candidate["query_id"] = "seedq2"
    candidates = pd.concat([candidates, seed_candidate], ignore_index=True)
    seeds = pd.concat([seeds, pd.DataFrame([{
        "study": "ST001122", "sample": "s2", "seed_query_id": "seedq2",
        "seed_feature_id": "fs2", "seed_compound_id": "AAAAAAAAAAAAAA",
        "seed_score": 0.94, "seed_margin": 0.31,
    }])], ignore_index=True)
    degree = pd.Series({"AAAAAAAAAAAAAA": 4})
    permuted, audit = permute_seed_events_across_samples(seeds, degree, 17)
    assert audit["moved_fraction"] == 1.0
    pairs, _ = reaction_pair_table(
        participants, set(seeds["seed_compound_id"]), set(candidates["candidate_id"]),
    )
    replayed = materialize_sample_events(
        candidates, queries, permuted, pairs,
        seed_reference_candidates=candidates,
    )
    assert isinstance(replayed, pd.DataFrame)


def test_candidate_identity_null_is_deranged_and_preserves_nuisance_columns() -> None:
    participants, queries, candidates, seeds = fixtures()
    one, audit = permute_candidate_identities_within_query_adduct(candidates, 11)
    assert audit["moved_rows"] == len(candidates)
    assert sorted(one["spectral_score"]) == sorted(candidates["spectral_score"])
    assert sorted(one["reference_spectra"]) == sorted(candidates["reference_spectra"])
    assert not one.duplicated(["query_id", "candidate_id"]).any()
    pairs, _ = reaction_pair_table(
        participants, set(seeds["seed_compound_id"]), set(candidates["candidate_id"]),
    )
    # The wrong-identity target table must still replay the authentic seed
    # query/identity/reference tuple rather than crashing on the permutation.
    materialize_sample_events(
        one, queries, seeds, pairs, seed_reference_candidates=candidates,
    )


def test_specificity_is_across_reactions_and_duplicate_reactions_do_not_stack() -> None:
    participants, queries, candidates, seeds = fixtures()
    extra = participants.copy()
    extra["reaction_id"] = "R2"
    participants = pd.concat([participants, extra], ignore_index=True)
    pairs, _ = reaction_pair_table(
        participants, set(seeds["seed_compound_id"]), set(candidates["candidate_id"]),
    )
    events = materialize_sample_events(candidates, queries, seeds, pairs)
    assert set(events["competing_query_candidate_count"]) == {2}
    assert not events["exclusive_event"].any()
    features = aggregate_candidate_features(candidates, events).set_index(["query_id", "candidate_id"])
    # Two catalogue records remain auditable, but one observed seed contributes once.
    assert int(features.loc[("q1", "BBBBBBBBBBBBBB"), "event_count"]) == 2
    expected = float(events.loc[
        (events["query_id"] == "q1") & (events["candidate_id"] == "BBBBBBBBBBBBBB"),
        "specificity_weighted_contribution",
    ].max())
    assert np.isclose(
        features.loc[("q1", "BBBBBBBBBBBBBB"), "candidate_specific_network_support"],
        expected,
    )


def test_raw_events_that_fail_mass_contract_never_contribute() -> None:
    participants, queries, candidates, seeds = fixtures()
    candidates = candidates.copy()
    candidates.loc[candidates["query_id"].eq("q1"), "candidate_formula"] = "BAD_FORMULA"
    pairs, _ = reaction_pair_table(
        participants, set(seeds["seed_compound_id"]), set(candidates["candidate_id"]),
    )
    events = materialize_sample_events(candidates, queries, seeds, pairs)
    assert len(events) > 0
    assert not events["event_eligible"].any()
    assert np.allclose(events["contribution"], 0.0)
    features = aggregate_candidate_features(candidates, events)
    affected = features[features["query_id"].eq("q1")]
    assert affected["raw_event_count"].gt(0).any()
    assert affected["event_count"].eq(0).all()
    assert np.allclose(affected["candidate_specific_network_support"], 0.0)


def test_zero_net_stoichiometric_carrier_is_not_paired() -> None:
    participants, _, candidates, seeds = fixtures()
    carrier = "DDDDDDDDDDDDDD"
    carrier_rows = pd.DataFrame([
        {"reaction_id": "R1", "side": "left", "compound_id": carrier,
         "is_currency": False, "stoichiometry": 1.0,
         "direction_semantics": "reactome_consensus_lr", "reaction_weight": 1.0,
         "exact_mass": 50.0, "formula": "CH2"},
        {"reaction_id": "R1", "side": "right", "compound_id": carrier,
         "is_currency": False, "stoichiometry": 1.0,
         "direction_semantics": "reactome_consensus_lr", "reaction_weight": 1.0,
         "exact_mass": 50.0, "formula": "CH2"},
    ])
    pairs, audit = reaction_pair_table(
        pd.concat([participants, carrier_rows], ignore_index=True),
        set(seeds["seed_compound_id"]), set(candidates["candidate_id"]),
    )
    assert carrier not in set(pairs["candidate_id"])
    assert audit["unchanged_participants_removed"] >= 1


def test_baseline_tie_is_not_converted_to_disagreement() -> None:
    participants, queries, candidates, seeds = fixtures()
    candidates.loc[candidates["query_id"].eq("q1"), "spectral_score"] = 0.7
    pairs, _ = reaction_pair_table(
        participants, set(seeds["seed_compound_id"]), set(candidates["candidate_id"]),
    )
    events = materialize_sample_events(candidates, queries, seeds, pairs)
    features = aggregate_candidate_features(candidates, events)
    opportunity = query_opportunities(features, queries).set_index("query_id")
    assert not bool(opportunity.loc["q1", "baseline_top_unique"])
    assert opportunity.loc["q1", "baseline_top_candidate"] == ""
    assert not bool(opportunity.loc["q1", "event_disagrees_with_baseline"])
    assert not bool(opportunity.loc["q1", "candidate_specific_intervention_opportunity"])


def test_only_decision_changing_exclusive_evidence_counts_as_intervention() -> None:
    rows = []
    for query_id, event_winner in (("change", "B"), ("agree", "A")):
        for candidate_id, spectral_score in (("A", 0.9), ("B", 0.8)):
            rows.append({
                "query_id": query_id,
                "candidate_id": candidate_id,
                "candidate_formula": "C5H14O" if candidate_id == "A" else "C6H12O",
                "spectral_score": spectral_score,
                "candidate_specific_network_support": (
                    0.5 if candidate_id == event_winner else 0.0
                ),
                "exclusive_seed_count": 1 if candidate_id == event_winner else 0,
                "event_count": 1 if candidate_id == event_winner else 0,
            })
    metadata = pd.DataFrame([
        {"query_id": query_id, "study": "S", "sample": "s", "feature_id": query_id}
        for query_id in ("change", "agree")
    ])
    opportunity = query_opportunities(pd.DataFrame(rows), metadata).set_index("query_id")
    assert bool(opportunity.loc["change", "candidate_specific_intervention_opportunity"])
    assert bool(opportunity.loc["change", "event_disagrees_with_baseline"])
    assert not bool(opportunity.loc["agree", "candidate_specific_intervention_opportunity"])


def test_null_record_preserves_source_strata_without_serializing_dicts() -> None:
    record = _scalar_null_record("x", 3, {
        "candidate_specific_intervention_queries": np.int64(12),
        "by_study": {
            "S1": {"candidate_specific_intervention_queries": np.int64(7)},
            "S2": {"candidate_specific_intervention_queries": np.int64(5)},
        },
        "direction_labels": {"unknown": 4},
    })
    assert record["candidate_specific_intervention_queries"] == 12
    assert record["study__S1__candidate_specific_intervention_queries"] == 7
    assert record["study__S2__candidate_specific_intervention_queries"] == 5
    assert "direction_labels" not in record


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print("[test_bioaware_b47_u3_event_yield] PASS", flush=True)
