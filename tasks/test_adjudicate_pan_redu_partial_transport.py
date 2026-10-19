from __future__ import annotations

import pandas as pd
import pytest

from adjudicate_pan_redu_partial_transport import adjudicate, validate_and_annotate


WORLDS = (
    "all", "known_only", "measurement_matched_known", "measurement_matched_unresolved",
    "unresolved_ms2_reassigned_left_panel", "unresolved_ms2_reassigned_right_panel",
    "measurement_matched_unresolved_ms2_reassigned_left_panel",
    "measurement_matched_unresolved_ms2_reassigned_right_panel",
)


def fixture(n_sources: int = 6, with_truth: bool = True):
    topology = pd.DataFrame([
        {
            "panel_id": f"P{i}", "contrast_id": "C", "preliminary_source_family": f"S{i}",
            "comparison_stratum": "exact_axis", "topology_status": "PASS",
        }
        for i in range(n_sources)
    ])
    rows = []
    for query in range(n_sources):
        for target in range(n_sources):
            if query == target:
                continue
            for world in WORLDS:
                positive = target == (query + 1) % n_sources
                if world == "all":
                    score = 1.0 if positive else 0.0
                elif world == "measurement_matched_unresolved":
                    score = 0.9 if positive else 0.0
                else:
                    score = 0.1 if positive else 0.2
                rows.append({
                    "world": world, "left_panel_id": f"P{query}", "left_contrast_id": "C",
                    "right_panel_id": f"P{target}", "right_contrast_id": "C",
                    "answerable": True, "signed_chemical_evidence": score,
                })
    truth = []
    if with_truth:
        for query in range(n_sources):
            target = (query + 1) % n_sources
            truth.append({
                "query_panel_id": f"P{query}", "query_contrast_id": "C",
                "target_panel_id": f"P{target}", "target_contrast_id": "C",
                "truth_class": "replication_expected", "expected_direction": "similar",
                "truth_family_id": f"F{query}", "eligible_primary": True,
                "evidence_basis": "synthetic", "source_independence_status": "confirmed_independent",
                "truth_frozen_before_scores": True, "selection_blind_to_transport_scores": True,
            })
    truth_frame = pd.DataFrame(truth, columns=[
        "query_panel_id", "query_contrast_id", "target_panel_id", "target_contrast_id",
        "truth_class", "expected_direction", "truth_family_id", "eligible_primary",
        "evidence_basis", "source_independence_status", "truth_frozen_before_scores",
        "selection_blind_to_transport_scores",
    ])
    return pd.DataFrame(rows), topology, truth_frame


def test_partial_label_adjudicator_uses_source_equal_unconditional_rank() -> None:
    matrix, topology, truth = fixture()
    report, outcomes, replicates, delete_one, coverage = adjudicate(
        matrix, topology, truth, draws=599, seed=7,
    )
    assert report["status"] == "PAN_REDU_PARTIAL_LABEL_ADJUDICATION_COMPLETE"
    assert report["world_metrics"]["all"]["source_equal_mrr"] == pytest.approx(1.0)
    assert report["gates"]["all_beats_strongest_known_ci"]
    assert report["gates"]["matched_unresolved_beats_reassignment_ci"]
    assert report["pass_systemic_unknown_increment"]
    assert len(outcomes) == 6 * len(WORLDS)
    assert not replicates.empty and not delete_one.empty
    assert int(coverage.iloc[0].unlabeled_ordered_pairs) == 24


def test_no_gold_abstains_without_converting_unknown_pairs_to_negatives() -> None:
    matrix, topology, truth = fixture(with_truth=False)
    report, outcomes, replicates, delete_one, coverage = adjudicate(matrix, topology, truth)
    assert report["status"] == "ABSTAIN_NO_CLAIM_BEARING_GOLD_TRUTH"
    assert not report["pass_systemic_unknown_increment"]
    assert outcomes.empty and replicates.empty and delete_one.empty
    assert int(coverage.iloc[0].unlabeled_ordered_pairs) == 30


def test_primary_truth_requires_score_blind_source_independence() -> None:
    matrix, topology, truth = fixture()
    truth.loc[0, "source_independence_status"] = "unknown"
    with pytest.raises(RuntimeError, match="source independence"):
        validate_and_annotate(matrix, topology, truth)
    truth.loc[0, "source_independence_status"] = "confirmed_independent"
    truth.loc[0, "selection_blind_to_transport_scores"] = False
    with pytest.raises(RuntimeError, match="score blind"):
        validate_and_annotate(matrix, topology, truth)
