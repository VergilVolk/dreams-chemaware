from __future__ import annotations

import json
from pathlib import Path


CONTRACT = Path(__file__).with_name(
    "contrast_atlas_kidney_counterresponse_contract_v1.json"
)
ENDPOINT_MANIFEST = Path(__file__).with_name(
    "contrast_atlas_kidney_endpoint_manifest_v1.json"
)


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_kidney_contract_preserves_independence_and_denominators() -> None:
    contract = load_contract()
    families = contract["source_families"]
    accessions = [
        accession
        for family in families
        for accession in family["accessions"]
    ]
    denominators = contract["denominators"]
    arms = denominators["arm_assigned_accession_sample_records"]

    assert len(families) == 13
    assert len(accessions) == len(set(accessions)) == 16
    assert arms["control"] + arms["disease"] + arms["intervention"] == 383
    assert arms["total"] == 383
    assert denominators["nonredundant_specimens_after_known_exact_duplicate_fold"] == 361
    assert denominators["known_exact_duplicate_specimens"]["accessions"] == [
        "ST003458",
        "ST003465",
    ]


def test_kidney_development_layers_are_disjoint_and_exhaustive() -> None:
    contract = load_contract()
    layers = contract["exclusive_development_layers"]
    named_layers = [
        set(layers["sample_level_ms2_primary_representatives"]),
        set(layers["ms1_or_annotation_only_pressure_representatives"]),
        set(layers["same_source_sensitivity_accessions"]),
    ]
    assert [len(layer) for layer in named_layers] == [6, 7, 3]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(named_layers)
        for right in named_layers[index + 1 :]
    )
    assert len(set.union(*named_layers)) == 16


def test_kidney_contract_fails_closed_on_truth_and_shared_arm_bias() -> None:
    contract = load_contract()
    assert contract["go_no_go"]["gold_relation_truth"] == "NO_GO"
    assert contract["go_no_go"]["sealed_external_validation"] == "NO_GO"
    assert contract["truth_boundary"]["published_phase_labels_are_truth"] is False
    assert contract["truth_boundary"]["published_h4_candidates_are_truth"] is False

    cohorts = {row["accession"]: row for row in contract["cohorts"]}
    assert cohorts["ST001855"]["recorded_timepoint"] == "UIR_day21"
    assert cohorts["ST001855"]["correct_phase_by_preregistered_rule"] == "chronic"
    assert cohorts["ST001856"]["recorded_timepoint"] == "UIR_day10"

    field_test = contract["directed_response_contract"][
        "shared_arm_debiased_field_test"
    ]
    assert "offdiagonal" in field_test["unbiased_estimator"]
    assert "never renormalize" in field_test["world_mask_rule"]
    assert contract["evaluation_worlds"]["same_candidate_universe_required"] is True
    assert set(contract["evaluation_worlds"]) >= {
        "all",
        "known_only",
        "measurement_matched_known",
        "all_identity_field",
        "unresolved_shuffled_ms2",
    }
    endpoint_tiers = contract["directed_response_contract"]["phenotype_guard"][
        "endpoint_linkage_tiers"
    ]
    assert set(endpoint_tiers) == {"A", "B", "C", "BLOCKED"}


def test_kidney_endpoint_manifest_blocks_sample_level_causal_claims() -> None:
    manifest = json.loads(ENDPOINT_MANIFEST.read_text(encoding="utf-8"))
    rows = {row["accession"]: row for row in manifest["accessions"]}
    assert len(rows) == 6
    assert all(not row["sample_level_randomization_eligible"] for row in rows.values())
    assert {
        accession
        for accession, row in rows.items()
        if row["directional_efficacy_anchor"]
    } == {"ST001855", "ST002390", "ST002532", "ST003042", "ST003111"}
    assert rows["ST001856"]["directional_efficacy_anchor"] is False
    assert rows["ST002532"]["parent_randomized"] is True
    assert rows["ST002532"]["metabolomics_subset_randomized"] == "unknown"
    assert rows["ST003111"]["machine_freezable_numeric_endpoint"] is True
    assert len(rows["ST003111"]["group_vectors"]) == 7
