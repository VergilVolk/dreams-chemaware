from pathlib import Path

import pandas as pd

from audit_contrast_atlas_open_response_qualification import run_audit


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip" if path.suffix == ".gz" else None)


def test_unsigned_registry_and_identity_opportunity_are_kept_separate(tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    identity_root = tmp_path / "identity"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "output"
    collection.mkdir()

    responses = pd.DataFrame([
        {"study_id": "A", "source_family": "A", "entity_id": "A::1", "contrast_id": "case", "beta": 2.0, "se": 0.5, "identity_status": "library_supported"},
        {"study_id": "A", "source_family": "A", "entity_id": "A::2", "contrast_id": "case", "beta": -1.0, "se": 1.0, "identity_status": "unresolved_entity"},
        {"study_id": "B", "source_family": "B", "entity_id": "B::1", "contrast_id": "case", "beta": 1.5, "se": 0.5, "identity_status": "library_supported"},
        {"study_id": "B", "source_family": "B", "entity_id": "B::2", "contrast_id": "case", "beta": 0.2, "se": 1.0, "identity_status": "unresolved_entity"},
        {"study_id": "C", "source_family": "C", "entity_id": "C::1", "contrast_id": "other", "beta": 0.0, "se": 1.0, "identity_status": "unresolved_entity"},
    ])
    write_table(responses, collection / "response_entities.tsv.gz")
    contrasts = pd.DataFrame([
        {"study_id": "A", "source_family": "A", "contrast_id": "case"},
        {"study_id": "B", "source_family": "B", "contrast_id": "case"},
        {"study_id": "C", "source_family": "C", "contrast_id": "other"},
    ])
    write_table(contrasts, collection / "contrast_manifest.tsv")
    sources = pd.DataFrame([
        {"study_id": "A", "source_family": "A"},
        {"study_id": "B", "source_family": "B"},
        {"study_id": "C", "source_family": "C"},
    ])
    write_table(sources, collection / "source_manifest.tsv")
    relations = pd.DataFrame([
        {"relation_direction": "registered", "left_study_id": "A", "left_contrast_id": "case", "right_study_id": "B", "right_contrast_id": "case", "relation_class": "same_condition", "expected_signed_direction": "unspecified", "eligible_primary_relevance": True, "evidence_boundary": "independent sources", "signed_truth_claimed": False},
        {"relation_direction": "reverse_registered", "left_study_id": "B", "left_contrast_id": "case", "right_study_id": "A", "right_contrast_id": "case", "relation_class": "same_condition", "expected_signed_direction": "unspecified", "eligible_primary_relevance": True, "evidence_boundary": "independent sources", "signed_truth_claimed": False},
    ])
    write_table(relations, collection / "registered_relations.tsv")

    for study, rows in {
        "A": [("A::1", "KEY", True), ("A::2", None, False)],
        "B": [("B::1", "KEY", True), ("B::2", None, False)],
        "C": [("C::1", None, False)],
    }.items():
        partition = pd.DataFrame([
            {"entity_id": entity, "library_inchikey": key, "library_supported": supported, "identity_status": "library_supported" if supported else "unresolved_entity"}
            for entity, key, supported in rows
        ])
        write_table(partition, identity_root / study / "identity_partition.tsv.gz")

    for study in ("A", "C"):
        directory = embeddings / study
        directory.mkdir(parents=True)
        (directory / "report.json").write_text("{}\n", encoding="utf-8")
        (directory / f"{study}.npz").write_bytes(b"npz")
        (directory / f"{study}.spectral_eligibility.tsv.gz").write_bytes(b"tsv")

    report = run_audit(collection, identity_root, output, embeddings)

    assert report["registered_relation_registry"]["directional_registered_rows"] == 2
    assert report["registered_relation_registry"]["canonical_registered_relations"] == 1
    assert report["registered_relation_registry"]["source_disjoint_directional_candidate_pairs"] == 6
    assert report["registered_relation_registry"]["unlabeled_source_disjoint_directional_pairs"] == 4
    assert report["registered_relation_registry"]["confirmed_negative_directional_pairs"] == 0
    assert report["strict_identity_opportunities"]["relation_specific_shared_identity_keys"] == 1
    assert report["strict_identity_opportunities"]["pairs_both_abs_z_ge_1_96"] == 1
    assert report["identity_harmonization"]["response_entities"] == 5
    assert report["identity_harmonization"]["identity_status_changes"] == 0
    assert report["identity_harmonization"]["common_policy_library_supported"] == 2
    assert report["qualification"]["signed_response_truth_available"] is False
    assert report["qualification"]["unregistered_pairs_may_be_used_as_negatives"] is False
    assert report["qualification"]["all_registered_relation_studies_have_embeddings"] is False
    assert report["unresolved_relation_readiness"]["missing_registered_relation_study_shards"] == ["B"]


def test_registry_requires_reverse_row(tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    collection.mkdir()
    write_table(pd.DataFrame([{"study_id": "A", "source_family": "A", "entity_id": "A::1", "contrast_id": "x", "beta": 0.0, "se": 1.0, "identity_status": "unresolved_entity"}]), collection / "response_entities.tsv.gz")
    write_table(pd.DataFrame([{"study_id": "A", "source_family": "A", "contrast_id": "x"}]), collection / "contrast_manifest.tsv")
    write_table(pd.DataFrame([{"study_id": "A", "source_family": "A"}]), collection / "source_manifest.tsv")
    write_table(pd.DataFrame([{"relation_direction": "registered", "left_study_id": "A", "left_contrast_id": "x", "right_study_id": "B", "right_contrast_id": "y", "relation_class": "test", "expected_signed_direction": "unspecified", "eligible_primary_relevance": True, "evidence_boundary": "test", "signed_truth_claimed": False}]), collection / "registered_relations.tsv")

    try:
        run_audit(collection, tmp_path / "identity", tmp_path / "output")
    except RuntimeError as error:
        assert "lacks reverse row" in str(error)
    else:
        raise AssertionError("missing reverse relation was accepted")
