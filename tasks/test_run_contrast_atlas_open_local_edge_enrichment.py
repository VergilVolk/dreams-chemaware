import numpy as np
import pandas as pd

from run_contrast_atlas_open_local_edge_enrichment import (
    StudyProfiles,
    bh_adjust,
    build_profile_reassignments,
    count_and_strength,
    audit_registered_relations,
    make_directed_ledger,
)


def profile(entities, status, weight, positive=None, negative=None):
    weight = np.asarray(weight, dtype=np.float32)
    shape = weight.shape
    return StudyProfiles(
        entities=np.asarray(entities, dtype=str),
        contrasts=[f"c{i}" for i in range(shape[1])],
        status=np.asarray(status, dtype=str),
        positive=np.zeros(shape, dtype=bool) if positive is None else np.asarray(positive),
        negative=np.zeros(shape, dtype=bool) if negative is None else np.asarray(negative),
        weight=weight,
        beta=weight.copy(),
        se=np.ones(shape, dtype=np.float32),
        q=np.ones(shape, dtype=np.float32),
    )


def test_continuous_strength_does_not_require_normal_approximation_fdr():
    left = profile(["a", "b"], ["unresolved_entity"] * 2, [[0.8], [-0.7]])
    right = profile(["x", "y"], ["unresolved_entity"] * 2, [[0.6], [-0.5]])
    same, opposite, same_strength, opposite_strength = count_and_strength(
        left,
        right,
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        np.asarray([1.0, 0.5]),
    )
    assert same.item() == 0
    assert opposite.item() == 0
    assert np.isclose(same_strength.item(), 0.8 * 0.6 + 0.5 * 0.7 * 0.5)
    assert opposite_strength.item() == 0


def test_profile_reassignment_preserves_identity_strata():
    profiles = {
        "study": profile(
            ["a", "b", "c", "d"],
            ["library_supported", "library_supported", "unresolved_entity", "unresolved_entity"],
            [[0.1], [0.2], [0.3], [0.4]],
        )
    }
    donors = build_profile_reassignments(profiles, permutations=32, seed=7)["study"]
    assert donors.shape == (32, 4)
    assert all(set(row[:2]) == {0, 1} for row in donors)
    assert all(set(row[2:]) == {2, 3} for row in donors)


def test_bh_adjust_restores_original_order():
    observed = bh_adjust(np.asarray([0.04, 0.001, 0.02, 0.5]))
    assert np.allclose(observed, [0.05333333333333334, 0.004, 0.04, 0.5])


def test_parent_project_disjoint_ledger_removes_shared_project_candidates():
    rows = []
    for world in ("all", "known", "unresolved"):
        for track in (
            "combined_mutual",
            "same_precursor_10ppm_identity_hypothesis",
            "nonmass_chemical_relation_hypothesis",
        ):
            for left, left_contrast, right, right_contrast, similar in (
                ("a", "ca", "b", "cb", 5.0),
                ("a", "ca", "c", "cc", 2.0),
                ("b", "cb", "c", "cc", 3.0),
            ):
                rows.append({
                    "undirected_id": len(rows),
                    "left_study_id": left,
                    "left_contrast_id": left_contrast,
                    "right_study_id": right,
                    "right_contrast_id": right_contrast,
                    "evidence_world": world,
                    "evidence_track": track,
                    "mutual_edges": 1,
                    "similar_z": similar,
                    "opposite_z": 1.0,
                })
    undirected = pd.DataFrame(rows)
    nulls = {
        int(row.undirected_id): (np.zeros(19), np.zeros(19))
        for row in undirected.itertuples(index=False)
    }
    contrasts = pd.DataFrame({
        "study_id": ["a", "b", "c"],
        "source_family": ["a", "b", "c"],
        "contrast_id": ["ca", "cb", "cc"],
        "source_disjoint_candidate_contrasts": [2, 2, 2],
    })
    ledger = make_directed_ledger(
        undirected,
        nulls,
        contrasts,
        parent_project_lookup={"a": "shared", "b": "shared", "c": "independent"},
    )
    assert not (
        ledger[["query_study_id", "candidate_study_id"]]
        .apply(frozenset, axis=1)
        .eq(frozenset(("a", "b")))
    ).any()
    assert set(ledger["candidate_universe_size"]) == {1, 2}


def test_registered_same_parent_relation_is_reported_as_excluded(tmp_path):
    ledger = pd.DataFrame(columns=[
        "query_study_id", "query_contrast_id", "candidate_study_id", "candidate_contrast_id"
    ])
    relations = pd.DataFrame([{
        "relation_direction": "registered",
        "left_study_id": "a",
        "left_contrast_id": "ca",
        "right_study_id": "b",
        "right_contrast_id": "cb",
    }])
    path = tmp_path / "relations.tsv"
    relations.to_csv(path, sep="\t", index=False)
    result = audit_registered_relations(
        ledger,
        path,
        parent_project_lookup={"a": "shared", "b": "shared"},
    )
    assert len(result) == 9
    assert not result["candidate_in_universe"].any()
    assert not result["relation_candidate_eligible"].any()
    assert set(result["evaluation_status"]) == {"not_evaluable_same_parent_project"}
    assert result["best_face_bonferroni_p"].isna().all()
