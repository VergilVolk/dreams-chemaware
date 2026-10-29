from __future__ import annotations

import numpy as np
import pandas as pd

from build_contrast_atlas_connection_evidence_ledger import (
    CONNECTION_KEY,
    IDENTITY_TRACK,
    RELATION_TRACK,
    canonical_reference_connections,
    iter_edge_contribution_blocks,
    prepare_method_edges,
    summarise_and_validate_contributions,
)


def partition() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("A", "A::known", "library_supported"),
            ("A", "A::unknown", "unresolved_entity"),
            ("B", "B::known", "library_supported"),
            ("B", "B::unknown", "unresolved_entity"),
        ],
        columns=["study_id", "entity_id", "identity_status"],
    ).assign(common_identity_policy="frozen_test_policy")


def edge_rows() -> pd.DataFrame:
    rows = []
    for world, left, right, score in (
        ("all", "A::unknown", "B::unknown", 0.8),
        ("known", "A::known", "B::known", 0.5),
        ("unresolved", "A::unknown", "B::unknown", 0.8),
    ):
        rows.append(
            {
                "left_study_id": "A",
                "right_study_id": "B",
                "evidence_world": world,
                "left_entity_id": left,
                "right_entity_id": right,
                "chemical_similarity": score,
                "directional_support": 2,
                "left_precursor_mz": 100.0,
                "right_precursor_mz": 100.0005,
                "precursor_ppm_difference": 5.0,
                "precursor_compatible_10ppm": True,
                "precursor_incompatible_over_10ppm": False,
            }
        )
    return pd.DataFrame(rows)


def effects() -> pd.DataFrame:
    rows = []
    values = {
        "A::known": 0.8,
        "A::unknown": 0.6,
        "B::known": 0.5,
        "B::unknown": -0.7,
    }
    for study, entity, status in (
        ("A", "A::known", "library_supported"),
        ("A", "A::unknown", "unresolved_entity"),
        ("B", "B::known", "library_supported"),
        ("B", "B::unknown", "unresolved_entity"),
    ):
        rows.append(
            {
                "study_id": study,
                "entity_id": entity,
                "contrast_id": f"{study}_contrast",
                "beta": values[entity],
                "se": 1.0,
                "effect_q": 0.01,
                "response_qualified": True,
                "retrieval_response_weight": values[entity],
                "common_identity_status": status,
            }
        )
    return pd.DataFrame(rows)


def references() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "undirected_id": index,
                "left_study_id": "A",
                "left_contrast_id": "A_contrast",
                "right_study_id": "B",
                "right_contrast_id": "B_contrast",
                "evidence_world": world,
                "mutual_edges": 1,
                "weighted_evidence_edges": 1,
                "same_direction_dual_fdr_edges": int(world == "known"),
                "opposite_direction_dual_fdr_edges": int(world != "known"),
                "same_direction_strength": 0.2 if world == "known" else 0.0,
                "opposite_direction_strength": 0.336 if world != "known" else 0.0,
            }
            for index, world in enumerate(("all", "known", "unresolved"))
        ]
    )


def test_identity_edges_are_hypotheses_and_scores_are_not_probabilities(tmp_path) -> None:
    path = tmp_path / "edges.tsv"
    edge_rows().to_csv(path, sep="\t", index=False)
    result = prepare_method_edges(path, partition(), axis="identity")
    assert len(result) == 3
    assert set(result["method_name"]) == {"weighted_spectral_entropy_complete_axis"}
    assert result["edge_claim_boundary"].str.contains("not_confirmed_structure").all()
    assert set(result["score_semantics"]) == {
        "nonnegative_spectral_evidence_weight_not_probability"
    }
    assert set(result.loc[result.evidence_world.eq("known"), "left_common_identity_status"]) == {
        "library_supported"
    }


def test_relation_edges_are_separate_and_never_claim_identity(tmp_path) -> None:
    relation = edge_rows().copy()
    relation["precursor_compatible_10ppm"] = False
    relation["precursor_incompatible_over_10ppm"] = True
    relation["right_precursor_mz"] = 125.0
    relation["precursor_ppm_difference"] = 222222.0
    path = tmp_path / "relations.tsv"
    relation.to_csv(path, sep="\t", index=False)
    result = prepare_method_edges(path, partition(), axis="legacy_relation")
    assert len(result) == 3
    assert set(result["method_name"]) == {
        "legacy_fragment_cosine_mutual_top1_relation_axis_exploratory"
    }
    assert result["edge_claim_boundary"].str.contains("never_identity").all()


def test_full_contributions_conserve_same_and_opposite_strength(tmp_path) -> None:
    path = tmp_path / "edges.tsv"
    edge_rows().to_csv(path, sep="\t", index=False)
    edges = prepare_method_edges(path, partition(), axis="identity")
    reference = references()
    blocks = list(iter_edge_contribution_blocks(edges, effects(), reference))
    assert sum(map(len, blocks)) == 3
    result, diagnostics = summarise_and_validate_contributions(
        blocks,
        reference,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )
    assert len(result) == 3
    assert result["contribution_conservation_pass"].all()
    assert diagnostics["max_abs_same_strength_difference"] < 1e-12
    assert diagnostics["max_abs_opposite_strength_difference"] < 1e-12
    assert np.isclose(
        result.loc[result.evidence_world.eq("known"), "reconstructed_same_direction_strength"].item(),
        0.5 * 0.8 * 0.5,
    )
    assert np.isclose(
        result.loc[result.evidence_world.eq("all"), "reconstructed_opposite_direction_strength"].item(),
        0.8 * 0.6 * 0.7,
    )


def test_missing_entity_effect_remains_an_explicit_zero_contribution(tmp_path) -> None:
    edge_path = tmp_path / "edges.tsv"
    edge_rows().to_csv(edge_path, sep="\t", index=False)
    edges = prepare_method_edges(edge_path, partition(), axis="identity")
    response = effects()
    # B_extra is observed in a second contrast, which makes the contrast part
    # of the study, while the edge endpoint B::unknown is unobserved there.
    response = pd.concat(
        [
            response,
            pd.DataFrame(
                [
                    {
                        "study_id": "B",
                        "entity_id": "B::extra",
                        "contrast_id": "B_second",
                        "beta": 0.4,
                        "se": 1.0,
                        "effect_q": 0.01,
                        "response_qualified": True,
                        "retrieval_response_weight": 0.4,
                        "common_identity_status": "unresolved_entity",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    reference = references()
    second = reference.copy()
    second["undirected_id"] += 10
    second["right_contrast_id"] = "B_second"
    second["same_direction_dual_fdr_edges"] = 0
    second["opposite_direction_dual_fdr_edges"] = 0
    second["same_direction_strength"] = 0.0
    second["opposite_direction_strength"] = 0.0
    reference = pd.concat([reference, second], ignore_index=True)
    blocks = list(iter_edge_contribution_blocks(edges, response, reference))
    missing = pd.concat(blocks, ignore_index=True).loc[
        lambda frame: frame["right_contrast_id"].eq("B_second")
    ]
    assert len(missing) == 3
    assert not missing["right_effect_observed"].any()
    assert missing["right_response_weight"].eq(0).all()
    assert missing["same_direction_contribution"].eq(0).all()
    assert missing["opposite_direction_contribution"].eq(0).all()
    summarise_and_validate_contributions(
        blocks,
        reference,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )


def test_reference_table_requires_two_matching_query_directions(tmp_path) -> None:
    rows = []
    for world_index, world in enumerate(("all", "known", "unresolved")):
        for reverse in (False, True):
            rows.append(
                {
                    "undirected_id": world_index,
                    "query_study_id": "B" if reverse else "A",
                    "query_contrast_id": "B_contrast" if reverse else "A_contrast",
                    "candidate_study_id": "A" if reverse else "B",
                    "candidate_contrast_id": "A_contrast" if reverse else "B_contrast",
                    "evidence_world": world,
                    "evidence_track": IDENTITY_TRACK,
                    "mutual_edges": 1,
                    "weighted_evidence_edges": 1,
                    "same_direction_dual_fdr_edges": 0,
                    "opposite_direction_dual_fdr_edges": 1,
                    "same_direction_strength": 0.0,
                    "opposite_direction_strength": 0.3,
                }
            )
    path = tmp_path / "scores.tsv"
    # Legacy score ledgers did not yet expose weighted_evidence_edges; every
    # retained edge was positive, so the loader must recover it from count.
    pd.DataFrame(rows).drop(columns="weighted_evidence_edges").to_csv(
        path, sep="\t", index=False
    )
    canonical = canonical_reference_connections(path, IDENTITY_TRACK)
    assert len(canonical) == 3
    assert canonical[CONNECTION_KEY].duplicated().sum() == 0
    assert canonical["weighted_evidence_edges"].eq(1).all()


def test_relation_track_constant_is_not_identity_track() -> None:
    assert RELATION_TRACK != IDENTITY_TRACK
