import numpy as np
import pandas as pd

from calibrate_contrast_atlas_sparse_local_response import (
    build_profile_matrices,
    connected_relation_components,
    directional_response_energy_matrices,
    directional_score_matrices,
    permutation_indices,
    prepare_graphs,
)


def test_response_energy_kernel_uses_complete_world_denominator():
    source = np.asarray([[1.0], [1.0]])
    target = np.asarray([[1.0], [1.0]])
    similar, opposite, unsigned = directional_response_energy_matrices(
        source,
        target,
        np.asarray([0]),
        np.asarray([0]),
        np.asarray([1.0]),
        np.asarray([0, 1]),
        np.asarray([0, 1]),
    )
    # One of two unit-energy entities is chemically supported; missing support
    # therefore contributes zero instead of being removed from the denominator.
    assert np.allclose(similar, [[0.5]])
    assert np.allclose(opposite, [[0.0]])
    assert np.allclose(unsigned, [[0.5]])


def test_directional_score_matrices_match_hand_calculation():
    source = np.asarray([[1.0, -1.0], [-0.5, 0.5]])
    target = np.asarray([[0.5, -0.5], [-1.0, 1.0]])
    similar, opposite, unsigned = directional_score_matrices(
        source,
        target,
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        np.asarray([1.0, 0.5]),
        denominator=2,
    )
    expected_signed = source.T @ (np.asarray([1.0, 0.5])[:, None] * target) / 2
    assert np.allclose(similar - opposite, expected_signed)
    assert np.allclose(unsigned, similar + opposite)
    assert np.all(similar >= 0)
    assert np.all(opposite >= 0)


def test_profile_permutation_preserves_identity_strata_and_whole_profiles():
    entities = pd.DataFrame(
        {
            "study_id": ["S"] * 6,
            "contrast_id": ["a", "a", "a", "b", "b", "b"],
            "entity_id": ["e1", "e2", "e3", "e1", "e2", "e3"],
            "beta": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "se": [1.0] * 6,
        }
    )
    contrasts = pd.DataFrame(
        {"study_id": ["S", "S"], "contrast_id": ["a", "b"]}
    )
    identity = pd.DataFrame(
        {
            "study_id": ["S"] * 3,
            "entity_id": ["e1", "e2", "e3"],
            "identity_status": [
                "library_supported",
                "unresolved_entity",
                "unresolved_entity",
            ],
        }
    )
    matrices, _entities, _contrasts, statuses = build_profile_matrices(
        entities, contrasts, identity
    )
    donors = permutation_indices(statuses, seed=7)["S"]
    assert donors[0] == 0
    assert set(donors[1:]) == {1, 2}
    permuted = matrices["S"][donors]
    for row in permuted:
        assert any(np.allclose(row, original) for original in matrices["S"])


def test_prepare_graphs_retains_expected_zero_edge_world():
    edges = pd.DataFrame(
        columns=[
            "left_study_id",
            "right_study_id",
            "evidence_world",
            "direction",
            "left_entity_id",
            "right_entity_id",
            "chemical_similarity",
        ]
    )
    graphs = prepare_graphs(
        edges,
        {"A": ["a"], "B": ["b"]},
        {
            "A": np.asarray(["library_supported"]),
            "B": np.asarray(["library_supported"]),
        },
        {("A", "B", "known")},
    )
    graph = graphs[("A", "B", "known")]
    assert graph["forward_denominator"] == 1
    assert graph["reverse_denominator"] == 1
    assert len(graph["forward_source"]) == 0
    assert len(graph["reverse_source"]) == 0


def test_connected_relation_components_ignores_reverse_duplicates():
    relations = pd.DataFrame(
        {
            "left_study_id": ["A", "B", "C"],
            "left_contrast_id": ["a", "b", "c"],
            "right_study_id": ["B", "A", "D"],
            "right_contrast_id": ["b", "a", "d"],
        }
    )
    assert connected_relation_components(relations) == 2
