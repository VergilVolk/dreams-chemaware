import numpy as np
import pandas as pd

from tasks.run_contrast_atlas_known_anchor_unresolved_test import (
    add_specificity_residuals,
    reciprocal_unresolved_edges,
    retrieval_rows,
    select_known_entities,
    signed_cosine,
)


def test_signed_cosine_respects_sign_and_positive_weights() -> None:
    left = np.asarray([1.0, -1.0, 2.0, np.nan])
    right = np.asarray([-1.0, 1.0, -2.0, 100.0])
    weight = np.asarray([1.0, 2.0, 1.0, 1.0])
    assert np.isclose(signed_cosine(left, right, weight), -1.0)


def test_known_entity_selection_is_response_blind_and_deterministic() -> None:
    inchikey = "ABCDEFGHIJKLMN-ABCDEFGHIJ-A"
    library = pd.DataFrame(
        {
            "study_id": ["S1", "S1", "S1"],
            "entity_id": ["lower_wse", "best", "unsupported"],
            "library_inchikey": [inchikey, inchikey, inchikey],
            "library_supported": [True, True, False],
            "wse": [0.8, 0.9, 1.0],
            "matched_peaks": [20, 10, 30],
            # A deliberately tempting response column must not affect selection.
            "beta": [1000.0, -1.0, 5000.0],
        }
    )
    selected = select_known_entities(library)
    assert selected["entity_id"].tolist() == ["best"]


def test_reciprocal_unresolved_edges_require_both_directions() -> None:
    edges = pd.DataFrame(
        {
            "left_study_id": ["A", "A", "A", "A"],
            "right_study_id": ["B", "B", "B", "B"],
            "left_entity_id": ["a1", "a1", "a2", "a3"],
            "right_entity_id": ["b1", "b1", "b2", "b3"],
            "direction": ["left_to_right", "right_to_left", "left_to_right", "left_to_right"],
            "evidence_world": ["unresolved", "unresolved", "unresolved", "known"],
            "chemical_similarity": [0.7, 0.8, 0.9, 1.0],
        }
    )
    reciprocal = reciprocal_unresolved_edges(edges)
    assert reciprocal[["left_entity_id", "right_entity_id"]].values.tolist() == [["a1", "b1"]]
    assert reciprocal["chemical_similarity"].tolist() == [0.8]


def test_double_centering_removes_both_contrast_axis_main_effects() -> None:
    rows = []
    for left_index, left_contrast in enumerate(("L1", "L2")):
        for right_index, right_contrast in enumerate(("R1", "R2", "R3")):
            rows.append(
                {
                    "left_study_id": "A",
                    "right_study_id": "B",
                    "left_contrast_id": left_contrast,
                    "right_contrast_id": right_contrast,
                    "known_signed_cosine": 2.0 * left_index + 3.0 * right_index,
                    "unresolved_signed_cosine": 5.0 * left_index - 4.0 * right_index,
                }
            )
    centered = add_specificity_residuals(pd.DataFrame(rows))
    assert centered["double_centering_eligible"].all()
    for prefix in ("known", "unresolved"):
        column = f"{prefix}_double_centered"
        left_means = centered.groupby("left_contrast_id")[column].mean().to_numpy()
        right_means = centered.groupby("right_contrast_id")[column].mean().to_numpy()
        assert np.allclose(left_means, 0.0)
        assert np.allclose(right_means, 0.0)


def test_open_retrieval_keeps_unresolved_candidates_without_known_anchor() -> None:
    joined = pd.DataFrame(
        {
            "left_study_id": ["A"],
            "left_contrast_id": ["qa"],
            "right_study_id": ["B"],
            "right_contrast_id": ["truth"],
            "known_exact_inchikey_support": [5],
            "known_leave_one_out_sign_stable": [True],
            "known_signed_cosine": [0.8],
            "unresolved_signed_cosine": [0.5],
            "unresolved_mutual_edge_support": [3],
        }
    )
    # A--C has no known anchor row but is a legitimate unresolved competitor.
    unresolved = pd.DataFrame(
        {
            "left_study_id": ["A", "A"],
            "left_contrast_id": ["qa", "qa"],
            "right_study_id": ["B", "C"],
            "right_contrast_id": ["truth", "competitor"],
            "unresolved_signed_cosine": [0.5, 0.9],
            "unresolved_mutual_edge_support": [3, 3],
        }
    )
    contrasts = pd.DataFrame(
        {
            "study_id": ["A", "B", "C"],
            "contrast_id": ["qa", "truth", "competitor"],
        }
    )
    result = retrieval_rows(
        joined,
        unresolved,
        contrasts,
        {"A": "PA", "B": "PB", "C": "PC"},
        min_known=5,
        min_unresolved=3,
        anchor_abs=0.5,
    )
    forward = result.loc[result["query_study_id"].eq("A")].iloc[0]
    assert forward["finite_unresolved_candidates"] == 2
    assert forward["worst_tie_rank"] == 2
