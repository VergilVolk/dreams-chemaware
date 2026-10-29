import pandas as pd

from summarize_contrast_atlas_parent_project_holdout import (
    project_coverage,
    query_stability,
    two_face_calls,
)


PROJECTS = {"a": "pa", "b": "pb", "c": "pc"}


def row(study, contrast, face, p, edges, z, candidate):
    return {
        "query_study_id": study,
        "query_contrast_id": contrast,
        "evidence_world": "unresolved",
        "evidence_track": "same_precursor_10ppm_identity_hypothesis",
        "response_face": face,
        "query_maxT_p": p,
        "weighted_evidence_edges": edges,
        "enrichment_z": z,
        "candidate_study_id": candidate,
        "candidate_contrast_id": f"{candidate}_contrast",
    }


def test_two_face_call_requires_bonferroni_and_support():
    top = pd.DataFrame([
        row("a", "ca", "similar", 0.03, 4, 2.0, "b"),
        row("a", "ca", "opposite", 0.04, 4, 2.0, "c"),
        row("b", "cb", "similar", 0.02, 2, 3.0, "a"),
        row("b", "cb", "opposite", 0.01, 1, 4.0, "c"),
        row("c", "cc", "similar", 0.02, 3, 2.0, "a"),
        row("c", "cc", "opposite", 0.40, 3, 1.0, "b"),
    ])
    calls = two_face_calls(top, PROJECTS, expected_queries=3)
    observed = dict(zip(calls["query_study_id"], calls["result_class"]))
    assert observed == {
        "a": "no_calibrated_match",
        "b": "abstain_insufficient_evidence",
        "c": "calibrated_cross_project_answer",
    }
    assert calls.loc[calls["query_study_id"].eq("c"), "two_face_bonferroni_p"].item() == 0.04


def test_project_coverage_is_macro_ready():
    calls = pd.DataFrame({
        "query_parent_project": ["large", "large", "small"],
        "query_contrast_id": ["a", "b", "c"],
        "result_class": [
            "calibrated_cross_project_answer",
            "no_calibrated_match",
            "calibrated_cross_project_answer",
        ],
    })
    coverage = project_coverage(calls)
    assert coverage.set_index("query_parent_project").loc["large", "answer_coverage"] == 0.5
    assert coverage.set_index("query_parent_project").loc["small", "answer_coverage"] == 1.0


def test_query_stability_keeps_sensitivity_graphs_descriptive():
    rows = []
    for replicate in range(32):
        rows.extend([
            {
                "replicate": replicate,
                "query_study_id": "a",
                "query_contrast_id": "ca",
                "query_parent_project": "pa",
                "result_class": "calibrated_cross_project_answer",
                "candidate_parent_project": "pb",
            },
            {
                "replicate": replicate,
                "query_study_id": "b",
                "query_contrast_id": "cb",
                "query_parent_project": "pb",
                "result_class": (
                    "calibrated_cross_project_answer"
                    if replicate < 5 else "no_calibrated_match"
                ),
                "candidate_parent_project": "pa" if replicate < 16 else "pc",
            },
            {
                "replicate": replicate,
                "query_study_id": "c",
                "query_contrast_id": "cc",
                "query_parent_project": "pc",
                "result_class": "no_calibrated_match",
                "candidate_parent_project": "pb",
            },
        ])
    observed = query_stability(pd.DataFrame(rows)).set_index("query_study_id")
    assert observed.loc["a", "answer_stability_class"] == "always_answered"
    assert observed.loc["b", "answer_stability_class"] == "variable_answer"
    assert observed.loc["c", "answer_stability_class"] == "never_answered"
    assert observed.loc["a", "selected_candidate_projects"] == 1
    assert observed.loc["b", "selected_candidate_projects"] == 2
    assert observed.loc["b", "answered_selections"] == 5
