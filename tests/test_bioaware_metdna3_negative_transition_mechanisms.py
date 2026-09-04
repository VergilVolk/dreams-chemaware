import pandas as pd

from tasks.audit_bioaware_metdna3_negative_transition_mechanisms import role_rows


def test_role_rows_uses_proposal_minus_displaced_baseline() -> None:
    features = pd.DataFrame([
        {
            "query_id": "q", "candidate_id": candidate,
            "spectral_score": spectral,
            "known_mass_candidate_fraction": mass,
            "known_path_fraction": path,
            "known_inverse_depth_mean": path,
            "known_log_seed_support_mean": path,
            "known_log_degree": degree,
            "edge0_complete_fraction": edge,
            "edge0_bottleneck_mean": bottleneck,
            "edge1_complete_fraction": edge,
            "edge1_bottleneck_mean": bottleneck,
            "predicted_edge_increment": 0.0,
        }
        for candidate, spectral, mass, path, degree, edge, bottleneck in [
            ("truth", 0.8, 1.0, 1.0, 1.0, 1.0, 0.7),
            ("wrong", 0.9, 0.0, 0.0, 0.2, 0.0, 0.0),
        ]
    ])
    transitions = pd.DataFrame([{
        "query_id": "q", "unit_id": "source", "truth_candidate_id": "truth",
        "truth_formula": "C1", "baseline_candidate_id": "wrong",
        "proposed_candidate_id": "truth", "final_candidate_id": "truth",
        "baseline_correct": False, "final_correct": True, "corrected": True,
        "introduced": False, "delta": 1, "intervene": True,
        "proposal_unique": True, "proposal_probability": 0.9,
        "baseline_margin": 0.1, "raw_edge_validated": True,
        "truth_identity_unseen_in_training_units": True,
        "truth_formula_unseen_in_training_units": True,
    }])
    result = role_rows(features, transitions).iloc[0]
    assert result["proposal_minus_baseline__known_path_fraction"] == 1.0
    assert bool(result["proposal_supported_while_baseline_has_no_path"])
    assert bool(result["proposal_raw_edge_while_baseline_has_none"])


def test_role_rows_keeps_introduction_direction() -> None:
    features = pd.DataFrame([
        {
            "query_id": "q", "candidate_id": candidate,
            "spectral_score": spectral,
            "known_mass_candidate_fraction": 0.0,
            "known_path_fraction": path,
            "known_inverse_depth_mean": path,
            "known_log_seed_support_mean": path,
            "known_log_degree": degree,
            "edge0_complete_fraction": edge,
            "edge0_bottleneck_mean": bottleneck,
            "edge1_complete_fraction": edge,
            "edge1_bottleneck_mean": bottleneck,
            "predicted_edge_increment": 0.0,
        }
        for candidate, spectral, path, degree, edge, bottleneck in [
            ("truth", 0.9, 0.0, 1.0, 0.2, 0.1),
            ("wrong", 0.8, 1.0, 3.0, 1.0, 0.6),
        ]
    ])
    transitions = pd.DataFrame([{
        "query_id": "q", "unit_id": "source", "truth_candidate_id": "truth",
        "truth_formula": "C1", "baseline_candidate_id": "truth",
        "proposed_candidate_id": "wrong", "final_candidate_id": "wrong",
        "baseline_correct": True, "final_correct": False, "corrected": False,
        "introduced": True, "delta": -1, "intervene": True,
        "proposal_unique": True, "proposal_probability": 0.9,
        "baseline_margin": 0.1, "raw_edge_validated": True,
        "truth_identity_unseen_in_training_units": True,
        "truth_formula_unseen_in_training_units": True,
    }])
    result = role_rows(features, transitions).iloc[0]
    assert result["proposal_minus_baseline__known_log_degree"] == 2.0
    assert bool(result["proposal_has_stronger_raw_bottleneck"])
    assert bool(result["introduced"])
