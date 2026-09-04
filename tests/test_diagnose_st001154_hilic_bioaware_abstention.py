import numpy as np
import pandas as pd

from annotation.bioaware_negative_expert import FrozenNegativeBioAwareExpert
from tasks.diagnose_st001154_hilic_bioaware_abstention import (
    build_proposals,
    gate_decomposition,
    within_query_variation,
)


def expert() -> FrozenNegativeBioAwareExpert:
    return FrozenNegativeBioAwareExpert(
        feature_names=("spectral_score", "edge0_complete_fraction", "edge0_bottleneck_mean"),
        scaler_mean=np.zeros(3),
        scaler_scale=np.ones(3),
        model_coef=np.array([1.0, 1.0, 1.0]),
        maximum_dreams_top1_top2_gap=0.05,
        minimum_pairwise_proposal_probability=0.75,
        requires_raw_step0_edge_validation=True,
        scope="test",
    )


def test_gate_decomposition_distinguishes_proposal_and_safety_gates():
    frame = pd.DataFrame({
        "query_id": ["q1", "q1", "q2", "q2"],
        "candidate_id": ["wrong", "truth", "truth2", "wrong2"],
        "spectral_score": [0.51, 0.50, 0.51, 0.50],
        "edge0_complete_fraction": [0.0, 1.0, 0.0, 1.0],
        "edge0_bottleneck_mean": [0.0, 1.0, 0.0, 1.0],
    })
    truth = pd.DataFrame({
        "query_id": ["q1", "q2"],
        "truth_candidate_id": ["truth", "truth2"],
    })
    proposals, _ = build_proposals(frame, truth, expert())
    result = gate_decomposition(proposals, expert())
    assert result["proposal_only"]["corrected"] == 1
    assert result["proposal_only"]["introduced"] == 1
    assert result["proposal_plus_dreams_gap"]["interventions"] == 2
    assert result["sequential_gap_then_probability_then_raw"]["interventions"] == 2


def test_within_query_variation_detects_constant_feature():
    frame = pd.DataFrame({
        "query_id": ["a", "a", "b", "b"],
        "constant": [1.0, 1.0, 1.0, 1.0],
        "variable": [0.0, 1.0, 2.0, 2.0],
    })
    result = within_query_variation(frame, ("constant", "variable"))
    assert result["constant"]["queries_with_nonzero_range"] == 0
    assert result["variable"]["queries_with_nonzero_range"] == 1
