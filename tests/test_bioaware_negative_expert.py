import numpy as np
import pandas as pd
import pytest

from annotation.bioaware_negative_expert import (
    FrozenNegativeBioAwareExpert,
    apply_frozen_negative_bioaware_expert,
)


def expert() -> FrozenNegativeBioAwareExpert:
    return FrozenNegativeBioAwareExpert(
        feature_names=("known_mass_candidate_fraction", "edge0_complete_fraction", "edge0_bottleneck_mean"),
        scaler_mean=np.zeros(3), scaler_scale=np.ones(3),
        model_coef=np.ones(3),
        maximum_dreams_top1_top2_gap=0.05,
        minimum_pairwise_proposal_probability=0.75,
        requires_raw_step0_edge_validation=True,
        scope="test",
    )


def candidates(gap: float = 0.01) -> pd.DataFrame:
    return pd.DataFrame({
        "query_id": ["q", "q"], "candidate_id": ["dreams", "network"],
        "spectral_score": [0.8, 0.8 - gap],
        "known_mass_candidate_fraction": [0.0, 1.0],
        "edge0_complete_fraction": [0.0, 1.0],
        "edge0_bottleneck_mean": [0.0, 1.0],
    })


def test_network_candidate_can_override_only_with_all_gates() -> None:
    _, decision = apply_frozen_negative_bioaware_expert(candidates(), expert())
    assert bool(decision.iloc[0].intervene)
    assert decision.iloc[0].final_candidate_id == "network"


def test_confident_dreams_abstains() -> None:
    _, decision = apply_frozen_negative_bioaware_expert(candidates(gap=0.10), expert())
    assert not bool(decision.iloc[0].intervene)
    assert "dreams_confident" in decision.iloc[0].abstention_reasons


def test_truth_and_phenotype_columns_are_rejected() -> None:
    frame = candidates()
    frame["truth_candidate_id"] = "network"
    with pytest.raises(ValueError, match="forbidden"):
        apply_frozen_negative_bioaware_expert(frame, expert())


def test_network_tie_abstains() -> None:
    frame = candidates()
    frame.loc[:, ["known_mass_candidate_fraction", "edge0_complete_fraction", "edge0_bottleneck_mean"]] = 1.0
    _, decision = apply_frozen_negative_bioaware_expert(frame, expert())
    assert not bool(decision.iloc[0].intervene)
    assert "network_top_tie" in decision.iloc[0].abstention_reasons
