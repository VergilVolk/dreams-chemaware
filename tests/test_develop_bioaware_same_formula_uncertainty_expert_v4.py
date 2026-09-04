from tasks.develop_bioaware_same_formula_uncertainty_expert_v4 import (
    MINIMUM_PROPOSAL_PROBABILITY,
)
from tasks.develop_bioaware_metdna3_negative_loso_ranker import (
    PRIMARY_BASELINE_MARGIN_MAX,
)


def test_v4_keeps_original_dreams_uncertainty_threshold():
    assert PRIMARY_BASELINE_MARGIN_MAX == 0.05


def test_v4_accepts_every_unique_score_improving_network_proposal():
    assert MINIMUM_PROPOSAL_PROBABILITY == 0.5
