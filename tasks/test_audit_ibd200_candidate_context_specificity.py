import numpy as np

from audit_ibd200_candidate_context_specificity import exact_cluster_signflip_p


def test_exact_cluster_signflip_is_one_sided_and_exact():
    assert exact_cluster_signflip_p(np.asarray([1.0])) == 0.5
    assert exact_cluster_signflip_p(np.asarray([1.0, 1.0])) == 0.25


def test_exact_cluster_signflip_handles_mixed_cluster_effects():
    p_value = exact_cluster_signflip_p(np.asarray([1.0, -0.5]))
    assert 0.0 < p_value <= 1.0
