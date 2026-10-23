import numpy as np

from analyze_contrast_atlas_unresolved_edge_replication import bh_adjust


def test_bh_adjust_is_monotone_in_rank_and_restores_order():
    p = np.asarray([0.04, 0.001, 0.02, 0.5])
    q = bh_adjust(p)
    assert np.allclose(q, [0.05333333333333334, 0.004, 0.04, 0.5])
    assert q[1] < q[2] < q[0] < q[3]
