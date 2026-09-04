from __future__ import annotations

import numpy as np
import pytest

from audit_lcnec_hsst3n_priority_annotation import competition_qvalues, score_margin


def test_competition_qvalues_are_conservative_for_decoy_wins() -> None:
    target = np.array([0.9, 0.8, 0.7, 0.6])
    decoy = np.array([0.1, 0.85, 0.2, 0.65])
    q = competition_qvalues(target, decoy)
    assert q.shape == target.shape
    assert q[1] == 1.0 and q[3] == 1.0
    assert np.all((0 <= q) & (q <= 1))


def test_score_margin() -> None:
    assert score_margin(np.array([0.8, 0.6, 0.1])) == pytest.approx(0.2)
    assert np.isinf(score_margin(np.array([0.8])))
