from __future__ import annotations

import pandas as pd

from tasks.audit_bioaware_metdna3_rt_headroom import conservative_rt_candidate, rt_score


def test_rt_score_uses_fixed_relative_window() -> None:
    error, score, passed = rt_score(110.0, 100.0, 0.30)
    assert abs(error - 0.10) < 1e-12
    assert abs(score - (2.0 / 3.0)) < 1e-12
    assert passed
    assert not rt_score(140.0, 100.0, 0.30)[2]


def test_conservative_rule_keeps_passing_dreams_top1() -> None:
    frame = pd.DataFrame([
        {"candidate_id": "base", "rt_pass": True, "spectral_score": 0.8},
        {"candidate_id": "other", "rt_pass": True, "spectral_score": 0.7},
    ])
    assert conservative_rt_candidate(frame, "base") == ("base", False)


def test_conservative_rule_replaces_only_failing_top1() -> None:
    frame = pd.DataFrame([
        {"candidate_id": "base", "rt_pass": False, "spectral_score": 0.8},
        {"candidate_id": "winner", "rt_pass": True, "spectral_score": 0.7},
        {"candidate_id": "other", "rt_pass": True, "spectral_score": 0.6},
    ])
    assert conservative_rt_candidate(frame, "base") == ("winner", True)
