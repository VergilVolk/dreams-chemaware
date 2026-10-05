#!/usr/bin/env python
"""Unit/source checks for the B27 candidate-spectrum veto."""
from __future__ import annotations

import ast
from pathlib import Path


def main() -> None:
    path = Path(__file__).with_name("audit_bioaware_b27_candidate_spectral_veto.py")
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    for value in (
        '"B17"', '"veto_unanimous_baseline"', '"require_any_support"',
        '"require_majority_support"', '"require_unanimous_support"',
        '"require_mean_advantage_gt_0"', '"require_mean_advantage_ge_0_01"',
        '"require_dual_view_advantage_gt_0"',
    ):
        assert value in source, value
    assert 'output["baseline_candidate_id"].astype(str)' in source
    assert '"veto_only_reverts_to_DreaMS": True' in source
    assert '"outer_outcome_used_for_policy_selection": False' in source
    assert '"truth_used_as_action_feature": False' in source
    assert '"P2b_used": False' in source
    print("[test_bioaware_b27_candidate_spectral_veto] PASS")


if __name__ == "__main__":
    main()
