#!/usr/bin/env python
"""Static checks for B28."""
from __future__ import annotations

import ast
from pathlib import Path


def main() -> None:
    path = Path(__file__).with_name("audit_bioaware_b28_branch_spectral_arbitration.py")
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    for required in (
        '"B17", "direct", "neutral_loss", "modified_cosine", "dual_view"',
        '"three_view_majority", "three_view_mean"',
        '"only_B12_B16_disagreements_changed": True',
        '"outer_outcome_used_for_policy_selection": False',
        '"truth_used_as_action_feature": False',
        '"P2b_used": False',
    ):
        assert required in source, required
    assert "proposal[disagree & (advantage > 0)]" in source
    assert "proposal[disagree & (advantage < 0)]" in source
    print("[test_bioaware_b28_branch_spectral_arbitration] PASS")


if __name__ == "__main__":
    main()
