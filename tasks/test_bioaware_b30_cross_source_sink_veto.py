#!/usr/bin/env python
"""Dependency-light contract checks for BioAware B30."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "tasks/audit_bioaware_b30_cross_source_sink_veto.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    required = (
        "minimum_development_sources = 2", "minimum_development_actions = 3",
        "risk_penalty = 2", "candidate_sink_history_excludes_outer_source",
        "physical_duplicates_have_one_history_vote", "router_can_only_revert_b17",
        "unseen_candidate_falls_back_to_b17", "outer_outcome_used_for_sink_definition",
    )
    lower = source.lower()
    missing = [token for token in required if token not in lower]
    assert not missing, missing
    assert "p2b_score" not in lower
    print("[test_bioaware_b30_cross_source_sink_veto] PASS", flush=True)


if __name__ == "__main__":
    main()
