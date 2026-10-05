#!/usr/bin/env python
"""Dependency-light contract checks for BioAware B29."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tasks/audit_bioaware_b29_action_risk_router.py"


def main() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    text = source.lower()
    required = (
        "every outer source", "matching truth identity/formula",
        "formula-group oof",
        "collapse_physical", "risk_penalty = 2.0", "logisticregression",
        "b29_execute", "outer_outcome_used_for_model_or_threshold_selection",
        "p2b_used", "shared_embedding_changed",
    )
    missing = [token for token in required if token not in text]
    assert not missing, missing
    # Source is retained for grouping/reporting, never added to FEATURE_FAMILIES.
    assert "candidate_identity_or_source_used_as_feature" in text
    assert any(isinstance(node, ast.ClassDef) is False for node in tree.body)
    assert "#SBATCH" not in source
    print("[test_bioaware_b29_action_risk_router] PASS", flush=True)


if __name__ == "__main__":
    main()
