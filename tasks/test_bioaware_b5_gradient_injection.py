#!/usr/bin/env python
"""Source contract checks for the BioAware B5 gradient gate."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "tasks/audit_bioaware_b5_gradient_injection.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    for token in (
        "teacher_model_score", "baseline_candidate_id",
        "corrected teacher target is not exact truth",
        "bioaware_vs_generic_gradient_cosine",
        "truth_vs_exact_baseline_wrong_margin_delta",
        "bioaware_logits_enter_loss\": True",
        "generic_onehot_is_control_only\": True",
        "outer_held_formula_excluded\": True",
        '"params": backbone', '"params": head',
        "pass_to_shared_embedding_pilot",
    ):
        if token not in text:
            raise AssertionError(f"B5 gradient audit lacks contract token: {token}")
    print("[BioAware B5 gradient-injection source checks] PASS")


if __name__ == "__main__":
    main()
