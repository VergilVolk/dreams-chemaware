"""Static contracts for direct clean-boundary ChemAware injection."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tasks/audit_chemaware_rule_clean_boundary_injection.py"
CORE = ROOT / "tasks/chemaware_rule_clean_boundary_core.py"


def main() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    core = CORE.read_text(encoding="utf-8")
    ast.parse(source)
    required = (
        'ARMS = ("correct_chemistry", "predicate_present_control")',
        'default=(0, 1, 2)',
        'default=3',
        'default=4',
        'default=0.05',
        '"correct_and_control_negative_references_always_different": True',
        '"controls_trained_as_independent_arm_only": True',
        '"control_gradient_subtraction": False',
        '"candidate_input_at_inference": False',
        '"perturbed_spectrum_training": False',
        '"teacher_embedding_distillation": False',
        'formula_balanced_weights',
        'formula_cluster_interval',
        '"conditioned_rules.json"',
        '"stratified_meta_rules.json"',
        '"official_chemistry_boundary_audit"',
        '"errors_margin_le_0"',
        'np.isin(query_domain, tuple(expected_domains))',
        '"--activation-margin"',
        '"CHEMAWARE_RULE_ACTIVE_BOUNDARY_ACTION_COVERAGE_FAIL"',
        '"nonactive_corrective_weight_exact_zero": True',
        '"rule_selected_active_clean_boundary_transfer"',
    )
    missing = [value for value in required if value not in source]
    assert not missing, missing
    assert "control_loss - correct_loss" not in source
    assert "correct_loss - control_loss" not in source
    assert "(1, 5, 10, 20, 50)" in core
    print("PASS: ChemAware clean-boundary injection contracts")


if __name__ == "__main__":
    main()
