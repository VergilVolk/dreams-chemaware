"""Contracts for executable ChemAware rules."""
from __future__ import annotations

from build_chemaware_executable_action_knowledge import validate_executable_rule


def valid_rule() -> dict:
    return {
        "rule_id": "R:TEST", "claim_type": "mechanistic",
        "parent_predicate": {"smarts_any": ["[CX3](=O)[OX2H1]"]},
        "context": {"ion_mode": "positive", "adducts": ["[M+H]+"]},
        "observation": {"kind": "neutral_loss", "formula": "H2O", "exact_mass_da": 18.010565},
        "evidence": {"source_id": "curated:test", "citation": "doi:test",
                     "independent_molecules": 25, "independent_formulas": 12,
                     "formula_disjoint_confirmation_pass": True},
    }


def main() -> None:
    rule = valid_rule()
    assert validate_executable_rule(rule) == []
    rule["parent_predicate"] = {"smarts_any": []}
    assert "invalid:parent_predicate.smarts_any" in validate_executable_rule(rule)
    rule = valid_rule(); rule["evidence"]["independent_molecules"] = 1
    assert "insufficient:evidence.independent_molecules" in validate_executable_rule(rule)
    rule = valid_rule(); rule["evidence"]["formula_disjoint_confirmation_pass"] = False
    assert "unconfirmed:evidence.formula_disjoint" in validate_executable_rule(rule)
    print("executable action knowledge contracts passed")


if __name__ == "__main__":
    main()
