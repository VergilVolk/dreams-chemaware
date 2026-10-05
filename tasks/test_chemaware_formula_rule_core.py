"""CPU contracts for formula- and polarity-aware ChemAware rules."""
from __future__ import annotations

import numpy as np

from chemaware_formula_rule_core import (
    RuleChannel,
    formula_gate,
    inverse_document_frequency,
    normalized_formula_response,
    parse_formula,
    shuffled_formula_assignment,
)


def main() -> None:
    assert parse_formula("C6H12O6") == {"C": 6, "H": 12, "O": 6}
    assert parse_formula("NaCl") == {"Na": 1, "Cl": 1}
    assert parse_formula("C8H18OP+") == {"C": 8, "H": 18, "O": 1, "P": 1}
    assert parse_formula("glucuronide") is None

    channels = [
        RuleChannel("NL", "pos", "H2O", "water", 18.0),
        RuleChannel("NL", "pos", "C7H14", "too-large", 98.0),
        RuleChannel("NL", "neg", "CO2", "negative-only", 44.0),
        RuleChannel("CF", "unspecified", None, "ungated", 91.0),
    ]
    gate = formula_gate(channels, "C6H12O6")
    assert gate.tolist() == [True, False, False, True]

    assignment = shuffled_formula_assignment(channels, 17)
    assert len(assignment) == len(channels)
    assert assignment[2] == "CO2"  # excluded negative-mode rules are untouched

    response = np.asarray([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    idf = inverse_document_frequency(response, np.asarray([True, True, False, True]))
    assert idf[2] == 0.0 and idf[1] > idf[0]
    normalized = normalized_formula_response(response, idf, gate)
    assert normalized.shape == response.shape
    assert np.allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])
    assert np.all(normalized[:, 1:3] == 0.0)
    print("PASS: 9 formula-aware ChemAware rule contracts")


if __name__ == "__main__":
    main()
