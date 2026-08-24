import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks"))

from audit_g8r_p3_candidate_ambiguity_router import candidate_ambiguity  # noqa: E402


def test_candidate_set_ambiguity_does_not_use_query_identity_label():
    identities = ["A", "B", "C"]
    relations = {"near": {("A", "B")}, "mid": {("B", "C")}, "far": set()}
    formulas = {"A": "X", "B": "X", "C": "Y"}
    result = candidate_ambiguity(identities, formulas, relations)
    assert result["has_any_near_pair"]
    assert result["has_any_nearmid_pair"]
    assert result["has_any_same_formula_pair"]
