"""Contracts for the charged-species skip in the MassBank source ledger."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from build_chemaware_massbank_candidate_source_ledger import (  # noqa: E402
    charged_species_formula,
    protonated_parent_formula,
)


def test_net_charge_notation_is_detected() -> None:
    assert charged_species_formula("C8H18OP+")
    assert charged_species_formula("C22H28N7O+")
    assert charged_species_formula("C8H17N-")


def test_neutral_formulas_are_scoreable() -> None:
    assert not charged_species_formula("C8H18OP")
    assert not charged_species_formula("C6H12O6")
    assert not charged_species_formula("")


def test_neutral_protonation_is_unchanged() -> None:
    parsed = protonated_parent_formula("C8H18OP")
    assert parsed["C"] == 8 and parsed["H"] == 19 and parsed["O"] == 1


def test_malformed_formula_still_fails_closed() -> None:
    with pytest.raises(ValueError, match="invalid candidate formula"):
        protonated_parent_formula("not-a-formula")


def test_skip_and_original_local_index_are_wired() -> None:
    text = (
        ROOT / "tasks/build_chemaware_massbank_candidate_source_ledger.py"
    ).read_text(encoding="utf-8")
    assert "charged_species_formula(rows[pos][\"formula\"])" in text
    assert "if not positions:" in text
    # After filtering, the emitted join key must be the registry's own
    # local_candidate, never the enumerate position.
    assert '"local_candidate": int(row["local_candidate"]),' in text
    assert '"local_candidate": local,' not in text
    assert '"charged_species_candidates_skipped": charged_skipped,' in text
