from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation.bioaware_relations import identity_noop_reactions, typed_reaction_pairs


def participant(compound, reaction, side, stoich=1, direction="unknown", currency=False):
    return {
        "compound_id": compound, "reaction_id": reaction, "side": side,
        "stoichiometry": stoich, "direction_semantics": direction,
        "is_currency": currency,
    }


def test_noop_and_typed_direction() -> None:
    rows = [
        participant("AAAAAAAAAAAAAA", "noop", "left"),
        participant("AAAAAAAAAAAAAA", "noop", "right"),
        participant("WATERWATERWATE", "noop", "right", currency=True),
        participant("AAAAAAAAAAAAAA", "r1", "left", direction="physiological_lr"),
        participant("BBBBBBBBBBBBBB", "r1", "right", direction="physiological_lr"),
        participant("BBBBBBBBBBBBBB", "r2", "left", direction="reversible"),
        participant("CCCCCCCCCCCCCC", "r2", "right", direction="reversible"),
    ]
    table = pd.DataFrame(rows)
    assert identity_noop_reactions(table) == {"noop"}
    pairs, report = typed_reaction_pairs(
        table, {"AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"}
    )
    observed = {(p.identity_a, p.identity_b, p.relation_type) for p in pairs}
    assert ("AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "reaction_forward") in observed
    assert ("BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC", "reaction_bidirectional") in observed
    assert report["identity_noop_reactions"] == 1


if __name__ == "__main__":
    test_noop_and_typed_direction()
    print("[test_bioaware_embedding_relation_manifest] PASS")
