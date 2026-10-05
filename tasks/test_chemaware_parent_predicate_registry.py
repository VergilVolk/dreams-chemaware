"""Contracts for the parent-predicate hypothesis vocabulary."""
from __future__ import annotations

from rdkit import Chem

from build_chemaware_parent_predicate_registry import rdkit_fragment_patterns


def main() -> None:
    patterns = rdkit_fragment_patterns()
    assert len(patterns) >= 80
    assert len({item["predicate_id"] for item in patterns}) == len(patterns)
    assert all(Chem.MolFromSmarts(item["smarts_any"][0]) is not None for item in patterns)
    assert all(item["may_define_fragmentation_rule"] is False for item in patterns)
    print("parent predicate registry contracts passed")


if __name__ == "__main__":
    main()
