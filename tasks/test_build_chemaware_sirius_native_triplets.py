"""CPU contracts for the SIRIUS-guided native triplet rule."""
from __future__ import annotations

import math

from build_chemaware_sirius_native_triplets import source_relation, top_source_flags


def main() -> None:
    cross = source_relation(
        "C10H12O4", "C9H16O4", 8.0, 5.0, math.nan, math.nan, True, False,
    )
    assert cross == ("cross_formula_tree", 3.0)
    assert source_relation(
        "C10H12O4", "C9H16O4", 8.0, 5.0, math.nan, math.nan, False, False,
    ) is None
    within = source_relation(
        "C10H12O4", "C10H12O4", 8.0, 8.0, 4.5, 3.0, True, True,
    )
    assert within == ("within_formula_csi", 1.5)
    assert source_relation(
        "C10H12O4", "C10H12O4", 8.0, 8.0, 4.5, 3.0, True, False,
    ) is None
    scores = {
        0: {"formula": "C10H12O4", "tree": 8.0, "csi": 4.5},
        1: {"formula": "C9H16O4", "tree": 5.0, "csi": 8.0},
        2: {"formula": "C10H12O4", "tree": 8.0, "csi": 3.0},
    }
    assert top_source_flags(scores, 0) == (True, True)
    scores[1]["tree"] = 9.0
    assert top_source_flags(scores, 0) == (False, True)
    print("PASS: ChemAware SIRIUS native-triplet contracts", flush=True)


if __name__ == "__main__":
    main()
