#!/usr/bin/env python
from __future__ import annotations

import numpy as np

from audit_bioaware_b9_reaction_spectral_specificity import (
    ACTION_CELLS,
    control_distance,
    formula_descriptor,
    reaction_spectral_views,
    select_controls,
)


def tensor(precursor: float, peaks: list[tuple[float, float]]) -> np.ndarray:
    rows = [[precursor, 1.0], *[list(item) for item in peaks]]
    while len(rows) < 8:
        rows.append([0.0, 0.0])
    return np.asarray(rows, dtype=np.float32)


def record(node: int, formula: str, precursor: float, degree: float = 1.0) -> dict:
    return {
        "node": node,
        "formula_descriptor": formula_descriptor(formula),
        "precursor": precursor,
        "peak_count": 3,
        "log_degree": degree,
    }


def main() -> None:
    # A high peak at 140 in the larger-precursor spectrum must be removed by
    # MetDNA3 truncation at precursor 120; the two remaining peaks match.
    query = tensor(150.0, [(50.0, 1.0), (80.0, 1.0), (140.0, 100.0)])
    seed = tensor(120.0, [(50.0, 1.0), (80.0, 1.0), (110.0, 1.0)])
    views = reaction_spectral_views(query, seed)
    assert views["truncated_direct"] > 0.60
    assert views["matched_fragments"] == 2.0
    assert 0.0 <= views["dual_view"] <= 1.0
    assert 0.0 <= views["kgmn_support"] <= 1.0

    a = formula_descriptor("C6H12O6")
    b = formula_descriptor("C6H12O6")
    c = formula_descriptor("C20H40Cl2")
    assert a is not None and b is not None and c is not None
    tier_same, cost_same = control_distance(
        {**a, "precursor": 181.0, "peak_count": 10.0, "log_degree": 1.0},
        {**b, "precursor": 181.1, "peak_count": 10.0, "log_degree": 1.0},
    )
    tier_far, cost_far = control_distance(
        {**a, "precursor": 181.0, "peak_count": 10.0, "log_degree": 1.0},
        {**c, "precursor": 400.0, "peak_count": 30.0, "log_degree": 3.0},
    )
    assert tier_same == 0 and tier_far == 2 and cost_same < cost_far

    target = [record(1, "C6H12O6", 181.0)]
    profiles = {}
    for identity, item in {
        "A": record(2, "C6H12O6", 181.1),
        "B": record(3, "C6H12O6", 181.2),
        "C": record(4, "C6H12O6", 180.9),
        "D": record(5, "C20H40Cl2", 400.0, 3.0),
    }.items():
        from audit_bioaware_b9_reaction_spectral_specificity import identity_profile
        profiles[identity] = ([item], identity_profile([item]))
    selected = select_controls(target, profiles, {"D"}, count=3)
    assert selected is not None
    controls, tier = selected
    assert [identity for identity, _ in controls] == ["A", "C", "B"]
    assert tier == 0
    assert len(ACTION_CELLS) == 6
    print("[BioAware B9 reaction-spectral specificity unit checks] PASS")


if __name__ == "__main__":
    main()
