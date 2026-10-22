from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from estimate_fbmn_response_object import collapse_units, estimate, estimate_paired, normalize


def test_weighted_multilevel_contrast_and_technical_replicates() -> None:
    # Two features x six files.  Files 0/1 are duplicate injections of unit A1.
    raw = np.asarray([
        [10, 10, 12, 20, 22, 21],
        [20, 20, 18, 10, 9, 11],
    ], dtype=float)
    normalized, _ = normalize(raw)
    design = pd.DataFrame({
        "biological_unit_id": ["A1", "A1", "B1", "C1", "C2", "C3"],
        "level": ["A", "A", "B", "C", "C", "C"],
    })
    units, unit_table = collapse_units(normalized, design)
    assert units.shape[1] == 5
    result = estimate(
        units, unit_table.level.to_numpy(str), {"C": 1.0, "A": -0.5, "B": -0.5},
        minimum_units=1, minimum_prevalence=0.5,
    )
    assert result.beta.iloc[0] > 0
    assert result.beta.iloc[1] < 0


def test_biological_unit_crossing_levels_fails() -> None:
    values = np.ones((2, 2), dtype=float)
    design = pd.DataFrame({
        "biological_unit_id": ["same", "same"], "level": ["A", "B"],
    })
    with pytest.raises(ValueError, match="cross contrast levels"):
        collapse_units(values, design)


def test_paired_estimate_uses_within_pair_differences() -> None:
    values = np.asarray([
        [2.0, 1.0, 4.0, 1.0, 8.0, 2.0],
        [1.0, 2.0, 1.0, 4.0, 2.0, 8.0],
    ])
    units = pd.DataFrame({
        "biological_unit_id": ["p1_L", "p1_N", "p2_L", "p2_N", "p3_L", "p3_N"],
        "level": ["lesion", "nonlesion"] * 3,
        "pair_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
    })
    result = estimate_paired(
        values, units, {"lesion": 1.0, "nonlesion": -1.0},
        minimum_pairs=3, minimum_prevalence=1.0,
    )
    assert result.beta.iloc[0] > 0
    assert result.beta.iloc[1] < 0
    assert result.n_measured_pairs.tolist() == [3, 3]
    assert result.qc_pass.all()
