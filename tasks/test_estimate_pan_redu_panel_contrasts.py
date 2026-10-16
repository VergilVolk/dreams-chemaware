from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from estimate_pan_redu_panel_contrasts import aggregate_units, estimate_effects, normalize_positive_log2


def test_repeated_files_are_aggregated_before_effect_estimation() -> None:
    raw = np.asarray([
        [10.0, 14.0, 8.0, 32.0, 40.0],
        [5.0, 7.0, 4.0, 16.0, 20.0],
    ])
    normalized, offsets = normalize_positive_log2(raw)
    assert np.isfinite(offsets).all()
    membership = pd.DataFrame({
        "sample_id": ["s1", "s2", "s3", "s4", "s5"],
        "biological_unit_id": ["u1", "u1", "u2", "u3", "u4"],
        "level": ["a", "a", "a", "b", "b"],
    })
    units, ids, levels = aggregate_units(normalized, np.asarray(["s1", "s2", "s3", "s4", "s5"]), membership)
    assert units.shape == (2, 4)
    assert ids.tolist() == ["u1", "u2", "u3", "u4"]
    effects = estimate_effects(units, levels, "a", "b", 2, 0.5)
    assert len(effects) == 2


def test_cross_level_unit_fails_closed() -> None:
    membership = pd.DataFrame({
        "sample_id": ["s1", "s2"],
        "biological_unit_id": ["u1", "u1"],
        "level": ["a", "b"],
    })
    with pytest.raises(ValueError, match="cross contrast levels"):
        aggregate_units(np.ones((2, 2)), np.asarray(["s1", "s2"]), membership)
