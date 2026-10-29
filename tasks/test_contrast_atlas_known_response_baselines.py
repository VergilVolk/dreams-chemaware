from __future__ import annotations

import math

import pandas as pd

from run_contrast_atlas_known_response_baselines import pair_metrics


def _response(keys: list[str], values: list[float], z: list[float] | None = None) -> pd.DataFrame:
    if z is None:
        z = values
    return pd.DataFrame({
        "library_inchikey": keys,
        "beta": values,
        "se": [1.0] * len(keys),
        "z": z,
        "shrunk_response": values,
    })


def test_continuous_response_is_directional() -> None:
    keys = ["A", "B", "C"]
    query = _response(keys, [1.0, -0.5, 0.25])
    same = pair_metrics(query, query, 3, 1.96, 2)
    opposite = _response(keys, [-1.0, 0.5, -0.25])
    reverse = pair_metrics(query, opposite, 3, 1.96, 2)
    assert math.isclose(same["continuous_similar_score"], 1.0)
    assert math.isclose(reverse["continuous_similar_score"], -1.0)
    assert math.isclose(reverse["continuous_opposite_score"], 1.0)


def test_discrete_same_and_opposite_responses_separate() -> None:
    keys = ["A", "B", "C", "D"]
    query = _response(keys, [1, 1, -1, -1], [3, 3, -3, -3])
    same = _response(keys, [1, 1, -1, -1], [4, 4, -4, -4])
    opposite = _response(keys, [-1, -1, 1, 1], [-4, -4, 4, 4])
    same_result = pair_metrics(query, same, 3, 1.96, 2)
    opposite_result = pair_metrics(query, opposite, 3, 1.96, 2)
    assert same_result["idmet_signed_log_odds"] > 0
    assert opposite_result["idmet_signed_log_odds"] < 0
    assert same_result["ora_similar_score"] > same_result["ora_opposite_score"]
    assert opposite_result["ora_opposite_score"] > opposite_result["ora_similar_score"]


def test_zero_overlap_is_retained_as_unavailable() -> None:
    query = _response(["A", "B", "C"], [1, 1, 1], [3, 3, 3])
    target = _response(["D", "E", "F"], [1, 1, 1], [3, 3, 3])
    result = pair_metrics(query, target, 3, 1.96, 2)
    assert result["shared_known_identity_count"] == 0
    assert not result["continuous_available"]
    assert result["continuous_unavailability_reason"] == "no_shared_known_identity"
