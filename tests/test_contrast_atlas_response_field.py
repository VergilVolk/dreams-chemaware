from __future__ import annotations

import numpy as np

from tasks.audit_contrast_atlas_response_field import (
    centred_cosine,
    fit_query_field,
    permute_within_strata,
    signed_reliability,
)


def test_signed_reliability_and_centred_cosine() -> None:
    transformed = signed_reliability(
        np.array([1.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    assert np.allclose(transformed, [1.0, -1.0, 0.0])
    assert centred_cosine(np.array([1.0, 2.0, 3.0]), np.array([3.0, 6.0, 9.0])) > 0.999
    assert centred_cosine(np.array([1.0, 2.0, 3.0]), np.array([-3.0, -6.0, -9.0])) < -0.999


def test_stratified_reassignment_never_crosses_identity_worlds() -> None:
    values = np.arange(8.0)
    strata = np.array(["known"] * 3 + ["unresolved"] * 5)
    shuffled = permute_within_strata(values, strata, np.random.default_rng(7))
    assert set(shuffled[:3]) == {0.0, 1.0, 2.0}
    assert set(shuffled[3:]) == {3.0, 4.0, 5.0, 6.0, 7.0}


def test_query_only_identifiability_accepts_a_reproducible_response_field() -> None:
    rng = np.random.default_rng(20261022)
    x = rng.normal(size=(120, 8))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = 0.8 * x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.03, size=len(x))
    model, diagnostics = fit_query_field(
        x, y, study="synthetic", contrast="linear_response", world="all"
    )
    assert model is not None
    assert diagnostics["query_field_identifiable"]
    assert float(diagnostics["query_oof_r2"]) > 0.5
