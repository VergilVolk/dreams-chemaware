"""CPU contracts for replicate-calibrated ChemAware reliability."""
from __future__ import annotations

import numpy as np

from chemaware_replicate_reliability_core import (
    empirical_gate_amplitude,
    fit_reliability_ridge,
    identity_balanced_sample_weight,
    leave_one_out_identity_consistency,
    predict_reliability,
)


def main() -> None:
    rng = np.random.default_rng(83)
    identities = np.repeat(np.arange(200), 5)
    quality = rng.uniform(size=len(identities))
    identity_centroid = rng.normal(size=(200, 32))
    identity_centroid /= np.linalg.norm(identity_centroid, axis=1, keepdims=True)
    centroid = 4.0 * np.repeat(identity_centroid, 5, axis=0)
    feature = centroid + (1.5 - quality)[:, None] * rng.normal(size=centroid.shape)
    index, target, report = leave_one_out_identity_consistency(feature, identities)
    assert len(index) == len(feature) and report["replicated_identities"] == 200
    observable = np.column_stack((quality, quality ** 2, rng.normal(size=(len(quality), 2))))
    model, model_report = fit_reliability_ridge(observable[index], target)
    prediction = predict_reliability(observable, model)
    assert np.corrcoef(prediction[index], target)[0, 1] > 0.25
    amplitude = empirical_gate_amplitude(prediction, prediction[index], 0.25, 1.0)
    reversed_amplitude = empirical_gate_amplitude(
        prediction, prediction[index], 0.25, 1.0, reversed_order=True,
    )
    assert np.all((amplitude >= 0.5) & (amplitude <= 1.0))
    assert np.corrcoef(amplitude, reversed_amplitude)[0, 1] < -0.9
    assert model_report["training_correlation"] > 0.25
    assert model_report["weighted_training_correlation"] > 0.25
    mixed_identity = np.asarray(["a", "a", "a", "b", "b", "c"])
    weight, weight_report = identity_balanced_sample_weight(mixed_identity)
    totals = [weight[mixed_identity == value].sum() for value in np.unique(mixed_identity)]
    assert np.allclose(totals, totals[0])
    assert weight_report["identities"] == 3
    weighted_model, weighted_report = fit_reliability_ridge(
        observable[index], target, sample_weight=np.ones(len(index)),
    )
    assert np.allclose(predict_reliability(observable, weighted_model), prediction)
    assert weighted_report["weighted_standardization"]
    small_identity = np.asarray(["a", "a", "b", "b", "b"])
    small_feature = rng.normal(size=(5, 4))
    strict_index, _strict_target, strict_report = leave_one_out_identity_consistency(
        small_feature, small_identity, min_replicates=3,
    )
    assert np.array_equal(strict_index, np.asarray([2, 3, 4]))
    assert strict_report["minimum_replicates"] == 3
    gram_feature = amplitude[:, None] * feature
    assert np.linalg.eigvalsh(gram_feature[:50] @ gram_feature[:50].T).min() > -1e-4
    print("PASS: 18 ChemAware replicate-reliability contracts")


if __name__ == "__main__":
    main()
