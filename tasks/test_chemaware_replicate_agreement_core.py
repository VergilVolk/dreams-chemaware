"""CPU contracts for ChemAware replicate-agreement maps."""
from __future__ import annotations

import numpy as np

from chemaware_replicate_agreement_core import (
    agreement_transform,
    apply_agreement_transform,
    fit_formula_conditional_contrast_basis,
    fit_replicate_agreement_basis,
)


def paired_cosine(value: np.ndarray) -> float:
    return float(np.mean(np.sum(value[0::3] * value[1::3], axis=1)))


def main() -> None:
    rng = np.random.default_rng(91)
    identities = np.repeat(np.arange(400), 3)
    latent = rng.normal(size=(400, 3))
    feature = np.zeros((len(identities), 12), dtype=np.float64)
    feature[:, :3] = np.repeat(latent, 3, axis=0) + 0.25 * rng.normal(size=(len(feature), 3))
    feature[:, 3:] = 2.0 * rng.normal(size=(len(feature), 9))
    feature /= np.linalg.norm(feature, axis=1, keepdims=True)
    basis, report = fit_replicate_agreement_basis(feature, identities, min_replicates=3)
    assert report["eligible_identities"] == 400
    assert report["ordered_self_pairs_excluded"]
    transform = agreement_transform(basis, floor=0.05, power=1.0)
    mapped = apply_agreement_transform(feature, transform)
    assert paired_cosine(mapped) > paired_cosine(feature) + 0.20
    constant = agreement_transform(basis, floor=1.0, power=2.0)
    preserved = apply_agreement_transform(feature, constant)
    assert np.allclose(feature @ feature.T, preserved @ preserved.T, atol=2e-5)
    assert np.linalg.eigvalsh(transform @ transform.T).min() > -1e-5
    formulas = np.repeat(np.arange(200), 6)
    isomer = np.tile(np.repeat(np.arange(2), 3), 200)
    conditional_identity = np.asarray([f"{formula}:{kind}" for formula, kind in zip(formulas, isomer)])
    common = rng.normal(size=(200, 12))
    differentiator = rng.normal(size=(400, 2))
    conditional = 2.0 * np.repeat(common, 6, axis=0)
    identity_number = np.repeat(np.arange(400), 3)
    conditional[:, :2] += 2.0 * differentiator[identity_number]
    conditional += rng.normal(size=conditional.shape)
    conditional /= np.linalg.norm(conditional, axis=1, keepdims=True)
    contrast_basis, contrast_report = fit_formula_conditional_contrast_basis(
        conditional, conditional_identity, formulas, min_replicates=3,
    )
    reverse_basis, reverse_report = fit_formula_conditional_contrast_basis(
        conditional, conditional_identity, formulas, min_replicates=3, reverse=True,
    )
    assert contrast_report["eligible_multi_identity_formulas"] == 200
    assert reverse_report["contrast_reversed"]
    contrast_map = agreement_transform(contrast_basis, floor=0.05, power=1.0)
    mapped_contrast = apply_agreement_transform(conditional, contrast_map)
    positive_similarity = np.sum(mapped_contrast[0::6] * mapped_contrast[1::6], axis=1)
    negative_similarity = np.sum(mapped_contrast[0::6] * mapped_contrast[3::6], axis=1)
    assert float(np.mean(positive_similarity - negative_similarity)) > 0.10
    assert not np.allclose(
        contrast_basis["normalized_agreement"], reverse_basis["normalized_agreement"],
    )
    print("PASS: 14 ChemAware replicate-agreement contracts")


if __name__ == "__main__":
    main()
