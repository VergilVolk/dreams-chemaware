"""CPU contracts for shrinkage-whitened shared chemical features."""
from __future__ import annotations

import numpy as np

from chemaware_shrinkage_whitening_core import (
    apply_whitener,
    fit_balanced_within_whitener,
    fit_shrinkage_whitener,
)


def main() -> None:
    rng = np.random.default_rng(31)
    latent = rng.normal(size=(600, 4))
    mix = rng.normal(size=(4, 7))
    value = (latent @ mix + 0.1 * rng.normal(size=(600, 7))).astype(np.float32)
    mean, transform, report = fit_shrinkage_whitener(value, 0.05)
    whitened = apply_whitener(value, mean, transform, normalize=False)
    covariance = np.cov(whitened, rowvar=False)
    assert np.isfinite(whitened).all() and report["shrunk_condition_number"] > 0
    assert np.diag(covariance).min() > 0.1
    normalized = apply_whitener(value[:20], mean, transform)
    assert np.max(np.abs(np.linalg.norm(normalized, axis=1) - 1.0)) < 1e-5
    gram = normalized @ normalized.T
    assert np.linalg.eigvalsh(gram).min() > -1e-5
    identity = np.repeat(np.arange(100), 6)
    within_value = (
        np.repeat(rng.normal(size=(100, 7)), 6, axis=0)
        + 0.2 * rng.normal(size=(600, 7))
    ).astype(np.float32)
    within_mean, within_transform, within_report = fit_balanced_within_whitener(
        within_value, identity, 0.1,
    )
    within = apply_whitener(within_value, within_mean, within_transform)
    assert within_report["replicated_identities"] == 100
    assert np.isfinite(within).all()
    print("PASS: 6 ChemAware shrinkage-whitening contracts")


if __name__ == "__main__":
    main()
