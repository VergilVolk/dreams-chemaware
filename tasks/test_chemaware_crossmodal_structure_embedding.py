"""Unit tests for the cross-modal structural product-space audit."""
from __future__ import annotations

import numpy as np

from audit_chemaware_crossmodal_structure_embedding import fit_ridge, normalize, predict_ridge


def main() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(200, 24)).astype(np.float32)
    truth = rng.normal(size=(24, 12)).astype(np.float32)
    y = normalize(x @ truth + 0.01 * rng.normal(size=(200, 12)))
    formula = np.asarray([f"F{i // 2}" for i in range(len(x))])
    model = fit_ridge(x[:160], y[:160], formula[:160], alpha=0.01)
    predicted = predict_ridge(x[160:], model)
    cosine = np.sum(predicted * y[160:], axis=1)
    assert float(np.mean(cosine)) > 0.95
    assert np.allclose(np.linalg.norm(predicted, axis=1), 1.0, atol=1e-5)

    # Product-space similarity is exactly the dot product of the concatenated
    # normalized embedding, not a query-only reranking trick.
    z = normalize(rng.normal(size=(2, 8)))
    c = normalize(rng.normal(size=(2, 4)))
    beta = 0.2
    shared = np.concatenate([z, np.sqrt(beta) * c], axis=1) / np.sqrt(1.0 + beta)
    expected = (float(z[0] @ z[1]) + beta * float(c[0] @ c[1])) / (1.0 + beta)
    assert abs(float(shared[0] @ shared[1]) - expected) < 1e-6
    print("crossmodal structure embedding tests passed")


if __name__ == "__main__":
    main()
