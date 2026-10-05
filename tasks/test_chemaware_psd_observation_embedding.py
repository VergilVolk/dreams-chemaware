"""Mathematical contracts for the PSD ChemAware product-space embedding."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from audit_chemaware_psd_observation_embedding import fit_metric, transform


def main() -> None:
    value = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    weight = torch.tensor([0.25, 1.75])
    embedded = transform(value, weight)
    assert torch.allclose(torch.linalg.vector_norm(embedded, dim=1), torch.ones(2))
    gram = embedded @ embedded.T
    assert float(torch.linalg.eigvalsh(gram).min()) >= -1e-6
    rng = np.random.default_rng(8); count = 300
    query = rng.normal(size=(count, 4)).astype(np.float32)
    positive = query + 0.1 * rng.normal(size=(count, 4)).astype(np.float32)
    negative = rng.normal(size=(count, 4)).astype(np.float32)
    # Only channel zero is made systematically useful.
    positive[:, 1:] = rng.normal(size=(count, 3)); negative[:, 1:] = rng.normal(size=(count, 3))
    args = SimpleNamespace(learning_rate=0.05, steps=200, beta=0.2, temperature=0.1,
                           uniform_regularization=0.001)
    learned, _ = fit_metric(query, positive, negative, np.zeros(count, np.float32),
                            np.asarray([f"F{i // 2}" for i in range(count)]), args)
    assert learned[0] > float(np.mean(learned[1:])), learned
    assert np.isclose(learned.mean(), 1.0, atol=1e-5)
    print("PSD normalization, mean-dose and informative-channel contracts passed")


if __name__ == "__main__":
    main()
