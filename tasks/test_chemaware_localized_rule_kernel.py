"""CPU contracts for localized ChemAware shared-kernel algebra."""
from __future__ import annotations

import numpy as np
import torch

from chemaware_localized_rule_kernel_core import (
    LocalizedRuleGate,
    hardest_margin_loss,
    localized_pair_scores,
    localized_shared_embedding,
)


def main() -> None:
    rng = np.random.default_rng(7)
    official = rng.normal(size=(5, 4)).astype(np.float32)
    nl = rng.normal(size=(5, 3)).astype(np.float32)
    cf = rng.normal(size=(5, 2)).astype(np.float32)
    gates = rng.uniform(0.05, 0.95, size=(5, 2)).astype(np.float32)
    q = np.asarray([0, 0, 1, 1], dtype=np.int64)
    r = np.asarray([2, 3, 3, 4], dtype=np.int64)
    phi = localized_shared_embedding(
        official, nl, cf, gates, neutral_loss_cap=0.8, fragment_ion_cap=1.6,
    )
    direct = localized_pair_scores(
        torch.as_tensor(np.sum(official[q] * official[r], axis=1)),
        torch.as_tensor(np.sum(nl[q] * nl[r], axis=1)),
        torch.as_tensor(np.sum(cf[q] * cf[r], axis=1)),
        torch.as_tensor(gates), torch.as_tensor(q), torch.as_tensor(r),
        neutral_loss_cap=0.8, fragment_ion_cap=1.6,
    ).numpy()
    assert np.allclose(direct, np.sum(phi[q] * phi[r], axis=1), atol=1e-6)

    gate = LocalizedRuleGate(6, initial_gate=0.25)
    observed = gate(torch.zeros(3, 6)).detach().numpy()
    assert np.allclose(observed, 0.25, atol=1e-6)

    pair_score = torch.tensor([0.7, 0.3, 0.4, 0.6], requires_grad=True)
    loss, margins = hardest_margin_loss(
        pair_score,
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([True, False, True, False]),
        torch.ones(2), temperature=0.05, margin=0.02,
    )
    loss.backward()
    assert margins.tolist() == [0.3999999761581421, -0.20000001788139343]
    assert torch.isfinite(loss) and torch.isfinite(pair_score.grad).all()
    print("PASS: 3 localized ChemAware shared-kernel contracts")


if __name__ == "__main__":
    main()
