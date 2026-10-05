from __future__ import annotations

import torch

from dreams.models.chem_aware.global_embedding_adapter import (
    RuleConditionedResidualEmbeddingAdapter,
)


def main() -> None:
    torch.manual_seed(7)
    model = RuleConditionedResidualEmbeddingAdapter(8, 5, 3, 0.25)
    value = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    rule = torch.randn(4, 5)
    assert torch.allclose(model(value, rule), value, atol=1e-7)
    loss = -model(value, rule)[:, 0].mean()
    loss.backward()
    assert model.up.weight.grad is not None and torch.any(model.up.weight.grad != 0)
    assert model.rule_down.weight.grad is not None
    # The zero output projection intentionally delays route learning until the
    # first update, as in standard zero-init residual adapters.
    assert torch.all(model.rule_down.weight.grad == 0)
    with torch.no_grad():
        model.up.weight.add_(0.01)
    changed = model(value, rule)
    assert not torch.equal(changed, value)
    assert torch.allclose(torch.linalg.vector_norm(changed, dim=1), torch.ones(4), atol=1e-6)
    print("PASS: ChemAware rule-conditioned shared-adapter contracts")


if __name__ == "__main__":
    main()
