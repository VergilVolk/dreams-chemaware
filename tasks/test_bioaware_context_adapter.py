#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from annotation.bioaware_context_adapter import (  # noqa: E402
    BiologicalContextAdapter,
    BiologicalEvidenceContextAdapter,
    context_training_loss,
)


def main() -> None:
    torch.manual_seed(7)
    batch, candidates, edges, dimension = 3, 4, 5, 16
    model = BiologicalContextAdapter(dimension, relation_types=6, hidden_dim=16, relation_dim=4)
    base = F.normalize(torch.randn(batch * candidates, dimension), dim=-1)
    seeds = F.normalize(torch.randn(batch * candidates, edges, dimension), dim=-1)
    relation = torch.randint(0, 6, (batch * candidates, edges))
    features = torch.rand(batch * candidates, edges, 4)
    mask = torch.rand(batch * candidates, edges) > 0.3
    mask[0] = False
    adapted, delta, gate = model(base, seeds, relation, features, mask)
    assert torch.equal(adapted, base), "zero-init must reproduce normalized universal embeddings exactly"
    assert torch.count_nonzero(delta) == 0
    assert gate[0] == 0

    # Exact fallback must not make the zero-initialized model untrainable.
    initial_loss = -(adapted[1] * F.normalize(torch.randn(dimension), dim=0)).sum()
    initial_loss.backward()
    assert model.output.weight.grad is not None
    assert float(model.output.weight.grad.abs().sum()) > 0
    model.zero_grad(set_to_none=True)

    # Edge order must not change the contextual result.
    with torch.no_grad():
        model.output.weight.normal_(0, 0.01)
    first, _, trained_gate = model(base, seeds, relation, features, mask)
    order = torch.tensor([4, 2, 0, 3, 1])
    second = model(base, seeds[:, order], relation[:, order], features[:, order], mask[:, order])[0]
    assert torch.allclose(first, second, atol=1e-6)

    # Identical biological messages on different candidate embeddings must not
    # collapse to one universal residual direction.
    shared_seeds = seeds[1:2].expand(2, -1, -1).clone()
    shared_relations = relation[1:2].expand(2, -1).clone()
    shared_features = features[1:2].expand(2, -1, -1).clone()
    shared_mask = mask[1:2].expand(2, -1).clone()
    _, candidate_deltas, _ = model(
        base[1:3], shared_seeds, shared_relations, shared_features, shared_mask,
    )
    assert not torch.allclose(candidate_deltas[0], candidate_deltas[1])

    contextual = first.reshape(batch, candidates, dimension)
    universal = base.reshape(batch, candidates, dimension)
    query = F.normalize(torch.randn(batch, dimension), dim=-1)
    output = context_training_loss(
        query, contextual, torch.zeros(batch, dtype=torch.long), universal,
        trained_gate.reshape(batch, candidates),
    )
    output["total"].backward()
    assert torch.isfinite(output["total"])

    # The reduced-evidence B2 adapter is also an embedding transform, not a
    # scalar residual.  Zero initialization and missing context must both be
    # exact fallbacks, while the zero-initialized output basis remains trainable.
    evidence_model = BiologicalEvidenceContextAdapter(
        dimension, evidence_dim=10, hidden_dim=16, update_rank=4,
    )
    evidence = torch.randn(batch * candidates, 10)
    context_mask = torch.ones(batch * candidates, dtype=torch.bool)
    context_mask[0] = False
    evidence_adapted, evidence_delta, evidence_gate = evidence_model(
        base, evidence, context_mask,
    )
    assert torch.equal(evidence_adapted, base)
    assert torch.count_nonzero(evidence_delta) == 0
    assert evidence_gate[0] == 0
    evidence_loss = -(evidence_adapted[1] * F.normalize(torch.randn(dimension), dim=0)).sum()
    evidence_loss.backward()
    assert evidence_model.output.weight.grad is not None
    assert float(evidence_model.output.weight.grad.abs().sum()) > 0
    with torch.no_grad():
        evidence_model.output.weight.normal_(0, .01)
    changed, _, _ = evidence_model(base, evidence, context_mask)
    assert torch.equal(changed[0], base[0])
    assert not torch.equal(changed[1], base[1])
    print("[test_bioaware_context_adapter] PASS")


if __name__ == "__main__":
    main()
