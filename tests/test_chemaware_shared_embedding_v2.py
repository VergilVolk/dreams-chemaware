from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import pytest

from dreams.models.chem_aware.shared_embedding_v2 import (
    ChemAwareSharedEncoder,
    SignedPeakResidualAdapter,
    chemical_weighted_listwise_loss,
    identity_equal_weights,
    masked_multilabel_loss,
    molecule_listwise_loss,
    molecule_scores_from_spectrum_pairs,
    positive_reference_increment_loss,
    protected_margin_loss,
)


class TinyBackbone(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.lift = nn.Linear(2, dimension, bias=False)

    def forward(self, spectra, charge=None):
        return self.lift(spectra)


class TinyOfficial(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.backbone = TinyBackbone(dimension)
        self.head = nn.Linear(dimension, dimension)


def spectra() -> torch.Tensor:
    return torch.tensor([
        [[300.0, 1.1], [80.0, 0.4], [120.0, 1.0], [0.0, 0.0]],
        [[301.0, 1.1], [81.0, 0.3], [180.0, 0.8], [220.0, 1.0]],
    ])


def test_zero_init_exactly_reproduces_official_shared_embedding():
    torch.manual_seed(7)
    official = TinyOfficial(16)
    adapter = SignedPeakResidualAdapter(16, hidden_dim=24, delta_bound=0.1)
    model = ChemAwareSharedEncoder(official, adapter)
    output = model(spectra())

    assert torch.equal(output.embedding, output.official_embedding)
    assert torch.count_nonzero(output.delta) == 0
    assert torch.allclose(output.embedding.norm(dim=1), torch.ones(2))
    assert torch.allclose(output.support_weights.sum(dim=1), torch.ones(2))
    assert torch.allclose(output.conflict_weights.sum(dim=1), torch.ones(2))
    assert output.support_weights[0, -1] == 0
    assert output.conflict_weights[0, -1] == 0


def test_contextual_peak_gate_is_candidate_independent_and_zero_init_exact():
    torch.manual_seed(71)
    official = TinyOfficial(16)
    adapter = SignedPeakResidualAdapter(
        16, hidden_dim=24, delta_bound=0.1, contextual_gate=True
    )
    model = ChemAwareSharedEncoder(official, adapter)
    output = model(spectra())
    assert torch.equal(output.embedding, output.official_embedding)
    assert torch.count_nonzero(output.delta) == 0
    assert torch.allclose(output.support_weights.sum(dim=1), torch.ones(2))
    assert torch.allclose(output.conflict_weights.sum(dim=1), torch.ones(2))
    assert not torch.allclose(output.support_weights, output.conflict_weights)


def test_global_residual_branch_remains_zero_init_exact():
    torch.manual_seed(72)
    official = TinyOfficial(16)
    adapter = SignedPeakResidualAdapter(
        16, hidden_dim=24, delta_bound=0.1, global_branch=True
    )
    output = ChemAwareSharedEncoder(official, adapter)(spectra())
    assert torch.equal(output.embedding, output.official_embedding)
    assert torch.count_nonzero(output.delta) == 0
    target = F.normalize(torch.randn_like(output.embedding), dim=1)
    (1.0 - torch.sum(output.embedding * target, dim=1).mean()).backward()
    assert torch.count_nonzero(adapter.output[-1].weight.grad) > 0


def test_zero_init_exact_forward_keeps_adapter_gradient_alive():
    torch.manual_seed(8)
    adapter = SignedPeakResidualAdapter(8, hidden_dim=16, delta_bound=0.1)
    official = F.normalize(torch.randn(2, 8), dim=1)
    tokens = torch.randn(2, 3, 8)
    mz = torch.tensor([[50.0, 90.0, 120.0], [51.0, 91.0, 121.0]])
    intensity = torch.tensor([[0.2, 1.0, 0.4], [0.3, 0.8, 0.5]])
    adapted, _, _, _ = adapter(
        official, tokens, mz, intensity, torch.tensor([250.0, 260.0]), mz > 0
    )
    assert torch.equal(adapted, official)
    target = F.normalize(torch.randn(2, 8), dim=1)
    loss = 1.0 - torch.sum(adapted * target, dim=1).mean()
    loss.backward()
    assert adapter.output[-1].weight.grad is not None
    assert torch.count_nonzero(adapter.output[-1].weight.grad) > 0


def test_adapter_can_move_embedding_and_keeps_residual_bounded():
    torch.manual_seed(9)
    adapter = SignedPeakResidualAdapter(8, hidden_dim=16, delta_bound=0.05)
    nn.init.normal_(adapter.output[-1].weight, std=0.02)
    official = F.normalize(torch.randn(3, 8), dim=1)
    tokens = torch.randn(3, 4, 8)
    mz = torch.tensor([[50.0, 70.0, 90.0, 0.0]]).repeat(3, 1)
    intensity = torch.tensor([[0.2, 0.5, 1.0, 0.0]]).repeat(3, 1)
    mask = mz > 0
    adapted, delta, _, _ = adapter(
        official, tokens, mz, intensity, torch.tensor([200.0, 250.0, 300.0]), mask
    )
    assert torch.any(torch.abs(adapted - official) > 1e-7)
    assert torch.all(delta.norm(dim=1) < 0.05)
    assert torch.allclose(adapted.norm(dim=1), torch.ones(3), atol=1e-6)


def test_molecule_max_and_full_listwise_objective_have_retrieval_gradients():
    query = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
    candidate = F.normalize(torch.tensor([
        [0.6, 0.8], [0.95, 0.05],  # query 0 positive molecule: max second spectrum
        [0.2, 0.98],               # query 0 negative molecule
        [0.1, 0.99],               # query 1 positive molecule
        [0.8, 0.2], [0.7, 0.3],    # query 1 negative molecule
    ]), dim=1).requires_grad_()
    pair_query = torch.tensor([0, 0, 0, 1, 1, 1])
    molecule_ptr = torch.tensor([0, 2, 3, 4, 6])
    query_ptr = torch.tensor([0, 2, 4])
    scores = molecule_scores_from_spectrum_pairs(
        query, candidate, pair_query, molecule_ptr
    )
    assert scores.shape == (4,)
    assert torch.allclose(scores[0], torch.sum(query[0] * candidate[1]))
    loss = molecule_listwise_loss(scores, query_ptr, temperature=0.2)
    loss.backward()
    assert torch.isfinite(loss)
    assert candidate.grad is not None
    assert torch.any(candidate.grad != 0)


def test_positive_reference_increment_is_zero_when_every_view_meets_target():
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    official_query = query.clone()
    official_references = torch.tensor([
        [0.80, 0.60], [0.60, 0.80], [0.60, 0.80],
    ])
    new_references = torch.tensor([
        [0.90, (1.0 - 0.90**2) ** 0.5],
        [0.70, (1.0 - 0.70**2) ** 0.5],
        [(1.0 - 0.90**2) ** 0.5, 0.90],
    ], requires_grad=True)
    loss = positive_reference_increment_loss(
        query, new_references, torch.tensor([0, 2, 3]),
        official_query, official_references, increment=0.05,
    )
    assert torch.equal(loss, torch.tensor(0.0))


def test_positive_reference_increment_penalizes_unimproved_views_and_worst():
    query = torch.tensor([[1.0, 0.0]])
    official_references = torch.tensor([[0.70, 0.7141428], [0.70, -0.7141428]])
    new_references = torch.tensor(
        [[0.78, 0.6257795], [0.72, -0.6939740]], requires_grad=True
    )
    ptr = torch.tensor([0, 2])
    mean_loss = positive_reference_increment_loss(
        query, new_references, ptr, query, official_references,
        increment=0.10, aggregation="mean",
    )
    worst_loss = positive_reference_increment_loss(
        query, new_references, ptr, query, official_references,
        increment=0.10, aggregation="worst",
    )
    assert torch.allclose(mean_loss, torch.tensor(0.05), atol=1e-6)
    assert torch.allclose(worst_loss, torch.tensor(0.08), atol=1e-6)
    assert worst_loss > mean_loss
    mean_loss.backward()
    assert new_references.grad is not None
    assert torch.count_nonzero(new_references.grad) > 0


def test_positive_reference_increment_rejects_invalid_contracts():
    query = torch.tensor([[1.0, 0.0]])
    references = torch.tensor([[0.8, 0.6]])
    ptr = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="non-negative"):
        positive_reference_increment_loss(
            query, references, ptr, query, references, increment=-0.01,
        )
    with pytest.raises(ValueError, match="aggregation"):
        positive_reference_increment_loss(
            query, references, ptr, query, references, aggregation="median",
        )


def test_chemical_hardness_reweights_deployable_candidate_scores_directly():
    scores = torch.tensor([0.7, 0.65, 0.55, 0.45], requires_grad=True)
    query_ptr = torch.tensor([0, 4])
    teacher = F.normalize(torch.tensor([
        [1.0, 0.0], [0.99, 0.01], [0.7, 0.7], [0.0, 1.0],
    ]), dim=1)
    weighted = chemical_weighted_listwise_loss(
        scores, teacher, query_ptr, temperature=0.2, hardness_beta=4.0
    )
    weighted.backward()
    assert scores.grad is not None
    # The chemically closest negative receives the strongest direct pressure.
    assert scores.grad[1] > scores.grad[2] > scores.grad[3]

    equal_teacher = F.normalize(torch.tensor([
        [1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5],
    ]), dim=1)
    clean = molecule_listwise_loss(scores.detach(), query_ptr, temperature=0.2)
    neutral = chemical_weighted_listwise_loss(
        scores.detach(), equal_teacher, query_ptr, temperature=0.2, hardness_beta=4.0
    )
    assert torch.allclose(clean, neutral, atol=1e-7)

    # The bounded absolute mode remains chemically informative when the
    # candidate group has only one negative (the common real-data case).
    binary_scores = torch.tensor([0.7, 0.65], requires_grad=True)
    binary_ptr = torch.tensor([0, 2])
    close_teacher = F.normalize(torch.tensor([[1.0, 0.0], [0.99, 0.01]]), dim=1)
    far_teacher = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
    close_loss = chemical_weighted_listwise_loss(
        binary_scores, close_teacher, binary_ptr, temperature=0.2,
        hardness_beta=4.0, weighting="absolute_bounded",
    )
    far_loss = chemical_weighted_listwise_loss(
        binary_scores, far_teacher, binary_ptr, temperature=0.2,
        hardness_beta=4.0, weighting="absolute_bounded",
    )
    assert close_loss > far_loss
    close_loss.backward()
    assert binary_scores.grad is not None
    assert torch.any(binary_scores.grad != 0)


def test_shared_adapter_can_overfit_and_flip_a_realistic_small_margin_group():
    torch.manual_seed(123)
    adapter = SignedPeakResidualAdapter(8, hidden_dim=24, delta_bound=0.12)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-3)

    def vector(cosine: float) -> torch.Tensor:
        return torch.tensor([cosine, (1.0 - cosine**2) ** 0.5, 0, 0, 0, 0, 0, 0])

    # Positive starts 0.03 below the hardest negative, a local-ordering error.
    official = F.normalize(torch.stack([
        vector(1.0), vector(0.78), vector(0.81),
        torch.tensor([0.2, 0.0, 0.98, 0, 0, 0, 0, 0]),
    ]), dim=1)
    tokens = torch.randn(4, 5, 8)
    mz = torch.arange(20, dtype=torch.float32).reshape(4, 5) + 50.0
    intensity = torch.tensor([[0.2, 0.4, 1.0, 0.6, 0.3]]).repeat(4, 1)
    precursor = torch.full((4,), 300.0)
    mask = torch.ones(4, 5, dtype=torch.bool)
    pair_query = torch.tensor([0, 0, 0])
    molecule_ptr = torch.tensor([0, 1, 2, 3])
    query_ptr = torch.tensor([0, 3])
    for _ in range(160):
        adapted, delta, _, _ = adapter(
            official, tokens, mz, intensity, precursor, mask
        )
        scores = molecule_scores_from_spectrum_pairs(
            adapted[:1], adapted[1:], pair_query, molecule_ptr
        )
        loss = molecule_listwise_loss(scores, query_ptr, temperature=0.07)
        loss = loss + 0.1 * torch.mean(1.0 - torch.sum(adapted * official, dim=1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert scores[0] > scores[1]
    assert torch.max(delta.norm(dim=1)) < 0.12
    assert torch.mean(torch.sum(adapted * official, dim=1)) > 0.99


def test_protection_and_observability_masks_are_fail_safe():
    query_ptr = torch.tensor([0, 3, 6])
    official = torch.tensor([0.9, 0.5, 0.4, 0.2, 0.7, 0.6])
    new = torch.tensor([0.7, 0.6, 0.5, 0.9, 0.8, 0.7], requires_grad=True)
    protection = protected_margin_loss(new, official, query_ptr)
    # Only the first group was official-correct and has lost margin.
    assert torch.allclose(protection, torch.tensor(0.3))

    logits = torch.tensor([[0.0, 2.0, -1.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    observable = torch.tensor([[True, False, True]])
    concept = masked_multilabel_loss(logits, targets, observable)
    concept.backward()
    assert logits.grad[0, 1] == 0
    assert logits.grad[0, 0] != 0 and logits.grad[0, 2] != 0


def test_identity_equal_weighting_is_invariant_to_query_multiplicity():
    weights = identity_equal_weights(["A", "A", "B"])
    assert torch.allclose(weights[:2].sum(), weights[2])
    assert torch.allclose(weights.sum(), torch.tensor(1.0))


def test_sparse_peak_gate_keeps_only_topk_valid_peaks_and_normalizes():
    logits = torch.tensor([[0.1, 0.9, 0.4, 2.0], [3.0, 2.0, 1.0, 0.0]])
    mask = torch.tensor([[True, True, True, False], [True, False, True, True]])
    weights = SignedPeakResidualAdapter._masked_weights(
        logits, mask, temperature=0.5, topk=2
    )
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))
    assert torch.all((weights > 0).sum(dim=1) == 2)
    assert torch.all(weights[~mask] == 0)
    assert weights[0, 1] > weights[0, 2]
