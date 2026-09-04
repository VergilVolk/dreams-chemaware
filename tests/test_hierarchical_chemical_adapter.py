import pytest
import torch
import torch.nn.functional as F

from dreams.models.chem_aware.hierarchical_chemical_adapter import (
    HierarchicalChemicalResidualAdapter,
    deployable_parameter_count,
)


def _batch(batch: int = 3, peaks: int = 7):
    generator = torch.Generator().manual_seed(23)
    official = F.normalize(torch.randn(batch, 1024, generator=generator), dim=-1)
    precursor = torch.linspace(420.0, 620.0, batch)
    mz = torch.rand(batch, peaks, generator=generator) * (precursor[:, None] - 10.0)
    intensity = torch.rand(batch, peaks, generator=generator)
    mask = torch.ones(batch, peaks, dtype=torch.bool)
    mask[0, -2:] = False
    return official, mz, intensity, precursor, mask


def test_zero_initialization_exactly_reproduces_official_embedding():
    model = HierarchicalChemicalResidualAdapter(dropout=0.0).eval()
    official, mz, intensity, precursor, mask = _batch()

    output = model(official, mz, intensity, precursor, mask)

    assert output.embedding.shape == official.shape
    assert output.formula_logits.shape == (*mz.shape, 36)
    assert torch.count_nonzero(output.delta) == 0
    assert torch.equal(output.embedding, official)
    assert deployable_parameter_count(model) == 259_270
    assert sum(parameter.numel() for parameter in model.parameters()) == 262_762


def test_padded_measurements_cannot_change_valid_peak_states():
    model = HierarchicalChemicalResidualAdapter(dropout=0.0).eval()
    official, mz, intensity, precursor, mask = _batch()
    changed_mz = mz.clone()
    changed_intensity = intensity.clone()
    changed_mz[~mask] = 1_000_000.0
    changed_intensity[~mask] = 1_000_000.0

    baseline = model(official, mz, intensity, precursor, mask)
    changed = model(official, changed_mz, changed_intensity, precursor, mask)

    assert torch.equal(baseline.peak_states[mask], changed.peak_states[mask])
    assert torch.equal(baseline.formula_logits[mask], changed.formula_logits[mask])


def test_formula_pretraining_reaches_chemical_encoder_but_not_official_embedding():
    model = HierarchicalChemicalResidualAdapter(dropout=0.0)
    official, mz, intensity, precursor, mask = _batch()
    output = model(official, mz, intensity, precursor, mask)
    target = torch.randn_like(output.formula_logits)
    loss = F.mse_loss(output.formula_logits[mask], target[mask])
    loss.backward()

    assert model.formula_head.weight.grad is not None
    assert torch.count_nonzero(model.formula_head.weight.grad) > 0
    assert model.peak_input[1].weight.grad is not None
    assert torch.count_nonzero(model.peak_input[1].weight.grad) > 0
    assert official.grad is None


def test_retrieval_loss_first_opens_residual_then_reaches_chemical_encoder():
    model = HierarchicalChemicalResidualAdapter(dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    official, mz, intensity, precursor, mask = _batch()
    target = F.normalize(torch.roll(official, shifts=1, dims=0), dim=-1)

    first = model(official, mz, intensity, precursor, mask)
    (1.0 - (first.embedding * target).sum(dim=-1)).mean().backward()
    residual_weight = model.residual_head[-1].weight
    assert residual_weight.grad is not None
    assert torch.count_nonzero(residual_weight.grad) > 0
    # Exact zero initialization deliberately blocks encoder gradients on step 1.
    encoder_input = model.peak_input[1].weight
    assert encoder_input.grad is None or torch.count_nonzero(encoder_input.grad) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second = model(official, mz, intensity, precursor, mask)
    (1.0 - (second.embedding * target).sum(dim=-1)).mean().backward()
    assert encoder_input.grad is not None
    assert torch.count_nonzero(encoder_input.grad) > 0


def test_forward_api_rejects_candidate_conditioning():
    model = HierarchicalChemicalResidualAdapter()
    official, mz, intensity, precursor, mask = _batch()

    with pytest.raises(TypeError):
        model(
            official,
            mz,
            intensity,
            precursor,
            mask,
            candidate_structure="CCO",
        )


def test_formula_moment_mode_is_zero_init_and_deploys_formula_head():
    model = HierarchicalChemicalResidualAdapter(
        dropout=0.0, use_formula_moments=True,
    ).eval()
    official, mz, intensity, precursor, mask = _batch()

    output = model(official, mz, intensity, precursor, mask)

    assert torch.equal(output.embedding, official)
    assert model.residual_head[-1].in_features == 96 + 4 * 36
    assert deployable_parameter_count(model) == sum(
        parameter.numel() for parameter in model.parameters()
    )
