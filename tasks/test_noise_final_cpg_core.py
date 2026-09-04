"""CPU unit and architecture-contract tests for noise_final_cpg_core."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from noise_final_cpg_core import (
    ActionKey,
    ContextualPeakGate,
    PeakGatedSharedEncoder,
    candidate_margin_vector,
    clean_candidate_residual_loss,
    molecule_max_scores,
    no_replacement_hierarchical_sample,
    paired_candidate_residual,
    project_auxiliary_against_reference,
)


class TinyBackbone(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.input = nn.Linear(2, dimension, bias=False)

    def forward(self, spectra: torch.Tensor, charge=None) -> torch.Tensor:
        del charge
        return self.input(spectra)


def test_full_candidate_math() -> None:
    pair = torch.tensor([0.2, 0.7, 0.5, 0.1, 0.4])
    molecule = molecule_max_scores(pair, [0, 2, 3, 5])
    assert torch.allclose(molecule, torch.tensor([0.7, 0.5, 0.4]))
    query = torch.tensor([1.0, 0.0])
    candidates = torch.tensor([[0.2, 0.0], [0.7, 0.0], [0.5, 0.0], [0.1, 0.0], [0.4, 0.0]])
    margins = candidate_margin_vector(query, candidates, [0, 2, 3, 5])
    assert torch.allclose(margins, torch.tensor([0.2, 0.3]))


def test_vector_residual_does_not_collapse() -> None:
    candidates = torch.eye(3)
    target = torch.tensor([0.8, 0.1, 0.1])
    # Deliberately asymmetric controls.  The former symmetric fixture produced
    # [0.45, 0.45] exactly and therefore could not test candidate resolution.
    controls = torch.tensor([[0.5, 0.4, 0.1], [0.6, 0.2, 0.2]])
    residual, target_margin, control_margin = paired_candidate_residual(
        target, controls, candidates, [0, 1, 2, 3],
    )
    assert residual.shape == (2,)
    assert torch.allclose(target_margin, torch.tensor([0.7, 0.7]))
    assert torch.allclose(control_margin, torch.tensor([0.25, 0.40]))
    assert torch.allclose(residual, torch.tensor([0.45, 0.30]))
    assert torch.allclose(residual, target_margin - control_margin)


def test_clean_loss_updates_query_and_references() -> None:
    current_query = torch.tensor([0.7, 0.2, 0.1], requires_grad=True)
    current_candidates = torch.eye(3, requires_grad=True)
    initial_query = current_query.detach().clone()
    initial_candidates = current_candidates.detach().clone()
    teacher = torch.tensor([0.05, -0.03])
    loss, detail = clean_candidate_residual_loss(
        current_query, current_candidates, initial_query, initial_candidates,
        [0, 1, 2, 3], teacher,
    )
    loss.backward()
    assert current_query.grad is not None and float(current_query.grad.abs().sum()) > 0
    assert current_candidates.grad is not None and float(current_candidates.grad.abs().sum()) > 0
    assert detail["predicted_residual"].shape == teacher.shape


def test_peak_gate_exact_identity_and_gradient() -> None:
    torch.manual_seed(7)
    batch, tokens, dimension = 3, 6, 8
    sequence = torch.randn(batch, tokens, dimension)
    spectra = torch.zeros(batch, tokens, 2)
    spectra[:, 0] = torch.tensor([500.0, 1.0])
    spectra[:, 1:4, 0] = torch.tensor([100.0, 200.0, 300.0])
    spectra[:, 1:4, 1] = torch.tensor([1.0, 0.5, 0.25])
    gate = ContextualPeakGate(dimension, hidden_dimension=16, strength=0.25)
    precursor, weights = gate(sequence, spectra)
    assert torch.equal(precursor, sequence[:, 0, :])
    assert torch.equal(weights[:, 1:3], torch.ones_like(weights[:, 1:3]))
    assert torch.equal(weights[:, 3:], torch.zeros_like(weights[:, 3:]))
    loss = precursor.square().mean()
    loss.backward()
    final = gate.scorer[-1]
    assert final.weight.grad is not None and float(final.weight.grad.abs().sum()) > 0


def test_shared_encoder_reproduces_base_and_never_adds_tokens() -> None:
    torch.manual_seed(9)
    dimension = 4
    backbone = TinyBackbone(dimension)
    head = nn.Linear(dimension, dimension)
    model = PeakGatedSharedEncoder(backbone, head, dimension, gate_hidden_dimension=8)
    spectra = torch.tensor([[[500.0, 1.0], [100.0, 1.0], [200.0, 0.5], [0.0, 0.0]]])
    base = torch.nn.functional.normalize(head(backbone(spectra)[:, 0]), dim=-1)
    output, gate = model(spectra, return_gate=True)
    assert torch.equal(output, base)
    assert gate.shape == spectra[:, 1:, 0].shape
    assert gate[0, -1].item() == 0.0


def test_sampler_is_bounded_and_no_replacement() -> None:
    actions = [
        ActionKey("A", 1, "N", 1), ActionKey("A", 1, "P", 2),
        ActionKey("A", 2, "N", 3), ActionKey("B", 3, "P", 4),
    ]
    selected = no_replacement_hierarchical_sample(actions, np.random.default_rng(11))
    assert len(selected) == 2
    assert len({item.identity for item in selected}) == 2
    assert len({item.action for item in selected}) == len(selected)


def test_gradient_projection_only_for_conflict() -> None:
    reference = [torch.tensor([1.0, 0.0])]
    aligned, audit = project_auxiliary_against_reference([torch.tensor([2.0, 0.0])], reference)
    assert audit["projected"] == 0.0 and torch.equal(aligned[0], torch.tensor([2.0, 0.0]))
    conflict, audit = project_auxiliary_against_reference([torch.tensor([-2.0, 1.0])], reference)
    assert audit["projected"] == 1.0
    assert float(torch.dot(conflict[0], reference[0])) >= -1e-7


def test_static_contract() -> None:
    source = inspect.getsource(clean_candidate_residual_loss)
    assert "current_query" in source and "current_candidates" in source
    assert "teacher_residual" in source and "action_z" not in source
    contract = (Path(__file__).resolve().parents[1] / "docs/NOISE_FINAL_CPG_CONTRACT_20260903.md").read_text(
        encoding="utf-8",
    )
    for phrase in (
        "candidate-independent", "full molecule", "no replacement", "exactly the initialization",
        "same-initialization arms",
    ):
        assert phrase in contract


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_final_cpg_core] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
