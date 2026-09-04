from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from dreams.models.chem_aware.peft_v3 import (
    DreaMSPEFTConfig,
    install_dreams_peft,
    load_peft_state_dict,
    peft_state_dict,
)
from dreams.models.chem_aware.frozen_probe_v3 import (
    FrozenChemicalProbe,
    FrozenProbeFit,
    fit_frozen_ridge_probe,
    targeted_probe_listwise_loss,
    targeted_probe_multiview_listwise_loss,
)
from dreams.models.chem_aware.shared_embedding_v2 import (
    chemical_margin_listwise_loss, chemical_weighted_listwise_loss,
    molecule_listwise_loss,
    molecule_scores_from_spectrum_pairs,
    targeted_chemical_margin_increment,
)
from tasks.train_chemaware_shared_v2 import gradient_geometry
from tasks.train_chemaware_shared_v3_peft import (
    model_selection_eligible,
    official_error_focus_weights,
)


class TinyAttention(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(4 * dimension, dimension) * 0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The real DreaMS attention also uses one fused matrix.  Using its final
        # quarter here exercises parametrization of a bare Parameter.
        dimension = x.shape[-1]
        return F.linear(x, self.weights[3 * dimension:])


def test_model_selection_rejects_single_spectrum_tail_drift():
    safe = {
        "preservation_mean": 0.998,
        "preservation_min": 0.951,
        "delta_recall1": 0.0,
        "delta_near_recall1": None,
    }
    assert model_selection_eligible(
        safe,
        minimum_preservation=0.995,
        minimum_single_spectrum_preservation=0.95,
        recall1_floor=-5e-4,
        near_recall1_floor=-1e-3,
    )

    unsafe_tail = dict(safe, preservation_min=0.949)
    assert not model_selection_eligible(
        unsafe_tail,
        minimum_preservation=0.995,
        minimum_single_spectrum_preservation=0.95,
        recall1_floor=-5e-4,
        near_recall1_floor=-1e-3,
    )


def test_official_error_curriculum_is_bounded_deterministic_and_nonchemical():
    graph = SimpleNamespace(
        query_ptr=np.asarray([0, 2, 4, 6], dtype=np.int64),
        molecule_ptr=np.arange(7, dtype=np.int64),
        features=np.asarray([[0.4], [0.5], [0.6], [0.5], [0.9], [0.4]], dtype=np.float32),
        dreams_column=0,
        query_ik14=np.asarray(["A", "A", "B"]),
    )
    queries = np.arange(3, dtype=np.int64)
    allowed = np.ones(6, dtype=bool)
    weights, audit = official_error_focus_weights(
        graph, queries, allowed, strength=3.0, temperature=0.02
    )
    repeated, repeated_audit = official_error_focus_weights(
        graph, queries, allowed, strength=3.0, temperature=0.02
    )

    assert weights == repeated
    assert audit["query_weight_ledger_sha256"] == repeated_audit[
        "query_weight_ledger_sha256"
    ]
    assert weights[0] > weights[2] > weights[1]
    assert 1.0 <= audit["focus_multiplier_min"]
    assert audit["focus_multiplier_max"] <= 4.0
    assert audit["official_wrong_queries"] == 1
    assert audit["uses_heldout_queries"] is False
    assert audit["uses_structure_teacher"] is False


def test_candidate_margin_is_clean_at_zero_and_emphasizes_close_decoy():
    scores = torch.tensor([0.60, 0.50], requires_grad=True)
    ptr = torch.tensor([0, 2])
    close = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    distant = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    clean = molecule_listwise_loss(scores, ptr, temperature=0.07)
    zero_margin = chemical_margin_listwise_loss(
        scores, close, ptr, temperature=0.07, margin_scale=0.0
    )
    close_loss = chemical_margin_listwise_loss(
        scores, close, ptr, temperature=0.07, margin_scale=0.03
    )
    distant_loss = chemical_margin_listwise_loss(
        scores, distant, ptr, temperature=0.07, margin_scale=0.03
    )

    assert torch.allclose(zero_margin, clean)
    assert close_loss > distant_loss
    assert torch.allclose(distant_loss, clean)


def test_targeted_candidate_margin_is_zero_for_correct_or_distant_groups():
    ptr = torch.tensor([0, 2])
    close = torch.tensor([[1.0, 0.0], [0.8, 0.6]])
    distant = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    scores = torch.tensor([0.50, 0.55], requires_grad=True)
    wrong_official = torch.tensor([0.50, 0.55])
    close_increment, active = targeted_chemical_margin_increment(
        scores, wrong_official, close, ptr,
        margin_scale=0.10, similarity_threshold=0.40,
    )
    assert active.tolist() == [True]
    assert close_increment > 0

    distant_increment, distant_active = targeted_chemical_margin_increment(
        scores, wrong_official, distant, ptr,
        margin_scale=0.10, similarity_threshold=0.40,
    )
    assert distant_active.tolist() == [True]
    assert torch.allclose(distant_increment, torch.tensor(0.0))

    correct_official = torch.tensor([0.60, 0.55])
    protected_increment, protected_active = targeted_chemical_margin_increment(
        scores, correct_official, close, ptr,
        margin_scale=0.10, similarity_threshold=0.40,
    )
    assert protected_active.tolist() == [False]
    assert torch.allclose(protected_increment, torch.tensor(0.0))
    protected_increment.backward()
    assert torch.count_nonzero(scores.grad) == 0


class TinyFeedForward(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.in_proj = nn.Linear(dimension, 4 * dimension)
        self.out_proj = nn.Linear(4 * dimension, dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.relu(self.in_proj(x)))


class TinyEncoder(nn.Module):
    def __init__(self, dimension: int, layers: int):
        super().__init__()
        self.n_layers = layers
        self.atts = nn.ModuleList([TinyAttention(dimension) for _ in range(layers)])
        self.ffs = nn.ModuleList([TinyFeedForward(dimension) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attention, feed_forward in zip(self.atts, self.ffs):
            x = x + attention(x)
            x = x + feed_forward(x)
        return x


class TinyBackbone(nn.Module):
    def __init__(self, dimension: int, layers: int = 3):
        super().__init__()
        self.lift = nn.Linear(2, dimension)
        self.transformer_encoder = TinyEncoder(dimension, layers)

    def forward(self, spectra: torch.Tensor, charge=None) -> torch.Tensor:
        return self.transformer_encoder(self.lift(spectra))


class TinyOfficialModel(nn.Module):
    def __init__(self, dimension: int = 12):
        super().__init__()
        self.backbone = TinyBackbone(dimension)
        self.head = nn.Linear(dimension, dimension)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        precursor = self.backbone(spectra, None)[:, 0]
        return F.normalize(self.head(precursor), dim=-1)


def test_peft_zero_init_is_exact_and_only_low_rank_parameters_train():
    torch.manual_seed(13)
    model = TinyOfficialModel()
    spectra = torch.randn(4, 6, 2)
    model.eval()
    expected = model(spectra).detach().clone()

    audit = install_dreams_peft(
        model, DreaMSPEFTConfig(last_blocks=1, rank=3, alpha=3.0)
    )
    observed = model(spectra)
    assert torch.equal(observed, expected)
    assert audit["adapted_parameter_matrices"] == [
        "backbone.transformer_encoder.atts.2.weights",
        "backbone.transformer_encoder.ffs.2.in_proj.weight",
        "backbone.transformer_encoder.ffs.2.out_proj.weight",
        "head.weight",
    ]
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    assert trainable
    assert all("parametrizations" in name for name, _ in trainable)

    target = F.normalize(torch.randn_like(observed), dim=-1)
    loss = (1.0 - torch.sum(observed * target, dim=1)).mean()
    loss.backward()
    b_gradients = [parameter.grad for name, parameter in trainable if name.endswith(".B")]
    assert b_gradients and all(gradient is not None for gradient in b_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in b_gradients)


def test_peft_checkpoint_round_trip_is_strict_and_changes_output_after_step():
    torch.manual_seed(14)
    base = TinyOfficialModel()
    clone = copy.deepcopy(base)
    spectra = torch.randn(3, 5, 2)
    config = DreaMSPEFTConfig(last_blocks=2, rank=2, alpha=4.0)
    install_dreams_peft(base, config)
    trainable = [parameter for parameter in base.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    before = base(spectra).detach().clone()
    target = F.normalize(torch.randn_like(before), dim=-1)
    loss = (1.0 - torch.sum(base(spectra) * target, dim=1)).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    after = base(spectra).detach()
    assert not torch.equal(after, before)

    saved = peft_state_dict(base)
    install_dreams_peft(clone, config)
    load_peft_state_dict(clone, saved)
    assert torch.equal(clone(spectra), after)


def test_chemical_candidate_geometry_changes_deployable_peft_gradient():
    torch.manual_seed(15)
    model = TinyOfficialModel(dimension=12)
    install_dreams_peft(
        model, DreaMSPEFTConfig(last_blocks=1, rank=3, alpha=3.0)
    )
    spectra = torch.randn(4, 6, 2)
    encoded = model(spectra)
    scores = molecule_scores_from_spectrum_pairs(
        encoded[:1], encoded[1:], torch.zeros(3, dtype=torch.long),
        torch.arange(4, dtype=torch.long),
    )
    query_ptr = torch.tensor([0, 3])
    teacher = F.normalize(torch.tensor([
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.0, 1.0, 0.0],
    ]), dim=1)
    clean = molecule_listwise_loss(scores, query_ptr, temperature=0.2)
    chemical = chemical_weighted_listwise_loss(
        scores, teacher, query_ptr, temperature=0.2,
        hardness_beta=4.0, weighting="absolute_bounded",
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    audit = gradient_geometry(clean, chemical - clean, trainable)
    assert audit["chemical_minus_clean_gradient_norm"] > 0
    assert audit["chemical_delta_nonzero_parameter_tensors"] > 0
    assert len(audit["chemical_delta_gradient_signature"]) > 0
    signature = torch.tensor(audit["chemical_delta_gradient_signature"])
    assert torch.allclose(signature.norm(), torch.tensor(1.0), atol=1e-6)


def test_frozen_ridge_probe_is_deterministic_and_has_no_trainable_parameters():
    rng = torch.Generator().manual_seed(21)
    inputs = torch.randn(20, 6, generator=rng).numpy()
    transform = torch.randn(6, 4, generator=rng).numpy()
    targets = inputs @ transform
    first = fit_frozen_ridge_probe(inputs, targets, alpha=0.1)
    second = fit_frozen_ridge_probe(inputs, targets, alpha=0.1)
    assert torch.equal(torch.from_numpy(first.weight), torch.from_numpy(second.weight))
    probe = FrozenChemicalProbe(first)
    assert sum(parameter.numel() for parameter in probe.parameters()) == 0
    prediction = probe(torch.from_numpy(inputs))
    expected = F.normalize(torch.from_numpy(targets), dim=-1)
    assert torch.mean(torch.sum(prediction * expected, dim=1)) > 0.99


def test_targeted_probe_loss_excludes_high_margin_official_correct_queries():
    torch.manual_seed(22)
    query = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_(True)
    fit = FrozenProbeFit(
        input_mean=torch.zeros(1, 4).numpy(),
        target_mean=torch.zeros(1, 4).numpy(),
        weight=torch.eye(4).numpy(),
        alpha=1.0,
        examples=8,
    )
    probe = FrozenChemicalProbe(fit)
    official = torch.tensor([
        0.8, 0.3, 0.2,  # high-margin correct: protected from chemistry
        0.2, 0.3, 0.1,  # official wrong
        0.25, 0.245, 0.1,  # low-margin official correct
    ])
    teacher = F.normalize(torch.randn(9, 4), dim=-1)
    query_ptr = torch.tensor([0, 3, 6, 9])
    loss, active = targeted_probe_listwise_loss(
        query, official, teacher, query_ptr, probe,
        margin_threshold=0.01, temperature=0.2,
    )
    assert active.tolist() == [False, True, True]
    loss.backward()
    assert torch.count_nonzero(query.grad[0]) == 0
    assert torch.count_nonzero(query.grad[1]) > 0
    assert torch.count_nonzero(query.grad[2]) > 0


def test_targeted_probe_loss_is_zero_without_observable_negative():
    query = F.normalize(torch.randn(1, 3), dim=-1).requires_grad_(True)
    fit = FrozenProbeFit(
        input_mean=torch.zeros(1, 3).numpy(),
        target_mean=torch.zeros(1, 3).numpy(),
        weight=torch.eye(3).numpy(),
        alpha=1.0,
        examples=3,
    )
    loss, active = targeted_probe_listwise_loss(
        query, torch.tensor([0.1, 0.2]), F.normalize(torch.randn(2, 3), dim=-1),
        torch.tensor([0, 2]), FrozenChemicalProbe(fit), observable=torch.tensor([True, False]),
    )
    assert not torch.any(active)
    loss.backward()
    assert torch.count_nonzero(query.grad) == 0


def test_targeted_multiview_probe_aligns_query_and_positive_references_only_when_active():
    torch.manual_seed(23)
    query = F.normalize(torch.randn(2, 4), dim=-1).requires_grad_(True)
    references = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_(True)
    fit = FrozenProbeFit(
        input_mean=torch.zeros(1, 4).numpy(),
        target_mean=torch.zeros(1, 4).numpy(),
        weight=torch.eye(4).numpy(),
        alpha=1.0,
        examples=8,
    )
    official = torch.tensor([
        0.8, 0.3, 0.2,  # high-margin correct
        0.2, 0.3, 0.1,  # official wrong
    ])
    teacher = F.normalize(torch.randn(6, 4), dim=-1)
    loss, active = targeted_probe_multiview_listwise_loss(
        query,
        references,
        torch.tensor([0, 2, 3]),
        official,
        teacher,
        torch.tensor([0, 3, 6]),
        FrozenChemicalProbe(fit),
        margin_threshold=0.01,
        temperature=0.2,
    )
    assert active.tolist() == [False, True]
    loss.backward()
    assert torch.count_nonzero(query.grad[0]) == 0
    assert torch.count_nonzero(references.grad[:2]) == 0
    assert torch.count_nonzero(query.grad[1]) > 0
    assert torch.count_nonzero(references.grad[2]) > 0
