from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from dreams.models.chem_aware.shared_embedding_v2 import SignedPeakResidualAdapter
from dreams.models.chem_aware.peft_v3 import (
    DreaMSPEFTConfig, install_dreams_peft, peft_state_dict,
)
from tasks.shared_dreams_inference import load_inference_model, sha256_file


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

    def forward(self, spectra):
        return F.normalize(self.head(self.backbone(spectra, None)[:, 0]), dim=-1)


class TinyAttention(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(4 * dimension, dimension) * 0.02)

    def forward(self, x):
        dimension = x.shape[-1]
        return F.linear(x, self.weights[3 * dimension:])


class TinyFeedForward(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.in_proj = nn.Linear(dimension, 4 * dimension)
        self.out_proj = nn.Linear(4 * dimension, dimension)

    def forward(self, x):
        return self.out_proj(F.relu(self.in_proj(x)))


class TinyTransformer(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.n_layers = 1
        self.atts = nn.ModuleList([TinyAttention(dimension)])
        self.ffs = nn.ModuleList([TinyFeedForward(dimension)])

    def forward(self, x):
        return x + self.atts[0](x) + self.ffs[0](x)


class TinyPeftBackbone(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.lift = nn.Linear(2, dimension, bias=False)
        self.transformer_encoder = TinyTransformer(dimension)

    def forward(self, spectra, charge=None):
        return self.transformer_encoder(self.lift(spectra))


class TinyPeftOfficial(TinyOfficial):
    def __init__(self, dimension: int):
        nn.Module.__init__(self)
        self.backbone = TinyPeftBackbone(dimension)
        self.head = nn.Linear(dimension, dimension)


def checkpoint(path: Path, official: Path, candidate_inputs: bool = False) -> None:
    adapter = SignedPeakResidualAdapter(8, hidden_dim=12, delta_bound=0.1)
    torch.save({
        "status": "chemaware_shared_v2_molecule_teacher",
        "adapter_state": adapter.state_dict(),
        "adapter_config": {"embedding_dim": 8, "hidden_dim": 12, "delta_bound": 0.1},
        "training_only_projector_used": False,
        "chemical_supervision": True,
        "teacher_control": "correct",
        "query_reference_encoder_shared": True,
        "candidate_inputs_at_inference": candidate_inputs,
        "P2b_used": False,
        "seed": 17,
        "outer_fold": 2,
        "provenance": {"official_checkpoint_sha256": sha256_file(official)},
    }, path)


def test_chemaware_checkpoint_loads_only_shared_spectrum_adapter(tmp_path: Path):
    official_path = tmp_path / "official.pt"
    official_path.write_bytes(b"official")
    shared_path = tmp_path / "adapter.pt"
    checkpoint(shared_path, official_path)
    official_model = TinyOfficial(8)
    spectra = torch.tensor([[
        [300.0, 1.1], [80.0, 0.4], [120.0, 1.0], [0.0, 0.0],
    ]])
    with patch(
        "tasks.shared_dreams_inference.load_base_model",
        return_value=(official_model, {"kind": "test"}),
    ):
        model, metadata = load_inference_model(
            official_path, tmp_path / "architecture.pt", torch.device("cpu"), 3,
            shared_path,
        )
    output = model(spectra)
    assert output.shape == (1, 8)
    assert torch.allclose(output.norm(dim=1), torch.ones(1), atol=1e-6)
    assert torch.equal(output, official_model(spectra))
    assert metadata["kind"] == "experimental_chemaware_shared_embedding"
    assert metadata["training_only_molecule_projector_loaded"] is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_chemaware_inference_rejects_candidate_dependent_checkpoint(tmp_path: Path):
    official_path = tmp_path / "official.pt"
    official_path.write_bytes(b"official")
    shared_path = tmp_path / "bad.pt"
    checkpoint(shared_path, official_path, candidate_inputs=True)
    with patch(
        "tasks.shared_dreams_inference.load_base_model",
        return_value=(TinyOfficial(8), {"kind": "test"}),
    ):
        try:
            load_inference_model(
                official_path, tmp_path / "architecture.pt", torch.device("cpu"), 3,
                shared_path,
            )
        except RuntimeError as error:
            assert "forbidden candidate inputs" in str(error)
        else:
            raise AssertionError("candidate-dependent checkpoint must be rejected")


def test_chemaware_v3_peft_checkpoint_reconstructs_shared_encoder(tmp_path: Path):
    torch.manual_seed(91)
    official_path = tmp_path / "official.pt"
    architecture_path = tmp_path / "architecture.pt"
    official_path.write_bytes(b"official")
    architecture_path.write_bytes(b"architecture")
    base = TinyPeftOfficial(8)
    trained = copy.deepcopy(base)
    config = DreaMSPEFTConfig(last_blocks=1, rank=2, alpha=2.0)
    install_dreams_peft(trained, config)
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if parameter.requires_grad and name.endswith(".B"):
                parameter.normal_(std=0.01)
    shared_path = tmp_path / "peft.pt"
    torch.save({
        "status": "chemaware_shared_v3_clean_peft",
        "format": "chemaware_shared_v3_peft_v1",
        "peft_state": peft_state_dict(trained),
        "peft_config": {
            "last_blocks": 1, "rank": 2, "alpha": 2.0,
            "adapt_attention": True, "adapt_feed_forward": True,
            "adapt_head": True,
        },
        "chemical_supervision": False,
        "training_only_projector_used": False,
        "query_reference_encoder_shared": True,
        "candidate_inputs_at_inference": False,
        "P2b_used": False,
        "seed": 17,
        "outer_fold": 0,
        "formal": False,
        "provenance": {
            "official_checkpoint_sha256": sha256_file(official_path),
            "raw_checkpoint_sha256": sha256_file(architecture_path),
        },
    }, shared_path)
    spectra = torch.randn(2, 4, 2)
    expected = trained(spectra).detach()
    with patch(
        "tasks.shared_dreams_inference.load_base_model",
        return_value=(copy.deepcopy(base), "test"),
    ):
        model, metadata = load_inference_model(
            official_path, architecture_path, torch.device("cpu"), 3, shared_path,
        )
    assert torch.equal(model(spectra), expected)
    assert metadata["kind"] == "experimental_chemaware_shared_peft_embedding"
    assert metadata["peft_config"]["rank"] == 2
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_chemaware_v3_frozen_probe_is_discarded_and_cannot_enter_inference(tmp_path: Path):
    official_path = tmp_path / "official.pt"
    architecture_path = tmp_path / "architecture.pt"
    official_path.write_bytes(b"official")
    architecture_path.write_bytes(b"architecture")
    base = TinyPeftOfficial(8)
    trained = copy.deepcopy(base)
    config = DreaMSPEFTConfig(last_blocks=1, rank=2, alpha=2.0)
    install_dreams_peft(trained, config)
    package = {
        "status": "chemaware_shared_v3_molecule_teacher_peft",
        "format": "chemaware_shared_v3_peft_v1",
        "peft_state": peft_state_dict(trained),
        "peft_config": {
            "last_blocks": 1, "rank": 2, "alpha": 2.0,
            "adapt_attention": True, "adapt_feed_forward": True,
            "adapt_head": True,
        },
        "chemical_supervision": True,
        "training_only_projector_used": False,
        "training_only_frozen_probe_used": True,
        "chemical_gradient_absorber_trainable": False,
        "query_reference_encoder_shared": True,
        "candidate_inputs_at_inference": False,
        "P2b_used": False,
        "seed": 17,
        "outer_fold": 0,
        "formal": False,
        "provenance": {
            "official_checkpoint_sha256": sha256_file(official_path),
            "raw_checkpoint_sha256": sha256_file(architecture_path),
        },
    }
    shared_path = tmp_path / "probe_peft.pt"
    torch.save(package, shared_path)
    with patch(
        "tasks.shared_dreams_inference.load_base_model",
        return_value=(copy.deepcopy(base), "test"),
    ):
        _, metadata = load_inference_model(
            official_path, architecture_path, torch.device("cpu"), 3, shared_path,
        )
    assert metadata["training_only_frozen_probe_loaded"] is False

    package["frozen_probe_state"] = {"weight": torch.eye(2)}
    torch.save(package, shared_path)
    with patch(
        "tasks.shared_dreams_inference.load_base_model",
        return_value=(copy.deepcopy(base), "test"),
    ), pytest.raises(RuntimeError, match="training-only chemistry"):
        load_inference_model(
            official_path, architecture_path, torch.device("cpu"), 3, shared_path,
        )
