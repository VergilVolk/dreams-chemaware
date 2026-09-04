from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
if str(TASKS) not in sys.path:
    sys.path.insert(0, str(TASKS))

from train_chemaware_shared_v3_peft import FrozenPrefixSpectrumStore  # noqa: E402


class _Attention:
    d_graphormer_params = 0


class _TinyEncoder(nn.Module):
    def __init__(self, dimension: int = 8, layers: int = 4):
        super().__init__()
        self.pre_norm = True
        self.n_layers = layers
        self.atts = [_Attention() for _ in range(layers)]
        self.scales = nn.ModuleList([nn.LayerNorm(dimension) for _ in range(2 * layers + 1)])
        self.blocks = nn.ModuleList([nn.Linear(dimension, dimension) for _ in range(layers)])

    def _layer_forward(self, layer, x, mask, graphormer_dists, chem_bias=None):
        del mask, graphormer_dists, chem_bias
        normalized = self.scales[2 * layer](x)
        x = x + torch.tanh(self.blocks[layer](normalized))
        return x + 0.1 * self.scales[2 * layer + 1](x)

    def forward(self, x, mask, graphormer_dists):
        for layer in range(self.n_layers):
            x = self._layer_forward(layer, x, mask, graphormer_dists)
        return self.scales[-1](x)


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_fourier = 1
        self.input = nn.Linear(2, 8)
        self.transformer_encoder = _TinyEncoder()
        self.ff_fourier = nn.Identity()

    @staticmethod
    def fourier_enc(value):
        return value

    def forward(self, spectra, _charge):
        x = self.input(spectra)
        mask = spectra[:, :, 0] == 0
        projected = spectra[..., [0]]
        bias = projected.unsqueeze(2) - projected.unsqueeze(1)
        return self.transformer_encoder(x, mask, bias)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _TinyBackbone()
        self.head = nn.Linear(8, 8, bias=False)

    def forward(self, spectra):
        tokens = self.backbone(spectra, None)
        return F.normalize(self.head(tokens[:, 0]), dim=-1)


class _TinyRawStore:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.rows = np.arange(len(tensor), dtype=np.int64)
        self.position = {int(row): int(row) for row in self.rows}

    def get(self, rows):
        return self.tensor[np.asarray(rows, dtype=np.int64)]


def test_multiblock_frozen_prefix_matches_full_forward():
    torch.manual_seed(14)
    model = _TinyModel().eval()
    spectra = torch.rand(7, 6, 2)
    spectra[:, 0, 0] += 1.0
    source = _TinyRawStore(spectra)
    expected = model(spectra).detach()
    cache = FrozenPrefixSpectrumStore(
        model, source, torch.device("cpu"), batch_size=3, last_blocks=2
    )
    observed = cache.forward(
        model, source.rows, torch.device("cpu"), batch_size=2, amp=False
    ).detach()
    assert cache.audit["adapted_layers"] == [2, 3]
    assert torch.max(torch.abs(expected - observed)).item() < 2e-6

