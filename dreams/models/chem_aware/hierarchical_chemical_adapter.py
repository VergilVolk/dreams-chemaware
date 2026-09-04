"""Small candidate-independent chemical branch for a shared DreaMS embedding.

The official DreaMS model stays frozen.  This branch reads only measurements
available from one spectrum at inference time (m/z, intensity, precursor m/z)
and produces a bounded zero-initialized residual on the official embedding.
Training-only peak heads can receive subformula and later high-confidence
fragment supervision, but no candidate or molecular structure enters forward.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class HierarchicalChemicalOutput:
    embedding: torch.Tensor
    official_embedding: torch.Tensor
    delta: torch.Tensor
    peak_states: torch.Tensor
    formula_logits: torch.Tensor
    peak_mask: torch.Tensor


class LogSpacedFourier(nn.Module):
    """Fixed mass features spanning high-resolution and coarse mass scales."""

    def __init__(self, n_frequencies: int = 16, min_period: float = 0.005, max_period: float = 1000.0):
        super().__init__()
        if n_frequencies < 2 or not 0 < min_period < max_period:
            raise ValueError("invalid Fourier configuration")
        periods = torch.logspace(
            torch.log10(torch.tensor(min_period)),
            torch.log10(torch.tensor(max_period)),
            n_frequencies,
        )
        self.register_buffer("periods", periods, persistent=True)

    @property
    def output_dim(self) -> int:
        return 2 * int(self.periods.numel())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * torch.pi * value.unsqueeze(-1).float() / self.periods
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)


class HierarchicalChemicalResidualAdapter(nn.Module):
    """Raw-spectrum chemical adapter with a training-only compositional head."""

    def __init__(
        self,
        embedding_dim: int = 1024,
        chemical_dim: int = 96,
        layers: int = 2,
        heads: int = 4,
        feed_forward_dim: int = 192,
        n_frequencies: int = 16,
        formula_dimensions: int = 36,
        delta_bound: float = 0.12,
        dropout: float = 0.0,
        use_formula_moments: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or chemical_dim < 8 or layers < 1 or heads < 1:
            raise ValueError("invalid adapter dimensions")
        if chemical_dim % heads:
            raise ValueError("chemical_dim must be divisible by heads")
        if formula_dimensions < 1 or delta_bound <= 0:
            raise ValueError("invalid output configuration")
        self.embedding_dim = int(embedding_dim)
        self.chemical_dim = int(chemical_dim)
        self.formula_dimensions = int(formula_dimensions)
        self.delta_bound = float(delta_bound)
        self.use_formula_moments = bool(use_formula_moments)

        self.mass_fourier = LogSpacedFourier(n_frequencies=n_frequencies)
        # fragment m/z Fourier, neutral-loss Fourier, sqrt intensity,
        # relative m/z, and a valid-peak indicator.
        peak_input_dim = 2 * self.mass_fourier.output_dim + 3
        self.peak_input = nn.Sequential(
            nn.LayerNorm(peak_input_dim),
            nn.Linear(peak_input_dim, chemical_dim),
            nn.GELU(),
        )
        self.precursor_input = nn.Sequential(
            nn.LayerNorm(self.mass_fourier.output_dim),
            nn.Linear(self.mass_fourier.output_dim, chemical_dim),
        )
        self.aggregate_token = nn.Parameter(torch.zeros(1, 1, chemical_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=chemical_dim,
            nhead=heads,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(chemical_dim),
        )
        self.formula_head = nn.Linear(chemical_dim, formula_dimensions)
        # Formula moments explicitly preserve peak-local chemistry in the
        # spectrum-level vector: composition marginal, intensity association,
        # fragment-mass association, and neutral-loss association.  This mode
        # makes ``formula_head`` part of deployment rather than a disposable
        # auxiliary head, while still reading only one raw spectrum.
        residual_input_dim = chemical_dim + (
            4 * formula_dimensions if self.use_formula_moments else 0
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(residual_input_dim),
            nn.Linear(residual_input_dim, embedding_dim),
        )
        # Exact official-embedding reproduction before any update.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        official_embedding: torch.Tensor,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        precursor_mz: torch.Tensor,
        peak_mask: torch.Tensor,
    ) -> HierarchicalChemicalOutput:
        if peak_mz.ndim != 2 or peak_intensity.shape != peak_mz.shape:
            raise RuntimeError("peak measurements must be aligned matrices")
        if peak_mask.shape != peak_mz.shape or peak_mask.dtype != torch.bool:
            raise RuntimeError("peak_mask must be a boolean matrix aligned to peaks")
        batch, _ = peak_mz.shape
        if precursor_mz.shape != (batch,):
            raise RuntimeError("precursor_mz must have shape (batch,)")
        if official_embedding.shape != (batch, self.embedding_dim):
            raise RuntimeError("official embedding shape mismatch")
        if not torch.all(peak_mask.any(dim=1)):
            raise RuntimeError("every spectrum needs at least one valid peak")

        mz = peak_mz.float()
        precursor = precursor_mz.float()
        loss = (precursor[:, None] - mz).clamp_min(0.0)
        intensity = torch.sqrt(peak_intensity.float().clamp_min(0.0)).unsqueeze(-1)
        relative = (mz / precursor[:, None].clamp_min(1e-6)).unsqueeze(-1)
        valid = peak_mask.unsqueeze(-1).to(mz.dtype)
        peak_features = torch.cat((
            self.mass_fourier(mz),
            self.mass_fourier(loss),
            intensity,
            relative,
            valid,
        ), dim=-1)
        peak_states = self.peak_input(peak_features)
        aggregate = self.aggregate_token.expand(batch, -1, -1) + self.precursor_input(
            self.mass_fourier(precursor)
        ).unsqueeze(1)
        sequence = torch.cat((aggregate, peak_states), dim=1)
        padding = torch.cat((
            torch.zeros((batch, 1), dtype=torch.bool, device=peak_mask.device),
            ~peak_mask,
        ), dim=1)
        sequence = self.encoder(sequence, src_key_padding_mask=padding)
        aggregate_state, contextual_peaks = sequence[:, 0], sequence[:, 1:]
        formula_logits = self.formula_head(contextual_peaks)
        residual_state = aggregate_state
        if self.use_formula_moments:
            mask_float = peak_mask.to(formula_logits.dtype)

            def weighted_formula_mean(weight: torch.Tensor) -> torch.Tensor:
                weight = weight * mask_float
                weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
                return torch.sum(formula_logits * weight.unsqueeze(-1), dim=1)

            relative_scalar = mz / precursor[:, None].clamp_min(1e-6)
            loss_scalar = loss / precursor[:, None].clamp_min(1e-6)
            formula_moments = torch.cat((
                weighted_formula_mean(torch.ones_like(mz)),
                weighted_formula_mean(torch.sqrt(peak_intensity.float().clamp_min(0.0))),
                weighted_formula_mean(relative_scalar),
                weighted_formula_mean(loss_scalar),
            ), dim=-1)
            residual_state = torch.cat((aggregate_state, formula_moments), dim=-1)
        raw_delta = self.residual_head(residual_state)
        raw_norm = raw_delta.norm(dim=1, keepdim=True)
        delta = self.delta_bound * raw_delta / (1.0 + raw_norm)
        official = official_embedding.float()
        adapted = F.normalize(official + delta, dim=-1)
        exact_zero = torch.all(delta == 0, dim=1, keepdim=True)
        straight_through = official + (adapted - adapted.detach())
        adapted = torch.where(exact_zero, straight_through, adapted)
        return HierarchicalChemicalOutput(
            embedding=adapted,
            official_embedding=official,
            delta=delta,
            peak_states=contextual_peaks,
            formula_logits=formula_logits,
            peak_mask=peak_mask,
        )


def deployable_parameter_count(module: nn.Module) -> int:
    """Exclude a formula head only when it is not used by deployment."""
    return sum(
        parameter.numel()
        for name, parameter in module.named_parameters()
        if getattr(module, "use_formula_moments", False)
        or not name.startswith("formula_head.")
    )
