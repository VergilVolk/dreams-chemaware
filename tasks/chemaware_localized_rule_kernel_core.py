"""Core algebra for a deployable localized ChemAware shared embedding."""
from __future__ import annotations

import numpy as np
import torch


class LocalizedRuleGate(torch.nn.Module):
    """Two low-capacity spectrum-only gates for NL and CF rule blocks."""

    def __init__(self, input_dim: int, initial_gate: float = 0.35) -> None:
        super().__init__()
        if input_dim <= 0 or not 0.0 < initial_gate < 1.0:
            raise ValueError("invalid localized gate shape or initialization")
        self.linear = torch.nn.Linear(input_dim, 2)
        torch.nn.init.zeros_(self.linear.weight)
        bias = float(np.log(initial_gate / (1.0 - initial_gate)))
        torch.nn.init.constant_(self.linear.bias, bias)

    def forward(self, clean_spectrum_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(clean_spectrum_features))


def localized_pair_scores(
    official_pair: torch.Tensor,
    neutral_loss_pair: torch.Tensor,
    fragment_ion_pair: torch.Tensor,
    gates: torch.Tensor,
    pair_query_node: torch.Tensor,
    pair_reference_node: torch.Tensor,
    *,
    neutral_loss_cap: float,
    fragment_ion_cap: float,
) -> torch.Tensor:
    """Score pairs exactly as an explicit spectrum-only shared embedding."""

    if gates.ndim != 2 or gates.shape[1] != 2:
        raise ValueError("gates must have shape [spectra, 2]")
    nl_scale = (
        gates[pair_query_node, 0] * gates[pair_reference_node, 0]
        * float(neutral_loss_cap)
    )
    cf_scale = (
        gates[pair_query_node, 1] * gates[pair_reference_node, 1]
        * float(fragment_ion_cap)
    )
    return official_pair + nl_scale * neutral_loss_pair + cf_scale * fragment_ion_pair


def localized_shared_embedding(
    official: np.ndarray,
    neutral_loss: np.ndarray,
    fragment_ion: np.ndarray,
    gates: np.ndarray,
    *,
    neutral_loss_cap: float,
    fragment_ion_cap: float,
) -> np.ndarray:
    """Materialize the primal map whose dot product equals the localized score."""

    gates = np.asarray(gates, dtype=np.float32)
    if gates.ndim != 2 or gates.shape != (len(official), 2):
        raise ValueError("gates and spectra disagree")
    return np.concatenate((
        np.asarray(official, dtype=np.float32),
        np.sqrt(float(neutral_loss_cap)) * gates[:, :1] * np.asarray(neutral_loss, dtype=np.float32),
        np.sqrt(float(fragment_ion_cap)) * gates[:, 1:] * np.asarray(fragment_ion, dtype=np.float32),
    ), axis=1)


def hardest_margin_loss(
    pair_scores: torch.Tensor,
    pair_molecule: torch.Tensor,
    molecule_query: torch.Tensor,
    molecule_label: torch.Tensor,
    query_weight: torch.Tensor,
    *,
    temperature: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Risk-weighted smooth max-negative versus positive retrieval margin."""

    molecule_count = int(molecule_label.numel())
    query_count = int(query_weight.numel())
    molecule_score = torch.full(
        (molecule_count,), -torch.inf, dtype=pair_scores.dtype, device=pair_scores.device,
    )
    molecule_score.scatter_reduce_(0, pair_molecule, pair_scores, reduce="amax", include_self=True)
    positive = torch.full(
        (query_count,), -torch.inf, dtype=pair_scores.dtype, device=pair_scores.device,
    )
    positive.scatter_reduce_(
        0, molecule_query[molecule_label], molecule_score[molecule_label],
        reduce="amax", include_self=True,
    )
    negative = torch.full_like(positive, -torch.inf)
    negative.scatter_reduce_(
        0, molecule_query[~molecule_label], molecule_score[~molecule_label],
        reduce="amax", include_self=True,
    )
    if not bool(torch.isfinite(positive).all() and torch.isfinite(negative).all()):
        raise RuntimeError("each training query must contain positive and negative molecules")
    shortfall = (negative - positive + float(margin)) / float(temperature)
    per_query = torch.nn.functional.softplus(shortfall) * float(temperature)
    return torch.sum(per_query * query_weight) / torch.sum(query_weight), positive - negative

