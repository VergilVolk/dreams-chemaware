"""Dose-neutral transfer-target construction for a future direct-v4 arm."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from noise_corrected_action_expansion_v4 import (
    mass_matched_monotone_transfer_by_group,
)


def v3_hard_capped_transfer_delta(
    delta_uncapped: torch.Tensor,
    edge_mask: torch.Tensor,
    *,
    hard_cap: float,
) -> torch.Tensor:
    if hard_cap <= 0 or delta_uncapped.shape != edge_mask.shape:
        raise ValueError("hard-cap transfer inputs are invalid")
    return (
        torch.relu(delta_uncapped.detach()).clamp_max(float(hard_cap))
        * edge_mask.to(delta_uncapped.dtype)
    )


def v4_mass_neutral_transfer_delta(
    delta_uncapped: torch.Tensor,
    edge_mask: torch.Tensor,
    groups: Sequence[str],
    *,
    hard_cap: float,
    maximum_cap_factor: float = 2.0,
) -> torch.Tensor:
    """Retain v3 total target mass while resolving strong action edges."""
    return mass_matched_monotone_transfer_by_group(
        delta_uncapped.detach(), edge_mask,
        groups,
        hard_cap=hard_cap,
        maximum_cap_factor=maximum_cap_factor,
    )
