#!/usr/bin/env python
"""Contracts for the sole WSE additive mechanism control."""
from __future__ import annotations

import torch

from train_wse_additive_mechanism_control import capacity_matched_model, parameter_count
from train_wse_conditional_residual import CHEMICAL_FEATURE_NAMES, SPECTRAL_FEATURE_NAMES


def main() -> None:
    feature_count = len(SPECTRAL_FEATURE_NAMES) + len(CHEMICAL_FEATURE_NAMES)
    model, chemical_hidden, target = capacity_matched_model(feature_count, hidden=16, bound=0.7)
    assert chemical_hidden == 40
    assert parameter_count(model) == target == 418
    assert model.residual_bound == 0.7

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.5, 0.5)
    first = torch.randn(5, feature_count)
    second = first.clone()
    second[:, :len(SPECTRAL_FEATURE_NAMES)] += 3.0
    zero_first = first.clone()
    zero_first[:, len(SPECTRAL_FEATURE_NAMES):] = 0.0
    zero_second = second.clone()
    zero_second[:, len(SPECTRAL_FEATURE_NAMES):] = 0.0
    chemical_effect_first = model.residual(first) - model.residual(zero_first)
    chemical_effect_second = model.residual(second) - model.residual(zero_second)
    assert torch.allclose(chemical_effect_first, chemical_effect_second, atol=1e-7, rtol=0)
    print("PASS: WSE additive mechanism-control contracts")


if __name__ == "__main__":
    main()
