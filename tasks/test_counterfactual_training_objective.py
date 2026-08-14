"""Fast unit test for counterfactual loss shapes and gradient flow."""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch
from torch import nn

from train_counterfactual_dreams import batch_objective


class TinyBackbone(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.projection = nn.Linear(2, dimension)

    def forward(self, spectra: torch.Tensor, _charge=None) -> torch.Tensor:
        tokens = self.projection(spectra)
        return tokens


class TinyModel(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.backbone = TinyBackbone(dimension)
        self.head = nn.Linear(dimension, dimension)


def main() -> None:
    torch.manual_seed(7)
    batch_size, peaks, dimension = 3, 11, 8
    batch = {}
    for name in ("clean", "identity_masked", "confounder_masked", "random_masked", "positive", "negative"):
        values = torch.rand(batch_size, peaks, 2)
        values[:, 0, 1] = 1.1
        batch[name] = values
    for name in ("teacher_query", "teacher_positive", "teacher_negative"):
        batch[name] = torch.nn.functional.normalize(torch.rand(batch_size, dimension), dim=-1)
    batch["has_identity"] = torch.tensor([True, True, False])
    batch["has_confounder"] = torch.tensor([True, False, True])
    batch["sample_weight"] = torch.tensor([1.5, 1.0, 1.0])
    args = Namespace(
        triplet_margin=0.05, counterfactual_margin=0.02,
        triplet_weight=1.0, counterfactual_weight=0.7,
        preserve_weight=5.0, random_consistency_weight=0.2,
    )
    model = TinyModel(dimension)
    loss, values = batch_objective(model, batch, args, torch.device("cpu"), False, True)
    if not torch.isfinite(loss):
        raise RuntimeError("Loss is not finite")
    loss.backward()
    gradient = sum(float(parameter.grad.abs().sum()) for parameter in model.parameters() if parameter.grad is not None)
    if not np.isfinite(gradient) or gradient <= 0:
        raise RuntimeError("No finite gradient reached the trainable model")
    required = {"clean_margin", "identity_effect", "confounder_effect", "preserve", "random_consistency"}
    if required - set(values):
        raise RuntimeError(f"Missing metrics: {required - set(values)}")
    print(f"Counterfactual objective test passed: loss={float(loss):.6f}, gradient_l1={gradient:.6f}")


if __name__ == "__main__":
    main()
