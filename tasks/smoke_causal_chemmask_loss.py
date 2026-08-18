"""Model-independent gradient smoke test for causal head loss wiring."""

from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_causal_chemmask_head import forward_head


class MockBackbone(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.projection = nn.Linear(2, dimension)

    def forward(self, spectra: torch.Tensor, _):
        return self.projection(spectra)


class MockModel(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.backbone = MockBackbone(dimension)
        self.head = nn.Linear(dimension, dimension)


def main() -> None:
    torch.manual_seed(20260815)
    model = MockModel(32)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    teacher_weight = model.head.weight.detach().clone()
    teacher_bias = model.head.bias.detach().clone()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    batch = {
        "anchor": torch.rand(4, 101, 2),
        "positive": torch.rand(4, 101, 2),
        "negative": torch.rand(4, 101, 2),
    }
    before_head = model.head.weight.detach().clone()
    before_backbone = model.backbone.projection.weight.detach().clone()
    _, positive, negative, preserve, _, _ = forward_head(
        model, batch, teacher_weight, teacher_bias, torch.device("cpu"), False
    )
    triplet = F.relu(0.1 - positive + negative)
    loss = triplet.mean() + 0.1 * preserve.mean()
    loss.backward()
    head_grad = float(model.head.weight.grad.norm())
    backbone_has_grad = any(parameter.grad is not None for parameter in model.backbone.parameters())
    optimizer.step()
    report = {
        "status": "causal_chemmask_loss_smoke",
        "loss_finite": bool(torch.isfinite(loss)),
        "triplet_shape": list(triplet.shape),
        "preserve_shape": list(preserve.shape),
        "head_gradient_norm": head_grad,
        "head_updated": bool(not torch.equal(before_head, model.head.weight.detach())),
        "backbone_has_gradient": backbone_has_grad,
        "backbone_updated": bool(not torch.equal(
            before_backbone, model.backbone.projection.weight.detach()
        )),
    }
    print(json.dumps(report, indent=2))
    if not (
        report["loss_finite"] and report["head_gradient_norm"] > 0
        and report["head_updated"] and not report["backbone_has_gradient"]
        and not report["backbone_updated"]
    ):
        raise RuntimeError("Causal head gradient smoke failed")


if __name__ == "__main__":
    main()
