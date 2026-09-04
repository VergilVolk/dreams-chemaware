"""Zero-initialized low-rank updates for the shared DreaMS encoder.

The module deliberately uses PyTorch parametrizations instead of replacing
the DreaMS layers.  This is important for the custom attention implementation,
whose Q/K/V/O projections are stored in one ``weights`` parameter rather than
four ``nn.Linear`` modules.  At installation time every base parameter is
frozen and every low-rank ``B`` matrix is exactly zero, so evaluation is
bitwise-equivalent to the supplied official model up to the ordinary floating
point operations already present in that model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn.utils import parametrize


class LowRankAdditiveParametrization(nn.Module):
    """Return ``weight + scale * (B @ A)`` with an exact zero delta at init."""

    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float):
        super().__init__()
        if rank < 1 or rank > min(out_features, in_features):
            raise ValueError("rank must be in 1..min(weight.shape)")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + (self.B @ self.A).to(dtype=weight.dtype) * self.scale


@dataclass(frozen=True)
class DreaMSPEFTConfig:
    last_blocks: int = 1
    rank: int = 8
    alpha: float = 8.0
    adapt_attention: bool = True
    adapt_feed_forward: bool = True
    adapt_head: bool = True


def _register(module: nn.Module, parameter_name: str, rank: int, alpha: float) -> None:
    parameter = getattr(module, parameter_name)
    if parameter.ndim != 2:
        raise ValueError(f"{parameter_name} must be a matrix, observed {parameter.shape}")
    out_features, in_features = map(int, parameter.shape)
    parametrize.register_parametrization(
        module,
        parameter_name,
        LowRankAdditiveParametrization(out_features, in_features, rank, alpha),
    )
    getattr(module.parametrizations, parameter_name).original.requires_grad_(False)


def install_dreams_peft(model: nn.Module, config: DreaMSPEFTConfig) -> dict:
    """Freeze ``model`` and install LoRA-style updates on its last blocks.

    The expected model contract is the repository's ``IdentityEmbeddingModel``:
    ``model.backbone.transformer_encoder`` is the custom DreaMS encoder and
    ``model.head`` is the official contrastive projection head.
    """

    encoder = model.backbone.transformer_encoder
    n_layers = int(encoder.n_layers)
    if config.last_blocks < 1 or config.last_blocks > n_layers:
        raise ValueError(f"last_blocks must be in 1..{n_layers}")
    if not (config.adapt_attention or config.adapt_feed_forward or config.adapt_head):
        raise ValueError("at least one PEFT target must be enabled")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    adapted: list[str] = []
    first_layer = n_layers - int(config.last_blocks)
    for layer in range(first_layer, n_layers):
        if config.adapt_attention:
            attention = encoder.atts[layer]
            _register(attention, "weights", config.rank, config.alpha)
            adapted.append(f"backbone.transformer_encoder.atts.{layer}.weights")
        if config.adapt_feed_forward:
            feed_forward = encoder.ffs[layer]
            _register(feed_forward.in_proj, "weight", config.rank, config.alpha)
            _register(feed_forward.out_proj, "weight", config.rank, config.alpha)
            adapted.extend((
                f"backbone.transformer_encoder.ffs.{layer}.in_proj.weight",
                f"backbone.transformer_encoder.ffs.{layer}.out_proj.weight",
            ))
    if config.adapt_head:
        _register(model.head, "weight", config.rank, config.alpha)
        adapted.append("head.weight")

    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("PEFT installation produced no trainable parameters")
    if any("parametrizations" not in name for name, _ in trainable):
        raise RuntimeError("a non-PEFT base parameter remained trainable")
    return {
        "config": asdict(config),
        "transformer_layers": n_layers,
        "adapted_parameter_matrices": adapted,
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameters": int(sum(parameter.numel() for _, parameter in trainable)),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def peft_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only deployable low-rank parameters, never the frozen backbone."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_peft_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Strictly restore a state produced by :func:`peft_state_dict`."""

    expected = {name: parameter for name, parameter in model.named_parameters()
                if parameter.requires_grad}
    if set(state) != set(expected):
        raise RuntimeError(
            "PEFT state keys differ from installed targets: "
            f"missing={sorted(set(expected) - set(state))}, "
            f"extra={sorted(set(state) - set(expected))}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            value = state[name]
            if value.shape != parameter.shape:
                raise RuntimeError(
                    f"PEFT shape mismatch for {name}: {value.shape} != {parameter.shape}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
