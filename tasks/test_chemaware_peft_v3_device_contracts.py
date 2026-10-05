"""Device/dtype and checkpoint contracts for the production PEFT installer."""
from __future__ import annotations

import torch
from torch import nn

from dreams.models.chem_aware.peft_v3 import (
    DreaMSPEFTConfig,
    install_dreams_peft,
    load_peft_state_dict,
    peft_state_dict,
)


class DummyAttention(nn.Module):
    def __init__(self, dimension: int, *, device: str, dtype: torch.dtype):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(4 * dimension, dimension, device=device, dtype=dtype)
        )


class DummyFeedForward(nn.Module):
    def __init__(self, dimension: int, *, device: str, dtype: torch.dtype):
        super().__init__()
        self.in_proj = nn.Linear(
            dimension, 2 * dimension, bias=False, device=device, dtype=dtype,
        )
        self.out_proj = nn.Linear(
            2 * dimension, dimension, bias=False, device=device, dtype=dtype,
        )


class DummyEncoder(nn.Module):
    def __init__(self, *, device: str, dtype: torch.dtype):
        super().__init__()
        self.n_layers = 2
        self.atts = nn.ModuleList([
            DummyAttention(8, device=device, dtype=dtype) for _ in range(2)
        ])
        self.ffs = nn.ModuleList([
            DummyFeedForward(8, device=device, dtype=dtype) for _ in range(2)
        ])


class DummyBackbone(nn.Module):
    def __init__(self, *, device: str, dtype: torch.dtype):
        super().__init__()
        self.transformer_encoder = DummyEncoder(device=device, dtype=dtype)


class DummyModel(nn.Module):
    def __init__(self, *, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.backbone = DummyBackbone(device=device, dtype=dtype)
        self.head = nn.Linear(8, 5, bias=False, device=device, dtype=dtype)


def assert_install(device: str, dtype: torch.dtype, *, checkpoint: bool) -> None:
    model = DummyModel(device=device, dtype=dtype)
    capacity = install_dreams_peft(
        model,
        DreaMSPEFTConfig(last_blocks=1, rank=2, alpha=2.0),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert capacity["trainable_parameter_tensors"] == len(trainable) == 8
    assert all(str(parameter.device) == device for parameter in trainable)
    assert all(parameter.dtype == dtype for parameter in trainable)
    adapted = [
        model.backbone.transformer_encoder.atts[-1].weights,
        model.backbone.transformer_encoder.ffs[-1].in_proj.weight,
        model.backbone.transformer_encoder.ffs[-1].out_proj.weight,
        model.head.weight,
    ]
    assert all(str(value.device) == device for value in adapted)
    if checkpoint:
        state = peft_state_dict(model)
        assert all(value.device.type == "cpu" for value in state.values())
        with torch.no_grad():
            trainable[0].add_(1.0)
        load_peft_state_dict(model, state)
        assert torch.equal(trainable[0], state[next(iter(state))].to(dtype=dtype))


def main() -> None:
    assert_install("cpu", torch.float64, checkpoint=True)
    # Meta is a genuine non-CPU device and catches the exact production bug
    # without requiring a GPU on the developer machine.
    assert_install("meta", torch.float32, checkpoint=False)
    if torch.cuda.is_available():
        assert_install("cuda:0", torch.float32, checkpoint=True)
    print("PASS: ChemAware PEFT device, dtype, state and non-CPU contracts")


if __name__ == "__main__":
    main()
