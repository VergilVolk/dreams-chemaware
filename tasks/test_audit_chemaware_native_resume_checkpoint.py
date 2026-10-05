from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from audit_chemaware_native_resume_checkpoint import audit_checkpoint


def checkpoint(path: Path, *, with_optimizer: bool = True, step: int = 2000) -> None:
    package = {
        "state_dict": {
            "backbone.x": torch.ones(1),
            "head.x": torch.ones(1),
        },
        "hyper_parameters": {"d_model": 1},
    }
    if with_optimizer:
        package["optimizer_states"] = [{
            "state": {0: {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            }},
            "param_groups": [{
                "lr": 5e-6, "weight_decay": 0.0, "params": [0],
            }],
        }]
    torch.save(package, path)


def test_native_adam_checkpoint_passes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "phasea.ckpt"
        checkpoint(path)
        report = audit_checkpoint(path, 5e-6, 0.0, 2000)
    assert report["continuation_authorized"] is True
    assert report["step_min"] == 2000


def test_missing_optimizer_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "weights_only.ckpt"
        checkpoint(path, with_optimizer=False)
        with pytest.raises(RuntimeError, match="lacks exactly one optimizer state"):
            audit_checkpoint(path, 5e-6, 0.0, 2000)


def test_stale_adam_step_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "stale.ckpt"
        checkpoint(path, step=1999)
        with pytest.raises(RuntimeError, match="below 2000"):
            audit_checkpoint(path, 5e-6, 0.0, 2000)
