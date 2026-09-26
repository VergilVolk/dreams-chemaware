"""Run the mature DreaMS trainer with ChemAware identity-equal event sampling.

The model, preprocessing, optimizer, cosine triplet loss, checkpoint format and
all numerical hyperparameters come from ``train_chemaware_dreams_native``.
This ChemAware-local wrapper changes only the training-event sampler when the
pool contains a validated ``sampling_weight`` vector.  Shared Noise runtimes
and files are not modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

import train_chemaware_dreams_native as native
import train_chemaware_specific_replay_native as compact


def _paths() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    return args.train_pool, args.output


def _weights(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as loaded:
        if "sampling_weight" not in loaded.files:
            raise RuntimeError("identity-balanced ChemAware pool lacks sampling_weight")
        weights = np.asarray(loaded["sampling_weight"], dtype=np.float64)
        anchors = np.asarray(loaded["anchor_idx"], dtype=np.int64)
    if weights.ndim != 1 or len(weights) != len(anchors):
        raise RuntimeError("sampling weights do not align with native triplet events")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise RuntimeError("sampling weights must be finite and strictly positive")
    if abs(float(weights.sum()) - 1.0) > 1e-10:
        raise RuntimeError("sampling weights are not normalized")
    return weights


def main() -> None:
    train_pool, output = _paths()
    weights = _weights(train_pool)
    original_loader = native.DataLoader
    calls = 0

    def weighted_loader(dataset: Any, *args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        if calls == 1:
            if len(dataset) != len(weights):
                raise RuntimeError(
                    "weighted sampler length does not match the training subset"
                )
            if not bool(kwargs.pop("shuffle", False)):
                raise RuntimeError("native training loader no longer requests shuffling")
            kwargs["sampler"] = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights), replacement=True,
            )
            print(
                "CHEMAWARE_IDENTITY_EQUAL_SAMPLER "
                f"events={len(weights)} replacement=true mass={weights.sum():.12f}",
                flush=True,
            )
        return original_loader(dataset, *args, **kwargs)

    native.DataLoader = weighted_loader
    try:
        compact.main()
    finally:
        native.DataLoader = original_loader

    report_path = output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["event_sampler"] = {
        "type": "torch.utils.data.WeightedRandomSampler",
        "replacement": True,
        "num_samples_per_epoch": int(len(weights)),
        "weight_source": str(train_pool.resolve()) + ":sampling_weight",
        "model_loss_optimizer_unchanged": True,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["event_sampler"], indent=2), flush=True)


if __name__ == "__main__":
    main()
