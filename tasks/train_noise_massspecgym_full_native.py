#!/usr/bin/env python
"""One query-balanced pass of native DreaMS on the full MassSpecGym library."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dreams.models.heads.heads import ContrastiveHead
from dreams.utils.data import ContrastiveSpectraDataset, SpectrumPreprocessor
from dreams.utils.spectra import MSnSpectrum
from noise_dreams_native_spectrum import make_action_spectrum
from noise_final_core import sha256_file
from noise_massspecgym_full_triplet_core import (
    query_disjoint_batches,
    query_rotation_positions,
)
from train_chemaware_dreams_native import SlurmLineProgress
from train_noise_dreams_native import make_hdf5_spectrum
from train_noise_dreams_native_residual_stage2 import construct_native_model


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--triplet-dir", type=Path, required=True)
    parser.add_argument("--stage1-action-bank", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--triplet-loss-margin", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        return {key: np.asarray(body[key]) for key in body.files}


def validate_pool(pool: dict[str, np.ndarray]) -> None:
    required = {
        "registry_kind", "registry_source_index", "anchor_idx",
        "positive_ptr", "positive_idx", "negative_ptr", "negative_idx",
        "event_query", "event_kind", "event_action_index",
    }
    if set(pool) != required:
        raise RuntimeError(f"full native pool keys drifted: {sorted(pool)}")
    events = len(pool["anchor_idx"])
    registry = len(pool["registry_kind"])
    if not (
        len(pool["registry_source_index"]) == registry
        and len(pool["positive_ptr"]) == events + 1
        and len(pool["negative_ptr"]) == events + 1
        and len(pool["event_query"]) == events
        and len(pool["event_kind"]) == events
        and len(pool["event_action_index"]) == events
        and np.all(np.diff(pool["positive_ptr"]) >= 1)
        and np.all(np.diff(pool["negative_ptr"]) >= 1)
    ):
        raise RuntimeError("full native pool arrays are not aligned")
    for key in ("anchor_idx", "positive_idx", "negative_idx"):
        if np.any((pool[key] < 0) | (pool[key] >= registry)):
            raise RuntimeError(f"full native pool has invalid {key}")


def native_dataset(
    pool: dict[str, np.ndarray],
    data: Path,
    noise_spectra: np.ndarray,
    action_spectra: np.ndarray,
    preprocessor: SpectrumPreprocessor,
) -> tuple[ContrastiveSpectraDataset, np.ndarray, dict[str, int | float | bool]]:
    validate_pool(pool)
    kind = np.asarray(pool["registry_kind"], dtype=np.int8)
    source = np.asarray(pool["registry_source_index"], dtype=np.int64)
    spectra: list[MSnSpectrum | None] = [None] * len(kind)
    hdf5_positions = np.flatnonzero(kind == 0)
    noise_positions = np.flatnonzero(kind == 1)
    action_positions = np.flatnonzero(kind == 2)
    if np.any(~np.isin(kind, np.asarray([0, 1, 2], dtype=np.int8))):
        raise RuntimeError("unknown full native spectrum source kind")
    with h5py.File(data, "r") as handle:
        rows = source[hdf5_positions]
        if np.any((rows < 0) | (rows >= len(handle["spectrum"]))):
            raise RuntimeError("HDF5 registry source is out of range")
        for position, row in zip(hdf5_positions, rows, strict=True):
            spectra[int(position)] = make_hdf5_spectrum(
                handle["spectrum"][int(row)], float(handle["precursor_mz"][int(row)]),
            )

    maximum_roundtrip_error = 0.0
    for positions, tensors, label in (
        (noise_positions, noise_spectra, "noise"),
        (action_positions, action_spectra, "action"),
    ):
        indices = source[positions]
        if len(indices) and np.any((indices < 0) | (indices >= len(tensors))):
            raise RuntimeError(f"{label} registry source is out of range")
        for position, index in zip(positions, indices, strict=True):
            tensor = np.asarray(tensors[int(index)], dtype=np.float32)
            spectrum = make_action_spectrum(tensor)
            replay = preprocessor(
                spectrum.get_peak_list(), prec_mz=spectrum.get_precursor_mz(),
                high_form=False, augment=False,
            )
            source_tokens = tensor[tensor[:, 0] > 0]
            replay_tokens = replay[replay[:, 0] > 0]
            if source_tokens.shape != replay_tokens.shape:
                raise RuntimeError(f"{label} native preprocessing lost a real token")
            maximum_roundtrip_error = max(
                maximum_roundtrip_error,
                float(np.max(np.abs(source_tokens - replay_tokens))),
            )
            spectra[int(position)] = spectrum
    if maximum_roundtrip_error > 1e-6:
        raise RuntimeError(
            f"full native spectrum replay error {maximum_roundtrip_error:.8g}"
        )
    if any(value is None for value in spectra):
        raise RuntimeError("full native spectrum registry is incomplete")

    registry_count = len(spectra)
    event_spectra = [spectra[int(index)] for index in pool["anchor_idx"]]
    event_positive: list[list[int]] = []
    event_negative: list[list[int]] = []
    for event in range(len(pool["anchor_idx"])):
        p0, p1 = map(int, pool["positive_ptr"][event:event + 2])
        n0, n1 = map(int, pool["negative_ptr"][event:event + 2])
        event_positive.append(list(map(int, pool["positive_idx"][p0:p1])))
        event_negative.append(list(map(int, pool["negative_idx"][n0:n1])))
    frame = pd.DataFrame({
        "MSnSpectrum": spectra + event_spectra,
        "pos_idx": ([[] for _ in spectra] + event_positive),
        "neg_idx": ([[] for _ in spectra] + event_negative),
    })
    dataset = ContrastiveSpectraDataset(
        frame, spec_preproc=preprocessor,
        n_pos_samples=1, n_neg_samples=1, return_smiles=False,
    )
    event_dataset_indices = registry_count + np.arange(
        len(event_spectra), dtype=np.int64,
    )
    return dataset, event_dataset_indices, {
        "registry_spectra": registry_count,
        "hdf5_spectra": int(len(hdf5_positions)),
        "noise_spectra": int(len(noise_positions)),
        "action_spectra": int(len(action_positions)),
        "event_rows": int(len(event_spectra)),
        "native_preprocessor_max_abs_error": maximum_roundtrip_error,
        "event_memberships_are_independent": True,
    }


def main() -> None:
    args = arguments()
    frozen = {
        "seed": 3407, "lr": 5e-6, "weight_decay": 0.0,
        "triplet_loss_margin": 0.1, "batch_size": 4,
        "max_epochs": 1, "n_highest_peaks": 100,
    }
    observed = {key: getattr(args, key) for key in frozen}
    if observed != frozen:
        raise RuntimeError(f"full native DreaMS settings drifted: {observed} != {frozen}")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("full native DreaMS training requires an allocated GPU")
    paths = [
        args.data, args.triplet_dir / "train_pool.npz",
        args.triplet_dir / "noise_spectra.npz",
        args.triplet_dir / "report.json", args.stage1_action_bank,
        args.warm_start_checkpoint, args.architecture_checkpoint,
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    triplet_report = json.loads(
        (args.triplet_dir / "report.json").read_text(encoding="utf-8")
    )
    if triplet_report.get("status") != "NOISE_MASSSPECGYM_FULL_NATIVE_TRIPLETS_COMPLETE":
        raise RuntimeError("full MassSpecGym triplet artifact is incomplete")

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    pool = load_npz(args.triplet_dir / "train_pool.npz")
    noise = load_npz(args.triplet_dir / "noise_spectra.npz")
    action_bank = load_npz(args.stage1_action_bank)
    action_spectra = np.asarray(action_bank["targeted_action_spectra"], dtype=np.float32)

    model, initialization_kind = construct_native_model(args)
    if type(model) is not ContrastiveHead:
        raise RuntimeError("full training did not construct the native ContrastiveHead")
    preprocessor = model.backbone.spec_preproc
    if (
        type(preprocessor) is not SpectrumPreprocessor
        or int(preprocessor.n_highest_peaks) != 100
        or abs(float(preprocessor.prec_intens) - 1.1) > 1e-12
    ):
        raise RuntimeError("full training preprocessor drifted from native DreaMS")
    dataset, event_dataset_indices, dataset_report = native_dataset(
        pool, args.data, np.asarray(noise["spectra"], dtype=np.float32),
        action_spectra, preprocessor,
    )
    selected_positions = query_rotation_positions(
        pool["event_query"], epoch=0, seed=args.seed,
    )
    batches = query_disjoint_batches(
        selected_positions, pool["event_query"], args.batch_size, args.seed, epoch=0,
    )
    dataset_batches = [
        [int(event_dataset_indices[position]) for position in batch]
        for batch in batches
    ]
    selected_queries = pool["event_query"][selected_positions]
    if not np.array_equal(
        np.sort(selected_queries), np.arange(len(np.unique(pool["event_query"])), dtype=np.int64),
    ):
        raise RuntimeError("formal full training did not select every query exactly once")
    selected_kinds = pool["event_kind"][selected_positions]
    loader = DataLoader(
        dataset, batch_sampler=dataset_batches, num_workers=0, pin_memory=True,
    )
    first_batch = next(iter(loader))
    if not {"spec", "pos_specs", "neg_specs"}.issubset(first_batch):
        raise RuntimeError("native full triplet loader returned an invalid batch")
    del first_batch
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output.mkdir(parents=True)
    started = time.time()
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=1, precision="32-true",
        logger=False, callbacks=[SlurmLineProgress(every_n_batches=250)],
        enable_checkpointing=False, enable_progress_bar=False,
        num_sanity_val_steps=0, log_every_n_steps=20,
    )
    trainer.fit(model, train_dataloaders=loader)
    if int(trainer.global_step) != len(dataset_batches):
        raise RuntimeError("native optimizer did not execute exactly one step per batch")
    checkpoint = args.output / "final_slim.pt"
    torch.save({
        "format": "official_embedding_slim_v1",
        "source_checkpoint": str(args.warm_start_checkpoint),
        "source_size_bytes": args.warm_start_checkpoint.stat().st_size,
        "backbone_state_dict": {
            key: value.detach().cpu() for key, value in model.backbone.state_dict().items()
        },
        "head_state_dict": {
            key: value.detach().cpu() for key, value in model.head.state_dict().items()
        },
    }, checkpoint)
    report = {
        "status": "NOISE_MASSSPECGYM_FULL_NATIVE_TRAINING_COMPLETE",
        "initialization": "Noise relation T1/T3 V1 champion",
        "initialization_kind": initialization_kind,
        "shared_encoder_updated": True,
        "native_components": {
            "dataset": "dreams.utils.data.ContrastiveSpectraDataset",
            "model": "dreams.models.heads.heads.ContrastiveHead",
            "loss": "ContrastiveHead.step native cosine triplet margin",
            "optimizer": "ContrastiveHead.configure_optimizers native Adam",
            "custom_model": False, "custom_loss": False, "custom_optimizer": False,
            "custom_triplet_content": True,
        },
        "training": {
            **frozen,
            "optimizer_steps": int(trainer.global_step),
            "queries": int(len(selected_queries)),
            "events_selected": int(len(selected_positions)),
            "selected_generic_noise_events": int(np.sum(selected_kinds == 0)),
            "selected_exact_action_events": int(np.sum(selected_kinds == 1)),
            "query_dose": "every corrected MassSpecGym query exactly once",
            "same_query_events_share_batch": False,
            "elapsed_seconds": time.time() - started,
        },
        "dataset": dataset_report,
        "adam_continuation": {
            "restored": False,
            "reason": "the V1 champion is a weights-only slim checkpoint and contains no optimizer state",
        },
        "provenance": {
            "warm_start_checkpoint_sha256": sha256_file(args.warm_start_checkpoint),
            "triplet_report_sha256": sha256_file(args.triplet_dir / "report.json"),
            "train_pool_sha256": sha256_file(args.triplet_dir / "train_pool.npz"),
            "noise_spectra_sha256": sha256_file(args.triplet_dir / "noise_spectra.npz"),
            "stage1_action_bank_sha256": sha256_file(args.stage1_action_bank),
            "final_checkpoint_sha256": sha256_file(checkpoint),
        },
        "claim_limit": "Training completion only; performance is decided by the frozen GNPS panels.",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
