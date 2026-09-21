"""Continue the official DreaMS contrastive model on ChemAware triplets.

The model, preprocessing, dynamic positive/negative sampling, loss and Adam
optimizer are the repository's native DreaMS implementations.  This file only
adapts HDF5 row-index candidate lists to the official dataset interface and
loads the official fine-tuned initialization.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from dreams.models.heads.heads import ContrastiveHead  # noqa: E402
from dreams.utils.data import ContrastiveSpectraDataset  # noqa: E402
from dreams.utils.spectra import MSnSpectrum  # noqa: E402
from train_e1_identity import load_base_model  # noqa: E402


class SlurmLineProgress(pl.Callback):
    """Newline-based progress that remains visible in redirected Slurm logs."""

    def __init__(self, every_n_batches: int = 50):
        super().__init__()
        self.every_n_batches = int(every_n_batches)
        self.started = 0.0

    def on_fit_start(self, trainer, pl_module) -> None:
        self.started = time.monotonic()
        print(
            f"FIT_START device={pl_module.device} strategy={type(trainer.strategy).__name__}",
            flush=True,
        )

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        print(
            f"TRAIN_EPOCH_START epoch={trainer.current_epoch} batches={trainer.num_training_batches}",
            flush=True,
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if batch_idx == 0:
            print(f"TRAIN_FIRST_BATCH_START epoch={trainer.current_epoch}", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx == 0 or (batch_idx + 1) % self.every_n_batches == 0:
            elapsed = time.monotonic() - self.started
            print(
                f"TRAIN_PROGRESS epoch={trainer.current_epoch} "
                f"batch={batch_idx + 1}/{trainer.num_training_batches} "
                f"global_step={trainer.global_step} elapsed_s={elapsed:.1f}",
                flush=True,
            )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--val-pool", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--triplet-loss-margin", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Keep zero for the one-GPU native run; CUDA-before-fork can deadlock workers.",
    )
    parser.add_argument("--max-epochs", type=int, default=301)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_pool(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        required = {"anchor_idx", "positive_ptr", "positive_idx", "negative_ptr", "negative_idx"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise RuntimeError(f"{path} lacks native triplet arrays: {missing}")
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def make_spectrum(raw: np.ndarray, precursor_mz: float) -> MSnSpectrum:
    raw = np.asarray(raw, dtype=np.float32)
    keep = (raw[0] > 0) & (raw[1] > 0)
    if not np.any(keep):
        raise RuntimeError("empty spectrum reached DreaMS contrastive dataset")
    return MSnSpectrum(
        peak_list=raw[:, keep], precursor_mz=float(precursor_mz),
        precursor_charge=1, assert_is_valid=False,
    )


def native_dataset(
    pool: dict[str, np.ndarray], data: Path, spec_preproc,
) -> tuple[ContrastiveSpectraDataset, np.ndarray]:
    anchors = np.asarray(pool["anchor_idx"], dtype=np.int64)
    reference_rows = np.unique(np.concatenate((
        pool["positive_idx"], pool["negative_idx"],
    )))
    rows = np.concatenate((anchors, reference_rows))
    with h5py.File(data, "r") as handle:
        if np.any((rows < 0) | (rows >= len(handle["spectrum"]))):
            raise RuntimeError("triplet dataset contains an out-of-range HDF5 row")
        spectra = [
            make_spectrum(handle["spectrum"][int(row)], handle["precursor_mz"][int(row)])
            for row in rows
        ]
    reference_position = {
        int(row): len(anchors) + index for index, row in enumerate(reference_rows)
    }
    positive_lists: list[list[int]] = []
    negative_lists: list[list[int]] = []
    for event in range(len(anchors)):
        p0, p1 = map(int, pool["positive_ptr"][event:event + 2])
        n0, n1 = map(int, pool["negative_ptr"][event:event + 2])
        positive_lists.append([
            reference_position[int(row)] for row in pool["positive_idx"][p0:p1]
        ])
        negative_lists.append([
            reference_position[int(row)] for row in pool["negative_idx"][n0:n1]
        ])
    frame = pd.DataFrame({
        "MSnSpectrum": spectra,
        "pos_idx": positive_lists + [[] for _ in reference_rows],
        "neg_idx": negative_lists + [[] for _ in reference_rows],
    })
    dataset = ContrastiveSpectraDataset(
        frame, spec_preproc=spec_preproc, n_pos_samples=1, n_neg_samples=1,
        return_smiles=False,
    )
    return dataset, np.arange(len(anchors), dtype=np.int64)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal native DreaMS continuation requires one CUDA GPU")
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    initialized, kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        torch.device("cpu"), args.n_highest_peaks,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Attribute 'backbone.*already saved during checkpointing.*",
        )
        model = ContrastiveHead(
            initialized.backbone, args.lr, args.weight_decay,
            triplet_loss_margin=args.triplet_loss_margin,
        )
    model.head.load_state_dict(initialized.head.state_dict(), strict=True)
    model.unfreeze_backbone_at_epoch = 0
    del initialized
    print("DreaMS model initialized; loading native triplet spectra", flush=True)
    if int(model.backbone.spec_preproc.n_highest_peaks) != args.n_highest_peaks:
        raise RuntimeError("DreaMS preprocessor peak count drifted from official configuration")
    if abs(float(model.backbone.spec_preproc.prec_intens) - 1.1) > 1e-12:
        raise RuntimeError("DreaMS precursor intensity is not the official 1.1")

    train_pool = load_pool(args.train_pool)
    val_pool = load_pool(args.val_pool)
    train_dataset, train_indices = native_dataset(
        train_pool, args.data, model.backbone.spec_preproc,
    )
    print(
        f"Training triplet spectra loaded: {len(train_indices)} events",
        flush=True,
    )
    val_dataset, val_indices = native_dataset(
        val_pool, args.data, model.backbone.spec_preproc,
    )
    print(
        f"Validation triplet spectra loaded: {len(val_indices)} events",
        flush=True,
    )
    train_loader = DataLoader(
        Subset(train_dataset, train_indices), batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices), batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0, pin_memory=True,
    )
    if not len(train_loader) or not len(val_loader):
        raise RuntimeError("native DreaMS triplet loader is empty")
    first_batch = next(iter(train_loader))
    if not {"spec", "pos_specs", "neg_specs"}.issubset(first_batch):
        raise RuntimeError("native DreaMS train loader returned an invalid batch")
    del first_batch
    # The preflight consumes sampler/Python RNG state; restore the formal seed
    # so the optimization trajectory is identical to a run without preflight.
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(
        f"Native train loader preflight passed: {len(train_loader)} batches/epoch; "
        "starting optimization",
        flush=True,
    )

    args.output.mkdir(parents=True)
    callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output, filename="best", monitor="Train loss",
        mode="min", save_top_k=1, save_last=True, every_n_train_steps=1000,
    )
    progress = SlurmLineProgress(every_n_batches=50)
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=args.max_epochs,
        precision="32-true", logger=False, callbacks=[callback, progress],
        num_sanity_val_steps=0, log_every_n_steps=5,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_path = Path(callback.best_model_path)
    if not best_path.is_file():
        raise RuntimeError("DreaMS native training produced no best checkpoint")
    canonical = args.output / "best.ckpt"
    if best_path.resolve() != canonical.resolve():
        shutil.copy2(best_path, canonical)
    report = {
        "status": "CHEMAWARE_DREAMS_NATIVE_TRAINING_COMPLETE",
        "custom_loss": False,
        "custom_model": False,
        "custom_optimizer": False,
        "custom_triplet_construction": True,
        "initialization_kind": kind,
        "native_components": {
            "dataset": "dreams.utils.data.ContrastiveSpectraDataset",
            "model": "dreams.models.heads.heads.ContrastiveHead",
            "preprocessor": type(model.backbone.spec_preproc).__name__,
            "optimizer": "ContrastiveHead.configure_optimizers (Adam)",
            "loss": "ContrastiveHead.step cosine triplet margin",
        },
        "official_hyperparameters": {
            "lr": args.lr, "batch_size": args.batch_size,
            "triplet_loss_margin": args.triplet_loss_margin,
            "n_pos_samples": 1, "n_neg_samples": 1,
            "n_highest_peaks": args.n_highest_peaks,
            "precision": 32, "unfreeze_backbone_at_epoch": 0,
            "max_epochs": args.max_epochs,
        },
        "train_triplets": int(len(train_indices)),
        "validation_triplets": int(len(val_indices)),
        "best_model_path": str(canonical.resolve()),
        "checkpoint_monitor": "Train loss every 1000 train steps",
        "best_train_loss": float(callback.best_model_score.cpu()),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
