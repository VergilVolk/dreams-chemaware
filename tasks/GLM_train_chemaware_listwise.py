"""GLM listwise two-arm trainer (pre-registered 2026-09-30).

Trains the shared DreaMS embedding with the candidate-set listwise objective of
docs/GLM_CHEMAWARE_LISTWISE_TWO_ARM_PREREGISTRATION_20260930.md on the pool
built by ``GLM_build_chemaware_listwise_two_arm_pools.py``.  The loss, dataset
and collation live in ``GLM_chemaware_listwise_loss`` (pure torch); this module
adds the native DreaMS Lightning wiring.

The model, preprocessor, optimizer family and checkpoint format are the
repository's native DreaMS components: this module subclasses
``ContrastiveHead`` and replaces ONLY the loss step -- the pairwise triplet is
upgraded to a reference-count-corrected smooth-maximum candidate-group
cross-entropy that matches the molecule-max retrieval decision the frozen
evaluation measures.  Arm 1 and Arm 2 read the same pool file and differ ONLY
through the selected margin vector; every other array is hash-anchored.

Fail-closed contracts:
- the pool's shared arrays must hash to ``--expected-shared-sha256``;
- arm1 margins must be identically zero; arm2 nonzero margins must equal
  ``--expected-nonzero-margins``; positive molecules never carry a margin;
- the trainer refuses existing outputs, requires one CUDA GPU, and saves
  fixed-step checkpoints exactly like the native chain (step-{step:06d}.ckpt).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from dreams.models.heads.heads import ContrastiveHead  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402
from GLM_chemaware_listwise_loss import (  # noqa: E402
    GroupTensorDataset,
    collate_groups,
    listwise_group_loss,
    shared_arrays_sha256,
)

TRAINER_STATUS = "GLM_LISTWISE_TWO_ARM_TRAINING_COMPLETE"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--arm", choices=("arm1", "arm2"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shared-sha256", required=True)
    parser.add_argument("--expected-nonzero-margins", type=int, required=True)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--save-every-n-steps", type=int, default=900)
    parser.add_argument("--maximum-spectra-per-batch", type=int, default=32)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class GLMListwiseCandidateHead(ContrastiveHead):
    """Native DreaMS contrastive head with the listwise candidate loss."""

    def __init__(self, backbone, lr: float, weight_decay: float,
                 tau: float, temperature: float):
        super().__init__(backbone, lr, weight_decay, triplet_loss_margin=0.1)
        self.listwise_tau = float(tau)
        self.listwise_temperature = float(temperature)

    def listwise_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        query_embeddings = self(batch["spec"], charge=None)
        reference_embeddings = self(batch["ref_specs"], charge=None)
        return listwise_group_loss(
            query_embeddings, reference_embeddings,
            batch["molecule_of_reference"], batch["slot_of_reference"],
            batch["reference_counts"], batch["margins"], batch["group_ptr"],
            self.listwise_tau, self.listwise_temperature,
        )

    def training_step(self, batch, batch_idx):
        loss = self.listwise_step(batch)
        self.log("Train loss", loss, sync_dist=True, batch_size=len(batch["spec"]))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.listwise_step(batch)
        self.log("Val loss", loss, sync_dist=True, batch_size=len(batch["spec"]))
        return loss


class SlurmLineProgress(pl.Callback):
    """Newline-based progress for redirected Slurm logs."""

    def __init__(self, every_n_batches: int = 50):
        super().__init__()
        self.every_n_batches = int(every_n_batches)
        self.started = 0.0

    def on_fit_start(self, trainer, pl_module) -> None:
        self.started = time.monotonic()
        print(
            f"FIT_START device={pl_module.device} "
            f"strategy={type(trainer.strategy).__name__}",
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx == 0 or (batch_idx + 1) % self.every_n_batches == 0:
            elapsed = time.monotonic() - self.started
            print(
                f"TRAIN_PROGRESS batch={batch_idx + 1}/"
                f"{trainer.num_training_batches} "
                f"global_step={trainer.global_step} elapsed_s={elapsed:.1f}",
                flush=True,
            )


def load_pool(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        required = {
            "query_row", "group_ptr", "molecule_ref_ptr", "ref_row",
            "molecule_label", "arm1_margin", "arm2_margin", "val_query_mask",
        }
        missing = sorted(required - set(loaded.files))
        if missing:
            raise RuntimeError(f"pool lacks required arrays: {missing}")
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the two-arm listwise run requires one CUDA GPU")
    if (
        args.tau <= 0 or args.temperature <= 0 or args.batch_size < 1
        or args.maximum_spectra_per_batch < 3
    ):
        raise ValueError("invalid listwise hyperparameters")
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    pool = load_pool(args.pool)
    digest = shared_arrays_sha256(pool)
    if digest != args.expected_shared_sha256:
        raise RuntimeError(
            f"pool shared-array hash drift: expected={args.expected_shared_sha256} "
            f"observed={digest}"
        )
    margins = np.asarray(pool[f"{args.arm}_margin"], dtype=np.float32)
    nonzero = int(np.count_nonzero(margins))
    if args.arm == "arm1" and nonzero != 0:
        raise RuntimeError("arm1 margins must be identically zero")
    if args.arm == "arm2" and nonzero != int(args.expected_nonzero_margins):
        raise RuntimeError(
            f"arm2 nonzero margins drifted: expected={args.expected_nonzero_margins} "
            f"observed={nonzero}"
        )
    labels = np.asarray(pool["molecule_label"], dtype=np.int8)
    if np.any(margins[labels == 1] != 0):
        raise RuntimeError("a positive molecule carries a chemical margin")

    initialized, kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        torch.device("cpu"), args.n_highest_peaks,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Attribute 'backbone.*already saved during checkpointing.*",
        )
        model = GLMListwiseCandidateHead(
            initialized.backbone, args.lr, args.weight_decay,
            args.tau, args.temperature,
        )
    model.head.load_state_dict(initialized.head.state_dict(), strict=True)
    model.unfreeze_backbone_at_epoch = 0
    del initialized
    if int(model.backbone.spec_preproc.n_highest_peaks) != args.n_highest_peaks:
        raise RuntimeError(
            "DreaMS preprocessor peak count drifted from official configuration"
        )

    val_mask = np.asarray(pool["val_query_mask"], dtype=bool)
    train_indices = np.flatnonzero(~val_mask).tolist()
    val_indices = np.flatnonzero(val_mask).tolist()
    if len(train_indices) < args.batch_size or not len(val_indices):
        raise RuntimeError("validation split leaves an empty train or val set")
    group_ptr = np.asarray(pool["group_ptr"], dtype=np.int64)
    molecule_ref_ptr = np.asarray(pool["molecule_ref_ptr"], dtype=np.int64)
    spectra_per_group = np.asarray([
        1 + int(
            molecule_ref_ptr[int(group_ptr[index + 1])]
            - molecule_ref_ptr[int(group_ptr[index])]
        )
        for index in train_indices
    ], dtype=np.int64)
    worst_case_batch = int(np.sort(spectra_per_group)[-args.batch_size:].sum())
    if worst_case_batch > args.maximum_spectra_per_batch:
        raise RuntimeError(
            "configured query-group batch can encode up to "
            f"{worst_case_batch} spectra, above the one-GPU safety limit "
            f"{args.maximum_spectra_per_batch}"
        )

    rows_needed = np.unique(np.concatenate((
        np.asarray(pool["query_row"], dtype=np.int64),
        np.asarray(pool["ref_row"], dtype=np.int64),
    )))
    spectra: dict[int, torch.Tensor] = {}
    with h5py.File(args.data, "r") as handle:
        total = int(len(handle["spectrum"]))
        if np.any((rows_needed < 0) | (rows_needed >= total)):
            raise RuntimeError("pool references an out-of-range HDF5 row")
        for row in rows_needed:
            spectra[int(row)] = preprocess_spectrum(
                np.asarray(handle["spectrum"][int(row)]),
                float(handle["precursor_mz"][int(row)]),
                args.n_highest_peaks,
            )
    dataset = GroupTensorDataset(
        pool["query_row"], pool["group_ptr"], pool["molecule_ref_ptr"],
        pool["ref_row"], margins, spectra,
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=args.num_workers,
        collate_fn=collate_groups, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=args.num_workers,
        collate_fn=collate_groups, pin_memory=True,
    )
    first_batch = next(iter(train_loader))
    if first_batch["spec"].shape[0] != args.batch_size:
        raise RuntimeError("train loader batch shape drifted")
    if int(first_batch["reference_counts"].shape[0]) != int(
        first_batch["group_ptr"][-1]
    ):
        raise RuntimeError("collated molecule layout is inconsistent")
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output.mkdir(parents=True)
    callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output, filename="step-{step:06d}",
        auto_insert_metric_name=False, save_top_k=-1, save_last=False,
        every_n_train_steps=args.save_every_n_steps,
        save_on_train_epoch_end=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        precision="32-true", logger=False,
        callbacks=[callback, SlurmLineProgress(every_n_batches=50)],
        num_sanity_val_steps=0, log_every_n_steps=5,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    checkpoint_paths = sorted(args.output.glob("step-*.ckpt"))
    if not checkpoint_paths:
        raise RuntimeError("fixed-step listwise training produced no checkpoint")
    report = {
        "status": TRAINER_STATUS,
        "arm": args.arm,
        "preregistration": (
            "docs/GLM_CHEMAWARE_LISTWISE_TWO_ARM_PREREGISTRATION_20260930.md"
        ),
        "objective": (
            "reference-count-corrected smooth-maximum candidate-group "
            "cross-entropy (molecule 0 target), native DreaMS components"
        ),
        "initialization_kind": kind,
        "native_components": {
            "model_base": "dreams.models.heads.heads.ContrastiveHead",
            "preprocessor": type(model.backbone.spec_preproc).__name__,
            "optimizer": "ContrastiveHead.configure_optimizers (Adam)",
        },
        "hyperparameters": {
            "tau": args.tau, "temperature": args.temperature,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "max_steps": args.max_steps,
            "max_epochs": args.max_epochs, "seed": args.seed,
            "n_highest_peaks": args.n_highest_peaks,
            "maximum_spectra_per_batch": args.maximum_spectra_per_batch,
        },
        "pool": {
            "path": str(args.pool.resolve()),
            "shared_arrays_sha256": digest,
            "training_groups": int(len(train_indices)),
            "validation_groups": int(len(val_indices)),
            "arm1_margin_all_zero": bool(np.all(
                np.asarray(pool["arm1_margin"], dtype=np.float32) == 0,
            )),
            "arm2_nonzero_margins": int(np.count_nonzero(
                np.asarray(pool["arm2_margin"], dtype=np.float32),
            )),
            "selected_margin_nonzero": nonzero,
            "maximum_training_spectra_per_group": int(spectra_per_group.max()),
            "mean_training_spectra_per_group": float(spectra_per_group.mean()),
        },
        "checkpoint_mode": "fixed_steps",
        "checkpoint_paths": [str(path.resolve()) for path in checkpoint_paths],
        "fairness_contract": (
            "both arms share the hash-anchored pool, initialization, seed, "
            "schedule and batching; only the margin vector differs"
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
