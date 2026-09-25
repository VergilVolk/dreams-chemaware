"""ChemAware-only compact checkpoint entry for specific replay.

The optimization runtime remains ``train_chemaware_dreams_native``.  This
process-local wrapper changes only Lightning checkpoint serialization so the
ChemAware residual run cannot consume the shared storage quota with optimizer
state and duplicate ``last.ckpt`` files.  Noise jobs continue to invoke the
shared runtime directly and never import this module.
"""
from __future__ import annotations

import time
from typing import Any

import train_chemaware_dreams_native as native


def _strip_module_hyperparameters(model: Any) -> list[str]:
    removed: set[str] = set()
    for attribute in ("_hparams", "_hparams_initial"):
        values = getattr(model, attribute, None)
        if values is None or not hasattr(values, "pop"):
            continue
        for key in ("backbone", "backbone_pth"):
            if key in values:
                values.pop(key)
                removed.add(key)
    return sorted(removed)


_SharedModelCheckpoint = native.pl.callbacks.ModelCheckpoint
_shared_arguments = native.arguments


class ChemAwareSlurmLineProgress(native.pl.Callback):
    """ChemAware-local progress callback; never changes the shared runtime."""

    def __init__(self, every_n_batches: int = 50):
        super().__init__()
        self.every_n_batches = int(every_n_batches)
        self.started = 0.0

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        self.started = time.monotonic()
        print(
            f"FIT_START device={pl_module.device} strategy={type(trainer.strategy).__name__}",
            flush=True,
        )

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        print(
            f"TRAIN_EPOCH_START epoch={trainer.current_epoch} "
            f"batches={trainer.num_training_batches}",
            flush=True,
        )

    def on_train_batch_start(
        self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int,
    ) -> None:
        if batch_idx == 0:
            print(f"TRAIN_FIRST_BATCH_START epoch={trainer.current_epoch}", flush=True)

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int,
    ) -> None:
        if batch_idx == 0 or (batch_idx + 1) % self.every_n_batches == 0:
            elapsed = time.monotonic() - self.started
            print(
                f"TRAIN_PROGRESS epoch={trainer.current_epoch} "
                f"batch={batch_idx + 1}/{trainer.num_training_batches} "
                f"global_step={trainer.global_step} elapsed_s={elapsed:.1f}",
                flush=True,
            )


class ChemAwareCompactModelCheckpoint(_SharedModelCheckpoint):
    """Checkpoint-only override; model training behavior is untouched."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["save_last"] = False
        kwargs["save_weights_only"] = True
        super().__init__(*args, **kwargs)

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        removed = _strip_module_hyperparameters(pl_module)
        print(
            "CHEMAWARE_COMPACT_CHECKPOINT "
            f"save_weights_only=true save_last=false removed_hparams={removed}",
            flush=True,
        )
        parent = getattr(super(), "on_train_start", None)
        if parent is not None:
            parent(trainer, pl_module)


def _chemaware_arguments():
    args = _shared_arguments()
    if args.checkpoint_mode != "fixed_steps":
        raise ValueError("ChemAware compact entry requires --checkpoint-mode fixed_steps")
    # Keep the shared runtime's report truthful when these optional fields are
    # available, without requiring Noise to use this wrapper.
    if hasattr(args, "checkpoint_save_weights_only"):
        args.checkpoint_save_weights_only = True
    if hasattr(args, "no_save_last_checkpoint"):
        args.no_save_last_checkpoint = True
    return args


def main() -> None:
    native.arguments = _chemaware_arguments
    native.SlurmLineProgress = ChemAwareSlurmLineProgress
    native.pl.callbacks.ModelCheckpoint = ChemAwareCompactModelCheckpoint
    native.main()


if __name__ == "__main__":
    main()
