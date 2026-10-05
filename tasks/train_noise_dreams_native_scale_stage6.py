"""Train the large Stage-6 corpus with the unmodified native DreaMS runtime."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from build_noise_dreams_native_scale_stage6 import (
    STAGE6_BUILDER_VERSION,
    STAGE6_MAXIMUM_OPTIMIZER_STEPS,
    STAGE6_STATUS,
)
from dreams.models.heads.heads import ContrastiveHead
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from train_noise_dreams_native import (
    load_npz,
    native_dataset,
    native_query_disjoint_one_pass_batches,
    sha256_file,
)
from train_noise_dreams_native_residual_stage2 import (
    RestoreNativeAdamState,
    adam_state_steps,
    construct_native_model,
)
from e1_checkpoint_io import torch_load_compat


STAGE6_TRAINER_VERSION = "noise_native_incremental_two_sided_trainer_v2"


class RestoreAndReleaseNativeAdamState(RestoreNativeAdamState):
    """Restore exact Stage-1 moments, then release the duplicate CPU payload."""

    def on_fit_start(self, trainer, pl_module) -> None:
        super().on_fit_start(trainer, pl_module)
        self.state = {}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--triplet-dir", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("targeted", "control"), required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--triplet-loss-margin", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-6 training requires an allocated GPU")
    expected = {
        "seed": 3407,
        "lr": 5e-6,
        "weight_decay": 0.0,
        "triplet_loss_margin": 0.1,
        "batch_size": 4,
        "max_epochs": 1,
        "n_highest_peaks": 100,
    }
    observed = {key: getattr(args, key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"Stage-6 native hyperparameters drifted: {observed}")

    paths = {
        "train_pool_sha256": args.triplet_dir / "train_pool.npz",
        "validation_pool_sha256": args.triplet_dir / "validation_pool.npz",
        "action_spectra_sha256": args.triplet_dir / "action_spectra.npz",
        "broad_events_sha256": args.triplet_dir / "broad_events.csv.gz",
    }
    for path in (
        args.data, args.warm_start_checkpoint, args.architecture_checkpoint,
        args.triplet_dir / "report.json", *paths.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    triplet_report = json.loads(
        (args.triplet_dir / "report.json").read_text(encoding="utf-8")
    )
    if (
        triplet_report.get("status") != STAGE6_STATUS
        or triplet_report.get("builder_version") != STAGE6_BUILDER_VERSION
        or triplet_report.get("training_initialization")
        != "stage1_targeted_champion_exact_continuation"
        or triplet_report.get("provenance", {}).get("stage1_checkpoint_sha256")
        != sha256_file(args.warm_start_checkpoint)
        or not triplet_report.get("gates")
        or not all(triplet_report["gates"].values())
    ):
        raise RuntimeError("Stage-6 triplet artifact did not pass its scientific gates")
    for key, path in paths.items():
        if triplet_report.get("output_artifacts", {}).get(key) != sha256_file(path):
            raise RuntimeError(f"Stage-6 artifact drifted: {key}")

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model, initialization_kind = construct_native_model(args)
    if type(model) is not ContrastiveHead:
        raise RuntimeError("Stage-6 did not construct the native ContrastiveHead")
    optimizer = model.configure_optimizers()
    if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
        raise RuntimeError("Stage-6 optimizer is not native torch.optim.Adam")
    group = optimizer.param_groups[0]
    if (
        abs(float(group["lr"]) - args.lr) > 1e-15
        or float(group["weight_decay"]) != 0.0
        or {id(parameter) for parameter in group["params"]}
        != {id(parameter) for parameter in model.parameters()}
    ):
        raise RuntimeError("Stage-6 native Adam parameter ownership drifted")
    del optimizer
    warm_package = torch_load_compat(args.warm_start_checkpoint, map_location="cpu")
    optimizer_states = warm_package.get("optimizer_states")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != 1:
        raise RuntimeError("Stage-1 champion lacks its single native Adam state")
    optimizer_restore = RestoreAndReleaseNativeAdamState(
        optimizer_states[0], args.lr,
    )
    del warm_package, optimizer_states

    preprocessor = SpectrumPreprocessor(
        dformat=DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    bank = load_npz(args.triplet_dir / "action_spectra.npz")
    action_key = (
        "targeted_action_spectra" if args.arm == "targeted"
        else "control_action_spectra"
    )
    actions = np.asarray(bank[action_key], dtype=np.float32)
    capable = np.asarray(bank["native_action_view_representable"], dtype=bool)
    role_code = np.asarray(bank["stage6_role_code"], dtype=np.int8)
    source_row = np.asarray(bank["stage6_source_row"], dtype=np.int64)
    source_formula = np.asarray(bank["stage6_source_formula"]).astype(str)
    stage1_action_count = int(np.asarray(bank["stage1_action_count"]).reshape(-1)[0])
    appendable_version = str(
        np.asarray(bank["appendable_library_version"]).reshape(-1)[0]
    )
    expected_role_code = np.concatenate((
        np.full(stage1_action_count, -1, dtype=np.int8),
        np.tile(
            np.asarray([0, 1, 2], dtype=np.int8),
            int(triplet_report["broad_two_sided_hard_events"]),
        ),
    ))
    if (
        capable.shape != (len(actions),)
        or role_code.shape != (len(actions),)
        or source_row.shape != (len(actions),)
        or source_formula.shape != (len(actions),)
        or not np.array_equal(role_code, expected_role_code)
        or np.any(source_row[stage1_action_count:] < 0)
        or np.any(source_formula[stage1_action_count:] == "")
        or stage1_action_count
        != int(triplet_report["stage1_action_spectra_retained"])
        or appendable_version != "noise_native_incremental_triplet_library_v1"
    ):
        raise RuntimeError("Stage-6 action bank is not aligned")
    train_pool = load_npz(args.triplet_dir / "train_pool.npz")
    validation_pool = load_npz(args.triplet_dir / "validation_pool.npz")
    registry_kind = np.asarray(train_pool["registry_kind"], dtype=np.int8)
    materialized = np.asarray(
        train_pool["registry_source_index"], dtype=np.int64,
    )[registry_kind == 1]
    if (
        np.any(materialized < 0)
        or np.any(materialized >= len(actions))
        or not np.all(capable[materialized])
    ):
        raise RuntimeError("Stage-6 materialized an unrepresentable action")

    train_dataset, train_indices, train_materialization = native_dataset(
        train_pool, args.data, actions, preprocessor,
    )
    validation_dataset, validation_indices, validation_materialization = native_dataset(
        validation_pool, args.data, actions, preprocessor,
    )
    train_batches, filler_events, schedule = native_query_disjoint_one_pass_batches(
        train_pool, batch_size=args.batch_size, seed=args.seed,
        event_dataset_indices=train_indices,
    )
    expected_semantic_units = int(
        triplet_report["stage1_action_spectra_retained"]
        + triplet_report["broad_two_sided_hard_events"]
    )
    if (
        schedule["every_base_event_exposed_exactly_once"] is not True
        or schedule["same_query_events_never_share_an_optimizer_batch"] is not True
        or schedule["query_balanced_oversampling"] is not False
        or schedule["synthetic_query_equalization_events"] != 0
        or schedule["forbidden_clean_to_action_events"] != 0
        or schedule["padding_is_final_batch_only"] is not True
        or schedule["minimum_action_events_per_semantic_unit"] != 1
        or schedule["maximum_action_events_per_semantic_unit"] != 1
        or schedule["semantic_action_units"] != expected_semantic_units
        or schedule["batches_per_epoch"] > STAGE6_MAXIMUM_OPTIMIZER_STEPS
    ):
        raise RuntimeError(f"Stage-6 native schedule failed: {schedule}")

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_batches, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0, pin_memory=True,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices), batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0, pin_memory=True,
    )
    args.output.mkdir(parents=True)
    callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output, filename="last", save_top_k=0, save_last=True,
        every_n_epochs=1, auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=1, precision="32-true",
        logger=False, callbacks=[callback, optimizer_restore], num_sanity_val_steps=0,
        log_every_n_steps=25, check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )
    trainer.validate(model, dataloaders=validation_loader)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
    if optimizer_restore.restored is not True:
        raise RuntimeError("Stage-6 did not restore the Stage-1 Adam state")
    final_checkpoint = args.output / "final.ckpt"
    last_checkpoint = Path(callback.last_model_path) if callback.last_model_path else args.output / "last.ckpt"
    if not last_checkpoint.is_file():
        raise RuntimeError("Stage-6 produced no native checkpoint")
    last_checkpoint.replace(final_checkpoint)
    checkpoint = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    if not {"state_dict", "optimizer_states", "epoch", "global_step"}.issubset(checkpoint):
        raise RuntimeError("Stage-6 final checkpoint is incomplete")
    if int(checkpoint["global_step"]) != int(schedule["batches_per_epoch"]):
        raise RuntimeError("Stage-6 optimizer step count differs from the frozen schedule")
    final_optimizer_states = checkpoint.get("optimizer_states")
    if not isinstance(final_optimizer_states, list) or len(final_optimizer_states) != 1:
        raise RuntimeError("Stage-6 final checkpoint lost its native Adam state")
    final_steps = adam_state_steps(final_optimizer_states[0])
    if set(final_steps) != set(optimizer_restore.initial_steps):
        raise RuntimeError("Stage-6 Adam parameter registry drifted")
    step_deltas = {
        parameter: final_steps[parameter] - optimizer_restore.initial_steps[parameter]
        for parameter in final_steps
    }
    if set(step_deltas.values()) != {int(schedule["batches_per_epoch"])}:
        raise RuntimeError("Stage-6 Adam did not advance by the exact one-pass schedule")

    report = {
        "status": "NOISE_DREAMS_NATIVE_SCALE_STAGE6_TRAINING_COMPLETE",
        "trainer_version": STAGE6_TRAINER_VERSION,
        "arm": args.arm,
        "training_runtime": {
            "dataset": "dreams.utils.data.ContrastiveSpectraDataset",
            "model": "dreams.models.heads.heads.ContrastiveHead",
            "loss": "native cosine triplet margin hinge",
            "optimizer": "native torch.optim.Adam",
            "custom_loss": False,
            "custom_model": False,
            "custom_optimizer": False,
            "custom_triplet_content": True,
        },
        "initialization": "stage1_targeted_champion_exact_continuation",
        "initialization_kind": initialization_kind,
        "warm_start_checkpoint_sha256": sha256_file(args.warm_start_checkpoint),
        "warm_start_tensor_equality_verified": True,
        "warm_start_adam_state_restored": True,
        "warm_start_adam_exact_continuation": {
            "registered_parameters": len(final_steps),
            "initial_step_min": min(optimizer_restore.initial_steps.values()),
            "initial_step_max": max(optimizer_restore.initial_steps.values()),
            "final_step_min": min(final_steps.values()),
            "final_step_max": max(final_steps.values()),
            "exact_step_advance": int(schedule["batches_per_epoch"]),
        },
        "frozen_hyperparameters": expected,
        "train_events": int(len(train_indices)),
        "validation_events": int(len(validation_indices)),
        "train_materialization": train_materialization,
        "validation_materialization": validation_materialization,
        "native_one_pass_schedule": schedule,
        "filler_event_indices": list(map(int, filler_events)),
        "triplet_report_sha256": sha256_file(args.triplet_dir / "report.json"),
        "runtime_local_final_checkpoint": str(final_checkpoint.resolve()),
        "final_checkpoint_artifact_name": "final.ckpt",
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "final_epoch": int(checkpoint["epoch"]),
        "final_global_step": int(checkpoint["global_step"]),
        "outer_performance_claimed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
