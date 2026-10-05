"""Short native-DreaMS continuation from the frozen Noise Stage-1 champion.

Only triplet membership and action spectra are Noise-specific. The model,
preprocessor, dynamic one-positive/one-negative dataset, cosine-margin loss and
Adam optimizer are the repository's unmodified DreaMS implementations.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from build_noise_dreams_native_residual_stage2 import STAGE2_BUILDER_VERSION
from build_noise_dreams_native_multidifficulty_stage3 import STAGE3_BUILDER_VERSION
from repair_noise_dreams_native_multidifficulty_stage3 import (
    STAGE3_REPAIR_BUILDER_VERSION,
    STAGE3_REPAIR_STATUS,
)
from build_noise_dreams_native_broad_positive_stage4 import (
    STAGE4_BUILDER_VERSION,
    STAGE4_OPTIMIZER_STEPS,
    STAGE4_STATUS,
)
from build_noise_dreams_native_negative_residual_stage5 import (
    STAGE5_BUILDER_VERSION,
    STAGE5_MAXIMUM_OPTIMIZER_STEPS,
    STAGE5_STATUS,
)
from dreams.models.heads.heads import ContrastiveHead
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from e1_checkpoint_io import official_backbone_state, official_head_state, torch_load_compat
from train_e1_identity import load_base_model
from train_noise_dreams_native import (
    load_npz,
    native_dataset,
    native_query_disjoint_one_pass_batches,
    sha256_file,
)


STAGE2_TRAINER_VERSION = "noise_native_champion_residual_continuation_v1"
STAGE3_TRAINER_VERSION = "noise_native_multidifficulty_continuation_v1"
STAGE3_REPAIR_TRAINER_VERSION = "noise_native_stage3_calibrated_continuation_v1"
STAGE4_TRAINER_VERSION = "noise_native_query_calibrated_residual_v2"
STAGE5_TRAINER_VERSION = "noise_native_action_mined_negative_continuation_v1"
STAGE2_MAX_OPTIMIZER_STEPS = 2000
STAGE3_MAX_OPTIMIZER_STEPS = 6000
STAGE3_REPAIR_MAX_OPTIMIZER_STEPS = 2500

CURRICULA = {
    "residual_stage2": {
        "status": "NOISE_DREAMS_NATIVE_RESIDUAL_STAGE2_COMPLETE",
        "builder_version": STAGE2_BUILDER_VERSION,
        "trainer_version": STAGE2_TRAINER_VERSION,
        "training_status": "NOISE_DREAMS_NATIVE_RESIDUAL_STAGE2_TRAINING_COMPLETE",
        "lr": 1e-6,
        "maximum_steps": STAGE2_MAX_OPTIMIZER_STEPS,
        "extra_artifacts": {
            "protection_sentinels_sha256": "protection_sentinels.csv.gz",
        },
    },
    "multidifficulty_stage3": {
        "status": "NOISE_DREAMS_NATIVE_MULTIDIFFICULTY_STAGE3_COMPLETE",
        "builder_version": STAGE3_BUILDER_VERSION,
        "trainer_version": STAGE3_TRAINER_VERSION,
        "training_status": (
            "NOISE_DREAMS_NATIVE_MULTIDIFFICULTY_STAGE3_TRAINING_COMPLETE"
        ),
        "lr": 5e-6,
        "maximum_steps": STAGE3_MAX_OPTIMIZER_STEPS,
        "extra_artifacts": {},
    },
    "multidifficulty_stage3_repair": {
        "status": STAGE3_REPAIR_STATUS,
        "builder_version": STAGE3_REPAIR_BUILDER_VERSION,
        "trainer_version": STAGE3_REPAIR_TRAINER_VERSION,
        "training_status": (
            "NOISE_DREAMS_NATIVE_MULTIDIFFICULTY_STAGE3_REPAIR_TRAINING_COMPLETE"
        ),
        # This is a late continuation from the frozen Stage-1 champion.  The
        # successful Stage-2 continuation established 1e-6 as the safe scale;
        # the failed Stage-3 run had restored the five-times larger from-base
        # learning rate while also tripling action dose.
        "lr": 1e-6,
        "maximum_steps": STAGE3_REPAIR_MAX_OPTIMIZER_STEPS,
        "extra_artifacts": {},
    },
    "broad_positive_stage4": {
        "status": STAGE4_STATUS,
        "builder_version": STAGE4_BUILDER_VERSION,
        "trainer_version": STAGE4_TRAINER_VERSION,
        "training_status": "NOISE_DREAMS_NATIVE_BROAD_POSITIVE_STAGE4_TRAINING_COMPLETE",
        # This is a late continuation, not a restart from official DreaMS.
        # Stage-2 and the calibrated Stage-3 repair established 1e-6 as the
        # safe scale.  Keep native Adam but initialize a fresh continuation
        # state, exactly as those positive continuations did.
        "lr": 1e-6,
        "maximum_steps": STAGE4_OPTIMIZER_STEPS,
        "extra_artifacts": {
            "selected_triplets_sha256": "selected_triplets.csv.gz",
        },
    },
    "negative_residual_stage5": {
        "status": STAGE5_STATUS,
        "builder_version": STAGE5_BUILDER_VERSION,
        "trainer_version": STAGE5_TRAINER_VERSION,
        "training_status": (
            "NOISE_DREAMS_NATIVE_NEGATIVE_RESIDUAL_STAGE5_TRAINING_COMPLETE"
        ),
        # The user-requested intervention is triplet membership only.  Keep
        # the exact successful Stage-1 native Adam hyperparameters.
        "lr": 5e-6,
        "maximum_steps": STAGE5_MAXIMUM_OPTIMIZER_STEPS,
        "extra_artifacts": {},
    },
}


def adam_state_steps(state: dict) -> dict[int, int]:
    """Return the exact Adam step counter for every registered parameter."""
    body = state.get("state")
    if not isinstance(body, dict) or not body:
        raise RuntimeError("native Adam state has no registered parameters")
    output: dict[int, int] = {}
    for parameter, slot in body.items():
        if not isinstance(slot, dict) or "step" not in slot:
            raise RuntimeError("native Adam parameter state lacks a step counter")
        step = slot["step"]
        output[int(parameter)] = int(step.item() if torch.is_tensor(step) else step)
    return output


class RestoreNativeAdamState(pl.Callback):
    """Restore the Stage-1 Adam moments without resuming its completed loop."""

    def __init__(self, state: dict, expected_lr: float) -> None:
        super().__init__()
        self.state = state
        self.expected_lr = float(expected_lr)
        self.restored = False
        self.initial_steps: dict[int, int] = {}

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if len(trainer.optimizers) != 1:
            raise RuntimeError("Stage-4 expected exactly one native optimizer")
        optimizer = trainer.optimizers[0]
        if type(optimizer) is not torch.optim.Adam:
            raise RuntimeError("Stage-4 optimizer is not native torch.optim.Adam")
        optimizer.load_state_dict(self.state)
        self.initial_steps = adam_state_steps(optimizer.state_dict())
        if (
            len(optimizer.param_groups) != 1
            or abs(float(optimizer.param_groups[0]["lr"]) - self.expected_lr) > 1e-15
            or float(optimizer.param_groups[0]["weight_decay"]) != 0.0
        ):
            raise RuntimeError("restored Stage-1 Adam parameters drifted")
        self.restored = True


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--triplet-dir", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("targeted", "control"), required=True)
    parser.add_argument(
        "--curriculum", choices=tuple(CURRICULA), default="residual_stage2",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--triplet-loss-margin", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def construct_native_model(args: argparse.Namespace) -> tuple[ContrastiveHead, str]:
    """Reconstruct Stage-1 weights inside an otherwise native head."""
    initialized, kind = load_base_model(
        args.warm_start_checkpoint,
        args.architecture_checkpoint,
        torch.device("cpu"),
        args.n_highest_peaks,
    )
    model = ContrastiveHead(
        initialized.backbone,
        args.lr,
        args.weight_decay,
        triplet_loss_margin=args.triplet_loss_margin,
    )
    model.head.load_state_dict(initialized.head.state_dict(), strict=True)
    model.unfreeze_backbone_at_epoch = 0
    del initialized
    if type(model) is not ContrastiveHead or not isinstance(model.head, torch.nn.Linear):
        raise RuntimeError("Stage-2 did not construct the native ContrastiveHead")
    optimizer = model.configure_optimizers()
    if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
        raise RuntimeError("Stage-2 native optimizer drifted from torch.optim.Adam")
    group = optimizer.param_groups[0]
    if (
        abs(float(group["lr"]) - args.lr) > 1e-15
        or float(group["weight_decay"]) != args.weight_decay
        or {id(p) for p in group["params"]} != {id(p) for p in model.parameters()}
    ):
        raise RuntimeError("Stage-2 native Adam does not own the exact model")
    del optimizer
    package = torch_load_compat(args.warm_start_checkpoint, map_location="cpu")
    source_backbone = official_backbone_state(package)
    source_head = official_head_state(package)
    model_backbone = model.backbone.state_dict()
    model_head = model.head.state_dict()
    if set(model_backbone) != set(source_backbone) or set(model_head) != set(source_head):
        raise RuntimeError("Stage-2 reconstruction changed warm-start state keys")
    for label, observed, expected in (
        ("backbone", model_backbone, source_backbone),
        ("head", model_head, source_head),
    ):
        for key in observed:
            if not torch.equal(observed[key].cpu(), expected[key].cpu()):
                raise RuntimeError(f"Stage-2 warm-start tensor drifted: {label}.{key}")
    del package, source_backbone, source_head, model_backbone, model_head
    return model, kind


def main() -> None:
    args = arguments()
    contract = CURRICULA[args.curriculum]
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 native continuation requires an allocated GPU")
    frozen = {
        "seed": 3407,
        "lr": contract["lr"],
        "weight_decay": 0.0,
        "triplet_loss_margin": 0.1,
        "batch_size": 4,
        "max_epochs": 1,
        "n_highest_peaks": 100,
    }
    observed = {key: getattr(args, key) for key in frozen}
    if observed != frozen:
        raise RuntimeError(f"Stage-2 continuation settings drifted: {observed} != {frozen}")
    artifact_paths = {
        "train_pool_sha256": args.triplet_dir / "train_pool.npz",
        "validation_pool_sha256": args.triplet_dir / "validation_pool.npz",
        "action_spectra_sha256": args.triplet_dir / "action_spectra.npz",
        "selected_actions_sha256": args.triplet_dir / "selected_actions.csv.gz",
        **{
            key: args.triplet_dir / value
            for key, value in contract["extra_artifacts"].items()
        },
    }
    for path in (
        args.data, args.warm_start_checkpoint, args.architecture_checkpoint,
        args.triplet_dir / "report.json", *artifact_paths.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    triplet_report = json.loads(
        (args.triplet_dir / "report.json").read_text(encoding="utf-8")
    )
    if (
        triplet_report.get("status") != contract["status"]
        or triplet_report.get("builder_version") != contract["builder_version"]
        or not all(triplet_report.get("gates", {}).values())
        or triplet_report.get("provenance", {}).get("stage1_checkpoint_sha256")
        != sha256_file(args.warm_start_checkpoint)
    ):
        raise RuntimeError("Stage-2 residual triplets do not bind the champion")
    for key, path in artifact_paths.items():
        if triplet_report.get("output_artifacts", {}).get(key) != sha256_file(path):
            raise RuntimeError(f"Stage-2 residual artifact drifted: {key}")

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model, initialization_kind = construct_native_model(args)
    optimizer_restore = None
    preprocessor = SpectrumPreprocessor(
        dformat=DataFormatA(), prec_intens=1.1,
        n_highest_peaks=args.n_highest_peaks,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    bank = load_npz(args.triplet_dir / "action_spectra.npz")
    action_key = (
        "targeted_action_spectra" if args.arm == "targeted"
        else "control_action_spectra"
    )
    action_spectra = np.asarray(bank[action_key], dtype=np.float32)
    capable = np.asarray(bank["native_action_view_representable"], dtype=bool)
    if capable.shape != (len(action_spectra),):
        raise RuntimeError("Stage-2 action representability ledger is misaligned")
    if (
        args.curriculum not in {
            "broad_positive_stage4", "negative_residual_stage5",
        }
        and not np.all(capable)
    ):
        raise RuntimeError("Stage-2 contains an unrepresentable action")
    train_pool = load_npz(args.triplet_dir / "train_pool.npz")
    validation_pool = load_npz(args.triplet_dir / "validation_pool.npz")
    train_dataset, train_indices, train_materialization = native_dataset(
        train_pool, args.data, action_spectra, preprocessor,
    )
    validation_dataset, validation_indices, validation_materialization = native_dataset(
        validation_pool, args.data, action_spectra, preprocessor,
    )
    if args.curriculum == "negative_residual_stage5":
        if (
            int(train_materialization["action_spectra"]) != 0
            or int(validation_materialization["action_spectra"]) != 0
            or np.any(np.asarray(train_pool["registry_kind"], dtype=np.int8) == 1)
            or np.any(np.asarray(validation_pool["registry_kind"], dtype=np.int8) == 1)
        ):
            raise RuntimeError(
                "Stage-5 action tensor entered the native model-input registry"
            )
    batches, filler_positions, schedule = native_query_disjoint_one_pass_batches(
        train_pool, train_indices, batch_size=args.batch_size, seed=args.seed,
    )
    if (
        schedule["every_base_event_exposed_exactly_once"] is not True
        or schedule["same_query_events_never_share_an_optimizer_batch"] is not True
        or schedule["maximum_action_events_per_semantic_unit"] != 1
        or schedule["minimum_action_events_per_semantic_unit"] != 1
        or schedule["batches_per_epoch"] > contract["maximum_steps"]
        or schedule["semantic_action_units"] != len(action_spectra)
        or (
            contract.get("exact_steps") is not None
            and schedule["batches_per_epoch"] != contract["exact_steps"]
        )
    ):
        raise RuntimeError(f"Stage-2 residual schedule failed: {schedule}")
    train_loader = DataLoader(
        train_dataset, batch_sampler=batches,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices), batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0, pin_memory=True,
    )
    if not len(train_loader) or not len(validation_loader):
        raise RuntimeError("Stage-2 native loader is empty")

    args.output.mkdir(parents=True)
    callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output, filename="last", save_top_k=0, save_last=True,
        every_n_epochs=1, auto_insert_metric_name=False,
    )
    callbacks: list[pl.Callback] = [callback]
    if optimizer_restore is not None:
        callbacks.append(optimizer_restore)
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=1, precision="32-true",
        logger=False, callbacks=callbacks, num_sanity_val_steps=0,
        log_every_n_steps=5, check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )
    trainer.validate(model, dataloaders=validation_loader)
    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=validation_loader,
    )
    if optimizer_restore is not None and optimizer_restore.restored is not True:
        raise RuntimeError("Stage-4 did not restore the Stage-1 Adam state")
    last_checkpoint = Path(callback.last_model_path) if callback.last_model_path else (
        args.output / "last.ckpt"
    )
    final_checkpoint = args.output / "final.ckpt"
    if not last_checkpoint.is_file() or final_checkpoint.exists():
        raise RuntimeError("Stage-2 native final checkpoint state is invalid")
    last_checkpoint.replace(final_checkpoint)
    package = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    required = {"state_dict", "optimizer_states", "epoch", "global_step"}
    if missing := required - set(package):
        raise RuntimeError(f"Stage-2 checkpoint lacks {sorted(missing)}")
    if int(package["global_step"]) != len(batches):
        raise RuntimeError(
            "Stage-2 optimizer-step count differs from the one-pass schedule"
        )
    adam_continuation = None
    if optimizer_restore is not None:
        final_optimizer_states = package["optimizer_states"]
        if not isinstance(final_optimizer_states, list) or len(final_optimizer_states) != 1:
            raise RuntimeError("Stage-4 final checkpoint lost the native Adam state")
        final_steps = adam_state_steps(final_optimizer_states[0])
        if set(final_steps) != set(optimizer_restore.initial_steps):
            raise RuntimeError("Stage-4 final Adam parameter registry drifted")
        deltas = {
            parameter: final_steps[parameter] - optimizer_restore.initial_steps[parameter]
            for parameter in final_steps
        }
        if set(deltas.values()) != {len(batches)}:
            raise RuntimeError(
                "Stage-4 Adam state did not continue by the exact one-pass step count"
            )
        adam_continuation = {
            "registered_parameters": len(final_steps),
            "initial_step_min": min(optimizer_restore.initial_steps.values()),
            "initial_step_max": max(optimizer_restore.initial_steps.values()),
            "final_step_min": min(final_steps.values()),
            "final_step_max": max(final_steps.values()),
            "exact_step_advance": len(batches),
        }
    report = {
        "status": contract["training_status"],
        "trainer_version": contract["trainer_version"],
        "curriculum": args.curriculum,
        "arm": args.arm,
        "initialization_kind": initialization_kind,
        "warm_start_checkpoint_sha256": sha256_file(args.warm_start_checkpoint),
        "triplet_report_sha256": sha256_file(args.triplet_dir / "report.json"),
        "native_runtime": {
            "model": "dreams.models.heads.heads.ContrastiveHead",
            "dataset": "dreams.utils.data.ContrastiveSpectraDataset",
            "preprocessor": "dreams.utils.data.SpectrumPreprocessor",
            "loss": "native cosine triplet margin hinge",
            "optimizer": "native torch.optim.Adam",
            "custom_loss": False,
            "custom_optimizer": False,
            "warm_start_tensor_equality_verified": True,
            "warm_start_adam_state_restored": optimizer_restore is not None,
            "warm_start_adam_exact_continuation": adam_continuation,
        },
        "continuation_settings": frozen,
        "schedule": schedule,
        "padding_event_positions": list(map(int, filler_positions)),
        "train_materialization": train_materialization,
        "validation_materialization": validation_materialization,
        "final_global_step": int(package["global_step"]),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "outer_performance_claimed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
