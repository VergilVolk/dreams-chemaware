#!/usr/bin/env python
"""Continue Noise V1 with one WSE-mined native DreaMS triplet per query."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dreams.models.heads.heads import ContrastiveHead
from dreams.utils.data import ContrastiveSpectraDataset, SpectrumPreprocessor
from noise_final_core import sha256_file
from noise_massspecgym_full_triplet_core import query_disjoint_batches
from train_chemaware_dreams_native import SlurmLineProgress
from train_noise_dreams_native_residual_stage2 import construct_native_model
from train_noise_massspecgym_full_native import load_npz, native_dataset


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--triplet-dir", type=Path, required=True)
    parser.add_argument("--stage1-action-bank", type=Path, required=True)
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--triplet-loss-margin", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    frozen = {
        "lr": 5e-6, "weight_decay": 0.0, "triplet_loss_margin": 0.1,
        "batch_size": 4, "max_epochs": 1, "n_highest_peaks": 100,
    }
    if {key: getattr(args, key) for key in frozen} != frozen:
        raise RuntimeError("WSE-native DreaMS settings drifted")
    if args.seed not in {3407, 3408}:
        raise RuntimeError("WSE-native seed is outside the registered pair")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("WSE-native training requires an allocated GPU")
    paths = [
        args.data, args.triplet_dir / "train_pool.npz",
        args.triplet_dir / "noise_spectra.npz",
        args.triplet_dir / "selection_ledger.npz",
        args.triplet_dir / "report.json", args.stage1_action_bank,
        args.warm_start_checkpoint, args.architecture_checkpoint,
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    triplet_report = json.loads(
        (args.triplet_dir / "report.json").read_text(encoding="utf-8")
    )
    if triplet_report.get("status") != "NOISE_WSE_DISCORDANT_NATIVE_TRIPLETS_COMPLETE":
        raise RuntimeError("WSE-native triplet artifact is incomplete")

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed); np.random.seed(args.seed)
    pool = load_npz(args.triplet_dir / "train_pool.npz")
    noise = load_npz(args.triplet_dir / "noise_spectra.npz")
    action_bank = load_npz(args.stage1_action_bank)
    action_spectra = np.asarray(
        action_bank["targeted_action_spectra"], dtype=np.float32,
    )
    queries = np.asarray(pool["event_query"], dtype=np.int64)
    if not np.array_equal(queries, np.arange(len(queries), dtype=np.int64)):
        raise RuntimeError("formal WSE training requires one ordered event per query")

    model, initialization_kind = construct_native_model(args)
    if type(model) is not ContrastiveHead:
        raise RuntimeError("WSE route did not construct native ContrastiveHead")
    preprocessor = model.backbone.spec_preproc
    if (
        type(preprocessor) is not SpectrumPreprocessor
        or int(preprocessor.n_highest_peaks) != 100
        or abs(float(preprocessor.prec_intens) - 1.1) > 1e-12
    ):
        raise RuntimeError("WSE route preprocessor drifted from native DreaMS")
    dataset, event_indices, dataset_report = native_dataset(
        pool, args.data, np.asarray(noise["spectra"], dtype=np.float32),
        action_spectra, preprocessor,
    )
    selected = np.arange(len(queries), dtype=np.int64)
    batches = query_disjoint_batches(
        selected, queries, args.batch_size, args.seed, epoch=0,
    )
    dataset_batches = [
        [int(event_indices[position]) for position in batch] for batch in batches
    ]
    loader = DataLoader(
        dataset, batch_sampler=dataset_batches, num_workers=0, pin_memory=True,
    )
    first_batch = next(iter(loader))
    if not {"spec", "pos_specs", "neg_specs"}.issubset(first_batch):
        raise RuntimeError("native WSE loader returned an invalid batch")
    del first_batch

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed); np.random.seed(args.seed)
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
        raise RuntimeError("WSE-native optimizer-step count drifted")
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
    kinds = np.asarray(pool["event_kind"], dtype=np.int8)
    report = {
        "status": "NOISE_WSE_DISCORDANT_NATIVE_TRAINING_COMPLETE",
        "seed": int(args.seed),
        "initialization": "Noise relation T1/T3 V1 champion",
        "initialization_kind": initialization_kind,
        "shared_encoder_updated": True,
        "native_components": {
            "dataset": "dreams.utils.data.ContrastiveSpectraDataset",
            "model": "dreams.models.heads.heads.ContrastiveHead",
            "loss": "native cosine triplet margin",
            "optimizer": "native Adam",
            "custom_model": False, "custom_loss": False,
            "custom_optimizer": False, "custom_triplet_content": True,
        },
        "training": {
            **frozen,
            "optimizer_steps": int(trainer.global_step),
            "queries": int(len(queries)),
            "events_selected": int(len(selected)),
            "wse_supported_events": int(np.sum(kinds == 0)),
            "stage1_replay_events": int(np.sum(kinds == 1)),
            "measured_preservation_events": int(np.sum(kinds == 2)),
            "singleton_noise_preservation_events": int(np.sum(kinds == 3)),
            "query_dose": "every corrected MassSpecGym query exactly once",
            "same_query_events_share_batch": False,
            "elapsed_seconds": time.time() - started,
        },
        "dataset": dataset_report,
        "adam_continuation": {
            "restored": False,
            "reason": "V1 is a weights-only slim checkpoint",
        },
        "provenance": {
            "warm_start_checkpoint_sha256": sha256_file(args.warm_start_checkpoint),
            "triplet_report_sha256": sha256_file(args.triplet_dir / "report.json"),
            "train_pool_sha256": sha256_file(args.triplet_dir / "train_pool.npz"),
            "selection_ledger_sha256": sha256_file(args.triplet_dir / "selection_ledger.npz"),
            "stage1_action_bank_sha256": sha256_file(args.stage1_action_bank),
            "final_checkpoint_sha256": sha256_file(checkpoint),
        },
        "claim_limit": "Training completion only; performance is decided on frozen GNPS panels.",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
