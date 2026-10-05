"""Continuous-Adam DreaMS fine-tuning with online ChemAware query packets.

Phase A is executed by the repository's mature Lightning/ContrastiveHead path.
The exact same Adam object is then retained while current molecule-max winners
are re-encoded and re-mined.  No checkpoint is loaded between phases and no
second optimizer is constructed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_online_packet_core import (  # noqa: E402
    DREAMS_NATIVE_REPLAY,
    allocate_role_steps,
    build_online_packet_pool,
    edges,
    phasea_role_fractions,
    role_entries,
)
from dreams.models.heads.heads import ContrastiveHead  # noqa: E402
from train_chemaware_dreams_native import (  # noqa: E402
    SlurmLineProgress,
    load_pool,
    make_spectrum,
    native_dataset,
)
from evaluate_chemaware_v2_direct_triplet import required_rows  # noqa: E402
from build_chemaware_dreams_native_triplets import audit_identity_edges  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--phasea-train-pool", type=Path, required=True)
    parser.add_argument("--phasea-val-pool", type=Path, required=True)
    parser.add_argument("--base-bank", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dreams-replay-pool", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--phasea-lr", type=float, default=5e-6)
    parser.add_argument("--online-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--triplet-loss-margin", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--phasea-steps", type=int, default=2000)
    parser.add_argument("--online-steps", type=int, default=1000)
    parser.add_argument("--remine-every-steps", type=int, default=250)
    parser.add_argument("--packet-width", type=int, default=3)
    parser.add_argument(
        "--online-schedule",
        choices=("query_packet", "phasea_role_budget"),
        default="query_packet",
    )
    parser.add_argument("--dreams-replay-events", type=int, default=1024)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_step_range(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    steps: list[int] = []
    for state in optimizer.state.values():
        raw = state.get("step")
        if raw is None:
            continue
        steps.append(int(raw.item() if torch.is_tensor(raw) else raw))
    if not steps:
        return 0, 0
    return min(steps), max(steps)


def save_weights(model: torch.nn.Module, path: Path, global_step: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    package = {
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "global_step": int(global_step),
        "chemaware_training": "continuous_adam_online_query_packet_v1",
    }
    torch.save(package, temporary)
    os.replace(temporary, path)
    del package
    gc.collect()
    return sha256(path)


def move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> None:
    """Move Adam moments without replacing the optimizer or its parameters."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def preprocess_rows(
    rows: np.ndarray, data: Path, n_highest_peaks: int,
    read_batch_size: int = 1024,
) -> torch.Tensor:
    """Preprocess the immutable candidate graph once for every online remine."""
    batches: list[torch.Tensor] = []
    with h5py.File(data, "r") as handle:
        for left in range(0, len(rows), read_batch_size):
            batch_rows = rows[left:left + read_batch_size]
            raw = np.asarray(handle["spectrum"][batch_rows])
            precursor = np.asarray(
                handle["precursor_mz"][batch_rows], dtype=np.float64,
            )
            batches.append(torch.stack([
                preprocess_spectrum(spectrum, float(mz), n_highest_peaks)
                for spectrum, mz in zip(raw, precursor, strict=True)
            ]))
    return torch.cat(batches, dim=0)


@torch.no_grad()
def encode_current_rows(
    model: ContrastiveHead, spectra: torch.Tensor,
    device: torch.device, batch_size: int,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    output: list[np.ndarray] = []
    for left in range(0, len(spectra), batch_size):
        batch = spectra[left:left + batch_size].to(device)
        encoded = F.normalize(model(batch), p=2, dim=-1)
        output.append(encoded.cpu().numpy())
    if was_training:
        model.train()
    result = np.concatenate(output).astype(np.float32, copy=False)
    norms = np.linalg.norm(result, axis=1)
    if np.any(~np.isfinite(result)) or np.max(np.abs(norms - 1.0)) > 2e-4:
        raise RuntimeError("online embeddings are non-finite or not normalized")
    return result


class OnlinePacketDataset(Dataset):
    """Native DreaMS spectra with a fixed number of negatives per query.

    Focused packets use every mined negative.  Official replay events retain
    native one-positive/one-negative sampling and are batched separately, so
    their loss and stochastic edge selection remain the mature DreaMS form.
    """

    def __init__(
        self, pool: Mapping[str, np.ndarray], data: Path, spec_preproc,
        packet_width: int, graph_rows: np.ndarray,
    ) -> None:
        self.spec_preproc = spec_preproc
        self.packet_width = int(packet_width)
        rows = np.unique(np.concatenate((
            np.asarray(graph_rows, dtype=np.int64),
            np.asarray(pool["anchor_idx"], dtype=np.int64),
            np.asarray(pool["positive_idx"], dtype=np.int64),
            np.asarray(pool["negative_idx"], dtype=np.int64),
        )))
        with h5py.File(data, "r") as handle:
            if np.any((rows < 0) | (rows >= len(handle["spectrum"]))):
                raise RuntimeError("online packet contains an out-of-range HDF5 row")
            self.spectra = {}
            for left in range(0, len(rows), 1024):
                batch_rows = rows[left:left + 1024]
                raw = np.asarray(handle["spectrum"][batch_rows])
                precursor = np.asarray(
                    handle["precursor_mz"][batch_rows], dtype=np.float64,
                )
                self.spectra.update({
                    int(row): make_spectrum(spectrum, float(mz))
                    for row, spectrum, mz in zip(
                        batch_rows, raw, precursor, strict=True,
                    )
                })
        self.replace_pool(pool)

    def replace_pool(self, pool: Mapping[str, np.ndarray]) -> None:
        required = {
            "anchor_idx", "positive_ptr", "positive_idx", "negative_ptr",
            "negative_idx", "event_kind",
        }
        missing = required - set(pool)
        if missing:
            raise RuntimeError(f"online packet pool lacks fields: {sorted(missing)}")
        rows = np.unique(np.concatenate((
            np.asarray(pool["anchor_idx"], dtype=np.int64),
            np.asarray(pool["positive_idx"], dtype=np.int64),
            np.asarray(pool["negative_idx"], dtype=np.int64),
        )))
        absent = [int(row) for row in rows if int(row) not in self.spectra]
        if absent:
            raise RuntimeError(f"remine introduced rows outside frozen graph: {absent[:10]}")
        self.pool = {key: np.asarray(value) for key, value in pool.items()}

    def __len__(self) -> int:
        return len(self.pool["anchor_idx"])

    def _preprocess(self, row: int) -> np.ndarray:
        spectrum = self.spectra[int(row)]
        return self.spec_preproc(
            spectrum.get_peak_list(), prec_mz=spectrum.get_precursor_mz(),
            high_form=False,
        )

    def __getitem__(self, event: int) -> dict[str, np.ndarray]:
        positives = edges(self.pool, int(event), "positive")
        negatives = edges(self.pool, int(event), "negative")
        replay = int(self.pool["event_kind"][event]) == DREAMS_NATIVE_REPLAY
        if replay:
            positive_rows = [random.choice(list(map(int, positives)))]
            negative = random.choice(list(map(int, negatives)))
            negative_rows = [negative]
        else:
            if len(positives) != 1 or len(negatives) != self.packet_width:
                raise RuntimeError("focused query packet changed its frozen shape")
            positive_rows = list(map(int, positives))
            negative_rows = list(map(int, negatives))
        return {
            "spec": self._preprocess(int(self.pool["anchor_idx"][event])),
            "pos_specs": np.stack([self._preprocess(row) for row in positive_rows]),
            "neg_specs": np.stack([self._preprocess(row) for row in negative_rows]),
        }

    def single_negative_item(self, event: int, negative_slot: int = 0) -> dict[str, np.ndarray]:
        positives = edges(self.pool, int(event), "positive")
        negatives = edges(self.pool, int(event), "negative")
        replay = int(self.pool["event_kind"][event]) == DREAMS_NATIVE_REPLAY
        positive = random.choice(list(map(int, positives)))
        if replay:
            negative = random.choice(list(map(int, negatives)))
        else:
            if len(positives) != 1 or not 0 <= negative_slot < len(negatives):
                raise RuntimeError("invalid focused role-triplet slot")
            negative = int(negatives[negative_slot])
        return {
            "spec": self._preprocess(int(self.pool["anchor_idx"][event])),
            "pos_specs": np.stack([self._preprocess(positive)]),
            "neg_specs": np.stack([self._preprocess(negative)]),
        }


class OnlineRoleDataset(Dataset):
    """One native single-negative triplet per role-balanced query event."""

    def __init__(self, parent: OnlinePacketDataset, entries: list[tuple[int, int]]):
        if not entries:
            raise RuntimeError("online role stream is empty")
        self.parent = parent
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        event, slot = self.entries[index]
        return self.parent.single_negative_item(event, slot)


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) for key, value in batch.items()
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal continuous-Adam training requires exactly one CUDA GPU")
    if args.num_workers != 0:
        raise ValueError("online mutable packet training requires --num-workers 0")
    if (
        args.phasea_steps < 1 or args.online_steps < 1
        or args.remine_every_steps < 1
        or args.online_steps % args.remine_every_steps
    ):
        raise ValueError("online steps must be a positive multiple of the remine interval")
    args.output.mkdir(parents=True)
    checkpoint_dir = args.output / "checkpoints"
    remine_dir = args.output / "remine"
    checkpoint_dir.mkdir()
    remine_dir.mkdir()

    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    initialized, kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        torch.device("cpu"), args.n_highest_peaks,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Attribute 'backbone.*already saved during checkpointing.*",
        )
        model = ContrastiveHead(
            initialized.backbone, args.phasea_lr, args.weight_decay,
            triplet_loss_margin=args.triplet_loss_margin,
        )
    model.head.load_state_dict(initialized.head.state_dict(), strict=True)
    model.unfreeze_backbone_at_epoch = 0
    del initialized

    phasea_train = load_pool(args.phasea_train_pool)
    phasea_val = load_pool(args.phasea_val_pool)
    role_fractions = phasea_role_fractions(phasea_train)
    train_dataset, train_indices = native_dataset(
        phasea_train, args.data, model.backbone.spec_preproc,
    )
    val_dataset, val_indices = native_dataset(
        phasea_val, args.data, model.backbone.spec_preproc,
    )
    train_loader = DataLoader(
        Subset(train_dataset, train_indices), batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices), batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=0, pin_memory=True,
    )
    first_batch = next(iter(train_loader))
    if not {"spec", "pos_specs", "neg_specs"}.issubset(first_batch):
        raise RuntimeError("native Phase-A loader returned an invalid batch")
    del first_batch
    pl.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # This is the exact mature DreaMS Phase-A training runtime.  Checkpointing
    # is deliberately disabled here; a compact weights-only artifact is saved
    # immediately after fit without touching the optimizer object.
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=10,
        max_steps=args.phasea_steps, precision="32-true", logger=False,
        callbacks=[SlurmLineProgress(every_n_batches=50)],
        num_sanity_val_steps=0, log_every_n_steps=5,
        enable_progress_bar=False, enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if int(trainer.global_step) != args.phasea_steps:
        raise RuntimeError(
            f"Phase A stopped at {trainer.global_step}, expected {args.phasea_steps}"
        )
    if len(trainer.optimizers) != 1:
        raise RuntimeError("mature DreaMS phase did not expose exactly one optimizer")
    optimizer = trainer.optimizers[0]
    if not isinstance(optimizer, torch.optim.Adam):
        raise RuntimeError(f"DreaMS optimizer drifted from Adam: {type(optimizer)}")
    if next(model.parameters()).device != device:
        model.to(device)
        move_optimizer_state(optimizer, device)
    phasea_optimizer_steps = optimizer_step_range(optimizer)
    if phasea_optimizer_steps != (args.phasea_steps, args.phasea_steps):
        raise RuntimeError(
            "Adam state does not prove an uninterrupted Phase-A trajectory: "
            f"{phasea_optimizer_steps}"
        )
    phasea_checkpoint = checkpoint_dir / f"step-{args.phasea_steps:06d}.ckpt"
    checkpoint_hashes = {
        str(args.phasea_steps): save_weights(model, phasea_checkpoint, args.phasea_steps)
    }
    print(
        "CONTINUOUS_ADAM_HANDOFF "
        f"global_step={args.phasea_steps} optimizer_step_range={phasea_optimizer_steps}",
        flush=True,
    )

    # The optimizer is not recreated.  Only its learning rate is reduced for
    # the online boundary-correction phase; first and second moments survive.
    for group in optimizer.param_groups:
        group["lr"] = args.online_lr
    model.train()
    model.backbone.unfreeze()

    base_report = json.loads(
        (args.base_bank / "report.json").read_text(encoding="utf-8")
    )
    if (
        base_report.get("status") != "CHEMAWARE_ACTION_HARD_NATIVE_TRIPLETS_COMPLETE"
        or base_report.get("roles", {}).get("optimization") != "formula roles 0-1"
        or base_report.get("roles", {}).get("model_validation") != "formula role 3"
        or base_report.get("roles", {}).get("outer") != "formula role 4 untouched"
    ):
        raise RuntimeError("base bank does not prove the frozen formula-role contract")
    base = load_npz(args.base_bank / "train_pool.npz")
    evidence = load_npz(args.evidence)
    manifest = load_npz(args.manifest)
    replay = load_npz(args.dreams_replay_pool)
    queries = np.asarray(evidence["query"], dtype=np.int64)
    graph_rows = required_rows(manifest, queries)
    print(
        f"Preprocessing immutable online candidate graph: {len(graph_rows)} spectra",
        flush=True,
    )
    graph_spectra = preprocess_rows(
        graph_rows, args.data, args.n_highest_peaks,
    )
    packet_dataset: OnlinePacketDataset | None = None
    remine_reports: list[dict[str, object]] = []
    online_losses: list[float] = []
    global_step = args.phasea_steps
    online_end = args.phasea_steps + args.online_steps
    started = time.monotonic()

    while global_step < online_end:
        current_embeddings = encode_current_rows(
            model, graph_spectra, device, args.encode_batch_size,
        )
        packet_pool, packet_report = build_online_packet_pool(
            manifest=manifest, evidence=evidence, base=base, replay=replay,
            embedding_rows=graph_rows, embeddings=current_embeddings,
            margin=args.triplet_loss_margin, packet_width=args.packet_width,
            replay_events=args.dreams_replay_events, seed=args.seed,
        )
        remine_step_dir = remine_dir / f"step_{global_step:06d}"
        remine_step_dir.mkdir()
        np.savez_compressed(remine_step_dir / "packet_pool.npz", **packet_pool)
        packet_report = dict(packet_report)
        packet_report["global_step_before_interval"] = int(global_step)
        packet_report["optimizer_step_range_before_interval"] = list(
            optimizer_step_range(optimizer)
        )
        if not remine_reports:
            packet_report["identity_audit"] = audit_identity_edges(
                packet_pool, args.data,
            )
        (remine_step_dir / "report.json").write_text(
            json.dumps(packet_report, indent=2), encoding="utf-8",
        )
        remine_reports.append(packet_report)
        print(json.dumps(packet_report, indent=2), flush=True)

        if packet_dataset is None:
            packet_dataset = OnlinePacketDataset(
                packet_pool, args.data, model.backbone.spec_preproc,
                args.packet_width, graph_rows,
            )
        else:
            packet_dataset.replace_pool(packet_pool)
        interval_end = min(global_step + args.remine_every_steps, online_end)
        interval_length = interval_end - global_step
        loaders: dict[str, DataLoader] = {}
        if args.online_schedule == "query_packet":
            focused_indices = np.flatnonzero(
                packet_pool["event_kind"] != DREAMS_NATIVE_REPLAY
            )
            replay_indices = np.flatnonzero(
                packet_pool["event_kind"] == DREAMS_NATIVE_REPLAY
            )
            loaders["focused"] = DataLoader(
                Subset(packet_dataset, focused_indices),
                batch_size=args.batch_size, shuffle=True,
                drop_last=True, num_workers=0, pin_memory=True,
            )
            loaders["replay"] = DataLoader(
                Subset(packet_dataset, replay_indices),
                batch_size=args.batch_size, shuffle=True,
                drop_last=True, num_workers=0, pin_memory=True,
            )
            replay_steps = int(round(
                interval_length * len(replay_indices) / len(packet_dataset)
            ))
            allocation = {
                "focused": interval_length - replay_steps,
                "replay": replay_steps,
            }
            stream_sizes = {
                "focused": int(len(focused_indices)),
                "replay": int(len(replay_indices)),
            }
        else:
            entries = role_entries(packet_pool)
            allocation = allocate_role_steps(role_fractions, interval_length)
            stream_sizes = {name: len(rows) for name, rows in entries.items()}
            loaders = {
                name: DataLoader(
                    OnlineRoleDataset(packet_dataset, rows),
                    batch_size=args.batch_size, shuffle=True,
                    drop_last=True, num_workers=0, pin_memory=True,
                )
                for name, rows in entries.items()
            }
        packet_report["online_schedule"] = args.online_schedule
        packet_report["stream_sizes"] = stream_sizes
        packet_report["optimizer_step_allocation"] = allocation
        (remine_step_dir / "report.json").write_text(
            json.dumps(packet_report, indent=2), encoding="utf-8",
        )
        iterators = {name: iter(loader) for name, loader in loaders.items()}
        schedule = np.concatenate([
            np.repeat(name, count) for name, count in allocation.items()
        ]).astype(object)
        np.random.default_rng(args.seed + global_step).shuffle(schedule)
        for role_name in schedule:
            role_name = str(role_name)
            try:
                batch = next(iterators[role_name])
            except StopIteration:
                iterators[role_name] = iter(loaders[role_name])
                batch = next(iterators[role_name])
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model.step(batch, global_step)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite online triplet loss at step {global_step}")
            loss.backward()
            optimizer.step()
            global_step += 1
            online_losses.append(float(loss.detach().cpu()))
            if global_step % 50 == 0 or global_step == interval_end:
                print(
                    "ONLINE_TRAIN_PROGRESS "
                    f"global_step={global_step}/{online_end} "
                    f"loss={online_losses[-1]:.8f} "
                    f"elapsed_s={time.monotonic() - started:.1f}",
                    flush=True,
                )
        expected_steps = (global_step, global_step)
        observed_steps = optimizer_step_range(optimizer)
        if observed_steps != expected_steps:
            raise RuntimeError(
                "Adam continuity failed after online interval: "
                f"expected={expected_steps} observed={observed_steps}"
            )
        checkpoint = checkpoint_dir / f"step-{global_step:06d}.ckpt"
        checkpoint_hashes[str(global_step)] = save_weights(model, checkpoint, global_step)

    report = {
        "status": "CHEMAWARE_CONTINUOUS_ADAM_ONLINE_PACKET_TRAINING_COMPLETE",
        "initialization_kind": kind,
        "shared_embedding": True,
        "reranker": False,
        "distillation": False,
        "custom_model": False,
        "custom_optimizer": False,
        "custom_loss": False,
        "optimizer_instances_constructed": 1,
        "optimizer_reloaded_from_checkpoint": False,
        "phasea_steps": int(args.phasea_steps),
        "online_steps": int(args.online_steps),
        "remine_every_steps": int(args.remine_every_steps),
        "phasea_lr": float(args.phasea_lr),
        "online_lr": float(args.online_lr),
        "batch_size": int(args.batch_size),
        "packet_width": int(args.packet_width),
        "online_schedule": args.online_schedule,
        "phasea_role_fractions": role_fractions,
        "triplet_margin": float(args.triplet_loss_margin),
        "phasea_optimizer_step_range": list(phasea_optimizer_steps),
        "final_optimizer_step_range": list(optimizer_step_range(optimizer)),
        "online_loss_mean": float(np.mean(online_losses)),
        "online_loss_last": float(online_losses[-1]),
        "checkpoints": {
            str(step): {
                "path": str((checkpoint_dir / f"step-{int(step):06d}.ckpt").resolve()),
                "sha256": digest,
            }
            for step, digest in checkpoint_hashes.items()
        },
        "remine_reports": remine_reports,
        "formula_roles_used_for_optimizer": [0, 1],
        "formula_roles_2_3_4_untouched": True,
        "native_components": {
            "phasea_dataset": "dreams.utils.data.ContrastiveSpectraDataset",
            "model": "dreams.models.heads.heads.ContrastiveHead",
            "optimizer": "single ContrastiveHead.configure_optimizers Adam instance",
            "loss": "ContrastiveHead.step cosine triplet margin",
        },
    }
    (args.output / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
