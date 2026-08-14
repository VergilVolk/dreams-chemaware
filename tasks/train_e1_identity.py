"""E1: identity-only DreaMS hard-triplet fine-tuning.

This script deliberately contains no MCES, chemical-rule, masked-spectrum, or
embedding-preservation loss.  It is the clean fine-tuning baseline required
before E2.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from e1_checkpoint_io import (
    checkpoint_kind,
    official_backbone_state,
    official_head_state,
    torch_load_compat,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_DATA = REPO_ROOT / "data" / "models" / "MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_BASE = REPO_ROOT / "dreams" / "models" / "pretrained" / "ssl_model_server.pt"
DEFAULT_OFFICIAL = REPO_ROOT / "dreams" / "models" / "pretrained" / "embedding_model.ckpt"
DEFAULT_TRAIN_POOL = REPO_ROOT / "data" / "e1" / "e1_train_triplet_pool.npz"
DEFAULT_VAL_POOL = REPO_ROOT / "data" / "e1" / "e1_val_triplet_pool.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1 identity hard-triplet fine-tuning")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train-pool", type=Path, default=DEFAULT_TRAIN_POOL)
    parser.add_argument("--val-pool", type=Path, default=DEFAULT_VAL_POOL)
    parser.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--architecture-ckpt", type=Path, default=DEFAULT_BASE,
        help="Raw SSL server checkpoint used only to reconstruct DreaMS architecture",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "e1" / "runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-triplets", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="Debug limit per epoch; 0 uses the complete pool")
    parser.add_argument("--max-val-batches", type=int, default=0,
                        help="Debug limit per epoch; 0 uses --val-triplets")
    return parser.parse_args()


def sha256_prefix(path: Path, length: int = 16) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_spectrum(raw_2_n: np.ndarray, precursor_mz: float, n_highest: int) -> torch.Tensor:
    """Exact lightweight equivalent of SpectrumPreprocessor for this HDF5 layout."""
    raw = np.asarray(raw_2_n)
    highest = np.argsort(raw[1], kind="stable")[-n_highest:]
    highest = np.sort(highest)
    peaks = raw[:, highest].T.astype(np.float32, copy=True)
    if len(peaks) < n_highest:
        peaks = np.pad(peaks, ((0, n_highest - len(peaks)), (0, 0)))
    maximum = float(peaks[:, 1].max())
    if maximum > 0:
        peaks[:, 1] /= maximum
    precursor = np.asarray([[precursor_mz, 1.1]], dtype=np.float32)
    return torch.from_numpy(np.vstack((precursor, peaks)))


class CandidatePool:
    def __init__(self, path: Path):
        data = np.load(path)
        self.anchor_idx = data["anchor_idx"]
        self.positive_ptr = data["positive_ptr"]
        self.positive_idx = data["positive_idx"]
        self.negative_ptr = data["negative_ptr"]
        self.negative_idx = data["negative_idx"]
        if not (
            len(self.positive_ptr) == len(self.anchor_idx) + 1
            and len(self.negative_ptr) == len(self.anchor_idx) + 1
        ):
            raise ValueError(f"Corrupt candidate pool: {path}")

    def __len__(self) -> int:
        return len(self.anchor_idx)

    def sample(self, pool_row: int, rng: np.random.RandomState) -> tuple[int, int, int]:
        p0, p1 = self.positive_ptr[pool_row:pool_row + 2]
        n0, n1 = self.negative_ptr[pool_row:pool_row + 2]
        positive = self.positive_idx[rng.randint(p0, p1)]
        negative = self.negative_idx[rng.randint(n0, n1)]
        return int(self.anchor_idx[pool_row]), int(positive), int(negative)


class DynamicTripletDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        pool: CandidatePool,
        n_highest_peaks: int,
        seed: int,
        length: int | None = None,
        fixed: bool = False,
    ):
        self.data_path = str(data_path)
        self.pool = pool
        self.n_highest_peaks = n_highest_peaks
        self.seed = seed
        self.length = len(pool) if length is None else length
        self.fixed = fixed
        self.epoch = 0
        self._h5 = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def _handle(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.data_path, "r")
        return self._h5

    def __getitem__(self, item: int):
        pool_row = item % len(self.pool)
        epoch = 0 if self.fixed else self.epoch
        rng = np.random.RandomState(self.seed + epoch * 1_000_003 + item)
        anchor_idx, positive_idx, negative_idx = self.pool.sample(pool_row, rng)
        handle = self._handle()

        def load(row: int) -> torch.Tensor:
            return preprocess_spectrum(
                handle["spectrum"][row],
                float(handle["precursor_mz"][row]),
                self.n_highest_peaks,
            )

        return {
            "anchor": load(anchor_idx),
            "positive": load(positive_idx),
            "negative": load(negative_idx),
        }

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


class IdentityEmbeddingModel(nn.Module):
    """DreaMS backbone plus the paper's 1024-to-1024 linear embedding head."""
    def __init__(self, backbone: nn.Module, dimension: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(dimension, dimension, bias=True)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        precursor_embedding = self.backbone(spectra, None)[:, 0, :]
        return F.normalize(self.head(precursor_embedding), p=2, dim=-1)


def load_base_model(
    path: Path,
    architecture_path: Path,
    device: torch.device,
    n_highest_peaks: int,
):
    print("Loading DreaMS initialization checkpoint (first import may be slow)...", flush=True)
    package = torch_load_compat(path, map_location="cpu")
    kind = checkpoint_kind(package)
    print(f"  Initialization format: {kind}", flush=True)
    if kind == "raw_ssl":
        args_package = package
        backbone_state = package["state_dict"]
        head_state = None
    elif kind in ("official_embedding", "official_embedding_slim"):
        args_package = torch_load_compat(architecture_path, map_location="cpu")
        if checkpoint_kind(args_package) != "raw_ssl":
            raise ValueError("--architecture-ckpt must be the raw ssl_model_server.pt package")
        backbone_state = official_backbone_state(package)
        head_state = official_head_state(package)
        if not head_state:
            raise ValueError("Official embedding checkpoint contains no projection head")
    else:
        raise ValueError("An existing E1 output cannot be used as --base-ckpt")

    from dreams.models.dreams.dreams import DreaMS
    from dreams.utils.data import SpectrumPreprocessor
    from dreams.utils.dformats import DataFormatA
    print("  DreaMS modules imported", flush=True)

    model_args = Namespace(**args_package["args"])
    model_args.dformat = DataFormatA()
    for attribute in (
        "max_mz", "max_peaks_n", "max_tbxic_stdev", "min_peaks_n",
        "min_charge", "max_charge", "max_prec_mz", "high_intensity_thld",
        "min_intensity_ampl", "max_ms_level",
    ):
        if attribute in args_package["args"]:
            setattr(model_args.dformat, attribute, args_package["args"][attribute])
    model_args.d_graphormer_params = 0
    preprocessor = SpectrumPreprocessor(model_args.dformat, n_highest_peaks=n_highest_peaks)
    print("  Constructing the 116M-parameter backbone", flush=True)
    backbone = DreaMS(model_args, preprocessor)
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if missing or unexpected:
        print(f"Checkpoint load: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model = IdentityEmbeddingModel(backbone, int(model_args.d_model))
    if head_state is not None:
        model.head.load_state_dict(head_state, strict=True)
        print("  Reused the official contrastive projection head", flush=True)
    model = model.to(device)
    del package, args_package
    gc.collect()
    print(f"  Model ready: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    return model, kind


def forward_triplet(model, batch, device, amp_enabled):
    anchor = batch["anchor"].to(device, non_blocking=True)
    positive = batch["positive"].to(device, non_blocking=True)
    negative = batch["negative"].to(device, non_blocking=True)
    spectra = torch.cat((anchor, positive, negative), dim=0)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        embedding = model(spectra)
        batch_size = len(anchor)
        anchor_embedding = embedding[:batch_size]
        positive_embedding = embedding[batch_size:2 * batch_size]
        negative_embedding = embedding[2 * batch_size:]
        positive_cosine = (anchor_embedding * positive_embedding).sum(dim=1)
        negative_cosine = (anchor_embedding * negative_embedding).sum(dim=1)
    return embedding, positive_cosine, negative_cosine


def summarize_epoch(losses, positive, negative, margin, embeddings=None) -> dict[str, float]:
    losses = np.concatenate(losses)
    positive = np.concatenate(positive)
    negative = np.concatenate(negative)
    summary = {
        "loss": float(losses.mean()),
        "positive_cosine": float(positive.mean()),
        "negative_cosine": float(negative.mean()),
        "separation": float((positive - negative).mean()),
        "triplet_accuracy": float(np.mean(positive > negative)),
        "margin_satisfaction": float(np.mean(positive >= negative + margin)),
        "margin_violation_rate": float(np.mean(losses > 0)),
    }
    if embeddings:
        sample = torch.cat(embeddings, dim=0)[:512]
        if len(sample) >= 2:
            similarity = sample @ sample.T
            off_diagonal = similarity[~torch.eye(len(sample), dtype=torch.bool)]
            summary["pairwise_cosine_mean"] = float(off_diagonal.mean())
            summary["pairwise_cosine_std"] = float(off_diagonal.std())
            summary["mean_dimension_std"] = float(sample.std(dim=0).mean())
    return summary


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this environment has CPU-only PyTorch. "
            "Use --device cpu only for a smoke test, or run formal E1 on a CUDA machine."
        )
    device = torch.device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    if args.margin <= 0:
        raise ValueError("--margin must be positive")

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; AMP: {amp_enabled}; run: {run_dir}")

    train_pool = CandidatePool(args.train_pool)
    val_pool = CandidatePool(args.val_pool)
    train_dataset = DynamicTripletDataset(
        args.data, train_pool, args.n_highest_peaks, args.seed, fixed=False
    )
    val_length = min(args.val_triplets, len(val_pool))
    val_dataset = DynamicTripletDataset(
        args.data, val_pool, args.n_highest_peaks, args.seed + 97, length=val_length, fixed=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        # Workers must be recreated after set_epoch() so each epoch samples new
        # positive/negative candidates instead of retaining a stale epoch copy.
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    print(f"Training anchors: {len(train_pool):,}; fixed validation triplets: {len(val_dataset):,}")

    model, initialization_kind = load_base_model(
        args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config.update({
        "loss": "relu(margin - cosine(anchor, positive) + cosine(anchor, negative))",
        "loss_terms": ["identity"],
        "base_checkpoint_sha256": sha256_prefix(args.base_ckpt),
        "architecture_checkpoint": str(args.architecture_ckpt.resolve()),
        "initialization_kind": initialization_kind,
        "train_pool_size": len(train_pool),
        "val_pool_size": len(val_pool),
    })
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        start_time = time.time()
        train_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses, train_positive, train_negative = [], [], []
        processed_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            _, positive_cosine, negative_cosine = forward_triplet(
                model, batch, device, amp_enabled
            )
            per_triplet = F.relu(args.margin - positive_cosine + negative_cosine)
            loss = per_triplet.mean() / args.grad_accum
            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_losses.append(per_triplet.detach().cpu().numpy())
            train_positive.append(positive_cosine.detach().cpu().numpy())
            train_negative.append(negative_cosine.detach().cpu().numpy())
            processed_batches += 1

        # Do not silently discard the final partial accumulation window.
        if processed_batches and processed_batches % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        model.eval()
        val_losses, val_positive, val_negative, val_embeddings = [], [], [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if args.max_val_batches and batch_idx >= args.max_val_batches:
                    break
                embedding, positive_cosine, negative_cosine = forward_triplet(
                    model, batch, device, amp_enabled
                )
                per_triplet = F.relu(args.margin - positive_cosine + negative_cosine)
                val_losses.append(per_triplet.cpu().numpy())
                val_positive.append(positive_cosine.cpu().numpy())
                val_negative.append(negative_cosine.cpu().numpy())
                if sum(len(item) for item in val_embeddings) < 512:
                    val_embeddings.append(embedding.cpu())

        train_metrics = summarize_epoch(
            train_losses, train_positive, train_negative, args.margin
        )
        val_metrics = summarize_epoch(
            val_losses, val_positive, val_negative, args.margin, val_embeddings
        )
        epoch_result = {
            "epoch": epoch + 1,
            "seconds": time.time() - start_time,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_result)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss={train_metrics['loss']:.4f} sep={train_metrics['separation']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} sep={val_metrics['separation']:.4f} "
            f"acc={val_metrics['triplet_accuracy']:.4f} "
            f"margin_ok={val_metrics['margin_satisfaction']:.4f} | "
            f"{epoch_result['seconds']:.0f}s",
            flush=True,
        )

        if val_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            checkpoint = {
                "format": "e1_identity_v1",
                "epoch": epoch + 1,
                "base_checkpoint": str(args.base_ckpt.resolve()),
                "architecture_checkpoint": str(args.architecture_ckpt.resolve()),
                "base_checkpoint_sha256": config["base_checkpoint_sha256"],
                "backbone_state_dict": cpu_state_dict(model.backbone),
                "head_state_dict": cpu_state_dict(model.head),
                "config": config,
                "val_metrics": val_metrics,
                "history": history,
            }
            torch.save(checkpoint, run_dir / "best_e1.pt")
            print(f"  Saved best checkpoint (val loss={best_val_loss:.4f})", flush=True)
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} unimproved epochs")
                break

    print(f"E1 training complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {run_dir / 'best_e1.pt'}")


if __name__ == "__main__":
    main()
