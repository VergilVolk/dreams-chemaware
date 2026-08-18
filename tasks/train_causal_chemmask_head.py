"""Budget causal fine-tuning of the official DreaMS projection head.

This first formal stage keeps the official DreaMS backbone frozen and trains
only the 1024-to-1024 projection head. It combines three validated ingredients:

1. identity triplets from the exact 10-ppm pool;
2. peak-overlap enriched different-IK14 hard negatives;
3. same-IK14 views with condition-specific unmatched peaks masked.

Chemical rules never define identity or structure labels. The official head
output is used as a preservation teacher without a second backbone forward.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_e1_identity import (
    CandidatePool,
    cpu_state_dict,
    load_base_model,
    preprocess_spectrum,
    seed_everything,
)
from e1_checkpoint_io import official_head_state, torch_load_compat


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_TRAIN = ROOT / "data/e1/e1_train_triplet_pool_10ppm.npz"
DEFAULT_VAL = ROOT / "data/e1/e1_val_triplet_pool_10ppm.npz"
DEFAULT_OFFICIAL = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_RAW = ROOT / "dreams/models/pretrained/ssl_model_server.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train-pool", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-pool", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--base-ckpt", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--initial-head-ckpt", type=Path,
        help="Optional validated head checkpoint used to initialize the candidate head.",
    )
    parser.add_argument(
        "--resume", type=Path,
        help="Resume an interrupted causal-head run, including optimizer state.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/e1/causal_chemmask_head"
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--lambda-preserve", type=float, default=0.1)
    parser.add_argument("--hard-negative-prob", type=float, default=0.5)
    parser.add_argument("--negative-probe-size", type=int, default=8)
    parser.add_argument("--identity-mask-prob", type=float, default=0.3)
    parser.add_argument("--identity-mask-max-fraction", type=float, default=0.3)
    parser.add_argument("--identity-mask-max-peaks", type=int, default=12)
    parser.add_argument("--noise-mask-prob", type=float, default=0.5)
    parser.add_argument("--noise-mask-fraction", type=float, default=0.3)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-triplets", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--sequential-anchors", action=argparse.BooleanOptionalAction, default=False,
        help="Use deterministic pool-row order; required for exact mid-epoch resume.",
    )
    parser.add_argument(
        "--checkpoint-every-batches", type=int, default=0,
        help="Write a lightweight resumable checkpoint after this many train batches.",
    )
    parser.add_argument(
        "--save-all-epochs", action=argparse.BooleanOptionalAction, default=True,
        help="Save each lightweight projection head for downstream multi-metric selection.",
    )
    return parser.parse_args()


def sha256_prefix(path: Path, length: int = 16) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def raw_peaks(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(raw)
    keep = np.flatnonzero((values[0] > 0) & (values[1] > 0))
    order = np.argsort(values[0, keep], kind="stable")
    indices = keep[order]
    return values[0, indices].astype(float), values[1, indices].astype(float), indices


def greedy_peak_matches(mz_a: np.ndarray, mz_b: np.ndarray, tolerance: float):
    matches = []
    i = j = 0
    while i < len(mz_a) and j < len(mz_b):
        delta = float(mz_a[i] - mz_b[j])
        if abs(delta) <= tolerance:
            matches.append((i, j))
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return matches


def shared_major_score(raw_a: np.ndarray, raw_b: np.ndarray, tolerance: float) -> float:
    mz_a, intensity_a, _ = raw_peaks(raw_a)
    mz_b, intensity_b, _ = raw_peaks(raw_b)
    if not len(mz_a) or not len(mz_b):
        return -1.0
    matches = greedy_peak_matches(mz_a, mz_b, tolerance)
    if not matches:
        return 0.0
    matched_a = np.asarray([a for a, _ in matches], dtype=int)
    matched_b = np.asarray([b for _, b in matches], dtype=int)
    shared_a = float(intensity_a[matched_a].sum() / max(intensity_a.sum(), 1e-12))
    shared_b = float(intensity_b[matched_b].sum() / max(intensity_b.sum(), 1e-12))
    top_a = set(np.argsort(intensity_a)[-min(10, len(intensity_a)):].tolist())
    top_b = set(np.argsort(intensity_b)[-min(10, len(intensity_b)):].tolist())
    top_matches = sum(a in top_a and b in top_b for a, b in matches)
    top_fraction = top_matches / max(min(10, len(top_a), len(top_b)), 1)
    return min(shared_a, shared_b) + top_fraction


MASK_VAL = -1.0  # DreaMS pretraining mask token: set on BOTH m/z and intensity


def mask_unique_peaks(
    t_source: torch.Tensor,
    t_reference: torch.Tensor,
    tolerance: float,
    max_fraction: float,
    max_peaks: int,
) -> tuple[torch.Tensor, int]:
    """Mask the highest-intensity peaks of a PREPROCESSED spectrum that have no
    m/z match in the reference view, in pretraining style: both m/z and intensity
    are set to MASK_VAL on the already-normalized tensor.  Masking after
    normalization keeps the intervention a clean "delete this peak" instead of a
    re-normalization of every remaining peak.  Row 0 is the precursor (never
    masked); rows with zero intensity are padding (never masked)."""
    peaks_s = t_source[1:]
    peaks_r = t_reference[1:]
    mz_s = peaks_s[:, 0].numpy()
    int_s = peaks_s[:, 1].numpy()
    mz_r = peaks_r[:, 0].numpy()
    int_r = peaks_r[:, 1].numpy()
    src_rows = np.flatnonzero(int_s > 0)
    ref_rows = np.flatnonzero(int_r > 0)
    if len(src_rows) <= 3:
        return t_source, 0
    order_s = np.argsort(mz_s[src_rows], kind="stable")
    order_r = np.argsort(mz_r[ref_rows], kind="stable")
    sorted_s = src_rows[order_s]
    sorted_r = ref_rows[order_r]
    matches = greedy_peak_matches(mz_s[sorted_s], mz_r[sorted_r], tolerance)
    matched_src = {int(sorted_s[a]) for a, _ in matches}
    unique = np.asarray([row for row in src_rows if row not in matched_src], dtype=int)
    capacity = min(
        len(unique), max_peaks, int(np.ceil(max_fraction * len(src_rows))), len(src_rows) - 3
    )
    if capacity <= 0:
        return t_source, 0
    order = np.argsort(int_s[unique])[::-1][:capacity]
    mask_rows = unique[order]
    masked = t_source.clone()
    masked[1 + mask_rows, 0] = MASK_VAL
    masked[1 + mask_rows, 1] = MASK_VAL
    return masked, int(capacity)


def mask_noise(t: torch.Tensor, fraction: float, rng: np.random.RandomState) -> torch.Tensor:
    """Pretraining-style intensity-proportional peak masking (MaskedSpectraDataset:
    mask_val=-1, strategy 'intens_p').  Protects the precursor (row 0) and the
    base peak (intensity 1.0).  Returns the tensor unchanged when too few peaks
    are maskable."""
    peaks = t[1:]
    inten = peaks[:, 1].numpy()
    maskable = (inten > 0) & (inten < 1.0)
    idx = np.flatnonzero(maskable)
    n_peaks = len(idx)
    n_masks = max(2, int(round(n_peaks * fraction)))
    if n_peaks <= n_masks or n_masks == 0:
        return t
    sampling_p = inten[idx] / inten[idx].sum()
    chosen = rng.choice(idx, size=n_masks, p=sampling_p, replace=False)
    masked = t.clone()
    masked[1 + chosen, 0] = MASK_VAL
    masked[1 + chosen, 1] = MASK_VAL
    return masked


class CausalDynamicTripletDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        pool: CandidatePool,
        n_highest_peaks: int,
        seed: int,
        hard_negative_prob: float,
        negative_probe_size: int,
        identity_mask_prob: float,
        identity_mask_max_fraction: float,
        identity_mask_max_peaks: int,
        fragment_tolerance: float,
        noise_mask_prob: float = 0.0,
        noise_mask_fraction: float = 0.3,
        length: int | None = None,
        fixed: bool = False,
    ):
        self.data_path = str(data_path)
        self.pool = pool
        self.n_highest_peaks = n_highest_peaks
        self.seed = seed
        self.hard_negative_prob = hard_negative_prob
        self.negative_probe_size = negative_probe_size
        self.identity_mask_prob = identity_mask_prob
        self.identity_mask_max_fraction = identity_mask_max_fraction
        self.identity_mask_max_peaks = identity_mask_max_peaks
        self.fragment_tolerance = fragment_tolerance
        self.noise_mask_prob = noise_mask_prob
        self.noise_mask_fraction = noise_mask_fraction
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
        anchor_idx = int(self.pool.anchor_idx[pool_row])
        p0, p1 = self.pool.positive_ptr[pool_row:pool_row + 2]
        n0, n1 = self.pool.negative_ptr[pool_row:pool_row + 2]
        positive_idx = int(self.pool.positive_idx[rng.randint(p0, p1)])
        negative_candidates = self.pool.negative_idx[n0:n1]
        handle = self._handle()
        raw_anchor = np.asarray(handle["spectrum"][anchor_idx])
        raw_positive = np.asarray(handle["spectrum"][positive_idx])

        use_hard = (
            self.hard_negative_prob > 0
            and len(negative_candidates) > 1
            and rng.rand() < self.hard_negative_prob
        )
        hard_score = -1.0
        if use_hard:
            probe_n = min(self.negative_probe_size, len(negative_candidates))
            probe = rng.choice(negative_candidates, size=probe_n, replace=False)
            scores = np.asarray([
                shared_major_score(
                    raw_anchor, np.asarray(handle["spectrum"][int(candidate)]),
                    self.fragment_tolerance,
                )
                for candidate in probe
            ])
            best = int(np.argmax(scores))
            negative_idx = int(probe[best])
            hard_score = float(scores[best])
        else:
            negative_idx = int(negative_candidates[rng.randint(len(negative_candidates))])

        def prepared(raw: np.ndarray, row: int) -> torch.Tensor:
            return preprocess_spectrum(
                raw, float(handle["precursor_mz"][row]), self.n_highest_peaks
            )

        anchor = prepared(raw_anchor, anchor_idx)
        positive = prepared(raw_positive, positive_idx)

        masked_count = 0
        if self.identity_mask_prob > 0 and rng.rand() < self.identity_mask_prob:
            if rng.rand() < 0.5:
                anchor, masked_count = mask_unique_peaks(
                    anchor, positive, self.fragment_tolerance,
                    self.identity_mask_max_fraction, self.identity_mask_max_peaks,
                )
            else:
                positive, masked_count = mask_unique_peaks(
                    positive, anchor, self.fragment_tolerance,
                    self.identity_mask_max_fraction, self.identity_mask_max_peaks,
                )

        if self.noise_mask_prob > 0 and rng.rand() < self.noise_mask_prob:
            anchor = mask_noise(anchor, self.noise_mask_fraction, rng)
            positive = mask_noise(positive, self.noise_mask_fraction, rng)

        return {
            "anchor": anchor,
            "positive": positive,
            "negative": prepared(np.asarray(handle["spectrum"][negative_idx]), negative_idx),
            "anchor_idx": anchor_idx,
            "positive_idx": positive_idx,
            "negative_idx": negative_idx,
            "hard_negative_selected": bool(use_hard),
            "hard_negative_score": np.float32(hard_score),
            "identity_masked": bool(masked_count > 0),
            "masked_peak_count": int(masked_count),
        }

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


def forward_head(
    model,
    batch,
    teacher_weight: torch.Tensor,
    teacher_bias: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
):
    anchor = batch["anchor"].to(device, non_blocking=True)
    positive = batch["positive"].to(device, non_blocking=True)
    negative = batch["negative"].to(device, non_blocking=True)
    spectra = torch.cat((anchor, positive, negative), dim=0)
    with torch.no_grad():
        dtype = next(model.backbone.parameters()).dtype
        precursor = model.backbone(spectra.to(dtype=dtype), None)[:, 0, :]
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        current = F.normalize(model.head(precursor), dim=-1)
        teacher = F.normalize(F.linear(precursor, teacher_weight, teacher_bias), dim=-1)
        size = len(anchor)
        positive_cosine = (current[:size] * current[size:2 * size]).sum(dim=1)
        negative_cosine = (current[:size] * current[2 * size:]).sum(dim=1)
        teacher_positive = (teacher[:size] * teacher[size:2 * size]).sum(dim=1)
        teacher_negative = (teacher[:size] * teacher[2 * size:]).sum(dim=1)
        preserve = 1.0 - (current * teacher).sum(dim=1)
    return (
        current, positive_cosine, negative_cosine, preserve,
        teacher_positive, teacher_negative,
    )


def epoch_summary(losses, triplets, preserves, positives, negatives, metadata):
    loss = np.concatenate(losses)
    triplet = np.concatenate(triplets)
    preserve = np.concatenate(preserves)
    positive = np.concatenate(positives)
    negative = np.concatenate(negatives)
    return {
        "loss": float(loss.mean()),
        "triplet_loss": float(triplet.mean()),
        "preserve_loss": float(preserve.mean()),
        "positive_cosine": float(positive.mean()),
        "negative_cosine": float(negative.mean()),
        "separation": float((positive - negative).mean()),
        "triplet_accuracy": float((positive > negative).mean()),
        "hard_negative_fraction": float(np.mean(metadata["hard"])),
        "identity_mask_fraction": float(np.mean(metadata["masked"])),
        "mean_masked_peaks_when_applied": float(
            np.mean([value for value in metadata["mask_count"] if value > 0])
        ) if any(value > 0 for value in metadata["mask_count"]) else 0.0,
        "mean_hard_peak_score": float(
            np.mean([value for value in metadata["hard_score"] if value >= 0])
        ) if any(value >= 0 for value in metadata["hard_score"]) else None,
    }


def reference_triplet_summary(positives, negatives, margin: float) -> dict:
    positive = np.concatenate(positives)
    negative = np.concatenate(negatives)
    separation = positive - negative
    return {
        "positive_cosine": float(positive.mean()),
        "negative_cosine": float(negative.mean()),
        "separation": float(separation.mean()),
        "triplet_accuracy": float((separation > 0).mean()),
        "triplet_loss": float(np.maximum(0.0, margin - separation).mean()),
    }


def validate_args(args: argparse.Namespace) -> None:
    for name in ("hard_negative_prob", "identity_mask_prob", "noise_mask_prob"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not 0 < args.identity_mask_max_fraction < 1:
        raise ValueError("--identity-mask-max-fraction must be in (0, 1)")
    if not 0 < args.noise_mask_fraction < 1:
        raise ValueError("--noise-mask-fraction must be in (0, 1)")
    if args.negative_probe_size < 1 or args.identity_mask_max_peaks < 1:
        raise ValueError("Probe size and maximum masked peaks must be positive")
    if args.margin <= 0 or args.lambda_preserve < 0:
        raise ValueError("Margin must be positive and preservation weight non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use CPU only for a smoke test")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))
        try:
            torch.set_num_interop_threads(min(2, max(1, args.cpu_threads)))
        except RuntimeError:
            pass
    amp_enabled = bool(args.amp and device.type == "cuda")
    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading strict 10-ppm pools on {device}...", flush=True)
    train_pool, val_pool = CandidatePool(args.train_pool), CandidatePool(args.val_pool)
    train_dataset = CausalDynamicTripletDataset(
        args.data, train_pool, args.n_highest_peaks, args.seed,
        args.hard_negative_prob, args.negative_probe_size,
        args.identity_mask_prob, args.identity_mask_max_fraction,
        args.identity_mask_max_peaks, args.fragment_tolerance,
        args.noise_mask_prob, args.noise_mask_fraction,
    )
    val_dataset = CausalDynamicTripletDataset(
        args.data, val_pool, args.n_highest_peaks, args.seed + 97,
        0.0, 1, 0.0, args.identity_mask_max_fraction,
        args.identity_mask_max_peaks, args.fragment_tolerance,
        0.0, args.noise_mask_fraction,
        length=min(args.val_triplets, len(val_pool)), fixed=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=not args.sequential_anchors,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    print(
        "Loading the official DreaMS backbone and projection head. "
        "On Windows CPU this can remain at this line for several minutes...",
        flush=True,
    )
    load_started = time.time()
    model, initialization_kind = load_base_model(
        args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks
    )
    print(f"Model loaded in {time.time() - load_started:.1f}s ({initialization_kind}).", flush=True)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    model.backbone.eval()
    teacher_weight = model.head.weight.detach().clone()
    teacher_bias = model.head.bias.detach().clone()
    initialization_head_kind = "official_embedding_head"
    if args.initial_head_ckpt is not None:
        initial_package = torch_load_compat(args.initial_head_ckpt, map_location="cpu")
        model.head.load_state_dict(official_head_state(initial_package), strict=True)
        initialization_head_kind = str(initial_package.get("format", "external_head"))
        print(
            f"Initialized candidate head from {args.initial_head_ckpt} "
            f"({initialization_head_kind}); official head remains the preservation teacher.",
            flush=True,
        )
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update({
        "format": "causal_chemmask_head_v1",
        "initialization_kind": initialization_kind,
        "candidate_head_initialization": initialization_head_kind,
        "base_checkpoint_sha256": sha256_prefix(args.base_ckpt),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "loss": "triplet_identity + lambda_preserve * official_head_distillation",
        "label_policy": "IK14 identity only; peak evidence affects augmentation/sampling only",
    })
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"Causal head training: train={len(train_pool):,}, val={len(val_dataset):,}, "
        f"trainable={config['trainable_parameters']:,}/{config['total_parameters']:,}",
        flush=True,
    )

    history = []
    best_val = float("inf")
    stale = 0
    start_epoch = 0
    resume_batch = 0
    latest_resume_path = run_dir / "latest_resume.pt"
    if args.resume is not None:
        resume_package = torch_load_compat(args.resume, map_location="cpu")
        if resume_package.get("format") != "causal_chemmask_resume_v1":
            raise ValueError("--resume must be a causal_chemmask_resume_v1 checkpoint")
        model.head.load_state_dict(resume_package["head_state_dict"], strict=True)
        optimizer.load_state_dict(resume_package["optimizer_state_dict"])
        if amp_enabled and resume_package.get("scaler_state_dict"):
            scaler.load_state_dict(resume_package["scaler_state_dict"])
        start_epoch = int(resume_package["next_epoch"])
        resume_batch = int(resume_package.get("next_batch", 0))
        history = list(resume_package.get("history", []))
        best_val = float(resume_package.get("best_val", best_val))
        stale = int(resume_package.get("stale", 0))
        if resume_batch and not args.sequential_anchors:
            raise ValueError("Mid-epoch resume requires --sequential-anchors")
        print(
            f"Resuming epoch {start_epoch + 1}, train batch {resume_batch:,} from {args.resume}",
            flush=True,
        )

    def save_resume(next_epoch: int, next_batch: int) -> None:
        package = {
            "format": "causal_chemmask_resume_v1",
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "head_state_dict": cpu_state_dict(model.head),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if amp_enabled else None,
            "best_val": best_val,
            "stale": stale,
            "history": history,
            "config": config,
        }
        temporary = latest_resume_path.with_suffix(".tmp")
        torch.save(package, temporary)
        temporary.replace(latest_resume_path)

    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        train_dataset.set_epoch(epoch)
        model.head.train()
        optimizer.zero_grad(set_to_none=True)
        train_items = ([], [], [], [], [], {"hard": [], "masked": [], "mask_count": [], "hard_score": []})
        processed = 0
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            _, positive, negative, preserve, _, _ = forward_head(
                model, batch, teacher_weight, teacher_bias, device, amp_enabled
            )
            triplet = F.relu(args.margin - positive + negative)
            per_item = triplet + args.lambda_preserve * preserve.view(3, -1).mean(dim=0)
            loss = per_item.mean() / args.grad_accum
            scaler.scale(loss).backward()
            if (batch_index + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if (
                    args.checkpoint_every_batches
                    and (batch_index + 1) % args.checkpoint_every_batches == 0
                ):
                    save_resume(epoch, batch_index + 1)
                    print(
                        f"  Resume checkpoint: epoch {epoch + 1}, "
                        f"batch {batch_index + 1:,}",
                        flush=True,
                    )
            for destination, values in zip(train_items[:5], (per_item, triplet, preserve, positive, negative)):
                destination.append(values.detach().float().cpu().numpy())
            meta = train_items[5]
            meta["hard"].extend(batch["hard_negative_selected"].numpy().tolist())
            meta["masked"].extend(batch["identity_masked"].numpy().tolist())
            meta["mask_count"].extend(batch["masked_peak_count"].numpy().tolist())
            meta["hard_score"].extend(batch["hard_negative_score"].numpy().tolist())
            processed += 1
        if processed and processed % args.grad_accum:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        model.head.eval()
        val_items = ([], [], [], [], [], {"hard": [], "masked": [], "mask_count": [], "hard_score": []})
        official_val_positive, official_val_negative = [], []
        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                if args.max_val_batches and batch_index >= args.max_val_batches:
                    break
                _, positive, negative, preserve, teacher_positive, teacher_negative = forward_head(
                    model, batch, teacher_weight, teacher_bias, device, amp_enabled
                )
                triplet = F.relu(args.margin - positive + negative)
                per_item = triplet + args.lambda_preserve * preserve.view(3, -1).mean(dim=0)
                for destination, values in zip(val_items[:5], (per_item, triplet, preserve, positive, negative)):
                    destination.append(values.detach().float().cpu().numpy())
                official_val_positive.append(teacher_positive.detach().float().cpu().numpy())
                official_val_negative.append(teacher_negative.detach().float().cpu().numpy())
                meta = val_items[5]
                meta["hard"].extend(batch["hard_negative_selected"].numpy().tolist())
                meta["masked"].extend(batch["identity_masked"].numpy().tolist())
                meta["mask_count"].extend(batch["masked_peak_count"].numpy().tolist())
                meta["hard_score"].extend(batch["hard_negative_score"].numpy().tolist())

        train_metrics = epoch_summary(*train_items)
        val_metrics = epoch_summary(*val_items)
        official_val_metrics = reference_triplet_summary(
            official_val_positive, official_val_negative, args.margin
        )
        result = {
            "epoch": epoch + 1,
            "seconds": time.time() - start,
            "train": train_metrics,
            "val": val_metrics,
            "official_val_reference": official_val_metrics,
        }
        history.append(result)
        resume_batch = 0
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"Epoch {epoch + 1:02d} | train={train_metrics['loss']:.4f} "
            f"sep={train_metrics['separation']:.4f} hard={train_metrics['hard_negative_fraction']:.3f} "
            f"mask={train_metrics['identity_mask_fraction']:.3f} | "
            f"val={val_metrics['loss']:.4f} acc={val_metrics['triplet_accuracy']:.4f} "
            f"official_acc={official_val_metrics['triplet_accuracy']:.4f} "
            f"delta={val_metrics['triplet_accuracy'] - official_val_metrics['triplet_accuracy']:+.4f} "
            f"preserve={1.0 - val_metrics['preserve_loss']:.4f} | {result['seconds']:.0f}s",
            flush=True,
        )
        checkpoint = {
            "format": "causal_chemmask_head_v1",
            "epoch": epoch + 1,
            "base_checkpoint": str(args.base_ckpt.resolve()),
            "base_checkpoint_sha256": config["base_checkpoint_sha256"],
            "head_state_dict": cpu_state_dict(model.head),
            "config": config,
            "val_metrics": val_metrics,
            "official_val_reference": official_val_metrics,
            "history": history,
        }
        if args.save_all_epochs:
            torch.save(checkpoint, run_dir / f"epoch_{epoch + 1:02d}_causal_head.pt")
        if val_metrics["loss"] < best_val - 1e-6:
            best_val = val_metrics["loss"]
            stale = 0
            torch.save(checkpoint, run_dir / "best_causal_head.pt")
            print("  Saved best causal head", flush=True)
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"Early stopping after {args.patience} unimproved epochs")
                break
        save_resume(epoch + 1, 0)

    del model
    gc.collect()
    print(f"Complete. Best validation loss={best_val:.6f}")
    print(f"Checkpoint: {run_dir / 'best_causal_head.pt'}")


if __name__ == "__main__":
    main()
