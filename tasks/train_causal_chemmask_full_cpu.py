"""Full CPU causal ChemMask head fine-tuning with rich progress output.

Same training objective as ``train_causal_chemmask_head.py``: the official
DreaMS backbone stays frozen and only the 1024-to-1024 projection head is
trained, on identity triplets from the strict 10-ppm pool, with peak-overlap
enriched hard negatives and same-identity peak masking. The official head acts
as a preservation teacher (no second backbone forward).

This variant is hard-pinned to CPU and instrumented with per-batch progress,
throughput, and ETA so a long full-data run stays observable. The GPU switches
(``--device cuda`` / ``--amp``) are intentionally removed: on a CPU-only box
they would crash immediately and hide a multi-hour run behind one line per
epoch. Warm-starts from the validated counterfactual head by default.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from e1_checkpoint_io import official_head_state, torch_load_compat
from train_e1_identity import (
    CandidatePool,
    cpu_state_dict,
    load_base_model,
    seed_everything,
)
from train_causal_chemmask_head import (
    CausalDynamicTripletDataset,
    forward_head,
    epoch_summary,
    reference_triplet_summary,
    sha256_prefix,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_TRAIN = ROOT / "data/e1/e1_train_triplet_pool_10ppm.npz"
DEFAULT_VAL = ROOT / "data/e1/e1_val_triplet_pool_10ppm.npz"
DEFAULT_OFFICIAL = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_RAW = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_INITIAL_HEAD = (
    ROOT / "data/e1/counterfactual_formal/head/seed_20260813/best_counterfactual.pt"
)
DEFAULT_OUTPUT = ROOT / "data/e1/strict_counterfactual_full_cpu"


def log(message: str) -> None:
    """Timestamped, flushed progress line — visible even mid-pipe."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(round(seconds))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train-pool", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-pool", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--base-ckpt", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--initial-head-ckpt", type=Path, default=DEFAULT_INITIAL_HEAD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", type=Path, help="Resume from a latest_resume.pt.")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--hard-negative-prob", type=float, default=0.5)
    parser.add_argument("--negative-probe-size", type=int, default=32)
    parser.add_argument("--identity-mask-prob", type=float, default=0.3)
    parser.add_argument("--identity-mask-max-fraction", type=float, default=0.3)
    parser.add_argument("--identity-mask-max-peaks", type=int, default=12)
    parser.add_argument("--noise-mask-prob", type=float, default=0.5)
    parser.add_argument("--noise-mask-fraction", type=float, default=0.3)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-triplets", type=int, default=21163)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument(
        "--sequential-anchors", action=argparse.BooleanOptionalAction, default=True,
        help="Deterministic pool-row order; required for exact mid-epoch resume.",
    )
    parser.add_argument("--checkpoint-every-batches", type=int, default=500)
    parser.add_argument(
        "--save-all-epochs", action=argparse.BooleanOptionalAction, default=True,
        help="Save each lightweight projection head for downstream selection.",
    )
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--val-log-every-batches", type=int, default=200)
    return parser.parse_args()


def print_banner(args: argparse.Namespace) -> None:
    log("=" * 78)
    log("Causal ChemMask head — full CPU training")
    log(f"torch {torch.__version__} | CPU threads {args.cpu_threads} | "
        f"device cpu (CUDA disabled by design)")
    log(f"seed {args.seed} | epochs {args.epochs} | batch {args.batch_size} "
        f"| grad-accum {args.grad_accum} | lr {args.lr} | margin {args.margin} "
        f"| lambda-preserve {args.lambda_preserve}")
    log(f"initial head: {args.initial_head_ckpt}")
    log(f"output dir:   {args.output_dir / f'seed_{args.seed}'}")
    log("=" * 78)


def validate_args(args: argparse.Namespace) -> None:
    for name in ("hard_negative_prob", "identity_mask_prob", "noise_mask_prob"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not 0 < args.identity_mask_max_fraction < 1:
        raise ValueError("--identity-mask-max-fraction must be in (0, 1)")
    if not 0 < args.noise_mask_fraction < 1:
        raise ValueError("--noise-mask-fraction must be in (0, 1)")
    if args.negative_probe_size < 1 or args.identity_mask_max_peaks < 1:
        raise ValueError("Probe size and maximum masked peaks must be positive")
    if args.margin <= 0 or args.lambda_preserve < 0:
        raise ValueError("Margin must be positive and preservation weight non-negative")
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("Epochs, batch size, and grad-accum must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    # Hard-pin to CPU: no CUDA path exists here on purpose.
    torch.set_num_threads(max(1, args.cpu_threads))
    try:
        torch.set_num_interop_threads(min(2, max(1, args.cpu_threads)))
    except RuntimeError:
        pass
    device = torch.device("cpu")
    amp_enabled = False  # CPU only; autocast/AMP is intentionally off.

    print_banner(args)
    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log("Loading strict 10-ppm pools and building datasets...")
    t0 = time.time()
    train_pool = CandidatePool(args.train_pool)
    val_pool = CandidatePool(args.val_pool)
    log(f"  pools loaded in {time.time() - t0:.1f}s "
        f"(train anchors {len(train_pool):,}, val anchors {len(val_pool):,})")

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
        train_dataset, batch_size=args.batch_size,
        shuffle=not args.sequential_anchors, num_workers=args.num_workers,
        pin_memory=False, persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, persistent_workers=False,
    )
    total_train_batches = len(train_loader)
    if args.max_train_batches:
        total_train_batches = min(total_train_batches, args.max_train_batches)
    total_val_batches = len(val_loader)
    if args.max_val_batches:
        total_val_batches = min(total_val_batches, args.max_val_batches)
    log(f"  train batches/epoch {total_train_batches:,} | "
        f"val batches/epoch {total_val_batches:,}")

    log("Loading the official DreaMS backbone and projection head. "
        "On Windows CPU this can sit at this line for several minutes...")
    load_started = time.time()
    model, initialization_kind = load_base_model(
        args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks
    )
    log(f"  model loaded in {time.time() - load_started:.1f}s ({initialization_kind})")
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    model.backbone.eval()

    teacher_weight = model.head.weight.detach().clone()
    teacher_bias = model.head.bias.detach().clone()

    initialization_head_kind = "official_embedding_head"
    if args.initial_head_ckpt is not None:
        log(f"Warm-starting candidate head from {args.initial_head_ckpt}...")
        initial_package = torch_load_compat(args.initial_head_ckpt, map_location="cpu")
        model.head.load_state_dict(official_head_state(initial_package), strict=True)
        initialization_head_kind = str(initial_package.get("format", "external_head"))
        log(f"  candidate head initialized ({initialization_head_kind}); "
            f"official head remains the preservation teacher.")

    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
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
        "device": "cpu",
    })
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log(f"trainable {config['trainable_parameters']:,}/{config['total_parameters']:,} "
        f"parameters | config written to {run_dir / 'config.json'}")

    history: list[dict] = []
    best_val = float("inf")
    stale = 0
    start_epoch = 0
    resume_batch = 0
    latest_resume_path = run_dir / "latest_resume.pt"

    if args.resume is not None:
        log(f"Resuming from {args.resume}...")
        resume_package = torch_load_compat(args.resume, map_location="cpu")
        if resume_package.get("format") != "causal_chemmask_resume_v1":
            raise ValueError("--resume must be a causal_chemmask_resume_v1 checkpoint")
        model.head.load_state_dict(resume_package["head_state_dict"], strict=True)
        optimizer.load_state_dict(resume_package["optimizer_state_dict"])
        start_epoch = int(resume_package["next_epoch"])
        resume_batch = int(resume_package.get("next_batch", 0))
        history = list(resume_package.get("history", []))
        best_val = float(resume_package.get("best_val", best_val))
        stale = int(resume_package.get("stale", 0))
        if resume_batch and not args.sequential_anchors:
            raise ValueError("Mid-epoch resume requires --sequential-anchors")
        log(f"  resuming epoch {start_epoch + 1}, train batch {resume_batch:,}")

    def save_resume(next_epoch: int, next_batch: int) -> None:
        package = {
            "format": "causal_chemmask_resume_v1",
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "head_state_dict": cpu_state_dict(model.head),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val": best_val,
            "stale": stale,
            "history": history,
            "config": config,
        }
        temporary = latest_resume_path.with_suffix(".tmp")
        torch.save(package, temporary)
        temporary.replace(latest_resume_path)

    total_run_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_dataset.set_epoch(epoch)
        model.head.train()
        optimizer.zero_grad(set_to_none=True)

        train_items = ([], [], [], [], [], {
            "hard": [], "masked": [], "mask_count": [], "hard_score": [],
        })
        window = {key: [] for key in ("loss", "triplet", "preserve", "positive", "negative")}
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
            loss.backward()

            if (batch_index + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if (
                    args.checkpoint_every_batches
                    and (batch_index + 1) % args.checkpoint_every_batches == 0
                ):
                    save_resume(epoch, batch_index + 1)
                    log(f"  [checkpoint] saved resume: epoch {epoch + 1}, "
                        f"batch {batch_index + 1:,} "
                        f"({format_duration(time.time() - total_run_start)} elapsed)")

            for destination, values in zip(
                train_items[:5], (per_item, triplet, preserve, positive, negative)
            ):
                destination.append(values.detach().float().cpu().numpy())
            meta = train_items[5]
            meta["hard"].extend(batch["hard_negative_selected"].numpy().tolist())
            meta["masked"].extend(batch["identity_masked"].numpy().tolist())
            meta["mask_count"].extend(batch["masked_peak_count"].numpy().tolist())
            meta["hard_score"].extend(batch["hard_negative_score"].numpy().tolist())

            window["loss"].append(float(per_item.mean().detach()))
            window["triplet"].append(float(triplet.mean().detach()))
            window["preserve"].append(float(preserve.mean().detach()))
            window["positive"].append(float(positive.mean().detach()))
            window["negative"].append(float(negative.mean().detach()))
            processed += 1

            if (batch_index + 1) % args.log_every_batches == 0:
                sep = np.mean(window["positive"]) - np.mean(window["negative"])
                acc = float(np.mean(
                    np.asarray(window["positive"]) > np.asarray(window["negative"])
                ))
                rate = processed / (time.time() - epoch_start)
                spectra_per_sec = rate * args.batch_size * 3
                epoch_eta = (total_train_batches - processed) / max(rate, 1e-6)
                remaining_epochs = args.epochs - epoch - 1
                total_eta = epoch_eta + remaining_epochs * total_train_batches / max(rate, 1e-6)
                pct = 100.0 * (batch_index + 1) / total_train_batches
                log(
                    f"[epoch {epoch + 1}/{args.epochs}] batch {batch_index + 1:,}/"
                    f"{total_train_batches:,} ({pct:5.1f}%) | "
                    f"loss {np.mean(window['loss']):.4f} "
                    f"trip {np.mean(window['triplet']):.4f} "
                    f"pres {np.mean(window['preserve']):.4f} | "
                    f"sep {sep:+.4f} acc {acc:.3f} | "
                    f"{rate:5.1f} it/s {spectra_per_sec:6.0f} sp/s | "
                    f"epoch ETA {format_duration(epoch_eta)} "
                    f"total ETA {format_duration(total_eta)}"
                )
                for key in window:
                    window[key].clear()

        if processed and processed % args.grad_accum:
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Validation.
        model.head.eval()
        val_items = ([], [], [], [], [], {
            "hard": [], "masked": [], "mask_count": [], "hard_score": [],
        })
        official_val_positive, official_val_negative = [], []
        val_start = time.time()
        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                if args.max_val_batches and batch_index >= args.max_val_batches:
                    break
                _, positive, negative, preserve, teacher_positive, teacher_negative = forward_head(
                    model, batch, teacher_weight, teacher_bias, device, amp_enabled
                )
                triplet = F.relu(args.margin - positive + negative)
                per_item = triplet + args.lambda_preserve * preserve.view(3, -1).mean(dim=0)
                for destination, values in zip(
                    val_items[:5], (per_item, triplet, preserve, positive, negative)
                ):
                    destination.append(values.detach().float().cpu().numpy())
                official_val_positive.append(teacher_positive.detach().float().cpu().numpy())
                official_val_negative.append(teacher_negative.detach().float().cpu().numpy())
                meta = val_items[5]
                meta["hard"].extend(batch["hard_negative_selected"].numpy().tolist())
                meta["masked"].extend(batch["identity_masked"].numpy().tolist())
                meta["mask_count"].extend(batch["masked_peak_count"].numpy().tolist())
                meta["hard_score"].extend(batch["hard_negative_score"].numpy().tolist())
                if (
                    args.val_log_every_batches
                    and (batch_index + 1) % args.val_log_every_batches == 0
                ):
                    rate = (batch_index + 1) / (time.time() - val_start)
                    val_eta = (total_val_batches - batch_index - 1) / max(rate, 1e-6)
                    log(
                        f"[epoch {epoch + 1}/{args.epochs} val] batch "
                        f"{batch_index + 1:,}/{total_val_batches:,} | "
                        f"{rate:5.1f} it/s | val ETA {format_duration(val_eta)}"
                    )

        train_metrics = epoch_summary(*train_items)
        val_metrics = epoch_summary(*val_items)
        official_val_metrics = reference_triplet_summary(
            official_val_positive, official_val_negative, args.margin
        )
        result = {
            "epoch": epoch + 1,
            "seconds": time.time() - epoch_start,
            "train": train_metrics,
            "val": val_metrics,
            "official_val_reference": official_val_metrics,
        }
        history.append(result)
        resume_batch = 0
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        log(
            f"=== EPOCH {epoch + 1:02d}/{args.epochs} done ({format_duration(result['seconds'])}) "
            f"| train loss {train_metrics['loss']:.4f} sep {train_metrics['separation']:+.4f} "
            f"acc {train_metrics['triplet_accuracy']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['triplet_accuracy']:.4f} | "
            f"official_acc {official_val_metrics['triplet_accuracy']:.4f} "
            f"delta {val_metrics['triplet_accuracy'] - official_val_metrics['triplet_accuracy']:+.4f} "
            f"preserve {1.0 - val_metrics['preserve_loss']:.4f}"
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
            log("  [best] saved best_causal_head.pt")
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                log(f"Early stopping after {args.patience} unimproved epochs")
                break
        save_resume(epoch + 1, 0)

    del model
    gc.collect()
    log("=" * 78)
    log(f"COMPLETE in {format_duration(time.time() - total_run_start)}. "
        f"Best validation loss = {best_val:.6f}")
    log(f"Checkpoint: {run_dir / 'best_causal_head.pt'}")
    log(f"History:    {run_dir / 'history.json'}")
    log("=" * 78)


if __name__ == "__main__":
    main()
