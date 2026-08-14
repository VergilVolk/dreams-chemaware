"""Formal staged fine-tuning of DreaMS with counterfactual peak evidence.

The model starts from the official fine-tuned DreaMS checkpoint.  Training
progressively unfreezes the projection head, the final Transformer layer, or
the final two layers.  Structure identity defines positives/negatives; peak
evidence only defines counterfactual interventions.
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
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

from e1_checkpoint_io import torch_load_compat
from pilot_paired_layer_cka import preprocess_spectrum
from run_large_targeted_peak_occlusion import parse_values, target_tokens
from train_e1_identity import cpu_state_dict, load_base_model


ROOT = Path(__file__).resolve().parent.parent


def stable_seed(*parts: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def matched_control(clean: torch.Tensor, targets: np.ndarray, excluded: set[int], seed: int) -> np.ndarray:
    values = clean.numpy()
    pool = np.asarray([
        index for index in range(1, len(values))
        if values[index, 0] > 0 and values[index, 1] > 0 and index not in excluded
    ], dtype=int)
    count = max(len(targets), max(1, int(round(0.20 * len(pool)))))
    count = min(count, len(pool))
    if count == 0:
        return np.empty(0, dtype=int)
    rng = np.random.default_rng(seed)
    if len(targets) and len(pool) >= len(targets):
        valid_mz = values[1:, 0][values[1:, 0] > 0]
        mz_scale = max(float(np.std(valid_mz)), 25.0)
        log_intensity = np.log10(np.clip(values[:, 1].astype(float), 1e-6, None))
        cost = np.empty((len(targets), len(pool)), dtype=float)
        for row, target in enumerate(targets):
            cost[row] = (
                4.0 * np.abs(log_intensity[pool] - log_intensity[target])
                + 8.0 * np.abs(values[pool, 1].astype(float) - float(values[target, 1]))
                + 0.15 * np.abs(values[pool, 0].astype(float) - float(values[target, 0])) / mz_scale
            )
        cost += rng.gumbel(0, 0.015, cost.shape)
        _, columns = linear_sum_assignment(cost)
        chosen = list(pool[columns])
        remaining = np.asarray([item for item in pool if item not in set(chosen)], dtype=int)
        if count > len(chosen):
            chosen.extend(rng.choice(remaining, count - len(chosen), replace=False).tolist())
        return np.asarray(chosen, dtype=int)
    return rng.choice(pool, count, replace=False).astype(int)


class CounterfactualDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data: Path, n_peaks: int, seed: int, training: bool):
        self.frame = frame.reset_index(drop=True)
        self.data = str(data)
        self.n_peaks = n_peaks
        self.seed = seed
        self.training = training
        self.epoch = 0
        self._handle = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.frame)

    def handle(self):
        if self._handle is None:
            self._handle = h5py.File(self.data, "r")
        return self._handle

    def spectrum(self, row: int) -> torch.Tensor:
        handle = self.handle()
        return preprocess_spectrum(
            np.asarray(handle["spectrum"][row]), float(handle["precursor_mz"][row]), self.n_peaks
        )

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[item]
        clean = self.spectrum(int(row.query_hdf5_row))
        identity_tokens = target_tokens(clean, parse_values(row.identity_peak_mz), 0.005)
        confounder_tokens = target_tokens(clean, parse_values(row.confounder_peak_mz), 0.005)
        identity_masked = clean.clone()
        confounder_masked = clean.clone()
        if len(identity_tokens):
            identity_masked[identity_tokens] = 0
        if len(confounder_tokens):
            confounder_masked[confounder_tokens] = 0
        target_union = np.asarray(sorted(set(identity_tokens) | set(confounder_tokens)), dtype=int)
        control = matched_control(
            clean, target_union, set(target_union.tolist()),
            stable_seed(self.seed, self.epoch if self.training else 0, item),
        )
        random_masked = clean.clone()
        if len(control):
            random_masked[control] = 0
        hard = str(row.transition) in {"fixed_oof", "residual_wrong"}
        return {
            "clean": clean, "identity_masked": identity_masked,
            "confounder_masked": confounder_masked, "random_masked": random_masked,
            "positive": self.spectrum(int(row.identity_hdf5_row)),
            "negative": self.spectrum(int(row.confounder_hdf5_row)),
            "teacher_query": torch.as_tensor(row.teacher_query, dtype=torch.float32),
            "teacher_positive": torch.as_tensor(row.teacher_positive, dtype=torch.float32),
            "teacher_negative": torch.as_tensor(row.teacher_negative, dtype=torch.float32),
            "has_identity": torch.tensor(bool(len(identity_tokens))),
            "has_confounder": torch.tensor(bool(len(confounder_tokens))),
            "sample_weight": torch.tensor(1.5 if hard else 1.0, dtype=torch.float32),
        }

    def __del__(self):
        if self._handle is not None:
            self._handle.close()


def attach_teacher(frame: pd.DataFrame, manifest: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    row_to_local = pd.Series(manifest.index.to_numpy(), index=manifest["hdf5_row"].astype(int)).to_dict()
    output = frame.copy()
    for label, column in (
        ("query", "query_hdf5_row"), ("positive", "identity_hdf5_row"), ("negative", "confounder_hdf5_row")
    ):
        missing = set(output[column].astype(int)) - set(row_to_local)
        if missing:
            raise ValueError(f"{len(missing)} {label} rows lack official teacher embeddings")
        output[f"teacher_{label}"] = [embeddings[row_to_local[int(row)]] for row in output[column]]
    return output


def configure_stage(model, stage: str) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    encoder = model.backbone.transformer_encoder
    if stage == "head":
        pass
    elif stage in {"last1", "last2"}:
        count = 1 if stage == "last1" else 2
        start = len(encoder.atts) - count
        for layer in range(start, len(encoder.atts)):
            for module in (encoder.atts[layer], encoder.ffs[layer], encoder.scales[2 * layer], encoder.scales[2 * layer + 1]):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        for parameter in encoder.scales[-1].parameters():
            parameter.requires_grad = True
        encoder.gradient_checkpointing_enable()
    elif stage == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
        encoder.gradient_checkpointing_enable()
    else:
        raise ValueError(stage)
    return {
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_resume(model, path: Path | None) -> None:
    if path is None:
        return
    package = torch_load_compat(path, map_location="cpu")
    if package.get("format") != "counterfactual_dreams_v1":
        raise ValueError("--resume must be a counterfactual_dreams_v1 checkpoint")
    model.backbone.load_state_dict(package["backbone_state_dict"], strict=True)
    model.head.load_state_dict(package["head_state_dict"], strict=True)
    print(f"Resumed weights from {path}")


def model_embeddings(model, spectra: torch.Tensor, backbone_trainable: bool) -> torch.Tensor:
    if backbone_trainable:
        hidden = model.backbone(spectra, None)[:, 0]
    else:
        with torch.no_grad():
            hidden = model.backbone(spectra, None)[:, 0]
    return F.normalize(model.head(hidden), dim=-1)


def batch_objective(model, batch, args, device, amp, training: bool) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    names = ["clean", "identity_masked", "confounder_masked", "random_masked", "positive", "negative"]
    size = len(batch["clean"])
    spectra = torch.cat([batch[name].to(device, non_blocking=True) for name in names], dim=0)
    backbone_trainable = any(parameter.requires_grad for parameter in model.backbone.parameters())
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        encoded = model_embeddings(model, spectra, backbone_trainable)
        q, qi, qc, qr, positive, negative = encoded.split(size)
        clean_margin = (q * positive).sum(1) - (q * negative).sum(1)
        identity_margin = (qi * positive).sum(1) - (qi * negative).sum(1)
        confounder_margin = (qc * positive).sum(1) - (qc * negative).sum(1)
        weights = batch["sample_weight"].to(device)
        identity_valid = batch["has_identity"].to(device).bool()
        confounder_valid = batch["has_confounder"].to(device).bool()
        triplet_per = F.relu(args.triplet_margin - clean_margin)
        triplet = (triplet_per * weights).sum() / weights.sum()
        identity_cf = (
            F.relu(args.counterfactual_margin + identity_margin[identity_valid] - clean_margin[identity_valid]).mean()
            if identity_valid.any() else clean_margin.new_zeros(())
        )
        confounder_cf = (
            F.relu(args.counterfactual_margin + clean_margin[confounder_valid] - confounder_margin[confounder_valid]).mean()
            if confounder_valid.any() else clean_margin.new_zeros(())
        )
        teacher_q = batch["teacher_query"].to(device)
        teacher_p = batch["teacher_positive"].to(device)
        teacher_n = batch["teacher_negative"].to(device)
        preserve_per = (3 - (q * teacher_q).sum(1) - (positive * teacher_p).sum(1) - (negative * teacher_n).sum(1)) / 3
        preserve = (preserve_per * weights).sum() / weights.sum()
        random_consistency = (1 - (q * qr).sum(1)).mean()
        loss = (
            args.triplet_weight * triplet
            + args.counterfactual_weight * 0.5 * (identity_cf + confounder_cf)
            + args.preserve_weight * preserve
            + args.random_consistency_weight * random_consistency
        )
    arrays = {
        "loss": np.asarray([float(loss.detach().cpu())]),
        "triplet": np.asarray([float(triplet.detach().cpu())]),
        "identity_cf_loss": np.asarray([float(identity_cf.detach().cpu())]),
        "confounder_cf_loss": np.asarray([float(confounder_cf.detach().cpu())]),
        "preserve": preserve_per.detach().float().cpu().numpy(),
        "random_consistency": (q * qr).sum(1).detach().float().cpu().numpy(),
        "clean_margin": clean_margin.detach().float().cpu().numpy(),
        "identity_effect": (clean_margin - identity_margin).detach().float().cpu().numpy()[batch["has_identity"].numpy().astype(bool)],
        "confounder_effect": (confounder_margin - clean_margin).detach().float().cpu().numpy()[batch["has_confounder"].numpy().astype(bool)],
    }
    return loss, arrays


def reduce_metrics(parts: list[dict[str, np.ndarray]]) -> dict[str, float]:
    def cat(key: str) -> np.ndarray:
        values = [part[key] for part in parts if len(part[key])]
        return np.concatenate(values) if values else np.empty(0)
    margin, identity, confounder = cat("clean_margin"), cat("identity_effect"), cat("confounder_effect")
    return {
        "loss": float(cat("loss").mean()), "triplet_loss": float(cat("triplet").mean()),
        "identity_cf_loss": float(cat("identity_cf_loss").mean()),
        "confounder_cf_loss": float(cat("confounder_cf_loss").mean()),
        "pairwise_accuracy": float(np.mean(margin > 0)), "mean_margin": float(margin.mean()),
        "identity_cf_order_accuracy": float(np.mean(identity > 0)) if len(identity) else float("nan"),
        "confounder_cf_order_accuracy": float(np.mean(confounder > 0)) if len(confounder) else float("nan"),
        "mean_cosine_to_official": float(1 - cat("preserve").mean()),
        "random_mask_cosine": float(cat("random_consistency").mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("head", "last1", "last2", "all"), required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "data/e1/counterfactual_peak_finetune/counterfactual_peak_finetune_split.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv")
    parser.add_argument("--teacher-embeddings", type=Path, default=ROOT / "data/validation/large_observability_embeddings_discovery/official_embeddings.npy")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-ckpt", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/e1/counterfactual_formal")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=3e-5)
    parser.add_argument("--backbone-lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--triplet-margin", type=float, default=0.05)
    parser.add_argument("--counterfactual-margin", type=float, default=0.02)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-weight", type=float, default=0.7)
    parser.add_argument("--preserve-weight", type=float, default=5.0)
    parser.add_argument("--random-consistency-weight", type=float, default=0.2)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-preservation", type=float, default=0.98)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu only for a smoke test.")
    device = torch.device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    run_dir = args.output_dir / args.stage / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.split)
    manifest = pd.read_csv(args.manifest)
    teacher = np.load(args.teacher_embeddings).astype(np.float32)
    frame = attach_teacher(frame, manifest, teacher)
    train_frame = frame.loc[frame["pilot_split"] == "train"].copy()
    val_frame = frame.loc[frame["pilot_split"] == "validation"].copy()
    train_data = CounterfactualDataset(train_frame, args.data, args.n_highest_peaks, args.seed, True)
    val_data = CounterfactualDataset(val_frame, args.data, args.n_highest_peaks, args.seed + 97, False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=False)
    model, initialization = load_base_model(args.base_ckpt, args.architecture_ckpt, device, args.n_highest_peaks)
    load_resume(model, args.resume)
    counts = configure_stage(model, args.stage)
    print(f"Stage={args.stage}; trainable={counts['trainable']:,}/{counts['total']:,}; AMP={amp}")
    groups = [
        {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": args.head_lr},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.backbone_lr},
    ]
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"initialization": initialization, "trainable_parameters": counts["trainable"], "total_parameters": counts["total"]})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history, best_score, stale = [], -float("inf"), 0
    for epoch in range(args.epochs):
        start = time.time()
        train_data.set_epoch(epoch)
        model.train()
        # A frozen backbone must remain deterministic; otherwise dropout would
        # silently turn head-only training into a moving-target experiment.
        if not any(parameter.requires_grad for parameter in model.backbone.parameters()):
            model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        train_parts = []
        batches = 0
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            loss, values = batch_objective(model, batch, args, device, amp, True)
            scaler.scale(loss / args.grad_accum).backward()
            if (batch_index + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_parts.append(values)
            batches += 1
        if batches and batches % args.grad_accum:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        model.eval()
        val_parts = []
        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                if args.max_val_batches and batch_index >= args.max_val_batches:
                    break
                _, values = batch_objective(model, batch, args, device, amp, False)
                val_parts.append(values)
        train_metrics, val_metrics = reduce_metrics(train_parts), reduce_metrics(val_parts)
        score = val_metrics["pairwise_accuracy"] + 0.05 * 0.5 * (
            val_metrics["identity_cf_order_accuracy"] + val_metrics["confounder_cf_order_accuracy"]
        )
        if val_metrics["mean_cosine_to_official"] < args.min_preservation:
            score -= 10 * (args.min_preservation - val_metrics["mean_cosine_to_official"])
        result = {"epoch": epoch + 1, "seconds": time.time() - start, "selection_score": score, "train": train_metrics, "val": val_metrics}
        history.append(result)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"Epoch {epoch + 1:02d} | train={train_metrics['loss']:.4f} | "
            f"val acc={val_metrics['pairwise_accuracy']:.4f} margin={val_metrics['mean_margin']:.4f} "
            f"idCF={val_metrics['identity_cf_order_accuracy']:.4f} confCF={val_metrics['confounder_cf_order_accuracy']:.4f} "
            f"preserve={val_metrics['mean_cosine_to_official']:.4f} | {result['seconds']:.0f}s",
            flush=True,
        )
        if score > best_score + 1e-6:
            best_score, stale = score, 0
            checkpoint = {
                "format": "counterfactual_dreams_v1", "stage": args.stage, "epoch": epoch + 1,
                "architecture_checkpoint": str(args.architecture_ckpt.resolve()),
                "base_checkpoint": str(args.base_ckpt.resolve()),
                "backbone_state_dict": cpu_state_dict(model.backbone), "head_state_dict": cpu_state_dict(model.head),
                "config": config, "val_metrics": val_metrics, "history": history,
            }
            torch.save(checkpoint, run_dir / "best_counterfactual.pt")
            print("  Saved best checkpoint", flush=True)
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"Early stopping after {args.patience} unimproved epochs")
                break
    print(f"Best checkpoint: {run_dir / 'best_counterfactual.pt'}")
    del model
    gc.collect()


if __name__ == "__main__":
    main()
