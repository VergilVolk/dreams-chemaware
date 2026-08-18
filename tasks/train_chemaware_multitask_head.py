"""Gate-controlled ChemAware head training after the P1 identity stage.

Labels have deliberately separate roles:
* IK14 identity supervises identity triplets;
* exact/proven MCES relations supervise only local relative order;
* observed spectrum motifs supervise a frozen chemical-concept decoder;
* the selected P1 head is the preservation teacher.

The DreaMS backbone stays frozen in this budget stage.  The concept decoder is
also frozen, preventing it from adapting around an unchanged embedding.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from e1_checkpoint_io import official_head_state, torch_load_compat
from train_causal_chemmask_head import CausalDynamicTripletDataset
from train_e1_identity import CandidatePool, cpu_state_dict, load_base_model, preprocess_spectrum, seed_everything
from train_frozen_spectrum_concept_probe import average_precision, capped_indices, split_molecules


ROOT = Path(__file__).resolve().parent.parent


class MCESRankDataset(Dataset):
    def __init__(self, data: Path, triplets: Path, n_highest_peaks: int):
        self.data = str(data)
        self.frame = pd.read_csv(triplets)
        self.n_highest_peaks = n_highest_peaks
        self._h5 = None

    def __len__(self) -> int:
        return len(self.frame)

    def _handle(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.data, "r")
        return self._h5

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[item]
        handle = self._handle()
        result = {}
        for name in ("anchor", "positive", "negative"):
            index = int(row[f"{name}_spectrum"])
            result[name] = preprocess_spectrum(
                np.asarray(handle["spectrum"][index]),
                float(handle["precursor_mz"][index]),
                self.n_highest_peaks,
            )
        return result

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


class RuleSpectrumDataset(Dataset):
    def __init__(
        self, data: Path, rows: np.ndarray, labels: np.ndarray,
        indices: np.ndarray, n_highest_peaks: int,
    ):
        self.data = str(data)
        self.rows = rows[indices].astype(np.int64)
        self.labels = labels[indices].astype(np.float32)
        self.n_highest_peaks = n_highest_peaks
        self._h5 = None

    def __len__(self) -> int:
        return len(self.rows)

    def _handle(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.data, "r")
        return self._h5

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        handle = self._handle()
        row = int(self.rows[item])
        spectrum = preprocess_spectrum(
            np.asarray(handle["spectrum"][row]),
            float(handle["precursor_mz"][row]),
            self.n_highest_peaks,
        )
        return {"spectrum": spectrum, "label": torch.from_numpy(self.labels[item])}

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
    )
    parser.add_argument(
        "--base-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt"
    )
    parser.add_argument(
        "--architecture-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument(
        "--identity-train", type=Path, default=ROOT / "data/e1/e1_train_triplet_pool_10ppm.npz"
    )
    parser.add_argument(
        "--identity-val", type=Path, default=ROOT / "data/e1/e1_val_triplet_pool_10ppm.npz"
    )
    parser.add_argument(
        "--mces-train", type=Path,
        default=ROOT / "data/e2/mces_local_rank/train_mces_rank_triplets.csv",
    )
    parser.add_argument(
        "--mces-val", type=Path,
        default=ROOT / "data/e2/mces_local_rank/val_mces_rank_triplets.csv",
    )
    parser.add_argument(
        "--rule-labels", type=Path,
        default=ROOT / "data/validation/double_mapping/spectrum_rule_labels.npz",
    )
    parser.add_argument(
        "--concept-probe", type=Path,
        default=ROOT / "data/validation/double_mapping/frozen_concept_probe/concept_probe.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/e3/chemaware_multitask_head"
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batches", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--identity-margin", type=float, default=0.05)
    parser.add_argument("--mces-margin", type=float, default=0.05)
    parser.add_argument("--lambda-mces", type=float, default=0.3)
    parser.add_argument("--lambda-rule", type=float, default=0.1)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--identity-safety-drop", type=float, default=0.005)
    parser.add_argument("--max-spectra-per-molecule", type=int, default=3)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cycle(loader: DataLoader):
    while True:
        yield from loader


def backbone_precursor(model, tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    spectra = torch.cat([tensor.to(device, non_blocking=True) for tensor in tensors], dim=0)
    with torch.no_grad():
        dtype = next(model.backbone.parameters()).dtype
        return model.backbone(spectra.to(dtype=dtype), None)[:, 0, :]


def pair_values(embedding: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        (embedding[:size] * embedding[size:2 * size]).sum(dim=1),
        (embedding[:size] * embedding[2 * size:3 * size]).sum(dim=1),
    )


def macro_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    values = [average_precision(labels[:, i], scores[:, i]) for i in range(labels.shape[1])]
    return float(np.nanmean(values))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    required = [
        args.p1_checkpoint, args.data, args.base_checkpoint, args.architecture_checkpoint,
        args.identity_train, args.identity_val, args.mces_train, args.mces_val,
        args.rule_labels, args.concept_probe,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing multitask inputs:\n" + "\n".join(missing))

    probe_package = torch_load_compat(args.concept_probe, map_location="cpu")
    cache = np.load(args.rule_labels, allow_pickle=False)
    ik14 = cache["ik14"].astype(str)
    hdf_rows = cache["hdf5_row"].astype(np.int64)
    rule_indices = probe_package["rule_indices"].numpy().astype(np.int64)
    rule_labels = cache["labels"][:, rule_indices].astype(np.float32)
    train_molecules, val_molecules, _ = split_molecules(ik14, args.seed)
    rule_train_idx = capped_indices(
        ik14, train_molecules, args.max_spectra_per_molecule, args.seed
    )
    rule_val_idx = capped_indices(
        ik14, val_molecules, args.max_spectra_per_molecule, args.seed + 1
    )
    preflight = {
        "status": "multitask_preflight_complete",
        "p1_checkpoint": str(args.p1_checkpoint.resolve()),
        "identity_train_anchors": len(CandidatePool(args.identity_train)),
        "mces_train_triplets": int(sum(1 for _ in args.mces_train.open(encoding="utf-8"))) - 1,
        "mces_val_triplets": int(sum(1 for _ in args.mces_val.open(encoding="utf-8"))) - 1,
        "rule_train_spectra": int(len(rule_train_idx)),
        "rule_val_spectra": int(len(rule_val_idx)),
        "rule_concepts": int(len(rule_indices)),
        "trainable_scope": "DreaMS projection head only; backbone and concept decoder frozen",
        "loss_roles": {
            "identity": "IK14 identity",
            "mces_rank": "local relative structure order only",
            "rule_decode": "observed spectrum motifs through a frozen decoder",
            "preserve": "selected P1 head teacher",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Formal multitask training requires CUDA; use --dry-run for CPU preflight")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    amp_enabled = bool(args.amp and device.type == "cuda")

    identity_train = CausalDynamicTripletDataset(
        args.data, CandidatePool(args.identity_train), args.n_highest_peaks, args.seed,
        0.0, 1, 0.0, 0.3, 12, 0.02,
    )
    identity_val = CausalDynamicTripletDataset(
        args.data, CandidatePool(args.identity_val), args.n_highest_peaks, args.seed + 97,
        0.0, 1, 0.0, 0.3, 12, 0.02,
        length=min(args.val_batches * args.batch_size, len(CandidatePool(args.identity_val))),
        fixed=True,
    )
    mces_train = MCESRankDataset(args.data, args.mces_train, args.n_highest_peaks)
    mces_val = MCESRankDataset(args.data, args.mces_val, args.n_highest_peaks)
    rule_train = RuleSpectrumDataset(
        args.data, hdf_rows, rule_labels, rule_train_idx, args.n_highest_peaks
    )
    rule_val = RuleSpectrumDataset(
        args.data, hdf_rows, rule_labels, rule_val_idx, args.n_highest_peaks
    )

    def loader(dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
        )

    id_train_loader, id_val_loader = loader(identity_train, True), loader(identity_val, False)
    mces_train_loader, mces_val_loader = loader(mces_train, True), loader(mces_val, False)
    rule_train_loader, rule_val_loader = loader(rule_train, True), loader(rule_val, False)
    steps = args.steps_per_epoch or len(mces_train_loader)

    model, initialization = load_base_model(
        args.base_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks
    )
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    model.backbone.eval()
    p1_package = torch_load_compat(args.p1_checkpoint, map_location="cpu")
    p1_state = official_head_state(p1_package)
    model.head.load_state_dict(p1_state, strict=True)
    teacher_weight = p1_state["weight"].to(device)
    teacher_bias = p1_state["bias"].to(device)

    concept_decoder = nn.Linear(1024, len(rule_indices)).to(device)
    concept_decoder.load_state_dict(probe_package["state_dict"], strict=True)
    concept_decoder.requires_grad_(False)
    concept_decoder.eval()
    embedding_mean = probe_package["embedding_mean"].to(device)
    embedding_std = probe_package["embedding_std"].to(device)
    train_positive = rule_labels[rule_train_idx].sum(axis=0)
    pos_weight = np.clip(
        (len(rule_train_idx) - train_positive) / np.maximum(train_positive, 1), 1, 20
    )
    pos_weight = torch.from_numpy(pos_weight.astype(np.float32)).to(device)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_score = -np.inf
    history = []
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update({
        "initialization": initialization,
        "backbone_frozen": True,
        "concept_decoder_frozen": True,
        "label_policy": (
            "IK14 for identity; MCES only for local rank; spectrum rules only for concept "
            "decode; P1 head for preservation"
        ),
    })
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def evaluate() -> dict:
        model.head.eval()
        result = {}
        with torch.inference_mode():
            for name, data_loader in (("identity", id_val_loader), ("mces", mces_val_loader)):
                candidate_pos, candidate_neg, teacher_pos, teacher_neg, preserves = [], [], [], [], []
                for batch_index, batch in enumerate(data_loader):
                    if batch_index >= args.val_batches:
                        break
                    size = len(batch["anchor"])
                    precursor = backbone_precursor(
                        model, [batch["anchor"], batch["positive"], batch["negative"]], device
                    )
                    current = F.normalize(model.head(precursor), dim=-1)
                    teacher = F.normalize(F.linear(precursor, teacher_weight, teacher_bias), dim=-1)
                    cp, cn = pair_values(current, size)
                    tp, tn = pair_values(teacher, size)
                    candidate_pos.append(cp.cpu().numpy()); candidate_neg.append(cn.cpu().numpy())
                    teacher_pos.append(tp.cpu().numpy()); teacher_neg.append(tn.cpu().numpy())
                    preserves.append((current * teacher).sum(dim=1).cpu().numpy())
                cp, cn = np.concatenate(candidate_pos), np.concatenate(candidate_neg)
                tp, tn = np.concatenate(teacher_pos), np.concatenate(teacher_neg)
                result[name] = {
                    "candidate_accuracy": float((cp > cn).mean()),
                    "teacher_accuracy": float((tp > tn).mean()),
                    "candidate_separation": float((cp - cn).mean()),
                    "teacher_separation": float((tp - tn).mean()),
                    "preservation_cosine": float(np.concatenate(preserves).mean()),
                }
            candidate_scores, teacher_scores, labels = [], [], []
            for batch_index, batch in enumerate(rule_val_loader):
                if batch_index >= args.val_batches:
                    break
                precursor = backbone_precursor(model, [batch["spectrum"]], device)
                current = F.normalize(model.head(precursor), dim=-1)
                teacher = F.normalize(F.linear(precursor, teacher_weight, teacher_bias), dim=-1)
                candidate_scores.append(torch.sigmoid(
                    concept_decoder((current - embedding_mean) / embedding_std)
                ).cpu().numpy())
                teacher_scores.append(torch.sigmoid(
                    concept_decoder((teacher - embedding_mean) / embedding_std)
                ).cpu().numpy())
                labels.append(batch["label"].numpy())
            labels_np = np.concatenate(labels)
            result["rule"] = {
                "candidate_macro_auprc": macro_ap(labels_np, np.concatenate(candidate_scores)),
                "teacher_macro_auprc": macro_ap(labels_np, np.concatenate(teacher_scores)),
            }
        return result

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.head.train()
        id_iter, mces_iter, rule_iter = cycle(id_train_loader), cycle(mces_train_loader), cycle(rule_train_loader)
        losses = []
        for _ in range(steps):
            id_batch, mces_batch, rule_batch = next(id_iter), next(mces_iter), next(rule_iter)
            id_size, mces_size = len(id_batch["anchor"]), len(mces_batch["anchor"])
            precursor = backbone_precursor(model, [
                id_batch["anchor"], id_batch["positive"], id_batch["negative"],
                mces_batch["anchor"], mces_batch["positive"], mces_batch["negative"],
                rule_batch["spectrum"],
            ], device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                current = F.normalize(model.head(precursor), dim=-1)
                teacher = F.normalize(F.linear(precursor, teacher_weight, teacher_bias), dim=-1)
                offset = 0
                id_current = current[offset:offset + 3 * id_size]; offset += 3 * id_size
                mces_current = current[offset:offset + 3 * mces_size]; offset += 3 * mces_size
                rule_current = current[offset:]
                id_pos, id_neg = pair_values(id_current, id_size)
                mces_pos, mces_neg = pair_values(mces_current, mces_size)
                identity_loss = F.relu(args.identity_margin - id_pos + id_neg).mean()
                mces_loss = F.relu(args.mces_margin - mces_pos + mces_neg).mean()
                rule_logits = concept_decoder((rule_current - embedding_mean) / embedding_std)
                rule_loss = F.binary_cross_entropy_with_logits(
                    rule_logits, rule_batch["label"].to(device), pos_weight=pos_weight
                )
                preserve_loss = (1.0 - (current * teacher).sum(dim=1)).mean()
                loss = (
                    identity_loss + args.lambda_mces * mces_loss
                    + args.lambda_rule * rule_loss + args.lambda_preserve * preserve_loss
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate()
        identity_drop = (
            metrics["identity"]["teacher_accuracy"] - metrics["identity"]["candidate_accuracy"]
        )
        mces_delta = metrics["mces"]["candidate_accuracy"] - metrics["mces"]["teacher_accuracy"]
        rule_delta = metrics["rule"]["candidate_macro_auprc"] - metrics["rule"]["teacher_macro_auprc"]
        safe = identity_drop <= args.identity_safety_drop
        score = mces_delta + 0.2 * rule_delta - 2 * max(identity_drop - args.identity_safety_drop, 0)
        record = {
            "epoch": epoch, "seconds": time.time() - started, "train_loss": float(np.mean(losses)),
            "identity_safe": bool(safe), "selection_score": float(score), **metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        checkpoint = {
            "format": "chemaware_multitask_head_v1", "epoch": epoch,
            "p1_checkpoint": str(args.p1_checkpoint.resolve()),
            "head_state_dict": cpu_state_dict(model.head), "metrics": metrics,
            "history": history, "initialization": initialization,
            "rule_indices": torch.from_numpy(rule_indices),
            "config": config,
        }
        torch.save(checkpoint, args.output_dir / f"epoch_{epoch:02d}.pt")
        if safe and score > best_score:
            best_score = score
            torch.save(checkpoint, args.output_dir / "best_chemaware_multitask.pt")
    report = {
        "status": "chemaware_multitask_training_complete",
        "best_selection_score": float(best_score), "history": history,
        "selection_rule": "identity drop<=gate, then maximize MCES accuracy delta + 0.2*rule AUPRC delta",
        "config": config,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
