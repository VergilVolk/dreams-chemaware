"""Probe molecule-mean frozen DreaMS embeddings for local structure environments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_frozen_spectrum_concept_probe import average_precision, roc_auc


ROOT = Path(__file__).resolve().parent.parent


def scaffold_split(scaffolds: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, scaffold in enumerate(scaffolds.tolist()):
        groups.setdefault(scaffold, []).append(index)
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: len(groups[key]), reverse=True)
    targets = np.asarray([0.70, 0.15, 0.15]) * len(scaffolds)
    allocations = [[], [], []]
    counts = np.zeros(3, dtype=int)
    for key in keys:
        deficits = targets - counts
        split = int(np.argmax(deficits))
        allocations[split].extend(groups[key])
        counts[split] += len(groups[key])
    return tuple(np.asarray(sorted(values), dtype=np.int64) for values in allocations)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, list[float]]:
    model.eval()
    with torch.inference_mode():
        score = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    per_rule = [average_precision(y[:, i], score[:, i]) for i in range(y.shape[1])]
    return float(np.nanmean(per_rule)), score, per_rule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/validation/double_mapping/structure_environment_probe_data.npz",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/double_mapping/frozen_structure_probe",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-train-prevalence", type=float, default=0.01)
    parser.add_argument("--max-train-prevalence", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data, allow_pickle=False)
    x_all = data["embeddings"].astype(np.float32)
    y_all = data["labels"].astype(np.float32)
    environments = data["environment"].astype(str)
    ik14 = data["ik14"].astype(str)
    scaffolds = data["scaffold"].astype(str)
    train_idx, val_idx, test_idx = scaffold_split(scaffolds, args.seed)
    prevalence = y_all[train_idx].mean(axis=0)
    selected = (prevalence >= args.min_train_prevalence) & (prevalence <= args.max_train_prevalence)
    environment_indices = np.flatnonzero(selected)
    mean = x_all[train_idx].mean(axis=0, keepdims=True)
    std = x_all[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    def prepare(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            ((x_all[indices] - mean) / std).astype(np.float32),
            y_all[indices][:, environment_indices].astype(np.float32),
        )
    x_train, y_train = prepare(train_idx)
    x_val, y_val = prepare(val_idx)
    x_test, y_test = prepare(test_idx)
    positive = y_train.sum(axis=0)
    pos_weight = np.clip((len(y_train) - positive) / np.maximum(positive, 1), 1, 20).astype(np.float32)
    model = nn.Linear(x_train.shape[1], y_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_ap, best_epoch, best_state = -np.inf, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_ap, _, _ = evaluate(model, x_val, y_val)
        print(f"epoch={epoch:02d} train={np.mean(losses):.5f} val_macro_auprc={val_ap:.5f}", flush=True)
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_macro, test_scores, per_ap = evaluate(model, x_test, y_test)
    rows = []
    for local, original in enumerate(environment_indices):
        truth = y_test[:, local]
        base = float(truth.mean())
        ap = float(per_ap[local])
        rows.append({
            "probe_output_index": local,
            "environment_index": int(original),
            "environment": environments[original],
            "test_positive_molecules": int(truth.sum()),
            "test_prevalence": base,
            "test_auprc": ap,
            "auprc_lift": ap / base if base else None,
            "test_roc_auc": roc_auc(truth, test_scores[:, local]),
        })
    metrics = pd.DataFrame(rows).sort_values(["auprc_lift", "test_auprc"], ascending=False)
    metrics.to_csv(args.output_dir / "per_environment_metrics.csv", index=False)
    torch.save({
        "format": "frozen_structure_environment_probe_v1",
        "state_dict": best_state,
        "embedding_mean": torch.from_numpy(mean.squeeze(0)),
        "embedding_std": torch.from_numpy(std.squeeze(0)),
        "environment_indices": torch.from_numpy(environment_indices),
        "environments": environments[environment_indices].tolist(),
        "best_epoch": best_epoch,
        "seed": args.seed,
    }, args.output_dir / "structure_probe.pt")
    test_scaffold_hash = hashlib.sha256(
        "|".join(sorted(set(scaffolds[test_idx].tolist()))).encode()
    ).hexdigest()[:16]
    report = {
        "status": "frozen_structure_environment_probe_complete",
        "split": "Murcko-scaffold-disjoint; acyclic molecules use molecule-specific keys",
        "train": {"molecules": len(train_idx), "scaffolds": len(set(scaffolds[train_idx]))},
        "validation": {"molecules": len(val_idx), "scaffolds": len(set(scaffolds[val_idx]))},
        "test": {
            "molecules": len(test_idx), "scaffolds": len(set(scaffolds[test_idx])),
            "scaffold_sha256": test_scaffold_hash,
        },
        "selected_environments": int(len(environment_indices)),
        "best_epoch": best_epoch,
        "validation_macro_auprc": float(best_ap),
        "test_macro_auprc": float(test_macro),
        "test_macro_prevalence_baseline": float(y_test.mean(axis=0).mean()),
        "environments_test_auprc_lift_at_least_2": int((metrics["auprc_lift"] >= 2).sum()),
        "environments_with_at_least_20_test_positives": int((metrics["test_positive_molecules"] >= 20).sum()),
        "claim_limit": (
            "Probe performance establishes decodability of local environment labels from "
            "molecule-mean spectral embeddings, not unique peak attribution."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
