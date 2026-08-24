"""Train a reusable linear probe from frozen DreaMS embeddings to spectral concepts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent.parent


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0 or positives == y_true.size:
        return float("nan")
    ranked = y_true[np.argsort(-score, kind="stable")].astype(np.float64)
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float((precision * ranked).sum() / positives)


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    positives = int(y_true.sum())
    negatives = int(y_true.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average tied ranks.
    values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for group in np.flatnonzero(counts > 1):
            mask = inverse == group
            ranks[mask] = ranks[mask].mean()
    return float((ranks[y_true.astype(bool)].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def split_molecules(ik14: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    molecules = np.unique(ik14)
    rng = np.random.default_rng(seed)
    molecules = molecules[rng.permutation(len(molecules))]
    train_end = int(round(0.70 * len(molecules)))
    val_end = int(round(0.85 * len(molecules)))
    return molecules[:train_end], molecules[train_end:val_end], molecules[val_end:]


def capped_indices(ik14: np.ndarray, allowed: np.ndarray, cap: int, seed: int) -> np.ndarray:
    allowed_set = set(allowed.tolist())
    groups: dict[str, list[int]] = {}
    for index, molecule in enumerate(ik14.tolist()):
        if molecule in allowed_set:
            groups.setdefault(molecule, []).append(index)
    rng = np.random.default_rng(seed)
    chosen = []
    for molecule in sorted(groups):
        values = np.asarray(groups[molecule], dtype=np.int64)
        if cap and len(values) > cap:
            values = rng.choice(values, size=cap, replace=False)
        chosen.extend(values.tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> tuple[float, list[float]]:
    model.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    per_rule = [average_precision(y[:, i], scores[:, i]) for i in range(y.shape[1])]
    return float(np.nanmean(per_rule)), per_rule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings", type=Path,
        default=ROOT / "data/validation/cosmic_retrieval/retrieval_embeddings.npy",
    )
    parser.add_argument(
        "--labels", type=Path,
        default=ROOT / "data/validation/double_mapping/spectrum_rule_labels.npz",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/cosmic_retrieval/frozen_concept_probe",
    )
    parser.add_argument("--categories", nargs="+", default=["CF", "NL", "ISO"])
    parser.add_argument("--min-prevalence", type=float, default=0.01)
    parser.add_argument("--max-prevalence", type=float, default=0.50)
    parser.add_argument("--max-spectra-per-molecule", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = np.load(args.labels, allow_pickle=False)
    ik14 = cache["ik14"].astype(str)
    labels_all = cache["labels"].astype(np.float32)
    names_all = cache["rule_name"].astype(str)
    categories_all = cache["rule_category"].astype(str)
    embeddings = np.load(args.embeddings, mmap_mode="r")
    # Cache-order retrieval embeddings (aligned row-for-row with labels/ik14). The
    # embeddings file may be a subset (local smoke), so slice labels to match.
    n = int(embeddings.shape[0])
    ik14 = ik14[:n]
    labels_all = labels_all[:n]
    x_all = np.asarray(embeddings, dtype=np.float32)

    train_mol, val_mol, test_mol = split_molecules(ik14, args.seed)
    train_idx = capped_indices(ik14, train_mol, args.max_spectra_per_molecule, args.seed)
    val_idx = capped_indices(ik14, val_mol, args.max_spectra_per_molecule, args.seed + 1)
    test_idx = capped_indices(ik14, test_mol, args.max_spectra_per_molecule, args.seed + 2)
    train_prevalence = labels_all[train_idx].mean(axis=0)
    selected = (
        np.isin(categories_all, np.asarray(args.categories))
        & (train_prevalence >= args.min_prevalence)
        & (train_prevalence <= args.max_prevalence)
    )
    rule_indices = np.flatnonzero(selected)
    if len(rule_indices) == 0:
        raise RuntimeError("No rules survived the category/prevalence filters")

    mean = x_all[train_idx].mean(axis=0, keepdims=True)
    std = x_all[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    def prepare(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            ((x_all[indices] - mean) / std).astype(np.float32),
            labels_all[indices][:, rule_indices].astype(np.float32),
        )
    x_train, y_train = prepare(train_idx)
    x_val, y_val = prepare(val_idx)
    x_test, y_test = prepare(test_idx)

    positive = y_train.sum(axis=0)
    negative = len(y_train) - positive
    pos_weight = np.clip(negative / np.maximum(positive, 1.0), 1.0, 20.0).astype(np.float32)
    model = nn.Linear(x_train.shape[1], y_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_ap = -np.inf
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_ap, _ = evaluate(model, x_val, y_val)
        print(f"epoch={epoch:02d} train={np.mean(losses):.5f} val_macro_auprc={val_ap:.5f}", flush=True)
        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_macro, test_per_rule_ap = evaluate(model, x_test, y_test)
    with torch.inference_mode():
        test_scores = torch.sigmoid(model(torch.from_numpy(x_test))).numpy()
    per_rule_rows = []
    for local, rule_index in enumerate(rule_indices):
        truth = y_test[:, local]
        prevalence = float(truth.mean())
        ap = float(test_per_rule_ap[local])
        per_rule_rows.append({
            "probe_output_index": local,
            "rule_index": int(rule_index),
            "rule_name": names_all[rule_index],
            "category": categories_all[rule_index],
            "test_prevalence": prevalence,
            "test_positive_spectra": int(truth.sum()),
            "test_auprc": ap,
            "auprc_lift": ap / prevalence if prevalence else None,
            "test_roc_auc": roc_auc(truth, test_scores[:, local]),
        })
    import pandas as pd
    metrics = pd.DataFrame(per_rule_rows).sort_values(
        ["auprc_lift", "test_auprc"], ascending=False
    )
    metrics.to_csv(args.output_dir / "per_rule_metrics.csv", index=False)
    checkpoint = {
        "format": "frozen_spectrum_concept_probe_v1",
        "state_dict": best_state,
        "embedding_mean": torch.from_numpy(mean.squeeze(0)),
        "embedding_std": torch.from_numpy(std.squeeze(0)),
        "rule_indices": torch.from_numpy(rule_indices),
        "rule_names": names_all[rule_indices].tolist(),
        "rule_categories": categories_all[rule_indices].tolist(),
        "seed": args.seed,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, args.output_dir / "concept_probe.pt")
    split_hash = hashlib.sha256("|".join(sorted(test_mol.tolist())).encode()).hexdigest()[:16]
    report = {
        "status": "frozen_spectrum_concept_probe_complete",
        "space": "retrieval (official_embedding_slim.pt, headed, 100 peaks)",
        "checkpoint": "official DreaMS retrieval embedding; backbone and embedding frozen",
        "label_semantics": "observed spectrum-level rule motif",
        "molecule_disjoint_split": True,
        "spectra_per_molecule_cap": args.max_spectra_per_molecule,
        "train": {"molecules": len(train_mol), "spectra": len(train_idx)},
        "validation": {"molecules": len(val_mol), "spectra": len(val_idx)},
        "test": {"molecules": len(test_mol), "spectra": len(test_idx), "ik14_sha256": split_hash},
        "selected_rules": int(len(rule_indices)),
        "categories": args.categories,
        "best_epoch": best_epoch,
        "validation_macro_auprc": float(best_ap),
        "test_macro_auprc": float(test_macro),
        "test_macro_prevalence_baseline": float(y_test.mean(axis=0).mean()),
        "rules_test_auprc_lift_at_least_2": int((metrics["auprc_lift"] >= 2).sum()),
        "claim_limit": (
            "Decodability establishes a predictive direction in frozen embedding space; "
            "chemical meaning still requires peak localization and intervention."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
