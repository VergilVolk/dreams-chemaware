"""CPU feasibility probe: decode 335 molecule-level rule labels from cached E0 embeddings.

This is intentionally a lightweight diagnostic for the second progress report.
It uses one [M+H]+ spectrum per IK14 and an IK14-disjoint 80/20 split.  Labels
come from the existing annotated01-derived rule-vector cache, so the result is a
proxy for molecule-level rule decodability, not a final spectrum-level E3 metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0 or positives == y_true.size:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ranked = y_true[order].astype(np.float64)
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float((precision * ranked).sum() / positives)


def macro_micro_ap(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float, list[float]]:
    per_rule = [average_precision(y_true[:, i], score[:, i]) for i in range(y_true.shape[1])]
    macro = float(np.nanmean(per_rule))
    micro = average_precision(y_true.reshape(-1), score.reshape(-1))
    return macro, micro, per_rule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "validation" / "e0_baseline" / "cached_rule_probe.json",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    embeddings = np.load(ROOT / "data" / "validation" / "e0_baseline" / "e0_embeddings.npy", mmap_mode="r")
    manifest = json.loads((ROOT / "data" / "validation" / "e0_baseline" / "e0_manifest.json").read_text(encoding="utf-8"))
    rule_cache = np.load(ROOT / "tasks" / "_cache" / "rule_vectors" / "ik_to_rvec.npz", allow_pickle=False)

    # One primary-adduct spectrum per molecule prevents spectrum-rich molecules
    # from dominating the probe.
    chosen: dict[str, int] = {}
    for row in manifest:
        ik = row["inchikey_14"]
        if row.get("adduct") == "[M+H]+" and ik in rule_cache and ik not in chosen:
            chosen[ik] = int(row["embedding_idx"])

    iks = np.array(sorted(chosen))
    x = np.asarray(embeddings[[chosen[ik] for ik in iks]], dtype=np.float32)
    y = np.stack([rule_cache[ik] for ik in iks]).astype(np.float32)
    rule_cache.close()

    permutation = rng.permutation(len(iks))
    split = int(round(0.8 * len(iks)))
    train_idx, val_idx = permutation[:split], permutation[split:]
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    positive = y_train.sum(axis=0)
    negative = y_train.shape[0] - positive
    pos_weight = np.clip(negative / np.maximum(positive, 1.0), 1.0, 20.0).astype(np.float32)
    supported = (positive >= 10) & (negative >= 10)

    model = nn.Linear(x_train.shape[1], y_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight), reduction="none")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            raw = criterion(model(xb), yb)
            loss = raw[:, torch.from_numpy(supported)].mean()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(torch.from_numpy(x_val))).numpy()

    macro, micro, per_rule = macro_micro_ap(y_val[:, supported], score[:, supported])
    prevalence = y_val[:, supported].mean(axis=0)
    baseline_macro = float(prevalence.mean())
    baseline_micro = float(y_val[:, supported].mean())

    rule_meta = json.loads(
        (ROOT / "dreams" / "models" / "chem_aware" / "chem_rules_data.json").read_text(encoding="utf-8")
    )["rules"]
    supported_indices = np.flatnonzero(supported)
    category_rows: dict[str, list[float]] = {}
    category_baselines: dict[str, list[float]] = {}
    for local_index, rule_index in enumerate(supported_indices):
        category = rule_meta[int(rule_index)].get("category", "UNKNOWN")
        value = per_rule[local_index]
        if np.isfinite(value):
            category_rows.setdefault(category, []).append(float(value))
            category_baselines.setdefault(category, []).append(float(prevalence[local_index]))

    category_macro = {key: float(np.mean(values)) for key, values in sorted(category_rows.items())}
    category_baseline = {key: float(np.mean(values)) for key, values in sorted(category_baselines.items())}

    report = {
        "status": "feasibility_proxy_not_final_e3",
        "seed": args.seed,
        "epochs": args.epochs,
        "one_spectrum_per_ik14": True,
        "adduct": "[M+H]+",
        "n_molecules": int(len(iks)),
        "n_train_molecules": int(len(train_idx)),
        "n_val_molecules": int(len(val_idx)),
        "n_rules_total": int(y.shape[1]),
        "n_rules_evaluated": int(supported.sum()),
        "macro_auprc": macro,
        "micro_auprc": micro,
        "prevalence_baseline_macro_auprc": baseline_macro,
        "prevalence_baseline_micro_auprc": baseline_micro,
        "macro_lift": macro / baseline_macro if baseline_macro else None,
        "micro_lift": micro / baseline_micro if baseline_micro else None,
        "category_macro_auprc": category_macro,
        "category_prevalence_baseline": category_baseline,
        "category_lift": {
            key: category_macro[key] / category_baseline[key] if category_baseline[key] else None
            for key in category_macro
        },
        "limitations": [
            "Labels are molecule-level vectors cached from annotated01, not condition-aware labels for each MassSpecGym spectrum.",
            "MassSpecGym molecules overlap annotated01; this is an IK14-disjoint probe split, not an independent-source benchmark.",
            "Use only as the frozen-E0 feasibility zero point; final E3 requires spectrum-level three-state labels.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
