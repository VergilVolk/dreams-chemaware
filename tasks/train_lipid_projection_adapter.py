"""CPU-friendly low-rank projection pilot for phospholipid hard negatives."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankDelta(nn.Module):
    def __init__(self, dimension: int, rank: int, scale: float):
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)
        self.scale = scale / rank

    def forward(self, backbone: torch.Tensor, official_raw: torch.Tensor) -> torch.Tensor:
        return F.normalize(official_raw + self.scale * self.up(self.down(backbone)), dim=-1)


def pools(records: list[dict], split: str, negative_mode: str, official_norm: torch.Tensor):
    by_ik, by_formula = defaultdict(list), defaultdict(set)
    for i, record in enumerate(records):
        if record["split"] != split: continue
        by_ik[record["ik14"]].append(i)
        by_formula[record["formula"]].add(record["ik14"])
    if negative_mode == "same_formula":
        negative_by_ik = {
            ik: sorted(by_formula[records[values[0]]["formula"]] - {ik})
            for ik, values in by_ik.items()
        }
    elif negative_mode == "different_formula_embedding_hard":
        identities = sorted(by_ik)
        centroids = []
        for ik in identities:
            value = official_norm[by_ik[ik]].mean(0)
            centroids.append(F.normalize(value, dim=0))
        similarity = torch.stack(centroids) @ torch.stack(centroids).T
        negative_by_ik = {}
        for i, ik in enumerate(identities):
            formula = records[by_ik[ik][0]]["formula"]
            candidates = [j for j, other in enumerate(identities)
                          if other != ik and records[by_ik[other][0]]["formula"] != formula]
            candidates.sort(key=lambda j: float(similarity[i, j]), reverse=True)
            negative_by_ik[ik] = [identities[j] for j in candidates[:5]]
    elif negative_mode == "mixed_equal":
        same_formula = pools(records, split, "same_formula", official_norm)
        different_formula = pools(records, split, "different_formula_embedding_hard", official_norm)
        by_ik = same_formula[0]
        negative_by_ik = {}
        for ik in sorted(set(same_formula[2]) & set(different_formula[2])):
            # Prefixes preserve source identity so the sampler can choose the
            # two supervision families with exactly equal probability.
            negative_by_ik[ik] = {
                "same_formula": same_formula[1][ik],
                "different_formula": different_formula[1][ik],
            }
        anchors = sorted(negative_by_ik)
        return by_ik, negative_by_ik, anchors
    else:
        raise ValueError(negative_mode)
    anchors = [ik for ik, values in by_ik.items()
               if len(values) >= 2 and negative_by_ik.get(ik)]
    return by_ik, negative_by_ik, anchors


def sample_triplets(records, by_ik, negative_by_ik, anchors, rng, repetitions=1):
    triplets = []
    for _ in range(repetitions):
        for ik in anchors:
            members = by_ik[ik]
            a, p = rng.choice(members, size=2, replace=False)
            choices = negative_by_ik[ik]
            if isinstance(choices, dict):
                family = "same_formula" if rng.random() < 0.5 else "different_formula"
                negative_ik = rng.choice(choices[family])
            else:
                negative_ik = rng.choice(choices)
            n = rng.choice(by_ik[negative_ik])
            triplets.append((a, p, n))
    return np.asarray(triplets, dtype=np.int64)


@torch.no_grad()
def evaluate(model, backbone, official_raw, official_norm, triplets, margin):
    adapted = model(backbone, official_raw)
    def metrics(values):
        a, p, n = (values[triplets[:, i]] for i in range(3))
        pos, neg = (a * p).sum(1), (a * n).sum(1)
        return {
            "triplets": len(triplets), "triplet_accuracy": float((pos > neg).float().mean()),
            "margin_satisfaction": float((pos >= neg + margin).float().mean()),
            "mean_separation": float((pos - neg).mean()),
            "mean_loss": float(F.relu(margin - pos + neg).mean()),
        }
    return {
        "official": metrics(official_norm), "adapted": metrics(adapted),
        "preservation_cosine": float((adapted * official_norm).sum(1).mean()),
        "mean_embedding_shift": float((1 - (adapted * official_norm).sum(1)).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/e1/lipid_projection_pilot"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/e1/lipid_projection_pilot/runs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--preserve-weight", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--negative-mode", choices=("same_formula", "different_formula_embedding_hard", "mixed_equal"), default="same_formula")
    parser.add_argument(
        "--selection-negative-mode",
        choices=("same_formula", "different_formula_embedding_hard"),
        default="same_formula",
        help="Shared internal validation task used for epoch selection across ablations.",
    )
    args = parser.parse_args()
    run_dir = args.output_dir / f"seed_{args.seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    cache = np.load(args.pilot_dir / "cache.npz")
    backbone = torch.from_numpy(cache["backbone"]).float()
    official_norm = torch.from_numpy(cache["official"]).float()
    head = torch.load(args.pilot_dir / "official_head.pt", map_location="cpu", weights_only=True)
    official_raw = F.linear(backbone, head["weight"].float(), head["bias"].float()).detach()
    records = json.loads((args.pilot_dir / "records.json").read_text())
    train_pool = pools(records, "train", args.negative_mode, official_norm)
    val_pool = pools(records, "val", args.selection_negative_mode, official_norm)
    fixed_val = sample_triplets(records, *val_pool, np.random.default_rng(args.seed + 991), repetitions=20)

    model = LowRankDelta(backbone.shape[1], args.rank, args.scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = None; history = []; stale = 0
    for epoch in range(args.epochs):
        model.train()
        triplets = sample_triplets(records, *train_pool, np.random.default_rng(args.seed + epoch * 1009), repetitions=2)
        order = np.random.default_rng(args.seed + epoch).permutation(len(triplets))
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = triplets[order[start:start + args.batch_size]]
            unique = np.unique(idx)
            adapted_all = model(backbone, official_raw)
            a, p, n = (adapted_all[idx[:, i]] for i in range(3))
            pos, neg = (a * p).sum(1), (a * n).sum(1)
            triplet_loss = F.relu(args.margin - pos + neg).mean()
            preserve = (1 - (adapted_all[unique] * official_norm[unique]).sum(1)).mean()
            loss = triplet_loss + args.preserve_weight * preserve
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval(); val = evaluate(model, backbone, official_raw, official_norm, fixed_val, args.margin)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val["adapted"].items() if k != "triplets"}, "preservation_cosine": val["preservation_cosine"]}
        history.append(row)
        # Choose by validation accuracy, separation, then preservation; internal validation only.
        score = (val["adapted"]["triplet_accuracy"], val["adapted"]["mean_separation"], val["preservation_cosine"])
        if best is None or score > best[0]:
            best = (score, epoch + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, val)
            stale = 0
        else:
            stale += 1
        if stale >= args.patience: break
    model.load_state_dict(best[2])
    train_fixed = sample_triplets(records, *train_pool, np.random.default_rng(args.seed + 1991), repetitions=10)
    result = {
        "status": "lipid_projection_adapter_pilot", "seed": args.seed,
        "config": {key: value for key, value in vars(args).items() if not isinstance(value, Path)},
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best[1],
        "train": evaluate(model, backbone, official_raw, official_norm, train_fixed, args.margin),
        "internal_formula_disjoint_val": evaluate(model, backbone, official_raw, official_norm, fixed_val, args.margin),
        "selection_rule": "shared internal validation task: triplet accuracy, then separation, then preservation; no external set used",
        "negative_mode": args.negative_mode,
        "selection_negative_mode": args.selection_negative_mode,
    }
    torch.save({"format": "lipid_projection_adapter_v1", "state_dict": model.state_dict(), "dimension": backbone.shape[1], "rank": args.rank, "scale": args.scale, "result": result}, run_dir / "best.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
