"""Exact paired validation of official and trained heads in one backbone pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from e1_checkpoint_io import official_head_state, torch_load_compat
from train_counterfactual_dreams import CounterfactualDataset, attach_teacher
from train_e1_identity import load_base_model


ROOT = Path(__file__).resolve().parents[1]


def summarize(parts: list[dict[str, np.ndarray]]) -> dict:
    def cat(key: str) -> np.ndarray:
        return np.concatenate([part[key] for part in parts])

    margin = cat("margin")
    identity = cat("identity")
    confounder = cat("confounder")
    random_cosine = cat("random_cosine")
    return {
        "n": int(len(margin)),
        "pairwise_accuracy": float(np.mean(margin > 0)),
        "pairwise_accuracy_margin_gt_1e-6": float(np.mean(margin > 1e-6)),
        "near_ties_abs_margin_le_1e-6": int(np.sum(np.abs(margin) <= 1e-6)),
        "mean_margin": float(margin.mean()),
        "median_margin": float(np.median(margin)),
        "identity_cf_order_accuracy": float(np.mean(identity > 0)),
        "confounder_cf_order_accuracy": float(np.mean(confounder > 0)),
        "mean_identity_effect": float(identity.mean()),
        "mean_confounder_effect": float(confounder.mean()),
        "random_mask_cosine": float(random_cosine.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "data/e1/counterfactual_peak_finetune/counterfactual_peak_finetune_split.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv")
    parser.add_argument("--teacher-embeddings", type=Path, default=ROOT / "data/validation/large_observability_embeddings_discovery/official_embeddings.npy")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--base-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/counterfactual_formal_cpu_head_audit/exact_head_comparison.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))

    split = pd.read_csv(args.split)
    manifest = pd.read_csv(args.manifest)
    teacher = np.load(args.teacher_embeddings).astype(np.float32)
    frame = attach_teacher(split, manifest, teacher)
    validation = frame.loc[frame["pilot_split"] == "validation"].copy()
    dataset = CounterfactualDataset(validation, args.data, args.n_highest_peaks, 20260910, False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model, _ = load_base_model(args.base_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks)
    package = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if package.get("stage") != "head":
        raise ValueError("This exact shared-backbone comparison is only valid for a head-stage checkpoint")
    base = torch_load_compat(args.base_checkpoint, map_location="cpu")
    official_state = official_head_state(base)
    trained_state = package["head_state_dict"]
    official_weight = official_state["weight"].to(device)
    official_bias = official_state["bias"].to(device)
    trained_weight = trained_state["weight"].to(device)
    trained_bias = trained_state["bias"].to(device)
    model.backbone.eval()
    names = ["clean", "identity_masked", "confounder_masked", "random_masked", "positive", "negative"]
    results: dict[str, list[dict[str, np.ndarray]]] = {"official": [], "model": []}
    with torch.inference_mode():
        for position, batch in enumerate(loader, start=1):
            size = len(batch["clean"])
            spectra = torch.cat([batch[name].to(device) for name in names], dim=0)
            hidden = model.backbone(spectra, None)[:, 0]
            for label, weight, bias in (
                ("official", official_weight, official_bias),
                ("model", trained_weight, trained_bias),
            ):
                encoded = F.normalize(F.linear(hidden, weight, bias), dim=-1)
                q, qi, qc, qr, positive, negative = encoded.split(size)
                margin = (q * positive).sum(1) - (q * negative).sum(1)
                identity_margin = (qi * positive).sum(1) - (qi * negative).sum(1)
                confounder_margin = (qc * positive).sum(1) - (qc * negative).sum(1)
                identity_valid = batch["has_identity"].numpy().astype(bool)
                confounder_valid = batch["has_confounder"].numpy().astype(bool)
                results[label].append({
                    "margin": margin.float().cpu().numpy(),
                    "identity": (margin - identity_margin).float().cpu().numpy()[identity_valid],
                    "confounder": (confounder_margin - margin).float().cpu().numpy()[confounder_valid],
                    "random_cosine": (q * qr).sum(1).float().cpu().numpy(),
                })
            if position % 10 == 0:
                print(f"evaluated {position}/{len(loader)} batches", flush=True)
    report = {"official": summarize(results["official"]), "model": summarize(results["model"])}
    report["delta"] = {
        key: report["model"][key] - report["official"][key]
        for key in (
            "pairwise_accuracy_margin_gt_1e-6",
            "mean_margin",
            "identity_cf_order_accuracy",
            "confounder_cf_order_accuracy",
            "mean_identity_effect",
            "mean_confounder_effect",
            "random_mask_cosine",
        )
    }
    report["claim_limit"] = "single seed; internal formula-isolated validation"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
