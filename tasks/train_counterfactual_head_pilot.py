"""CPU pilot: train only the official DreaMS embedding head.

Backbone embeddings are frozen.  The head learns clean triplet ranking plus
counterfactual ordering under identity-only and confounder-only peak masks,
with preservation against the official head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pilot_multilevel_factor_activations as multi
from e1_checkpoint_io import official_head_state
from pilot_paired_layer_cka import preprocess_spectrum
from run_large_targeted_peak_occlusion import parse_values, target_tokens


def encode_backbone(model, tensors: list[torch.Tensor], batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.stack(tensors)), batch_size=batch_size, shuffle=False)
    values = []
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for (batch,) in loader:
            values.append(model(batch.to(device=device, dtype=dtype), None)[:, 0].float().cpu().numpy())
    return np.concatenate(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=Path("data/e1/counterfactual_peak_finetune/counterfactual_peak_finetune_split.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/validation/large_observability_embeddings_discovery/manifest.csv"))
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--raw-checkpoint", type=Path, default=multi.DEFAULT_RAW)
    parser.add_argument("--official-checkpoint", type=Path, default=multi.DEFAULT_OFFICIAL)
    parser.add_argument("--output-dir", type=Path, default=Path("data/e1/counterfactual_head_pilot"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/e1/counterfactual_head_cache"))
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--encode-batch-size", type=int, default=24)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--triplet-margin", type=float, default=0.05)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-margin", type=float, default=0.02)
    parser.add_argument("--counterfactual-weight", type=float, default=0.5)
    parser.add_argument("--preserve-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    frame = pd.read_csv(args.split)
    manifest = pd.read_csv(args.manifest).set_index("hdf5_row")

    hidden_path = args.cache_dir / "hidden.npy"
    index_path = args.cache_dir / "index.csv"
    if hidden_path.exists() and index_path.exists():
        hidden = np.load(hidden_path)
        index = pd.read_csv(index_path)
    else:
        tensors: list[torch.Tensor] = []
        index_rows = []
        with h5py.File(args.data, "r") as handle:
            for row in frame.itertuples(index=False):
                raw = np.asarray(handle["spectrum"][int(row.query_hdf5_row)])
                precursor = float(manifest.at[int(row.query_hdf5_row), "precursor_mz"])
                clean = preprocess_spectrum(raw, precursor, args.n_highest_peaks)
                identity_mask = target_tokens(clean, parse_values(row.identity_peak_mz), 0.005)
                confounder_mask = target_tokens(clean, parse_values(row.confounder_peak_mz), 0.005)
                variants = [clean]
                identity_variant = clean.clone()
                if len(identity_mask):
                    identity_variant[identity_mask] = 0
                variants.append(identity_variant)
                confounder_variant = clean.clone()
                if len(confounder_mask):
                    confounder_variant[confounder_mask] = 0
                variants.append(confounder_variant)
                for hdf5_row in (row.identity_hdf5_row, row.confounder_hdf5_row):
                    candidate_raw = np.asarray(handle["spectrum"][int(hdf5_row)])
                    candidate_precursor = float(manifest.at[int(hdf5_row), "precursor_mz"])
                    variants.append(preprocess_spectrum(candidate_raw, candidate_precursor, args.n_highest_peaks))
                start = len(tensors)
                tensors.extend(variants)
                index_rows.append({
                    "clean": start, "identity_masked": start + 1, "confounder_masked": start + 2,
                    "positive": start + 3, "negative": start + 4,
                    "has_identity": bool(len(identity_mask)), "has_confounder": bool(len(confounder_mask)),
                    "pilot_split": row.pilot_split, "formula": row.formula,
                })
        index = pd.DataFrame(index_rows)
        raw_package = multi.torch_load_compat(args.raw_checkpoint, map_location="cpu")
        official_for_backbone = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
        backbone = multi.reconstruct_backbone(raw_package, multi.official_backbone_state(official_for_backbone), device)
        backbone.eval()
        hidden = encode_backbone(backbone, tensors, args.encode_batch_size, device)
        np.save(hidden_path, hidden)
        index.to_csv(index_path, index=False)
        del backbone, tensors, raw_package, official_for_backbone

    official_package = multi.torch_load_compat(args.official_checkpoint, map_location="cpu")
    head_state = official_head_state(official_package)
    official_weight = head_state["weight"].float().to(device)
    official_bias = head_state["bias"].float().to(device)
    head = torch.nn.Linear(official_weight.shape[1], official_weight.shape[0]).to(device)
    head.weight.data.copy_(official_weight)
    head.bias.data.copy_(official_bias)
    hidden_tensor = torch.from_numpy(hidden).float().to(device)
    with torch.inference_mode():
        official_embedding = F.normalize(F.linear(hidden_tensor, official_weight, official_bias), dim=-1)

    train_ids = np.flatnonzero(index["pilot_split"].to_numpy() == "train")
    validation_ids = np.flatnonzero(index["pilot_split"].to_numpy() == "validation")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    def evaluate(ids: np.ndarray) -> dict[str, float]:
        with torch.inference_mode():
            rows = index.iloc[ids]
            q = F.normalize(head(hidden_tensor[rows["clean"].to_numpy()]), dim=-1)
            p = F.normalize(head(hidden_tensor[rows["positive"].to_numpy()]), dim=-1)
            n = F.normalize(head(hidden_tensor[rows["negative"].to_numpy()]), dim=-1)
            margin = (q * p).sum(1) - (q * n).sum(1)
            official_q = official_embedding[rows["clean"].to_numpy()]
            preservation = (q * official_q).sum(1)
            qi = F.normalize(head(hidden_tensor[rows["identity_masked"].to_numpy()]), dim=-1)
            qc = F.normalize(head(hidden_tensor[rows["confounder_masked"].to_numpy()]), dim=-1)
            identity_margin = (qi * p).sum(1) - (qi * n).sum(1)
            confounder_margin = (qc * p).sum(1) - (qc * n).sum(1)
            identity_mask = torch.as_tensor(rows["has_identity"].to_numpy(bool), device=device)
            confounder_mask = torch.as_tensor(rows["has_confounder"].to_numpy(bool), device=device)
            return {
                "pairwise_accuracy": float((margin > 0).float().mean().cpu()),
                "mean_margin": float(margin.mean().cpu()),
                "mean_cosine_to_official": float(preservation.mean().cpu()),
                "identity_counterfactual_order_accuracy": float(
                    ((margin[identity_mask] - identity_margin[identity_mask]) > 0).float().mean().cpu()
                ),
                "confounder_counterfactual_order_accuracy": float(
                    ((confounder_margin[confounder_mask] - margin[confounder_mask]) > 0).float().mean().cpu()
                ),
            }

    history, best_state, best_metric = [], None, -np.inf
    rng = np.random.default_rng(args.seed)
    baseline = evaluate(validation_ids)
    for epoch in range(args.epochs):
        rng.shuffle(train_ids)
        head.train()
        epoch_loss = []
        for start in range(0, len(train_ids), args.train_batch_size):
            ids = train_ids[start:start + args.train_batch_size]
            rows = index.iloc[ids]
            q = F.normalize(head(hidden_tensor[rows["clean"].to_numpy()]), dim=-1)
            p = F.normalize(head(hidden_tensor[rows["positive"].to_numpy()]), dim=-1)
            n = F.normalize(head(hidden_tensor[rows["negative"].to_numpy()]), dim=-1)
            clean_margin = (q * p).sum(1) - (q * n).sum(1)
            loss = args.triplet_weight * F.relu(args.triplet_margin - clean_margin).mean()
            q_official = official_embedding[rows["clean"].to_numpy()]
            p_official = official_embedding[rows["positive"].to_numpy()]
            n_official = official_embedding[rows["negative"].to_numpy()]
            preservation = 3 - (q * q_official).sum(1) - (p * p_official).sum(1) - (n * n_official).sum(1)
            loss = loss + args.preserve_weight * preservation.mean() / 3
            identity_mask = torch.as_tensor(rows["has_identity"].to_numpy(bool), device=device)
            if identity_mask.any():
                qi = F.normalize(head(hidden_tensor[rows["identity_masked"].to_numpy()]), dim=-1)
                masked = (qi * p).sum(1) - (qi * n).sum(1)
                loss = loss + args.counterfactual_weight * F.relu(
                    args.counterfactual_margin + masked[identity_mask] - clean_margin[identity_mask]
                ).mean()
            confounder_mask = torch.as_tensor(rows["has_confounder"].to_numpy(bool), device=device)
            if confounder_mask.any():
                qc = F.normalize(head(hidden_tensor[rows["confounder_masked"].to_numpy()]), dim=-1)
                masked = (qc * p).sum(1) - (qc * n).sum(1)
                loss = loss + args.counterfactual_weight * F.relu(
                    args.counterfactual_margin + clean_margin[confounder_mask] - masked[confounder_mask]
                ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(float(loss.detach().cpu()))
        head.eval()
        validation = evaluate(validation_ids)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(epoch_loss))} | validation)
        selection = validation["pairwise_accuracy"] + 0.05 * validation["mean_cosine_to_official"]
        if selection > best_metric:
            best_metric = selection
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    head.load_state_dict(best_state)
    final = evaluate(validation_ids)
    torch.save({"head_state_dict": best_state, "source": str(args.official_checkpoint)}, args.output_dir / "counterfactual_head.pt")
    report = {
        "status": "counterfactual_head_cpu_pilot",
        "backbone": "frozen official DreaMS fine-tuned backbone",
        "trainable": "official 1024x1024 embedding projection head only",
        "train_examples": len(train_ids), "validation_examples": len(validation_ids),
        "loss_weights": {"triplet": args.triplet_weight, "counterfactual": args.counterfactual_weight, "preserve": args.preserve_weight},
        "baseline_validation": baseline, "final_validation": final,
        "delta_pairwise_accuracy": final["pairwise_accuracy"] - baseline["pairwise_accuracy"],
        "claim_limit": "internal formula-isolated pilot; confirmation and test not evaluated",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
