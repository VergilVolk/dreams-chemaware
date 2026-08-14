"""Minimal Base--Finetune Crosscoder smoke test on paired DreaMS activations.

The goal is limited to checking that a sparse shared code can reconstruct raw
and official activations on molecule-disjoint data. No chemical interpretation
or rule matching is performed in this script.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DISCOVERY = ROOT / "data/validation/multilevel_factor_pilot1000_qc"
DEFAULT_CONFIRMATION = ROOT / "data/validation/multilevel_factor_confirm1000_qc"
DEFAULT_OUTPUT = ROOT / "data/validation/crosscoder_smoke_precursor_l7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument(
        "--token-type", choices=("precursor", "peak"), default="precursor"
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_layer(directory: Path, layer: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "multilevel_activation_pilot":
        raise RuntimeError(f"Invalid activation run: {directory} ({report.get('status')})")
    layers = report["config"]["layers"]
    if layer not in layers:
        raise ValueError(f"Layer {layer} not in saved layers {layers}")
    layer_index = layers.index(layer)
    raw = np.load(directory / "raw_precursor.npy")[:, layer_index].astype(np.float32)
    official = np.load(directory / "official_precursor.npy")[:, layer_index].astype(np.float32)
    pairs = json.loads((directory / "pairs.json").read_text(encoding="utf-8"))
    return raw, official, [item["pair_id"] for item in pairs]


def load_activations(
    directory: Path,
    layer: int,
    token_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load activations and a molecule id for every training sample."""
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "multilevel_activation_pilot":
        raise RuntimeError(f"Invalid activation run: {directory} ({report.get('status')})")
    layers = report["config"]["layers"]
    if layer not in layers:
        raise ValueError(f"Layer {layer} not in saved layers {layers}")
    layer_index = layers.index(layer)
    n_molecules = int(report["config"]["n_molecules"])
    n_spectra = 2 * n_molecules
    spectrum_molecule = np.arange(n_spectra, dtype=np.int64) // 2
    if token_type == "precursor":
        raw = np.load(directory / "raw_precursor.npy")[:, layer_index].astype(np.float32)
        official = np.load(directory / "official_precursor.npy")[:, layer_index].astype(np.float32)
        sample_molecule = spectrum_molecule
    elif token_type == "peak":
        mask = np.load(directory / "peak_mask.npy").reshape(-1)
        raw_array = np.load(directory / "raw_peak.npy", mmap_mode="r")[:, layer_index]
        official_array = np.load(directory / "official_peak.npy", mmap_mode="r")[:, layer_index]
        raw = np.asarray(raw_array, dtype=np.float32).reshape(-1, raw_array.shape[-1])[mask]
        official = (
            np.asarray(official_array, dtype=np.float32)
            .reshape(-1, official_array.shape[-1])[mask]
        )
        sample_molecule = np.repeat(
            spectrum_molecule, raw_array.shape[1]
        )[mask]
    else:
        raise ValueError(token_type)
    if len(raw) != len(official) or len(raw) != len(sample_molecule):
        raise RuntimeError("Activation/sample mapping mismatch")
    return raw, official, sample_molecule, n_molecules


class Crosscoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, top_k: int):
        super().__init__()
        if top_k <= 0 or top_k > hidden_dim:
            raise ValueError("top_k must lie in [1, hidden_dim]")
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.encoder_raw = nn.Linear(input_dim, hidden_dim, bias=False)
        self.encoder_official = nn.Linear(input_dim, hidden_dim, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder_raw = nn.Linear(hidden_dim, input_dim, bias=False)
        self.decoder_official = nn.Linear(hidden_dim, input_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.decoder_raw.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.decoder_official.weight, a=np.sqrt(5))
        self.normalize_decoders()
        with torch.no_grad():
            self.encoder_raw.weight.copy_(0.5 * self.decoder_raw.weight.T)
            self.encoder_official.weight.copy_(0.5 * self.decoder_official.weight.T)
            self.latent_bias.zero_()

    @torch.no_grad()
    def normalize_decoders(self) -> None:
        self.decoder_raw.weight.div_(
            self.decoder_raw.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        )
        self.decoder_official.weight.div_(
            self.decoder_official.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        )

    def encode(self, raw: torch.Tensor, official: torch.Tensor) -> torch.Tensor:
        preactivation = (
            self.encoder_raw(raw) + self.encoder_official(official) + self.latent_bias
        )
        positive = torch.relu(preactivation)
        values, indices = torch.topk(positive, k=self.top_k, dim=-1)
        latent = torch.zeros_like(positive)
        latent.scatter_(dim=-1, index=indices, src=values)
        return latent

    def forward(
        self, raw: torch.Tensor, official: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(raw, official)
        return self.decoder_raw(latent), self.decoder_official(latent), latent


def normalize_train(
    raw: np.ndarray,
    official: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    raw_mean = raw[train_indices].mean(axis=0)
    official_mean = official[train_indices].mean(axis=0)
    raw_centered = raw - raw_mean
    official_centered = official - official_mean
    raw_rms = float(np.sqrt(np.mean(np.square(raw_centered[train_indices]))))
    official_rms = float(np.sqrt(np.mean(np.square(official_centered[train_indices]))))
    raw_normalized = raw_centered / max(raw_rms, 1e-8)
    official_normalized = official_centered / max(official_rms, 1e-8)
    stats = {
        "raw_mean": raw_mean,
        "official_mean": official_mean,
        "raw_rms": raw_rms,
        "official_rms": official_rms,
    }
    return raw_normalized, official_normalized, stats


def apply_normalization(raw: np.ndarray, official: np.ndarray, stats: dict):
    return (
        (raw - stats["raw_mean"]) / max(stats["raw_rms"], 1e-8),
        (official - stats["official_mean"]) / max(stats["official_rms"], 1e-8),
    )


def make_loader(
    raw: np.ndarray,
    official: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(raw[indices]), torch.from_numpy(official[indices])
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate(
    model: Crosscoder,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    raw_squared, official_squared = 0.0, 0.0
    raw_total, official_total = 0.0, 0.0
    raw_cosines, official_cosines = [], []
    usage = torch.zeros(model.hidden_dim, dtype=torch.long)
    n_rows = 0
    with torch.inference_mode():
        for raw, official in loader:
            raw = raw.to(device)
            official = official.to(device)
            raw_hat, official_hat, latent = model(raw, official)
            raw_squared += float(torch.square(raw_hat - raw).sum().cpu())
            official_squared += float(torch.square(official_hat - official).sum().cpu())
            raw_total += float(torch.square(raw).sum().cpu())
            official_total += float(torch.square(official).sum().cpu())
            raw_cosines.append(torch.nn.functional.cosine_similarity(raw_hat, raw).cpu())
            official_cosines.append(
                torch.nn.functional.cosine_similarity(official_hat, official).cpu()
            )
            usage += (latent > 0).sum(dim=0).cpu()
            n_rows += len(raw)
    usage_fraction = usage.float() / max(n_rows, 1)
    return {
        "n_rows": n_rows,
        "raw_nmse": raw_squared / max(raw_total, 1e-12),
        "official_nmse": official_squared / max(official_total, 1e-12),
        "raw_reconstruction_cosine": float(torch.cat(raw_cosines).mean()),
        "official_reconstruction_cosine": float(torch.cat(official_cosines).mean()),
        "used_features": int((usage > 0).sum()),
        "dead_features": int((usage == 0).sum()),
        "median_feature_usage": float(usage_fraction.median()),
        "max_feature_usage": float(usage_fraction.max()),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but unavailable")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        discovery_raw,
        discovery_official,
        discovery_molecule,
        discovery_n_molecules,
    ) = load_activations(
        args.discovery, args.layer, args.token_type
    )
    (
        confirmation_raw,
        confirmation_official,
        confirmation_molecule,
        confirmation_n_molecules,
    ) = load_activations(
        args.confirmation, args.layer, args.token_type
    )

    # Split by molecule so all views and peak tokens remain together.
    rng = np.random.RandomState(args.seed)
    molecule_order = rng.permutation(discovery_n_molecules)
    n_train_molecules = int(round(0.8 * len(molecule_order)))
    train_molecules = molecule_order[:n_train_molecules]
    validation_molecules = molecule_order[n_train_molecules:]
    train_indices = np.flatnonzero(np.isin(discovery_molecule, train_molecules))
    validation_indices = np.flatnonzero(
        np.isin(discovery_molecule, validation_molecules)
    )
    external_indices = np.arange(len(confirmation_raw), dtype=np.int64)

    discovery_raw, discovery_official, stats = normalize_train(
        discovery_raw, discovery_official, train_indices
    )
    confirmation_raw, confirmation_official = apply_normalization(
        confirmation_raw, confirmation_official, stats
    )
    train_loader = make_loader(
        discovery_raw, discovery_official, train_indices, args.batch_size, True
    )
    validation_loader = make_loader(
        discovery_raw, discovery_official, validation_indices, args.batch_size, False
    )
    external_loader = make_loader(
        confirmation_raw, confirmation_official, external_indices, args.batch_size, False
    )

    model = Crosscoder(
        input_dim=discovery_raw.shape[1],
        hidden_dim=args.hidden_dim,
        top_k=args.top_k,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = []
    best_validation = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for raw, official in train_loader:
            raw = raw.to(device)
            official = official.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw_hat, official_hat, _ = model(raw, official)
            loss_raw = torch.mean(torch.square(raw_hat - raw))
            loss_official = torch.mean(torch.square(official_hat - official))
            loss = loss_raw + loss_official
            loss.backward()
            optimizer.step()
            model.normalize_decoders()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(model, validation_loader, device)
        validation_loss = validation["raw_nmse"] + validation["official_nmse"]
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation": validation,
        })
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:02d}: train={np.mean(losses):.4f}; "
                f"val NMSE raw/off={validation['raw_nmse']:.4f}/"
                f"{validation['official_nmse']:.4f}; "
                f"used={validation['used_features']}/{args.hidden_dim}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("No model state was recorded")
    model.load_state_dict(best_state)
    train_metrics = evaluate(model, train_loader, device)
    validation_metrics = evaluate(model, validation_loader, device)
    external_metrics = evaluate(model, external_loader, device)

    with torch.no_grad():
        raw_decoder = model.decoder_raw.weight.detach().cpu().numpy().T
        official_decoder = model.decoder_official.weight.detach().cpu().numpy().T
        decoder_difference = (
            np.linalg.norm(raw_decoder - official_decoder, axis=1)
            / np.clip(
                np.linalg.norm(raw_decoder, axis=1)
                + np.linalg.norm(official_decoder, axis=1),
                1e-12,
                None,
            )
        )
    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    report = {
        "status": "crosscoder_pipeline_smoke_test",
        "warning": (
            "Reconstruction and external generalization only. Feature semantics "
            "and stability are not established by one seed."
        ),
        "config": config,
        "split": {
            "train_molecules": len(train_molecules),
            "validation_molecules": len(validation_molecules),
            "external_molecules": confirmation_n_molecules,
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "external_samples": len(external_indices),
        },
        "train": train_metrics,
        "validation": validation_metrics,
        "external_confirmation": external_metrics,
        "decoder_difference": {
            "median": float(np.median(decoder_difference)),
            "p90": float(np.quantile(decoder_difference, 0.9)),
            "max": float(decoder_difference.max()),
        },
        "history": history,
        "decision_rule": (
            "Pipeline passes only if external raw and official NMSE are finite, "
            "below 1.0, and at least 25% of latent features are used."
        ),
    }
    report["pipeline_pass"] = bool(
        np.isfinite(external_metrics["raw_nmse"])
        and np.isfinite(external_metrics["official_nmse"])
        and external_metrics["raw_nmse"] < 1.0
        and external_metrics["official_nmse"] < 1.0
        and external_metrics["used_features"] >= 0.25 * args.hidden_dim
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save({
        "state_dict": best_state,
        "normalization": {
            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in stats.items()
        },
        "config": config,
    }, args.output_dir / "crosscoder.pt")

    print("\nCrosscoder smoke test")
    print("  validation:", validation_metrics)
    print("  external:", external_metrics)
    print("  pipeline pass:", report["pipeline_pass"])
    print("  output:", args.output_dir)


if __name__ == "__main__":
    main()
