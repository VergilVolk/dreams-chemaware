"""Train a molecule-disjoint Top-K SAE on official DreaMS peak tokens.

The pilot is deliberately narrow: it tests sparse reconstruction and external
reproducibility.  Existing chemical rules are not used for fitting, feature
selection, or evaluation.  Chemical semantics are assigned only in later
spectral/structural enrichment and perturbation steps.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pca-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--peaks-per-spectrum", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=20260812,
        help="Controls molecule split, balanced peak draws, and PCA only.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--token-transform",
        choices=("raw", "within_spectrum_centered"),
        default="raw",
        help=(
            "within_spectrum_centered removes the mean valid peak token of "
            "each spectrum before PCA/SAE fitting."
        ),
    )
    return parser.parse_args()


class TiedTopKSAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, top_k: int):
        super().__init__()
        if not 0 < top_k <= hidden_dim:
            raise ValueError("top_k must lie in [1, hidden_dim]")
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.top_k = top_k
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        dense = F.relu(self.encoder(x))
        values, indices = torch.topk(dense, k=self.top_k, dim=1)
        sparse = torch.zeros_like(dense)
        sparse.scatter_(1, indices, values)
        return sparse

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.encode(x)
        return code @ self.encoder.weight + self.decoder_bias, code


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_peak_split(directory: Path) -> dict:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "mass_dense_all_peak_tokens":
        raise RuntimeError(f"Unexpected activation artifact: {directory}")
    spectra = json.loads((directory / "spectra.json").read_text(encoding="utf-8"))
    mask = np.load(directory / "peak_mask.npy")
    values = np.load(directory / "peak_values.npy")
    tokens = np.load(directory / "official_peak.npy", mmap_mode="r")
    if tokens.shape[:2] != mask.shape or values.shape[:2] != mask.shape:
        raise RuntimeError("Peak-token/value/mask shapes do not align")
    spectrum_index = np.repeat(np.arange(len(spectra)), mask.sum(axis=1))
    peak_slot = np.concatenate([np.flatnonzero(row) for row in mask])
    spectrum_mean = np.empty((len(spectra), tokens.shape[-1]), dtype=np.float32)
    for spectrum in range(len(spectra)):
        valid_slots = np.flatnonzero(mask[spectrum])
        spectrum_mean[spectrum] = np.asarray(
            tokens[spectrum, valid_slots], dtype=np.float32
        ).mean(axis=0)
    return {
        "directory": directory,
        "report": report,
        "spectra": spectra,
        "mask": mask,
        "values": values,
        "tokens": tokens,
        "spectrum_index": spectrum_index,
        "peak_slot": peak_slot,
        "ik14": np.asarray([str(item["ik14"]) for item in spectra]),
        "spectrum_mean": spectrum_mean,
    }


def balanced_observations(split: dict, spectrum_ids: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Return flat valid-token indices with exactly n draws per spectrum."""
    rng = np.random.default_rng(seed)
    offsets = np.concatenate([[0], np.cumsum(split["mask"].sum(axis=1))])
    selected = []
    for spectrum in spectrum_ids:
        available = np.arange(offsets[spectrum], offsets[spectrum + 1])
        if not len(available):
            raise RuntimeError(f"Spectrum {spectrum} has no valid peak")
        chosen = rng.choice(available, size=n, replace=len(available) < n)
        selected.append(chosen)
    return np.concatenate(selected).astype(np.int64)


def materialize_tokens(
    split: dict, flat_indices: np.ndarray, token_transform: str
) -> np.ndarray:
    spectrum = split["spectrum_index"][flat_indices]
    slot = split["peak_slot"][flat_indices]
    output = np.asarray(split["tokens"][spectrum, slot], dtype=np.float32)
    if token_transform == "within_spectrum_centered":
        output = output - split["spectrum_mean"][spectrum]
    elif token_transform != "raw":
        raise ValueError(token_transform)
    return output


def transform(x: np.ndarray, pca: PCA, center: np.ndarray, rms: float) -> np.ndarray:
    return ((pca.transform(x) - center) / max(rms, 1e-8)).astype(np.float32)


def encode_in_batches(model: TiedTopKSAE, x: np.ndarray, batch_size: int) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size])
            outputs.append(model.encode(batch).numpy().astype(np.float16))
    return np.concatenate(outputs)


def evaluate(model: TiedTopKSAE, x: np.ndarray, batch_size: int) -> dict:
    squared_error = 0.0
    squared_total = 0.0
    cosines = []
    usage = np.zeros(model.encoder.out_features, dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size])
            reconstruction, code = model(batch)
            squared_error += float(torch.square(reconstruction - batch).sum())
            squared_total += float(torch.square(batch).sum())
            cosines.append(F.cosine_similarity(reconstruction, batch).numpy())
            usage += (code > 0).sum(dim=0).numpy()
    fractions = usage / max(len(x), 1)
    return {
        "n_peak_draws": len(x),
        "nmse": squared_error / max(squared_total, 1e-12),
        "reconstruction_cosine_mean": float(np.concatenate(cosines).mean()),
        "used_features": int(np.sum(usage > 0)),
        "dead_features": int(np.sum(usage == 0)),
        "median_activation_fraction": float(np.median(fractions)),
        "p90_activation_fraction": float(np.quantile(fractions, 0.9)),
        "max_activation_fraction": float(fractions.max()),
    }


def all_codes(
    split: dict,
    model: TiedTopKSAE,
    pca: PCA,
    center: np.ndarray,
    rms: float,
    batch_size: int,
    token_transform: str,
) -> np.ndarray:
    outputs = []
    total = len(split["spectrum_index"])
    for start in range(0, total, batch_size):
        ids = np.arange(start, min(start + batch_size, total))
        x = transform(
            materialize_tokens(split, ids, token_transform), pca, center, rms
        )
        outputs.append(encode_in_batches(model, x, batch_size))
    return np.concatenate(outputs)


def main() -> None:
    args = parse_args()
    if args.pca_dim <= 1 or args.hidden_dim <= args.top_k:
        raise ValueError("Invalid PCA/hidden/top-k dimensions")
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    discovery = load_peak_split(args.discovery)
    confirmation = load_peak_split(args.confirmation)

    unique_molecules = np.unique(discovery["ik14"])
    rng = np.random.default_rng(args.data_seed)
    order = rng.permutation(unique_molecules)
    n_train = int(round(0.8 * len(order)))
    train_molecules = set(order[:n_train])
    validation_molecules = set(order[n_train:])
    train_spectra = np.flatnonzero(np.isin(discovery["ik14"], list(train_molecules)))
    validation_spectra = np.flatnonzero(np.isin(discovery["ik14"], list(validation_molecules)))
    confirmation_spectra = np.arange(len(confirmation["spectra"]))
    train_ids = balanced_observations(discovery, train_spectra, args.peaks_per_spectrum, args.data_seed)
    validation_ids = balanced_observations(discovery, validation_spectra, args.peaks_per_spectrum, args.data_seed + 1)
    external_ids = balanced_observations(confirmation, confirmation_spectra, args.peaks_per_spectrum, args.data_seed + 2)

    print("Materializing spectrum-balanced peak draws", flush=True)
    train_raw = materialize_tokens(discovery, train_ids, args.token_transform)
    validation_raw = materialize_tokens(
        discovery, validation_ids, args.token_transform
    )
    external_raw = materialize_tokens(
        confirmation, external_ids, args.token_transform
    )
    print(f"  train/validation/external={len(train_raw)}/{len(validation_raw)}/{len(external_raw)}", flush=True)

    print("Fitting PCA on discovery-train molecules only", flush=True)
    pca = PCA(n_components=args.pca_dim, svd_solver="randomized", random_state=args.data_seed)
    train_reduced = pca.fit_transform(train_raw)
    center = train_reduced.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.square(train_reduced - center))))
    train_x = ((train_reduced - center) / max(rms, 1e-8)).astype(np.float32)
    validation_x = transform(validation_raw, pca, center, rms)
    external_x = transform(external_raw, pca, center, rms)
    del train_raw, validation_raw, external_raw, train_reduced

    model = TiedTopKSAE(args.pca_dim, args.hidden_dim, args.top_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x)), batch_size=args.batch_size, shuffle=True, num_workers=0)
    history = []
    best_nmse = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for (batch,) in loader:
            reconstruction, _ = model(batch)
            loss = F.mse_loss(reconstruction, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_metrics = evaluate(model, validation_x, args.batch_size)
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "validation": validation_metrics})
        if validation_metrics["nmse"] < best_nmse:
            best_nmse = validation_metrics["nmse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:02d}: train MSE={np.mean(losses):.4f}; val NMSE={validation_metrics['nmse']:.4f}; used={validation_metrics['used_features']}/{args.hidden_dim}", flush=True)
    if best_state is None:
        raise RuntimeError("No best state recorded")
    model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_x, args.batch_size)
    validation_metrics = evaluate(model, validation_x, args.batch_size)
    external_metrics = evaluate(model, external_x, args.batch_size)
    print("Encoding every retained peak for downstream audits", flush=True)
    discovery_codes = all_codes(
        discovery, model, pca, center, rms, args.batch_size, args.token_transform
    )
    confirmation_codes = all_codes(
        confirmation,
        model,
        pca,
        center,
        rms,
        args.batch_size,
        args.token_transform,
    )
    np.save(args.output_dir / "discovery_codes.npy", discovery_codes)
    np.save(args.output_dir / "confirmation_codes.npy", confirmation_codes)

    package = {
        "state_dict": best_state,
        "pca_components": torch.from_numpy(pca.components_.astype(np.float32)),
        "pca_mean": torch.from_numpy(pca.mean_.astype(np.float32)),
        "reduced_center": torch.from_numpy(center.astype(np.float32)),
        "reduced_rms": rms,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    torch.save(package, args.output_dir / "peak_token_sae.pt")
    report = {
        "status": "peak_token_sae_pilot",
        "claim_limit": "Sparse reconstruction and external recurrence only; chemical semantics are not yet established.",
        "rules_used_as_labels": False,
        "config": package["config"],
        "split": {
            "discovery_unique_molecules": len(unique_molecules),
            "train_molecules": len(train_molecules),
            "validation_molecules": len(validation_molecules),
            "confirmation_unique_molecules": int(len(np.unique(confirmation["ik14"]))),
            "train_spectra": len(train_spectra),
            "validation_spectra": len(validation_spectra),
            "confirmation_spectra": len(confirmation_spectra),
        },
        "pca_variance_retained": float(pca.explained_variance_ratio_.sum()),
        "train": train_metrics,
        "validation": validation_metrics,
        "external_confirmation": external_metrics,
        "all_peak_codes": {
            "discovery_shape": list(discovery_codes.shape),
            "confirmation_shape": list(confirmation_codes.shape),
        },
        "history": history,
    }
    report["pipeline_pass"] = bool(
        external_metrics["nmse"] < 1.0
        and external_metrics["used_features"] >= 0.5 * args.hidden_dim
        and np.isfinite(external_metrics["reconstruction_cosine_mean"])
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pca_variance_retained": report["pca_variance_retained"], "validation": validation_metrics, "external_confirmation": external_metrics, "pipeline_pass": report["pipeline_pass"]}, indent=2))


if __name__ == "__main__":
    main()
