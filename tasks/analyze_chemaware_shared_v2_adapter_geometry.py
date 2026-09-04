"""Diagnose whether trained ChemAware adapters learn diverse peak-level changes.

The report is descriptive only.  It measures residual rank/directionality,
peak-gate concentration, official-embedding preservation, and agreement across
two or more checkpoints.  It never selects a checkpoint or evaluates a sealed
test set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))

from chemaware_shared_v2_core import ChemAwareTokenStore  # noqa: E402
from dreams.models.chem_aware.shared_embedding_v2 import SignedPeakResidualAdapter  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
import pilot_multilevel_factor_activations as multi  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        help="LABEL=path/to/adapter.pt; repeat for checkpoint comparisons",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(
            ("min", "q25", "median", "q75", "q90", "q99", "max"),
            np.quantile(values, (0, 0.25, 0.5, 0.75, 0.9, 0.99, 1)),
        )
    }


def normalized_entropy(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    safe = np.clip(weights, 1e-12, None)
    entropy = -np.sum(np.where(mask, safe * np.log(safe), 0.0), axis=1)
    denominator = np.log(np.maximum(mask.sum(axis=1), 2))
    return entropy / denominator


def residual_geometry(delta: np.ndarray) -> dict[str, object]:
    norms = np.linalg.norm(delta, axis=1)
    active = norms > 1e-10
    if not np.any(active):
        return {
            "delta_norm": quantiles(norms),
            "active_spectra": 0,
            "effective_rank": 0.0,
            "top_singular_energy_fraction": 0.0,
            "mean_direction_fraction": 0.0,
            "off_diagonal_delta_cosine_mean": 0.0,
            "off_diagonal_delta_cosine_median": 0.0,
        }
    centered = delta[active] - delta[active].mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular.astype(np.float64) ** 2
    probability = energy / np.clip(energy.sum(), 1e-30, None)
    probability = probability[probability > 0]
    effective_rank = float(np.exp(-np.sum(probability * np.log(probability))))
    unit = delta[active] / norms[active, None]
    cosine = unit @ unit.T
    off_diagonal = cosine[~np.eye(len(unit), dtype=bool)]
    rms = float(np.sqrt(np.mean(norms[active] ** 2)))
    return {
        "delta_norm": quantiles(norms),
        "active_spectra": int(np.sum(active)),
        "effective_rank": effective_rank,
        "top_singular_energy_fraction": float(energy[0] / energy.sum()),
        "mean_direction_fraction": float(
            np.linalg.norm(delta[active].mean(axis=0)) / max(rms, 1e-12)
        ),
        "off_diagonal_delta_cosine_mean": float(np.mean(off_diagonal)),
        "off_diagonal_delta_cosine_median": float(np.median(off_diagonal)),
    }


def load_adapter(path: Path, device: torch.device) -> tuple[SignedPeakResidualAdapter, dict]:
    package = multi.torch_load_compat(path, map_location="cpu")
    config = package.get("adapter_config")
    state = package.get("adapter_state")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise RuntimeError(f"malformed ChemAware adapter checkpoint: {path}")
    adapter = SignedPeakResidualAdapter(
        int(config["embedding_dim"]), int(config["hidden_dim"]),
        float(config["delta_bound"]), float(config.get("gate_temperature", 1.0)),
        int(config.get("gate_topk", 0)),
        bool(config.get("contextual_gate", False)),
        bool(config.get("global_branch", False)),
    ).to(device)
    adapter.load_state_dict(state, strict=True)
    adapter.eval()
    return adapter, package


def encode_adapter(
    adapter: SignedPeakResidualAdapter,
    store: ChemAwareTokenStore,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    output = {name: [] for name in ("official", "adapted", "delta", "support", "conflict")}
    with torch.inference_mode():
        for left in range(0, len(store.rows), batch_size):
            rows = store.rows[left:left + batch_size]
            official, adapted, delta, support, conflict = store.adapt(adapter, rows, device)
            for name, value in (
                ("official", official), ("adapted", adapted), ("delta", delta),
                ("support", support), ("conflict", conflict),
            ):
                output[name].append(value.cpu().numpy())
    return {name: np.concatenate(blocks) for name, blocks in output.items()}


def summarize(values: dict[str, np.ndarray], valid: np.ndarray) -> dict[str, object]:
    official = values["official"]
    adapted = values["adapted"]
    support = values["support"]
    conflict = values["conflict"]
    gate_cosine = np.sum(support * conflict, axis=1) / np.clip(
        np.linalg.norm(support, axis=1) * np.linalg.norm(conflict, axis=1), 1e-12, None
    )
    return {
        **residual_geometry(values["delta"]),
        "official_cosine_preservation": quantiles(np.sum(official * adapted, axis=1)),
        "support_gate_normalized_entropy": quantiles(normalized_entropy(support, valid)),
        "conflict_gate_normalized_entropy": quantiles(normalized_entropy(conflict, valid)),
        "support_gate_top_mass": quantiles(np.max(support, axis=1)),
        "conflict_gate_top_mass": quantiles(np.max(conflict, axis=1)),
        "support_conflict_gate_cosine": quantiles(gate_cosine),
    }


def compare(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, float]:
    left_delta = left["delta"]
    right_delta = right["delta"]
    denominator = np.clip(
        np.linalg.norm(left_delta, axis=1) * np.linalg.norm(right_delta, axis=1),
        1e-12,
        None,
    )
    return {
        "delta_cosine_mean": float(np.mean(np.sum(left_delta * right_delta, axis=1) / denominator)),
        "delta_l2_difference_mean": float(np.mean(np.linalg.norm(left_delta - right_delta, axis=1))),
        "adapted_cosine_mean": float(np.mean(np.sum(left["adapted"] * right["adapted"], axis=1))),
        "support_gate_l1_mean": float(np.mean(np.sum(np.abs(left["support"] - right["support"]), axis=1))),
        "conflict_gate_l1_mean": float(np.mean(np.sum(np.abs(left["conflict"] - right["conflict"]), axis=1))),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite geometry report: {args.output}")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    labels: dict[str, Path] = {}
    for value in args.adapter:
        if "=" not in value:
            raise ValueError("--adapter must be LABEL=PATH")
        label, raw_path = value.split("=", 1)
        if not label or label in labels:
            raise ValueError(f"duplicate or empty adapter label: {label}")
        labels[label] = Path(raw_path)
    store = ChemAwareTokenStore(
        args.token_dir, args.graph, args.official_checkpoint, require_formal=False
    )
    valid = np.asarray(store.valid, dtype=bool)
    device = torch.device(args.device)
    encoded: dict[str, dict[str, np.ndarray]] = {}
    reports = {}
    for label, path in labels.items():
        adapter, package = load_adapter(path, device)
        values = encode_adapter(adapter, store, device, args.batch_size)
        encoded[label] = values
        reports[label] = {
            "checkpoint_sha256": sha256_file(path),
            "best_epoch": package.get("best_epoch"),
            "teacher_control": package.get("teacher_control"),
            "chemical_objective": package.get("chemical_objective"),
            "geometry": summarize(values, valid),
        }
    comparisons = {}
    ordered = list(labels)
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1:]:
            comparisons[f"{left}__vs__{right}"] = compare(encoded[left], encoded[right])
    report = {
        "status": "chemaware_shared_v2_adapter_geometry_complete",
        "formal": False,
        "spectra": int(len(store.rows)),
        "adapters": reports,
        "comparisons": comparisons,
        "claim_limit": "descriptive mechanism diagnostic only; no retrieval or chemical claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
