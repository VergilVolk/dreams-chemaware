"""Align peak-token SAE factors across seeds on held-out molecules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-support", type=int, default=100)
    parser.add_argument("--direction-threshold", type=float, default=0.70)
    parser.add_argument("--activation-threshold", type=float, default=0.50)
    return parser.parse_args()


def load_run(path: Path) -> dict:
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    package = torch.load(path / "peak_token_sae.pt", map_location="cpu", weights_only=True)
    weight = package["state_dict"]["encoder.weight"].numpy().astype(np.float64)
    components = package["pca_components"].numpy().astype(np.float64)
    # Tied decoder directions in the original 1024-dimensional token space.
    directions = weight @ components
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-12)
    codes = np.load(path / "confirmation_codes.npy", mmap_mode="r").astype(np.float32)
    return {"path": path, "report": report, "directions": directions, "codes": codes}


def feature_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a -= a.mean(axis=0, keepdims=True)
    b -= b.mean(axis=0, keepdims=True)
    denominator = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    return np.divide(
        np.sum(a * b, axis=0), denominator,
        out=np.zeros(a.shape[1], dtype=float), where=denominator > 0,
    )


def quantiles(values: np.ndarray) -> dict:
    return {str(q): float(np.quantile(values, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}


def main() -> None:
    args = parse_args()
    if len(args.runs) < 3:
        raise ValueError("At least three seeds are required")
    runs = [load_run(path) for path in args.runs]
    shapes = {(run["directions"].shape, run["codes"].shape) for run in runs}
    if len(shapes) != 1:
        raise RuntimeError(f"Run shapes differ: {shapes}")
    reference = runs[0]
    reference_support = np.sum(reference["codes"] > 0, axis=0)
    direction_columns = []
    activation_columns = []
    matched_columns = []
    comparisons = []
    for candidate in runs[1:]:
        similarity = reference["directions"] @ candidate["directions"].T
        ref_idx, cand_idx = linear_sum_assignment(-similarity)
        order = np.argsort(ref_idx)
        cand_idx = cand_idx[order]
        matched_direction = similarity[np.arange(len(ref_idx)), cand_idx]
        matched_activation = feature_correlations(
            reference["codes"], candidate["codes"][:, cand_idx]
        )
        candidate_support = np.sum(candidate["codes"][:, cand_idx] > 0, axis=0)
        supported = (reference_support >= args.minimum_support) & (candidate_support >= args.minimum_support)
        direction_columns.append(matched_direction)
        activation_columns.append(matched_activation)
        matched_columns.append(cand_idx)
        comparisons.append({
            "candidate": str(candidate["path"]),
            "direction_cosine": quantiles(matched_direction),
            "activation_correlation": quantiles(matched_activation),
            "supported_features": int(supported.sum()),
            "supported_direction_ge_threshold": int(np.sum(supported & (matched_direction >= args.direction_threshold))),
            "supported_activation_ge_threshold": int(np.sum(supported & (matched_activation >= args.activation_threshold))),
            "supported_both_ge_threshold": int(np.sum(supported & (matched_direction >= args.direction_threshold) & (matched_activation >= args.activation_threshold))),
        })
    directions = np.stack(direction_columns, axis=1)
    activations = np.stack(activation_columns, axis=1)
    matches = np.stack(matched_columns, axis=1)
    stable = (
        (reference_support >= args.minimum_support)
        & np.all(directions >= args.direction_threshold, axis=1)
        & np.all(activations >= args.activation_threshold, axis=1)
    )
    dtype = [
        ("reference_factor", "i4"), ("reference_support", "i4"),
        ("mean_direction_cosine", "f8"), ("min_direction_cosine", "f8"),
        ("mean_activation_correlation", "f8"), ("min_activation_correlation", "f8"),
        ("stable", "?"),
    ]
    table = np.empty(len(stable), dtype=dtype)
    table["reference_factor"] = np.arange(len(stable))
    table["reference_support"] = reference_support
    table["mean_direction_cosine"] = directions.mean(axis=1)
    table["min_direction_cosine"] = directions.min(axis=1)
    table["mean_activation_correlation"] = activations.mean(axis=1)
    table["min_activation_correlation"] = activations.min(axis=1)
    table["stable"] = stable
    args.output_dir.mkdir(parents=True, exist_ok=True)
    header = ",".join(table.dtype.names)
    rows = np.column_stack([table[name] for name in table.dtype.names])
    np.savetxt(args.output_dir / "factor_stability.csv", rows, delimiter=",", header=header, comments="", fmt=["%d", "%d", "%.8f", "%.8f", "%.8f", "%.8f", "%s"])
    np.save(args.output_dir / "matched_factor_indices.npy", matches)
    report = {
        "status": "peak_token_sae_seed_stability",
        "reference_run": str(reference["path"]),
        "n_runs": len(runs),
        "n_factors": len(stable),
        "minimum_confirmation_peak_support": args.minimum_support,
        "direction_threshold": args.direction_threshold,
        "activation_threshold": args.activation_threshold,
        "stable_features_all_comparisons": int(stable.sum()),
        "stable_feature_ids": np.flatnonzero(stable).astype(int).tolist(),
        "mean_direction_cosine_quantiles": quantiles(directions.mean(axis=1)),
        "mean_activation_correlation_quantiles": quantiles(activations.mean(axis=1)),
        "comparisons": comparisons,
        "claim_limit": "Stability is necessary but not sufficient for fragmentation semantics.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
