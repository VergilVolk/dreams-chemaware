"""Discover condition-invariant directions in frozen DreaMS activations.

This is a deliberately small, deterministic baseline.  It uses no curated
chemical rules and trains no neural network.  For same-molecule, same-adduct
pairs collected under different conditions, it finds directions that maximize
between-molecule variance while minimizing within-molecule condition variance.

The discovery and confirmation cohorts are fitted independently so that axis
reproducibility can be audited instead of inferred from reconstruction quality.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCOVERY = ROOT / "data/validation/multilevel_factor_pilot1000_qc"
DEFAULT_CONFIRMATION = ROOT / "data/validation/multilevel_factor_confirm1000_qc"
DEFAULT_OUTPUT = ROOT / "data/validation/condition_invariant_subspace_l7.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--n-factors", type=int, default=16)
    parser.add_argument("--ridge-fraction", type=float, default=0.1)
    return parser.parse_args()


def load_same_adduct_pairs(directory: Path, layer: int, kind: str) -> np.ndarray:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    layers = report["config"]["layers"]
    if layer not in layers:
        raise ValueError(f"Layer {layer} is absent from {directory}; available={layers}")
    layer_index = layers.index(layer)
    activations = np.load(directory / f"{kind}_precursor.npy", mmap_mode="r")
    pairs = json.loads((directory / "pairs.json").read_text(encoding="utf-8"))
    if len(activations) != 2 * len(pairs):
        raise ValueError("Expected exactly two activation rows per molecule pair")
    selected = [
        pair_id for pair_id, pair in enumerate(pairs)
        if pair["adduct"][0] == pair["adduct"][1]
        and (
            pair["condition_difference"]["instrument"]
            or pair["condition_difference"]["collision_energy_ge_10"]
        )
    ]
    if not selected:
        raise ValueError(f"No same-adduct, cross-condition pairs in {directory}")
    rows = np.asarray([[2 * i, 2 * i + 1] for i in selected], dtype=np.int64)
    return np.asarray(activations[rows, layer_index], dtype=np.float64)


@dataclass
class Factorization:
    center: np.ndarray
    directions: np.ndarray
    eigenvalues: np.ndarray
    pca_variance: float
    within_ridge: float


def fit_factorization(
    pairs: np.ndarray,
    pca_dim: int,
    n_factors: int,
    ridge_fraction: float,
) -> Factorization:
    flat = pairs.reshape(-1, pairs.shape[-1])
    pca_dim = min(pca_dim, len(flat) - 1, flat.shape[1])
    n_factors = min(n_factors, pca_dim)
    pca = PCA(n_components=pca_dim, svd_solver="full")
    reduced = pca.fit_transform(flat).reshape(len(pairs), 2, pca_dim)

    pair_means = reduced.mean(axis=1)
    pair_differences = reduced[:, 0] - reduced[:, 1]
    between = np.cov(pair_means, rowvar=False)
    within = 0.5 * np.cov(pair_differences, rowvar=False)
    scale = float(np.trace(within) / pca_dim)
    ridge = max(ridge_fraction * scale, 1e-8)
    values, vectors = eigh(between, within + ridge * np.eye(pca_dim))
    order = np.argsort(values)[::-1][:n_factors]
    values = values[order]
    vectors = vectors[:, order]

    directions = pca.components_.T @ vectors
    directions /= np.linalg.norm(directions, axis=0, keepdims=True).clip(1e-12)
    return Factorization(
        center=pca.mean_,
        directions=directions,
        eigenvalues=values,
        pca_variance=float(pca.explained_variance_ratio_.sum()),
        within_ridge=ridge,
    )


def external_metrics(pairs: np.ndarray, fit: Factorization) -> dict:
    scores = (pairs - fit.center) @ fit.directions
    means = scores.mean(axis=1)
    differences = scores[:, 0] - scores[:, 1]
    mean_variance = np.var(means, axis=0, ddof=1)
    difference_variance = np.var(differences, axis=0, ddof=1)
    within_noise = 0.5 * difference_variance
    between_signal = np.maximum(mean_variance - 0.25 * difference_variance, 0.0)
    ratio = between_signal / np.maximum(within_noise, 1e-12)
    icc = between_signal / np.maximum(between_signal + within_noise, 1e-12)
    paired_cosine = np.sum(scores[:, 0] * scores[:, 1], axis=1) / np.maximum(
        np.linalg.norm(scores[:, 0], axis=1) * np.linalg.norm(scores[:, 1], axis=1),
        1e-12,
    )
    return {
        "n_pairs": int(len(pairs)),
        "factor_icc": icc.tolist(),
        "factor_between_within_ratio": ratio.tolist(),
        "median_factor_icc": float(np.median(icc)),
        "median_between_within_ratio": float(np.median(ratio)),
        "multifactor_paired_cosine_mean": float(np.mean(paired_cosine)),
        "multifactor_paired_cosine_median": float(np.median(paired_cosine)),
    }


def direction_replication(a: Factorization, b: Factorization) -> dict:
    similarity = np.abs(a.directions.T @ b.directions)
    row, column = linear_sum_assignment(-similarity)
    matched = similarity[row, column]
    basis_a, _ = np.linalg.qr(a.directions)
    basis_b, _ = np.linalg.qr(b.directions)
    principal_cosines = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    ambient_dim = a.directions.shape[0]
    random_mean_squared_cosine = len(principal_cosines) / ambient_dim
    return {
        "matched_absolute_cosines": matched.tolist(),
        "median_absolute_cosine": float(np.median(matched)),
        "minimum_absolute_cosine": float(np.min(matched)),
        "directions_ge_0_7": int(np.sum(matched >= 0.7)),
        "directions_ge_0_5": int(np.sum(matched >= 0.5)),
        "principal_cosines": principal_cosines.tolist(),
        "subspace_mean_squared_cosine": float(np.mean(principal_cosines ** 2)),
        "random_subspace_reference": float(random_mean_squared_cosine),
    }


def analyze_kind(args: argparse.Namespace, kind: str) -> dict:
    discovery_pairs = load_same_adduct_pairs(args.discovery, args.layer, kind)
    confirmation_pairs = load_same_adduct_pairs(args.confirmation, args.layer, kind)
    discovery_fit = fit_factorization(
        discovery_pairs, args.pca_dim, args.n_factors, args.ridge_fraction
    )
    confirmation_fit = fit_factorization(
        confirmation_pairs, args.pca_dim, args.n_factors, args.ridge_fraction
    )
    return {
        "discovery_pairs": int(len(discovery_pairs)),
        "confirmation_pairs": int(len(confirmation_pairs)),
        "discovery_pca_variance": discovery_fit.pca_variance,
        "confirmation_pca_variance": confirmation_fit.pca_variance,
        "discovery_eigenvalues": discovery_fit.eigenvalues.tolist(),
        "confirmation_eigenvalues": confirmation_fit.eigenvalues.tolist(),
        "discovery_fit_on_confirmation": external_metrics(
            confirmation_pairs, discovery_fit
        ),
        "independent_direction_replication": direction_replication(
            discovery_fit, confirmation_fit
        ),
    }


def main() -> None:
    args = parse_args()
    if args.n_factors > args.pca_dim:
        raise ValueError("--n-factors cannot exceed --pca-dim")
    result = {
        "status": "condition_invariant_subspace_pilot",
        "method": (
            "Deterministic generalized eigendecomposition: maximize between-molecule "
            "variance while minimizing within-molecule cross-condition variance."
        ),
        "supervision": (
            "Same molecule and same adduct; instrument and/or collision energy differs. "
            "No curated chemical-rule labels are used."
        ),
        "config": {
            "discovery": str(args.discovery),
            "confirmation": str(args.confirmation),
            "layer": args.layer,
            "pca_dim": args.pca_dim,
            "n_factors": args.n_factors,
            "ridge_fraction": args.ridge_fraction,
        },
        "raw_ssl": analyze_kind(args, "raw"),
        "official_finetuned": analyze_kind(args, "official"),
        "interpretation_limit": (
            "A reproducible invariant direction is only a candidate chemistry-associated "
            "factor. It is not a named chemical mechanism until peak-level perturbation "
            "and independent chemical annotation validate it."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key in ("raw_ssl", "official_finetuned"):
        metrics = result[key]["discovery_fit_on_confirmation"]
        replication = result[key]["independent_direction_replication"]
        print(
            f"{key}: external median ICC={metrics['median_factor_icc']:.3f}; "
            f"paired cosine={metrics['multifactor_paired_cosine_mean']:.3f}; "
            f"axis median |cos|={replication['median_absolute_cosine']:.3f}; "
            f"axes >=0.7={replication['directions_ge_0_7']}/{args.n_factors}; "
            f"subspace={replication['subspace_mean_squared_cosine']:.3f} "
            f"(random={replication['random_subspace_reference']:.3f})"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
