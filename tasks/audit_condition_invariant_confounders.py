"""Audit whether condition-invariant DreaMS directions survive confounder removal."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

import discover_condition_invariant_subspace as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HDF5 = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_OUTPUT = ROOT / "data/validation/condition_invariant_subspace_l7_adjusted.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=base.DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation", type=Path, default=base.DEFAULT_CONFIRMATION)
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--n-factors", type=int, default=8)
    parser.add_argument("--ridge-fraction", type=float, default=0.1)
    parser.add_argument("--regression-ridge", type=float, default=1e-3)
    return parser.parse_args()


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@dataclass
class PairBundle:
    raw: np.ndarray
    official: np.ndarray
    rows: np.ndarray
    precursor_mz: np.ndarray
    collision_energy: np.ndarray
    peak_count: np.ndarray
    instrument: np.ndarray
    adduct: np.ndarray


def load_bundle(directory: Path, hdf5_path: Path, layer: int) -> PairBundle:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    layers = report["config"]["layers"]
    layer_index = layers.index(layer)
    pairs = json.loads((directory / "pairs.json").read_text(encoding="utf-8"))
    selected = [
        i for i, pair in enumerate(pairs)
        if pair["adduct"][0] == pair["adduct"][1]
        and (
            pair["condition_difference"]["instrument"]
            or pair["condition_difference"]["collision_energy_ge_10"]
        )
    ]
    activation_rows = np.asarray([[2 * i, 2 * i + 1] for i in selected])
    hdf5_rows = np.asarray([pairs[i]["rows"] for i in selected], dtype=np.int64)
    raw_all = np.load(directory / "raw_precursor.npy", mmap_mode="r")
    official_all = np.load(directory / "official_precursor.npy", mmap_mode="r")
    raw = np.asarray(raw_all[activation_rows, layer_index], dtype=np.float64)
    official = np.asarray(official_all[activation_rows, layer_index], dtype=np.float64)

    flat_rows = hdf5_rows.reshape(-1)
    order = np.argsort(flat_rows)
    inverse = np.argsort(order)
    with h5py.File(hdf5_path, "r") as handle:
        sorted_rows = flat_rows[order]
        precursor_mz = np.asarray(handle["precursor_mz"][sorted_rows])[inverse]
        collision_energy = np.asarray(handle["COLLISION_ENERGY"][sorted_rows])[inverse]
        instrument = np.asarray(
            [decode(v) for v in handle["INSTRUMENT_TYPE"][sorted_rows]]
        )[inverse]
        adduct = np.asarray([decode(v) for v in handle["adduct"][sorted_rows]])[inverse]
        spectra = np.asarray(handle["spectrum"][sorted_rows])[inverse]
    peak_count = np.count_nonzero(spectra[:, 1, :] > 0, axis=1)
    shape = (len(selected), 2)
    return PairBundle(
        raw=raw,
        official=official,
        rows=hdf5_rows,
        precursor_mz=precursor_mz.reshape(shape),
        collision_energy=collision_energy.reshape(shape),
        peak_count=peak_count.reshape(shape),
        instrument=instrument.reshape(shape),
        adduct=adduct.reshape(shape),
    )


@dataclass
class DesignSchema:
    continuous_mean: np.ndarray
    continuous_std: np.ndarray
    collision_energy_fill: float
    instruments: list[str]
    adducts: list[str]


def fit_schema(bundle: PairBundle) -> DesignSchema:
    collision = bundle.collision_energy.reshape(-1).astype(float)
    finite = np.isfinite(collision)
    collision_fill = float(np.median(collision[finite])) if finite.any() else 0.0
    collision = np.where(finite, collision, collision_fill)
    continuous = np.column_stack([
        np.log(bundle.precursor_mz.reshape(-1).clip(1e-8)),
        np.log1p(bundle.peak_count.reshape(-1)),
        collision,
    ])
    mean = continuous.mean(axis=0)
    std = continuous.std(axis=0)
    std[std < 1e-8] = 1.0
    return DesignSchema(
        continuous_mean=mean,
        continuous_std=std,
        collision_energy_fill=collision_fill,
        instruments=sorted(set(bundle.instrument.reshape(-1).tolist())),
        adducts=sorted(set(bundle.adduct.reshape(-1).tolist())),
    )


def design_matrix(bundle: PairBundle, schema: DesignSchema) -> np.ndarray:
    collision = bundle.collision_energy.reshape(-1).astype(float)
    collision = np.where(
        np.isfinite(collision), collision, schema.collision_energy_fill
    )
    continuous = np.column_stack([
        np.log(bundle.precursor_mz.reshape(-1).clip(1e-8)),
        np.log1p(bundle.peak_count.reshape(-1)),
        collision,
    ])
    continuous = (continuous - schema.continuous_mean) / schema.continuous_std
    instrument = bundle.instrument.reshape(-1)
    adduct = bundle.adduct.reshape(-1)
    categorical = [
        (instrument == level).astype(float) for level in schema.instruments[1:]
    ] + [
        (adduct == level).astype(float) for level in schema.adducts[1:]
    ]
    columns = [np.ones(len(continuous)), *continuous.T.tolist(), *categorical]
    return np.column_stack(columns)


def fit_residualizer(
    activations: np.ndarray, design: np.ndarray, ridge: float
) -> tuple[np.ndarray, float]:
    x = activations.reshape(-1, activations.shape[-1])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ x)
    residual = x - design @ coefficients
    centered = x - x.mean(axis=0, keepdims=True)
    removed_fraction = 1.0 - np.sum(residual ** 2) / np.maximum(
        np.sum(centered ** 2), 1e-12
    )
    return coefficients, float(removed_fraction)


def apply_residualizer(
    activations: np.ndarray, design: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    x = activations.reshape(-1, activations.shape[-1])
    residual = x - design @ coefficients
    return residual.reshape(activations.shape)


def analyze(
    discovery: PairBundle,
    confirmation: PairBundle,
    kind: str,
    discovery_design: np.ndarray,
    confirmation_design: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    discovery_x = getattr(discovery, kind)
    confirmation_x = getattr(confirmation, kind)
    coefficients, removed = fit_residualizer(
        discovery_x, discovery_design, args.regression_ridge
    )
    discovery_residual = apply_residualizer(
        discovery_x, discovery_design, coefficients
    )
    confirmation_residual = apply_residualizer(
        confirmation_x, confirmation_design, coefficients
    )
    discovery_fit = base.fit_factorization(
        discovery_residual,
        args.pca_dim,
        args.n_factors,
        args.ridge_fraction,
    )
    confirmation_fit = base.fit_factorization(
        confirmation_residual,
        args.pca_dim,
        args.n_factors,
        args.ridge_fraction,
    )
    metrics = base.external_metrics(confirmation_residual, discovery_fit)
    replication = base.direction_replication(discovery_fit, confirmation_fit)
    return {
        "discovery_variance_removed_fraction": removed,
        "external_metrics_after_adjustment": metrics,
        "independent_direction_replication_after_adjustment": replication,
    }


def strict_mass_neighbor_counts(directory: Path) -> dict:
    pairs = json.loads((directory / "pairs.json").read_text(encoding="utf-8"))
    masses = np.asarray([np.mean(pair["precursor_mz"]) for pair in pairs])
    adducts = np.asarray([
        pair["adduct"][0] if pair["adduct"][0] == pair["adduct"][1] else ""
        for pair in pairs
    ])
    links = []
    for i, mass in enumerate(masses):
        valid = (
            (adducts != "")
            & (adducts == adducts[i])
            & (np.arange(len(pairs)) != i)
            & (np.abs(masses - mass) / mass * 1e6 <= 10)
        )
        links.append(int(valid.sum()) if adducts[i] else 0)
    return {
        "eligible_molecules": int(np.sum(adducts != "")),
        "anchors_with_same_adduct_10ppm_neighbor": int(np.sum(np.asarray(links) > 0)),
        "directed_candidate_links": int(np.sum(links)),
    }


def main() -> None:
    args = parse_args()
    discovery = load_bundle(args.discovery, args.hdf5, args.layer)
    confirmation = load_bundle(args.confirmation, args.hdf5, args.layer)
    schema = fit_schema(discovery)
    discovery_design = design_matrix(discovery, schema)
    confirmation_design = design_matrix(confirmation, schema)
    result = {
        "status": "confounder_adjusted_condition_invariant_subspace",
        "confounders": [
            "log precursor m/z",
            "log peak count",
            "collision energy",
            "instrument type",
            "adduct",
        ],
        "config": vars(args) | {
            "discovery": str(args.discovery),
            "confirmation": str(args.confirmation),
            "hdf5": str(args.hdf5),
            "output": str(args.output),
        },
        "discovery_mass_neighbor_audit": strict_mass_neighbor_counts(args.discovery),
        "confirmation_mass_neighbor_audit": strict_mass_neighbor_counts(args.confirmation),
        "raw_ssl": analyze(
            discovery, confirmation, "raw", discovery_design, confirmation_design, args
        ),
        "official_finetuned": analyze(
            discovery,
            confirmation,
            "official",
            discovery_design,
            confirmation_design,
            args,
        ),
        "decision_rule": (
            "Do not compute a formal 10 ppm retrieval AUC unless the cohort contains "
            "enough same-adduct mass-neighbor anchors. Stable axes remain candidates, "
            "not named chemistry, until peak perturbation validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    for key in ("raw_ssl", "official_finetuned"):
        item = result[key]
        metrics = item["external_metrics_after_adjustment"]
        replication = item["independent_direction_replication_after_adjustment"]
        print(
            f"{key}: variance removed={item['discovery_variance_removed_fraction']:.3f}; "
            f"external ICC={metrics['median_factor_icc']:.3f}; "
            f"axes >=0.7={replication['directions_ge_0_7']}/{args.n_factors}; "
            f"subspace={replication['subspace_mean_squared_cosine']:.3f}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
