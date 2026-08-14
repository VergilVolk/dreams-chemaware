"""Probe transparent chemical properties from frozen DreaMS embeddings.

This is the first, low-cost stage of cross-modal chemical-factor discovery.
Models are fitted only on the molecule-disjoint discovery cohort and evaluated
on the confirmation cohort.  Every descriptor is evaluated both as observed
and after removing the linear/quadratic effect of exact molecular mass.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_DISCOVERY = ROOT / "data/validation/mass_dense_factor_discovery"
DEFAULT_CONFIRMATION = ROOT / "data/validation/mass_dense_factor_confirmation"
DEFAULT_OUTPUT = ROOT / "data/validation/crossmodal_descriptor_probe"


ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--n-factors", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_embeddings(directory: Path, kind: str, layer: int) -> np.ndarray:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    layer_index = report["config"]["layers"].index(layer)
    values = np.load(directory / f"{kind}_precursor.npy")[:, layer_index]
    pairs = json.loads((directory / "pairs.json").read_text(encoding="utf-8"))
    values = values.reshape(len(pairs), 2, values.shape[-1]).astype(np.float64)
    values /= np.linalg.norm(values, axis=-1, keepdims=True).clip(1e-12)
    values = values.mean(axis=1)
    return values / np.linalg.norm(values, axis=-1, keepdims=True).clip(1e-12)


def load_smiles(data: Path, directory: Path) -> tuple[list[str], list[str]]:
    pairs = json.loads((directory / "pairs.json").read_text(encoding="utf-8"))
    rows = np.asarray([pair["rows"][0] for pair in pairs], dtype=np.int64)
    order = np.argsort(rows)
    inverse = np.argsort(order)
    with h5py.File(data, "r") as handle:
        smiles = handle["smiles"].asstr()[rows[order]][inverse].tolist()
    return smiles, [pair["ik14"] for pair in pairs]


def descriptor_values(mol: Chem.Mol) -> dict[str, float]:
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    heavy = max(1, mol.GetNumHeavyAtoms())
    values = {
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "MolMR": Crippen.MolMR(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "HeteroatomCount": Descriptors.NumHeteroatoms(mol),
        "RingCount": Lipinski.RingCount(mol),
        "AromaticRingCount": Lipinski.NumAromaticRings(mol),
        "AliphaticRingCount": Lipinski.NumAliphaticRings(mol),
        "SaturatedRingCount": Lipinski.NumSaturatedRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "NHOHCount": Lipinski.NHOHCount(mol),
        "NOCount": Lipinski.NOCount(mol),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        "BalabanJ": Descriptors.BalabanJ(mol),
        "HallKierAlpha": Descriptors.HallKierAlpha(mol),
        "Kappa1": Descriptors.Kappa1(mol),
        "Kappa2": Descriptors.Kappa2(mol),
        "Kappa3": Descriptors.Kappa3(mol),
        "Chi0v": Descriptors.Chi0v(mol),
        "Chi1v": Descriptors.Chi1v(mol),
    }
    for element in ELEMENTS:
        count = atoms.count(element)
        values[f"Count_{element}"] = count
        values[f"Fraction_{element}"] = count / heavy
    return values


def compute_descriptors(smiles: list[str]) -> tuple[list[str], np.ndarray]:
    rows = []
    for value in smiles:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"RDKit failed to parse SMILES: {value}")
        rows.append(descriptor_values(mol))
    names = list(rows[0])
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite molecular descriptor encountered")
    return names, matrix


def residualize_targets(
    discovery: np.ndarray, confirmation: np.ndarray, mass_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mass_discovery = discovery[:, mass_index]
    mass_confirmation = confirmation[:, mass_index]
    scaler = StandardScaler()
    mass_d = scaler.fit_transform(mass_discovery[:, None])
    mass_c = scaler.transform(mass_confirmation[:, None])
    spline = SplineTransformer(
        n_knots=12, degree=3, knots="quantile", include_bias=True
    )
    design_discovery = spline.fit_transform(mass_d)
    design_confirmation = spline.transform(mass_c)
    prediction_d = np.zeros_like(discovery)
    prediction_c = np.zeros_like(confirmation)
    for index in range(discovery.shape[1]):
        model = RidgeCV(alphas=np.logspace(-5, 4, 20))
        model.fit(design_discovery, discovery[:, index])
        prediction_d[:, index] = model.predict(design_discovery)
        prediction_c[:, index] = model.predict(design_confirmation)
    return (
        discovery - prediction_d,
        confirmation - prediction_c,
        prediction_c,
    )


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def embedding_transform(
    discovery: np.ndarray, confirmation: np.ndarray, pca_dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler()
    scaled_discovery = scaler.fit_transform(discovery)
    scaled_confirmation = scaler.transform(confirmation)
    pca = PCA(
        n_components=min(pca_dim, len(discovery) - 1, discovery.shape[1]),
        svd_solver="randomized",
        random_state=seed,
    )
    return pca.fit_transform(scaled_discovery), pca.transform(scaled_confirmation), scaler, pca


def probe_descriptors(
    z_discovery: np.ndarray,
    z_confirmation: np.ndarray,
    y_discovery: np.ndarray,
    y_confirmation: np.ndarray,
    mass_prediction: np.ndarray,
    names: list[str],
    kind: str,
    target_mode: str,
) -> list[dict]:
    rows = []
    for index, name in enumerate(names):
        model = RidgeCV(alphas=np.logspace(-3, 4, 16))
        model.fit(z_discovery, y_discovery[:, index])
        prediction = model.predict(z_confirmation)
        rows.append({
            "embedding": kind,
            "target_mode": target_mode,
            "descriptor": name,
            "ridge_alpha": float(model.alpha_),
            "confirmation_r2": float(r2_score(y_confirmation[:, index], prediction)),
            "confirmation_spearman": safe_spearman(y_confirmation[:, index], prediction),
            "mass_only_r2": float(
                r2_score(y_confirmation[:, index], mass_prediction[:, index])
            ) if target_mode == "raw" else 0.0,
        })
    return rows


def fit_pls_factorization(
    x: np.ndarray, y: np.ndarray, pca_dim: int, n_factors: int, seed: int
) -> dict:
    x_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)
    pca = PCA(
        n_components=min(pca_dim, len(x) - 1, x.shape[1]),
        svd_solver="randomized",
        random_state=seed,
    )
    z = pca.fit_transform(x_scaled)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)
    pls = PLSRegression(
        n_components=min(n_factors, z.shape[1], y.shape[1]),
        scale=False,
        max_iter=2000,
    )
    pls.fit(z, y_scaled)
    direction_scaled = pca.components_.T @ pls.x_weights_
    direction_original = direction_scaled / x_scaler.scale_[:, None]
    direction_original /= np.linalg.norm(direction_original, axis=0, keepdims=True).clip(1e-12)
    return {
        "x_scaler": x_scaler,
        "pca": pca,
        "y_scaler": y_scaler,
        "pls": pls,
        "directions": direction_original,
    }


def external_pls_metrics(
    fit: dict,
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
) -> tuple[list[dict], list[float]]:
    z = fit["pca"].transform(fit["x_scaler"].transform(x))
    y_scaled = fit["y_scaler"].transform(y)
    x_scores, y_scores = fit["pls"].transform(z, y_scaled)
    correlations = []
    factors = []
    for factor in range(x_scores.shape[1]):
        corr = safe_spearman(x_scores[:, factor], y_scores[:, factor])
        correlations.append(corr)
        loadings = fit["pls"].y_loadings_[:, factor]
        top = np.argsort(np.abs(loadings))[::-1][:5]
        factors.append({
            "factor": factor + 1,
            "confirmation_crossview_spearman": corr,
            "top_descriptor_loadings": [
                {"descriptor": names[i], "loading": float(loadings[i])}
                for i in top
            ],
        })
    return factors, correlations


def replication(discovery_fit: dict, confirmation_fit: dict) -> dict:
    similarity = np.abs(discovery_fit["directions"].T @ confirmation_fit["directions"])
    row, col = linear_sum_assignment(-similarity)
    matched = similarity[row, col]
    singular = np.linalg.svd(
        discovery_fit["directions"].T @ confirmation_fit["directions"],
        compute_uv=False,
    )
    return {
        "matched_direction_cosines": matched.tolist(),
        "directions_ge_0_7": int(np.sum(matched >= 0.7)),
        "median_matched_cosine": float(np.median(matched)),
        "maximum_matched_cosine": float(np.max(matched)),
        "subspace_singular_values": singular.tolist(),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    smiles_d, ik_d = load_smiles(args.data, args.discovery)
    smiles_c, ik_c = load_smiles(args.data, args.confirmation)
    overlap = set(ik_d) & set(ik_c)
    if overlap:
        raise RuntimeError(f"Molecule leakage detected: {len(overlap)}")
    names_d, y_d = compute_descriptors(smiles_d)
    names_c, y_c = compute_descriptors(smiles_c)
    if names_d != names_c:
        raise RuntimeError("Descriptor schemas differ")
    names = names_d
    mass_index = names.index("ExactMolWt")
    residual_d_all, residual_c_all, mass_prediction_c = residualize_targets(
        y_d, y_c, mass_index
    )
    # ExactMolWt is the nuisance variable itself.  Its residual is numerical
    # round-off and must never be treated as a chemical target.  Also remove
    # descriptors with effectively no residual variation in discovery.
    residual_indices = [
        index for index, name in enumerate(names)
        if name != "ExactMolWt" and np.std(residual_d_all[:, index]) > 1e-8
    ]
    residual_names = [names[index] for index in residual_indices]
    residual_d = residual_d_all[:, residual_indices]
    residual_c = residual_c_all[:, residual_indices]

    all_probe_rows = []
    summary = {
        "status": "crossmodal_descriptor_probe",
        "purpose": (
            "Test whether frozen DreaMS embeddings contain externally replicable, "
            "transparent chemical-property information beyond exact molecular mass."
        ),
        "audit": {
            "discovery_molecules": len(ik_d),
            "confirmation_molecules": len(ik_c),
            "molecule_overlap": len(overlap),
            "n_raw_descriptors": len(names),
            "n_mass_residual_descriptors": len(residual_names),
        },
        "config": {
            "data": str(args.data),
            "discovery": str(args.discovery),
            "confirmation": str(args.confirmation),
            "layer": args.layer,
            "pca_dim": args.pca_dim,
            "n_factors": args.n_factors,
            "mass_residualization": (
                "discovery-fit quantile cubic spline (12 knots) on ExactMolWt; "
                "mass is also reported as a standalone external baseline"
            ),
        },
        "models": {},
    }
    for kind in ("raw", "official"):
        x_d = load_embeddings(args.discovery, kind, args.layer)
        x_c = load_embeddings(args.confirmation, kind, args.layer)
        z_d, z_c, _, _ = embedding_transform(x_d, x_c, args.pca_dim, args.seed)
        raw_rows = probe_descriptors(
            z_d, z_c, y_d, y_c, mass_prediction_c, names, kind, "raw"
        )
        residual_rows = probe_descriptors(
            z_d, z_c, residual_d, residual_c,
            np.zeros_like(residual_c), residual_names, kind, "mass_residual"
        )
        all_probe_rows.extend(raw_rows + residual_rows)

        discovery_fit = fit_pls_factorization(
            x_d, residual_d, args.pca_dim, args.n_factors, args.seed
        )
        confirmation_fit = fit_pls_factorization(
            x_c, residual_c, args.pca_dim, args.n_factors, args.seed + 1
        )
        factors, correlations = external_pls_metrics(
            discovery_fit, x_c, residual_c, residual_names
        )
        residual_lookup = {row["descriptor"]: row for row in residual_rows}
        ranked = sorted(
            residual_lookup.values(),
            key=lambda row: row["confirmation_spearman"]
            if np.isfinite(row["confirmation_spearman"]) else -np.inf,
            reverse=True,
        )
        summary["models"][kind] = {
            "top_mass_residual_descriptors": ranked[:12],
            "n_mass_residual_spearman_ge_0_3": int(sum(
                np.isfinite(row["confirmation_spearman"])
                and row["confirmation_spearman"] >= 0.3
                for row in residual_rows
            )),
            "n_mass_residual_positive_r2": int(sum(
                row["confirmation_r2"] > 0 for row in residual_rows
            )),
            "external_pls_factors": factors,
            "median_absolute_external_factor_spearman": float(
                np.median(np.abs(correlations))
            ),
            "independent_factor_replication": replication(
                discovery_fit, confirmation_fit
            ),
        }
        print(
            f"{kind}: residual rho>=0.3 "
            f"{summary['models'][kind]['n_mass_residual_spearman_ge_0_3']}/{len(residual_names)}; "
            f"positive R2 "
            f"{summary['models'][kind]['n_mass_residual_positive_r2']}/{len(residual_names)}; "
            f"stable PLS axes>=0.7 "
            f"{summary['models'][kind]['independent_factor_replication']['directions_ge_0_7']}"
        )

    write_csv(args.output_dir / "descriptor_probe.csv", all_probe_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
