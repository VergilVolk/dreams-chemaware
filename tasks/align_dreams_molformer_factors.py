"""Align frozen DreaMS and MolFormer views and test external chemical factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import SplineTransformer, StandardScaler

import probe_crossmodal_chemical_descriptors as descriptors


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCOVERY = ROOT / "data/validation/mass_dense_factor_discovery"
DEFAULT_CONFIRMATION = ROOT / "data/validation/mass_dense_factor_confirmation"
DEFAULT_MOLFORMER = ROOT / "data/validation/molformer_factor_embeddings"
DEFAULT_OUTPUT = ROOT / "data/validation/dreams_molformer_alignment.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--molformer", type=Path, default=DEFAULT_MOLFORMER)
    parser.add_argument("--data", type=Path, default=descriptors.DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--n-factors", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=-1, keepdims=True).clip(1e-12)


def load_dreams(directory: Path, kind: str, layer: int) -> np.ndarray:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    layer_index = report["config"]["layers"].index(layer)
    values = np.load(directory / f"{kind}_precursor.npy")[:, layer_index]
    pairs = json.loads((directory / "pairs.json").read_text(encoding="utf-8"))
    return values.reshape(len(pairs), 2, values.shape[-1]).astype(np.float64)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def fit_projection(x: np.ndarray, y: np.ndarray, dim: int, seed: int) -> dict:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y)
    n_components = min(dim, len(x) - 1, x.shape[1], y.shape[1])
    x_pca = PCA(n_components=n_components, whiten=True, random_state=seed)
    y_pca = PCA(n_components=n_components, whiten=True, random_state=seed + 1)
    x_score = x_pca.fit_transform(x_scaled)
    y_score = y_pca.fit_transform(y_scaled)
    ridge = RidgeCV(alphas=np.logspace(-3, 4, 24))
    ridge.fit(x_score, y_score)
    return {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "x_pca": x_pca,
        "y_pca": y_pca,
        "ridge": ridge,
    }


def project_spectra(fit: dict, values: np.ndarray) -> np.ndarray:
    shape = values.shape
    flat = values.reshape(-1, shape[-1])
    score = fit["x_pca"].transform(fit["x_scaler"].transform(flat))
    mapped = fit["ridge"].predict(score)
    return normalize(mapped.reshape(shape[0], shape[1], -1))


def project_molecules(fit: dict, values: np.ndarray) -> np.ndarray:
    score = fit["y_pca"].transform(fit["y_scaler"].transform(values))
    return normalize(score)


def retrieval_records(
    query: np.ndarray, candidates: np.ndarray, pairs: list[dict]
) -> list[dict]:
    records = []
    for pair_id, pair in enumerate(pairs):
        negatives = np.asarray(pair["negative_pair_ids"], dtype=int)
        if len(negatives) == 0:
            continue
        for view in (0, 1):
            positive = float(query[pair_id, view] @ candidates[pair_id])
            negative = candidates[negatives] @ query[pair_id, view]
            records.append({"positive": positive, "negatives": negative})
    return records


def summarize_retrieval(records: list[dict], bootstrap: int, seed: int) -> dict:
    positives = np.asarray([record["positive"] for record in records])
    negatives = np.concatenate([record["negatives"] for record in records])
    pairwise = np.asarray([
        np.mean((record["positive"] > record["negatives"]).astype(float)
                + 0.5 * (record["positive"] == record["negatives"]).astype(float))
        for record in records
    ])
    top1 = np.asarray([
        float(record["positive"] > np.max(record["negatives"]))
        for record in records
    ])
    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    scores = np.concatenate([positives, negatives])
    rng = np.random.RandomState(seed)
    boot_pairwise, boot_top1 = [], []
    for _ in range(bootstrap):
        sample = rng.randint(0, len(records), len(records))
        boot_pairwise.append(float(np.mean(pairwise[sample])))
        boot_top1.append(float(np.mean(top1[sample])))
    return {
        "n_queries": len(records),
        "pooled_roc_auc": float(roc_auc_score(labels, scores)),
        "query_macro_pairwise_accuracy": float(np.mean(pairwise)),
        "query_macro_pairwise_ci95": np.quantile(
            boot_pairwise, [0.025, 0.975]
        ).tolist(),
        "top1_accuracy": float(np.mean(top1)),
        "top1_ci95": np.quantile(boot_top1, [0.025, 0.975]).tolist(),
        "positive_similarity_median": float(np.median(positives)),
        "negative_similarity_median": float(np.median(negatives)),
    }


def mass_residualize(
    discovery: np.ndarray,
    confirmation: np.ndarray,
    mass_discovery: np.ndarray,
    mass_confirmation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    mass_d = scaler.fit_transform(mass_discovery[:, None])
    mass_c = scaler.transform(mass_confirmation[:, None])
    spline = SplineTransformer(
        n_knots=12, degree=3, knots="quantile", include_bias=True
    )
    basis_d = spline.fit_transform(mass_d)
    basis_c = spline.transform(mass_c)
    model = Ridge(alpha=1.0)
    model.fit(basis_d, discovery)
    return discovery - model.predict(basis_d), confirmation - model.predict(basis_c)


def fit_factorization(
    x: np.ndarray, y: np.ndarray, dim: int, n_factors: int, seed: int
) -> dict:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y)
    n_components = min(dim, len(x) - 1, x.shape[1], y.shape[1])
    x_pca = PCA(n_components=n_components, random_state=seed)
    y_pca = PCA(n_components=n_components, random_state=seed + 1)
    x_score = x_pca.fit_transform(x_scaled)
    y_score = y_pca.fit_transform(y_scaled)
    pls = PLSCanonical(
        n_components=min(n_factors, n_components),
        scale=True,
        algorithm="nipals",
        max_iter=2000,
        tol=1e-7,
    )
    pls.fit(x_score, y_score)
    x_direction = x_pca.components_.T @ pls.x_weights_
    y_direction = y_pca.components_.T @ pls.y_weights_
    x_direction = x_direction / x_scaler.scale_[:, None]
    y_direction = y_direction / y_scaler.scale_[:, None]
    x_direction = normalize(x_direction.T).T
    y_direction = normalize(y_direction.T).T
    return {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "x_pca": x_pca,
        "y_pca": y_pca,
        "pls": pls,
        "x_direction": x_direction,
        "y_direction": y_direction,
    }


def external_factor_scores(fit: dict, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_score = fit["x_pca"].transform(fit["x_scaler"].transform(x))
    y_score = fit["y_pca"].transform(fit["y_scaler"].transform(y))
    return fit["pls"].transform(x_score, y_score)


def factor_replication(discovery: dict, confirmation: dict) -> dict:
    x_cos = np.abs(discovery["x_direction"].T @ confirmation["x_direction"])
    y_cos = np.abs(discovery["y_direction"].T @ confirmation["y_direction"])
    joint = np.sqrt(x_cos * y_cos)
    row, col = linear_sum_assignment(-joint)
    records = []
    for left, right in zip(row, col):
        records.append({
            "discovery_factor": int(left + 1),
            "confirmation_factor": int(right + 1),
            "dreams_direction_cosine": float(x_cos[left, right]),
            "molformer_direction_cosine": float(y_cos[left, right]),
            "joint_geometric_mean": float(joint[left, right]),
        })
    return {
        "matches": records,
        "both_directions_ge_0_7": int(sum(
            record["dreams_direction_cosine"] >= 0.7
            and record["molformer_direction_cosine"] >= 0.7
            for record in records
        )),
        "joint_geometric_mean_ge_0_7": int(sum(
            record["joint_geometric_mean"] >= 0.7 for record in records
        )),
    }


def remove_descriptor_explainable(
    molecular_discovery: np.ndarray,
    molecular_confirmation: np.ndarray,
    descriptor_discovery: np.ndarray,
    descriptor_confirmation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    scaler = StandardScaler()
    descriptor_d = scaler.fit_transform(descriptor_discovery)
    descriptor_c = scaler.transform(descriptor_confirmation)
    model = RidgeCV(alphas=np.logspace(-3, 4, 20))
    model.fit(descriptor_d, molecular_discovery)
    prediction_d = model.predict(descriptor_d)
    prediction_c = model.predict(descriptor_c)
    residual_d = molecular_discovery - prediction_d
    residual_c = molecular_confirmation - prediction_c
    external_fraction = 1.0 - (
        np.sum(residual_c**2) / np.sum(
            (molecular_confirmation - molecular_discovery.mean(axis=0)) ** 2
        )
    )
    return residual_d, residual_c, {
        "ridge_alpha": float(model.alpha_),
        "confirmation_fraction_variance_explained": float(external_fraction),
    }


def summarize_factor_model(
    fit_discovery: dict,
    fit_confirmation: dict,
    x_confirmation: np.ndarray,
    y_confirmation: np.ndarray,
    residual_descriptors: np.ndarray,
    descriptor_names: list[str],
) -> dict:
    x_external, y_external = external_factor_scores(
        fit_discovery, x_confirmation, y_confirmation
    )
    annotations = descriptor_annotations(
        x_external, residual_descriptors, descriptor_names
    )
    factors = []
    for index in range(x_external.shape[1]):
        factors.append({
            "factor": index + 1,
            "confirmation_dreams_molecular_spearman": safe_spearman(
                x_external[:, index], y_external[:, index]
            ),
            "top_mass_residual_descriptor_correlations": annotations[index],
        })
    return {
        "factors": factors,
        "independent_factor_replication": factor_replication(
            fit_discovery, fit_confirmation
        ),
    }


def descriptor_annotations(
    factor_scores: np.ndarray, residual_descriptors: np.ndarray, names: list[str]
) -> list[list[dict]]:
    result = []
    for factor in range(factor_scores.shape[1]):
        values = []
        for index, name in enumerate(names):
            rho = safe_spearman(factor_scores[:, factor], residual_descriptors[:, index])
            values.append({"descriptor": name, "spearman": rho})
        result.append(sorted(
            values,
            key=lambda item: abs(item["spearman"])
            if np.isfinite(item["spearman"]) else -np.inf,
            reverse=True,
        )[:6])
    return result


def analyze_kind(
    args: argparse.Namespace,
    kind: str,
    mol_d: np.ndarray,
    mol_c: np.ndarray,
    pairs_c: list[dict],
    mass_d: np.ndarray,
    mass_c: np.ndarray,
    residual_desc_d: np.ndarray,
    residual_desc_c: np.ndarray,
    descriptor_names: list[str],
) -> dict:
    spectra_d = normalize(load_dreams(args.discovery, kind, args.layer))
    spectra_c = normalize(load_dreams(args.confirmation, kind, args.layer))
    molecule_d = normalize(spectra_d.mean(axis=1))
    molecule_c = normalize(spectra_c.mean(axis=1))

    projection = fit_projection(molecule_d, mol_d, args.pca_dim, args.seed)
    query_c = project_spectra(projection, spectra_c)
    candidate_c = project_molecules(projection, mol_c)
    retrieval = summarize_retrieval(
        retrieval_records(query_c, candidate_c, pairs_c),
        args.bootstrap,
        args.seed,
    )

    x_d, x_c = mass_residualize(molecule_d, molecule_c, mass_d, mass_c)
    y_d, y_c = mass_residualize(mol_d, mol_c, mass_d, mass_c)
    factor_d = fit_factorization(
        x_d, y_d, args.pca_dim, args.n_factors, args.seed
    )
    factor_c = fit_factorization(
        x_c, y_c, args.pca_dim, args.n_factors, args.seed + 17
    )
    shared_summary = summarize_factor_model(
        factor_d, factor_c, x_c, y_c, residual_desc_c, descriptor_names
    )

    innovation_d, innovation_c, innovation_audit = remove_descriptor_explainable(
        y_d, y_c, residual_desc_d, residual_desc_c
    )
    innovation_factor_d = fit_factorization(
        x_d, innovation_d, args.pca_dim, args.n_factors, args.seed + 101
    )
    innovation_factor_c = fit_factorization(
        x_c, innovation_c, args.pca_dim, args.n_factors, args.seed + 118
    )
    innovation_summary = summarize_factor_model(
        innovation_factor_d,
        innovation_factor_c,
        x_c,
        innovation_c,
        residual_desc_c,
        descriptor_names,
    )
    return {
        "ridge_projection_alpha": float(projection["ridge"].alpha_),
        "strict_10ppm_spectrum_to_molecule_retrieval": retrieval,
        "mass_residual_shared_factors": shared_summary["factors"],
        "independent_factor_replication": shared_summary[
            "independent_factor_replication"
        ],
        "descriptor_incremental_test": {
            "audit": innovation_audit,
            "description": (
                "MolFormer variation predictable from the 40 transparent, "
                "mass-residual descriptors is removed before alignment."
            ),
            "shared_factors": innovation_summary["factors"],
            "independent_factor_replication": innovation_summary[
                "independent_factor_replication"
            ],
        },
    }


def main() -> None:
    args = parse_args()
    pairs_d = json.loads((args.discovery / "pairs.json").read_text(encoding="utf-8"))
    pairs_c = json.loads((args.confirmation / "pairs.json").read_text(encoding="utf-8"))
    ik_d = {pair["ik14"] for pair in pairs_d}
    ik_c = {pair["ik14"] for pair in pairs_c}
    if ik_d & ik_c:
        raise RuntimeError("Discovery/confirmation molecule leakage")
    mol_d = np.load(args.molformer / "discovery.npy").astype(np.float64)
    mol_c = np.load(args.molformer / "confirmation.npy").astype(np.float64)
    if len(mol_d) != len(pairs_d) or len(mol_c) != len(pairs_c):
        raise RuntimeError("MolFormer and DreaMS cohort lengths differ")

    smiles_d, _ = descriptors.load_smiles(args.data, args.discovery)
    smiles_c, _ = descriptors.load_smiles(args.data, args.confirmation)
    names_d, desc_d = descriptors.compute_descriptors(smiles_d)
    names_c, desc_c = descriptors.compute_descriptors(smiles_c)
    if names_d != names_c:
        raise RuntimeError("Descriptor schemas differ")
    mass_index = names_d.index("ExactMolWt")
    residual_d_all, residual_c_all, _ = descriptors.residualize_targets(
        desc_d, desc_c, mass_index
    )
    keep = [
        index for index, name in enumerate(names_d)
        if name != "ExactMolWt" and np.std(residual_d_all[:, index]) > 1e-8
    ]
    descriptor_names = [names_d[index] for index in keep]
    residual_desc_d = residual_d_all[:, keep]
    residual_desc_c = residual_c_all[:, keep]
    mass_d = desc_d[:, mass_index]
    mass_c = desc_c[:, mass_index]

    result = {
        "status": "frozen_dreams_molformer_alignment",
        "protocol": (
            "All projection and factor models are fit on 464 discovery molecules. "
            "Retrieval and factor correlations are evaluated on 464 molecule-disjoint "
            "confirmation molecules with different-molecule candidates within 10 ppm."
        ),
        "audit": {
            "discovery_molecules": len(pairs_d),
            "confirmation_molecules": len(pairs_c),
            "ik14_overlap": 0,
            "molformer_dimension": int(mol_d.shape[1]),
            "descriptor_count_after_mass_filter": len(descriptor_names),
        },
        "config": {
            "layer": args.layer,
            "pca_dim": args.pca_dim,
            "n_factors": args.n_factors,
            "bootstrap": args.bootstrap,
            "molformer": str(args.molformer),
        },
        "raw_ssl": analyze_kind(
            args, "raw", mol_d, mol_c, pairs_c, mass_d, mass_c,
            residual_desc_d, residual_desc_c, descriptor_names,
        ),
        "official_finetuned": analyze_kind(
            args, "official", mol_d, mol_c, pairs_c, mass_d, mass_c,
            residual_desc_d, residual_desc_c, descriptor_names,
        ),
        "interpretation_limit": (
            "A shared factor is not yet a fragmentation mechanism. It must also "
            "replicate with a second molecular teacher and pass peak-level evidence tests."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key in ("raw_ssl", "official_finetuned"):
        retrieval = result[key]["strict_10ppm_spectrum_to_molecule_retrieval"]
        replication = result[key]["independent_factor_replication"]
        print(
            f"{key}: cross-modal AUC={retrieval['pooled_roc_auc']:.3f}, "
            f"Top1={retrieval['top1_accuracy']:.3f}, replicated factors both>=0.7 "
            f"{replication['both_directions_ge_0_7']}/{args.n_factors}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
