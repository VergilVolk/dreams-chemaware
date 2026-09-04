#!/usr/bin/env python
"""Audit candidate-specific fragmentation-to-structure headroom.

The model decodes a structural fingerprint from DreaMS embeddings using only
Level-1 seed identities.  Every query truth identity is held out, and every
formula present in the 117-query development task is excluded from decoder
training.  This is a consumed-development headroom audit, not a deployable
identity claim and not a P2b reranker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


EPS = 1e-12


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def canonical_smiles(value: object) -> str | None:
    molecule = Chem.MolFromSmiles(decode(value))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def load_structures(
    hdf5_path: Path,
    truth: pd.DataFrame,
    wanted: set[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in truth.itertuples(index=False):
        identity = str(row.ik14)
        if identity in wanted:
            smiles = canonical_smiles(row.smiles)
            if smiles:
                result.setdefault(identity, smiles)
    with h5py.File(hdf5_path, "r") as handle:
        for start in range(0, len(handle["INCHIKEY"]), 16384):
            stop = min(len(handle["INCHIKEY"]), start + 16384)
            keys = handle["INCHIKEY"][start:stop]
            structures = handle["smiles"][start:stop]
            for raw_key, raw_structure in zip(keys, structures, strict=True):
                identity = decode(raw_key)[:14]
                if identity not in wanted or identity in result:
                    continue
                smiles = canonical_smiles(raw_structure)
                if smiles:
                    result[identity] = smiles
    return result


def fingerprints(structures: dict[str, str], dimensions: int) -> dict[str, np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=dimensions)
    result: dict[str, np.ndarray] = {}
    for identity, smiles in structures.items():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        bitvector = generator.GetFingerprint(molecule)
        array = np.zeros(dimensions, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bitvector, array)
        result[identity] = array
    return result


def normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), EPS)


def fit_kernel_decoder(
    embeddings: np.ndarray,
    targets: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = normalize(embeddings)
    target_mean = targets.mean(axis=0, keepdims=True)
    centered = targets - target_mean
    kernel = np.maximum(x @ x.T, 0.0) ** 2
    dual = np.linalg.solve(kernel + alpha * np.eye(len(kernel)), centered)
    return x, dual, target_mean[0]


def decoder_prediction(
    queries: np.ndarray,
    train_x: np.ndarray,
    dual: np.ndarray,
    target_mean: np.ndarray,
) -> np.ndarray:
    query = normalize(queries)
    kernel = np.maximum(query @ train_x.T, 0.0) ** 2
    return kernel @ dual + target_mean[None, :]


def candidate_score(prediction: np.ndarray, fingerprint: np.ndarray, mean: np.ndarray) -> float:
    predicted = prediction - mean
    candidate = fingerprint - mean
    denominator = max(float(np.linalg.norm(predicted) * np.linalg.norm(candidate)), EPS)
    return float(predicted @ candidate / denominator)


def top1(group: pd.DataFrame, column: str) -> tuple[str, bool]:
    maximum = float(group[column].max())
    tied = group[np.isclose(group[column], maximum, rtol=0, atol=1e-12)]
    predicted = str(tied.sort_values("candidate_id").iloc[0].candidate_id)
    truth = str(group["truth_candidate_id"].iloc[0])
    return predicted, bool(len(tied) == 1 and predicted == truth)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--query-cache", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_cache_v1/queries.csv.gz"))
    parser.add_argument("--query-embeddings", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_official_v1/embeddings.npz"))
    parser.add_argument("--external-spectra", type=Path, default=Path(
        "data/validation/bioaware_metdna3_dreams_cache_v2/external_spectra.csv.gz"))
    parser.add_argument("--external-embeddings", type=Path, default=Path(
        "data/validation/bioaware_metdna3_data_layer_embeddings_v2.npz"))
    parser.add_argument("--splits", type=Path, default=Path(
        "data/validation/bioaware_metdna3_development_v1/identity_splits.csv.gz"))
    parser.add_argument("--truth", type=Path, default=Path(
        "data/validation/bioaware_metdna3_development_v1/development_level1.csv.gz"))
    parser.add_argument("--hdf5", type=Path, default=Path(
        "data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--unresolved", type=Path, default=Path(
        "data/validation/bioaware_10pp_headroom_v1/unresolved_error_queries.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "data/validation/bioaware_candidate_fragment_decoder_v1"))
    parser.add_argument("--fingerprint-dimensions", type=int, default=1024)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()

    inputs = [args.scores, args.query_cache, args.query_embeddings, args.external_spectra,
              args.external_embeddings, args.splits, args.truth, args.hdf5, args.unresolved]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores)
    queries = pd.read_csv(args.query_cache)
    external = pd.read_csv(args.external_spectra)
    splits = pd.read_csv(args.splits)
    truth = pd.read_csv(args.truth)
    unresolved = pd.read_csv(args.unresolved)
    query_embedding = np.load(args.query_embeddings)["query_embedding"]
    external_embedding = np.load(args.external_embeddings)["embedding"]
    if len(queries) != len(query_embedding) or len(external) != len(external_embedding):
        raise RuntimeError("embedding/table row mismatch")
    query_index = {str(value): index for index, value in enumerate(queries["query_id"])}
    if set(scores["query_id"].astype(str)) != set(query_index):
        raise RuntimeError("scores and query cache cover different queries")

    candidate_ids = set(scores["candidate_id"].astype(str))
    external_ids = set(external["truth_ik14"].astype(str))
    structures = load_structures(args.hdf5, truth, candidate_ids | external_ids)
    fp = fingerprints(structures, args.fingerprint_dimensions)
    missing_candidates = sorted(candidate_ids - set(fp))
    if missing_candidates:
        raise RuntimeError(f"candidate structures missing for {missing_candidates[:10]}")

    # Average multiple independent spectra per seed identity.
    external_vectors: dict[str, np.ndarray] = {}
    external_formula: dict[str, str] = {}
    for identity, group in external.groupby("truth_ik14", sort=True):
        positions = group.index.to_numpy(np.int64)
        vector = normalize(external_embedding[positions]).mean(axis=0, keepdims=True)
        external_vectors[str(identity)] = normalize(vector)[0]
        external_formula[str(identity)] = str(group["truth_formula"].iloc[0])

    excluded_formulas = set(scores["truth_formula"].astype(str))
    rotation_rows: list[dict] = []
    for fold in sorted(splits["fold"].unique()):
        seed_ids = set(splits.loc[(splits["fold"] == fold) & (splits["role"] == "seed"), "ik14"].astype(str))
        train_ids = sorted(
            identity for identity in seed_ids
            if identity in external_vectors and identity in fp
            and external_formula.get(identity) not in excluded_formulas
        )
        if len(train_ids) < 50:
            raise RuntimeError(f"fold {fold} has only {len(train_ids)} formula-isolated seed identities")
        train_x = np.stack([external_vectors[identity] for identity in train_ids])
        train_y = np.stack([fp[identity] for identity in train_ids])
        kernel_x, dual, mean = fit_kernel_decoder(train_x, train_y, args.ridge_alpha)
        heldout = set(splits.loc[(splits["fold"] == fold) & (splits["role"] == "heldout"), "ik14"].astype(str))
        eligible_queries = queries[queries["truth_ik14"].astype(str).isin(heldout)]
        positions = eligible_queries.index.to_numpy(np.int64)
        prediction = decoder_prediction(query_embedding[positions], kernel_x, dual, mean)
        for local, query_row in enumerate(eligible_queries.itertuples(index=False)):
            query_id = str(query_row.query_id)
            candidates = scores[scores["query_id"].astype(str) == query_id]
            for candidate in candidates.itertuples(index=False):
                rotation_rows.append({
                    "fold": int(fold),
                    "query_id": query_id,
                    "candidate_id": str(candidate.candidate_id),
                    "truth_candidate_id": str(candidate.truth_candidate_id),
                    "truth_formula": str(candidate.truth_formula),
                    "spectral_score": float(candidate.spectral_score),
                    "decoder_score": candidate_score(prediction[local], fp[str(candidate.candidate_id)], mean),
                    "training_identities": int(len(train_ids)),
                })
    rotations = pd.DataFrame(rotation_rows)
    if rotations.empty:
        raise RuntimeError("no held-out decoder rotations")
    aggregate = rotations.groupby(
        ["query_id", "candidate_id", "truth_candidate_id", "truth_formula"], as_index=False
    ).agg(
        spectral_score=("spectral_score", "first"),
        decoder_score=("decoder_score", "median"),
        heldout_rotations=("fold", "nunique"),
    )
    if aggregate["heldout_rotations"].min() != 7:
        raise RuntimeError("every query/candidate must have seven held-out rotations")

    query_rows: list[dict] = []
    unresolved_ids = set(unresolved["query_id"].astype(str))
    for query_id, group in aggregate.groupby("query_id", sort=False):
        spectral_candidate, spectral_correct = top1(group, "spectral_score")
        decoder_candidate, decoder_correct = top1(group, "decoder_score")
        truth_id = str(group["truth_candidate_id"].iloc[0])
        truth_score = float(group.loc[group["candidate_id"] == truth_id, "decoder_score"].iloc[0])
        wrong_score = float(group.loc[group["candidate_id"] != truth_id, "decoder_score"].max())
        query_rows.append({
            "query_id": str(query_id),
            "truth_candidate_id": truth_id,
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "spectral_candidate_id": spectral_candidate,
            "decoder_candidate_id": decoder_candidate,
            "spectral_correct": spectral_correct,
            "decoder_correct": decoder_correct,
            "decoder_truth_minus_wrong": truth_score - wrong_score,
            "unresolved_before_g2": str(query_id) in unresolved_ids,
        })
    query_table = pd.DataFrame(query_rows)
    corrected = query_table[~query_table["spectral_correct"] & query_table["decoder_correct"]]
    introduced = query_table[query_table["spectral_correct"] & ~query_table["decoder_correct"]]
    new_unresolved = corrected[corrected["unresolved_before_g2"]]
    aggregate_path = output / "candidate_scores.csv.gz"
    query_path = output / "query_headroom.csv.gz"
    rotation_path = output / "rotation_scores.csv.gz"
    aggregate.to_csv(aggregate_path, index=False)
    query_table.to_csv(query_path, index=False)
    rotations.to_csv(rotation_path, index=False)
    payload = {
        "status": "bioaware_candidate_fragment_decoder_headroom_complete",
        "formal": True,
        "protocol": (
            "seven held-out identity rotations; every development-task formula excluded "
            "from decoder training; fixed radius-2 Morgan and kernel-ridge recipe"
        ),
        "queries": int(len(query_table)),
        "identities": int(query_table["truth_candidate_id"].nunique()),
        "baseline_recall1": float(query_table["spectral_correct"].mean()),
        "decoder_recall1": float(query_table["decoder_correct"].mean()),
        "decoder_vs_dreams_corrected": int(len(corrected)),
        "decoder_vs_dreams_introduced": int(len(introduced)),
        "unresolved_errors_before_g2": int(len(unresolved_ids)),
        "new_independent_unresolved_headroom": int(len(new_unresolved)),
        "new_independent_unresolved_identities": int(new_unresolved["truth_candidate_id"].nunique()),
        "new_unresolved_query_ids": sorted(new_unresolved["query_id"].astype(str)),
        "gates": {
            "new_headroom_ge_3": bool(len(new_unresolved) >= 3),
            "decoder_corrected_gt_introduced": bool(len(corrected) > len(introduced)),
        },
        "provenance": {
            "scores_sha256": sha256(args.scores),
            "query_embeddings_sha256": sha256(args.query_embeddings),
            "external_embeddings_sha256": sha256(args.external_embeddings),
            "splits_sha256": sha256(args.splits),
            "unresolved_sha256": sha256(args.unresolved),
            "candidate_scores_sha256": sha256(aggregate_path),
            "query_headroom_sha256": sha256(query_path),
        },
        "claim_limit": (
            "Consumed-development structural headroom audit. Decoder-only accuracy is not "
            "a deployable fusion result; RP remains sealed."
        ),
    }
    atomic_json(output / "report.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
