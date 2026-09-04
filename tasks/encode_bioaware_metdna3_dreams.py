#!/usr/bin/env python
"""Encode frozen MetDNA3 queries/references with official DreaMS and rank them."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation._inference import SpectrumRows  # noqa: E402
from annotation.embed import load_embedder  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def encode_loader(loader, model, weight, bias, device: torch.device) -> np.ndarray:
    output: list[np.ndarray] = []
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for position, batch in enumerate(loader, 1):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device=device, dtype=dtype)
            precursor = model(batch, None)[:, 0]
            embedding = F.normalize(F.linear(precursor, weight, bias), dim=-1)
            output.append(embedding.float().cpu().numpy())
            if position % 50 == 0:
                print(f"[encode] batches={position}", flush=True)
    return np.concatenate(output).astype(np.float32)


def unique_top(group: pd.DataFrame, score: str, truth: str) -> bool:
    maximum = float(group[score].max())
    top = group[np.isclose(group[score].astype(float), maximum, rtol=0, atol=1e-12)]
    return bool(len(top) == 1 and str(top.iloc[0].candidate_id) == truth)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_dreams_cache_v1"),
    )
    parser.add_argument(
        "--reference-hdf5", type=Path,
        default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_dreams_official_v1"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    report_path = args.cache_dir / "report.json"
    query_path = args.cache_dir / "queries.csv.gz"
    candidate_path = args.cache_dir / "candidate_references.csv.gz"
    tensor_path = args.cache_dir / "query_tensors.npz"
    for path in (report_path, query_path, candidate_path, tensor_path, args.reference_hdf5):
        if not path.exists():
            raise FileNotFoundError(path)
    cache_report = json.loads(report_path.read_text(encoding="utf-8"))
    if not cache_report.get("formal") or cache_report["contracts"].get("P2b") != "forbidden":
        raise RuntimeError("MetDNA3 cache contract is not valid")
    query = pd.read_csv(query_path)
    candidate = pd.read_csv(candidate_path)
    tensor = np.load(tensor_path, allow_pickle=False)["query_tensor"].astype(np.float32)
    if len(query) != len(tensor):
        raise RuntimeError("query tensor/manifest length mismatch")
    unique_reference_rows = np.sort(candidate["reference_row"].unique().astype(np.int64))
    reference_position = {row: position for position, row in enumerate(unique_reference_rows)}

    device = torch.device(args.device)
    model, weight, bias = load_embedder(
        device=device, n_highest_peaks=args.n_highest_peaks
    )
    query_loader = DataLoader(
        TensorDataset(torch.from_numpy(tensor)), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    query_embedding = encode_loader(query_loader, model, weight, bias, device)
    reference_loader = DataLoader(
        SpectrumRows(args.reference_hdf5, unique_reference_rows, args.n_highest_peaks),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    reference_embedding = encode_loader(reference_loader, model, weight, bias, device)
    if not np.isfinite(query_embedding).all() or not np.isfinite(reference_embedding).all():
        raise RuntimeError("non-finite official embedding")

    query_position = {value: position for position, value in enumerate(query["query_id"])}
    score_rows: list[dict] = []
    for (query_id, candidate_id), group in candidate.groupby(
        ["query_id", "candidate_id"], sort=False
    ):
        q = query_embedding[query_position[query_id]]
        positions = [reference_position[int(row)] for row in group["reference_row"]]
        similarities = reference_embedding[positions] @ q
        best_local = int(np.argmax(similarities))
        best_source = group.iloc[best_local]
        score_rows.append({
            "query_id": query_id, "candidate_id": str(candidate_id),
            "spectral_score": float(similarities[best_local]),
            "best_reference_row": int(best_source.reference_row),
            "reference_spectra": int(len(group)),
            "truth_candidate_id": str(best_source.truth_candidate_id),
            "truth_formula": str(best_source.truth_formula),
            "adduct": str(best_source.adduct),
        })
    scores = pd.DataFrame(score_rows)
    correctness: list[bool] = []
    margins: list[float] = []
    for query_id, group in scores.groupby("query_id", sort=False):
        truth = str(group["truth_candidate_id"].iloc[0])
        correctness.append(unique_top(group, "spectral_score", truth))
        ordered = np.sort(group["spectral_score"].to_numpy(float))[::-1]
        margins.append(float(ordered[0] - ordered[1]))

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = [output / "candidate_scores.csv.gz", output / "embeddings.npz", output / "report.json"]
    if any(path.exists() for path in files):
        raise RuntimeError(f"fail-closed: official DreaMS output already exists: {output}")
    scores.to_csv(files[0], index=False, compression="gzip")
    np.savez_compressed(
        files[1], query_embedding=query_embedding,
        reference_embedding=reference_embedding,
        reference_rows=unique_reference_rows,
    )
    payload = {
        "status": "bioaware_metdna3_official_dreams_complete", "formal": True,
        "queries": int(len(query)), "identities": int(query["truth_ik14"].nunique()),
        "candidate_rows": int(len(scores)), "reference_spectra": int(len(unique_reference_rows)),
        "official_dreams_recall1": float(np.mean(correctness)),
        "official_dreams_errors": int(len(correctness) - sum(correctness)),
        "spectral_margin": {
            "median": float(np.median(margins)), "p10": float(np.quantile(margins, 0.1)),
        },
        "contracts": {
            "shared_official_encoder": True, "candidate_aggregation": "maximum cosine per IK14",
            "ties": "count against truth", "P2b": "forbidden",
            "reaction_network_used": False, "internal_validation_or_external_test_opened": False,
        },
        "provenance": {
            "cache_report_sha256": sha256(report_path), "queries_sha256": sha256(query_path),
            "candidates_sha256": sha256(candidate_path), "tensors_sha256": sha256(tensor_path),
            "reference_hdf5_sha256": sha256(args.reference_hdf5),
            "scores_sha256": sha256(files[0]), "embeddings_sha256": sha256(files[1]),
        },
        "claim_limit": "Official DreaMS baseline on the consumed HILIC development set; not BioAware gain or external validation.",
    }
    atomic_json(files[2], payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

