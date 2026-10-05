#!/usr/bin/env python
"""Encode the frozen B47 queries and candidate references with official DreaMS.

This stage is deliberately truth blind.  It emits only query/reference vectors
and alignment manifests; it does not score candidates, select seeds, inspect
annotation truth, or fit a BioAware model.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from annotation._inference import SpectrumRows, preprocess_spectrum  # noqa: E402
from annotation.embed import load_embedder  # noqa: E402
from bioaware_b47_truthblind_io import parse_mgf_records, sha256_file  # noqa: E402


EXPECTED = {
    "reference_hdf5": "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f",
    "official_checkpoint": "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245",
    "architecture_checkpoint": "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2",
}


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_query_ids(path: Path) -> tuple[list[str], dict[str, int]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "query_id" not in reader.fieldnames:
            raise RuntimeError("query manifest has no query_id")
        rows = list(reader)
    query_ids = [str(row["query_id"]) for row in rows]
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise RuntimeError("query IDs are empty or duplicated")
    return query_ids, {query_id: index for index, query_id in enumerate(query_ids)}


def load_reference_rows(path: Path, valid_query_ids: set[str]) -> tuple[np.ndarray, int]:
    rows: set[int] = set()
    observed_queries: set[str] = set()
    total = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", "reference_row"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"candidate manifest misses {sorted(required)}")
        for record in reader:
            query_id = str(record["query_id"])
            if query_id not in valid_query_ids:
                raise RuntimeError(f"candidate has unknown query_id: {query_id}")
            row = int(record["reference_row"])
            if row < 0:
                raise RuntimeError(f"negative reference row: {row}")
            rows.add(row)
            observed_queries.add(query_id)
            total += 1
    if observed_queries != valid_query_ids:
        raise RuntimeError(
            f"candidate/query coverage mismatch missing={len(valid_query_ids-observed_queries)}"
        )
    if not rows:
        raise RuntimeError("candidate manifest has no reference rows")
    return np.asarray(sorted(rows), dtype=np.int64), total


class QueryTensorDataset(Dataset):
    def __init__(self, records: list[dict[str, object]], n_highest_peaks: int):
        self.records = records
        self.n_highest_peaks = n_highest_peaks

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        record = self.records[index]
        return preprocess_spectrum(
            record["peaks"], float(record["precursor_mz"]), self.n_highest_peaks
        )


def encode_to_npy(
    loader: DataLoader,
    model: torch.nn.Module,
    weight: torch.Tensor,
    bias: torch.Tensor,
    device: torch.device,
    destination: Path,
    rows: int,
    label: str,
) -> dict[str, float | int]:
    dimension = int(weight.shape[0])
    output = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=(rows, dimension)
    )
    dtype = next(model.parameters()).dtype
    offset = 0
    norm_min = float("inf")
    norm_max = 0.0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device=device, dtype=dtype, non_blocking=True)
            precursor = model(batch, None)[:, 0]
            embedding = F.normalize(F.linear(precursor, weight, bias), dim=-1)
            array = embedding.float().cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(array).all():
                raise RuntimeError(f"non-finite {label} embedding")
            norms = np.linalg.norm(array, axis=1)
            if not np.allclose(norms, 1.0, rtol=2e-4, atol=2e-4):
                raise RuntimeError(f"non-unit {label} embedding")
            end = offset + len(array)
            if end > rows:
                raise RuntimeError(f"{label} encoder emitted too many rows")
            output[offset:end] = array
            offset = end
            norm_min = min(norm_min, float(norms.min()))
            norm_max = max(norm_max, float(norms.max()))
            if batch_index % 25 == 0 or offset == rows:
                print(f"[B47 {label}] {offset:,}/{rows:,}", flush=True)
    output.flush()
    del output
    if offset != rows:
        raise RuntimeError(f"{label} encoder emitted {offset:,}, expected {rows:,}")
    return {"rows": rows, "dimension": dimension, "norm_min": norm_min, "norm_max": norm_max}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--reference-hdf5", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.batch_size <= 0 or args.n_highest_peaks <= 0:
        raise ValueError("batch size and peak count must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    graph_dir = args.graph_dir.resolve()
    query_path = graph_dir / "queries.csv.gz"
    candidate_path = graph_dir / "candidate_references.csv.gz"
    mgf_path = graph_dir / "queries.mgf"
    graph_report_path = graph_dir / "report.json"
    required = [query_path, candidate_path, mgf_path, graph_report_path,
                args.reference_hdf5, args.official_checkpoint, args.architecture_checkpoint]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    if graph_report.get("status") != "bioaware_b47_truthblind_candidate_graph_frozen":
        raise RuntimeError("unexpected B47 graph status")
    if not graph_report.get("pass_to_query_embedding_and_seed_construction"):
        raise RuntimeError("B47 graph did not pass its scientific gate")
    if graph_report.get("contracts") != {
        "P2b_used": False, "algorithm_outputs_opened": False,
        "phenotype_used": False, "truth_opened": False,
    }:
        raise RuntimeError("B47 graph truth-blind contract changed")
    graph_provenance = graph_report.get("provenance", {})
    observed_graph_hashes = {
        "queries_sha256": sha256_file(query_path),
        "candidate_references_sha256": sha256_file(candidate_path),
        "queries_mgf_sha256": sha256_file(mgf_path),
    }
    for key, observed in observed_graph_hashes.items():
        if graph_provenance.get(key) != observed:
            raise RuntimeError(f"B47 graph provenance mismatch: {key}")

    artifacts = {
        "reference_hdf5": args.reference_hdf5.resolve(),
        "official_checkpoint": args.official_checkpoint.resolve(),
        "architecture_checkpoint": args.architecture_checkpoint.resolve(),
    }
    artifact_hashes = {key: sha256_file(path) for key, path in artifacts.items()}
    for key, expected in EXPECTED.items():
        if artifact_hashes[key] != expected:
            raise RuntimeError(f"frozen artifact SHA256 mismatch: {key}")
    if graph_provenance.get("reference_hdf5_sha256") != artifact_hashes["reference_hdf5"]:
        raise RuntimeError("candidate graph/reference HDF5 mismatch")

    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    query_ids, query_position = load_query_ids(query_path)
    if len(query_ids) != int(graph_report.get("queries", -1)):
        raise RuntimeError("query count does not replay frozen graph report")
    mgf_titles, mgf_records = parse_mgf_records(mgf_path)
    if set(mgf_titles) != set(query_ids):
        raise RuntimeError(
            f"MGF/query ID set mismatch missing={len(set(query_ids)-set(mgf_titles))} "
            f"extra={len(set(mgf_titles)-set(query_ids))}"
        )
    mgf_by_title = dict(zip(mgf_titles, mgf_records, strict=True))
    ordered_records = [mgf_by_title[query_id] for query_id in query_ids]
    del mgf_by_title, mgf_records
    reference_rows, candidate_rows = load_reference_rows(candidate_path, set(query_ids))
    if candidate_rows != int(graph_report.get("candidate_reference_rows", -1)):
        raise RuntimeError("candidate-reference count does not replay frozen graph report")
    reference_universe_rows = int(graph_report.get("reference", {}).get("rows", -1))
    if reference_universe_rows <= 0 or int(reference_rows[-1]) >= reference_universe_rows:
        raise RuntimeError("candidate reference row is outside frozen HDF5 universe")

    device = torch.device(args.device)
    model, weight, bias = load_embedder(
        device=device,
        raw_path=artifacts["architecture_checkpoint"],
        official_path=artifacts["official_checkpoint"],
        n_highest_peaks=args.n_highest_peaks,
    )
    query_loader = DataLoader(
        QueryTensorDataset(ordered_records, args.n_highest_peaks),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    query_embedding_path = output / "query_embeddings.npy"
    query_summary = encode_to_npy(
        query_loader, model, weight, bias, device, query_embedding_path,
        len(query_ids), "queries",
    )
    del ordered_records, query_loader

    reference_loader = DataLoader(
        SpectrumRows(artifacts["reference_hdf5"], reference_rows, args.n_highest_peaks),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    reference_embedding_path = output / "reference_embeddings.npy"
    reference_summary = encode_to_npy(
        reference_loader, model, weight, bias, device, reference_embedding_path,
        len(reference_rows), "references",
    )
    reference_rows_path = output / "reference_rows.npy"
    np.save(reference_rows_path, reference_rows, allow_pickle=False)
    query_index_path = output / "query_index.csv"
    with query_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "embedding_position"])
        writer.writerows((query_id, query_position[query_id]) for query_id in query_ids)

    report = {
        "status": "bioaware_b47_truthblind_embeddings_complete",
        "formal": True,
        "queries": query_summary,
        "references": reference_summary,
        "candidate_reference_rows": candidate_rows,
        "unique_reference_rows": int(len(reference_rows)),
        "alignment": {
            "query_manifest_rows": len(query_ids),
            "mgf_titles": len(mgf_titles),
            "title_set_exact": True,
            "query_embedding_order": "queries.csv.gz query_id order",
            "reference_embedding_order": "reference_rows.npy sorted row order",
        },
        "contracts": {
            "one_shared_official_encoder": True,
            "truth_opened": False,
            "phenotype_used": False,
            "algorithm_outputs_opened": False,
            "candidate_scores_computed": False,
            "seed_selection_performed": False,
            "model_fitted": False,
            "P2b_used": False,
        },
        "parameters": {
            "batch_size": args.batch_size,
            "n_highest_peaks": args.n_highest_peaks,
            "device": args.device,
        },
        "provenance": {
            "graph_report_sha256": sha256_file(graph_report_path),
            **observed_graph_hashes,
            **{f"{key}_sha256": value for key, value in artifact_hashes.items()},
            "query_embeddings_sha256": sha256_file(query_embedding_path),
            "reference_embeddings_sha256": sha256_file(reference_embedding_path),
            "reference_rows_sha256": sha256_file(reference_rows_path),
            "query_index_sha256": sha256_file(query_index_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "pass_to_truthblind_seed_construction": True,
        "claim_limit": (
            "Execution cache only. No annotation truth, BioAware gain, DreaMS error, "
            "reaction-reachable error, or SOTA claim has been evaluated."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
