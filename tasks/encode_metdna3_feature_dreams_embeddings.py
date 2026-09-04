#!/usr/bin/env python
"""Encode exact MetDNA3 feature spectra with one frozen shared DreaMS encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from annotation._inference import preprocess_spectrum  # noqa: E402
from shared_dreams_inference import load_inference_model, sha256_file  # noqa: E402


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_mgf(path: Path, n_highest_peaks: int) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    tensors: list[np.ndarray] = []
    header: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    in_record = False

    def finish() -> None:
        nonlocal header, peaks, in_record
        if not in_record:
            return
        name = header.get("TITLE", "").strip()
        precursor_text = header.get("PEPMASS", "").split()[0]
        if not name or not precursor_text or not peaks:
            raise RuntimeError("incomplete MetDNA3 MGF record")
        precursor = float(precursor_text)
        raw = np.asarray(peaks, dtype=np.float32).T
        tensor = preprocess_spectrum(raw, precursor, n_highest_peaks).numpy().astype(np.float32)
        names.append(name)
        tensors.append(tensor)
        header, peaks, in_record = {}, [], False

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                if in_record:
                    raise RuntimeError("nested BEGIN IONS")
                in_record, header, peaks = True, {}, []
            elif line == "END IONS":
                finish()
            elif in_record and "=" in line:
                key, value = line.split("=", 1)
                header[key.upper()] = value
            elif in_record:
                fields = line.split()
                if len(fields) < 2:
                    raise RuntimeError(f"invalid MGF peak line: {line!r}")
                peaks.append((float(fields[0]), float(fields[1])))
    if in_record:
        raise RuntimeError("unterminated MGF record")
    if not names or len(set(names)) != len(names):
        raise RuntimeError("MGF feature names are empty or duplicated")
    return names, np.stack(tensors)


def load_feature_cache(path: Path) -> tuple[list[str], np.ndarray, dict[str, str]]:
    metadata_path = path / "feature_ms2.csv.gz"
    tensor_path = path / "feature_ms2_tensors.npz"
    report_path = path / "report.json"
    for item in (metadata_path, tensor_path, report_path):
        if not item.is_file():
            raise FileNotFoundError(item)
    metadata = pd.read_csv(metadata_path)
    if "feature_node" not in metadata:
        raise RuntimeError("feature cache misses feature_node")
    names = metadata["feature_node"].astype(str).tolist()
    tensors = np.load(tensor_path, allow_pickle=False)["feature_ms2_tensor"].astype(np.float32)
    if len(names) != len(tensors) or len(set(names)) != len(names):
        raise RuntimeError("feature cache metadata/tensor mismatch or duplicate nodes")
    return names, tensors, {
        "metadata_sha256": sha256_file(metadata_path),
        "tensors_sha256": sha256_file(tensor_path),
        "cache_report_sha256": sha256_file(report_path),
    }


def encode(model, tensors: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(tensors)), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    dtype = next(model.parameters()).dtype
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_index, (batch,) in enumerate(loader, 1):
            z = model(batch.to(device=device, dtype=dtype))
            result.append(z.float().cpu().numpy())
            if batch_index % 50 == 0:
                print(f"[MetDNA3 DreaMS] encoded batches={batch_index}", flush=True)
    embedding = np.concatenate(result).astype(np.float32)
    if not np.isfinite(embedding).all():
        raise RuntimeError("non-finite feature embedding")
    norms = np.linalg.norm(embedding, axis=1)
    if not np.allclose(norms, 1.0, rtol=2e-4, atol=2e-4):
        raise RuntimeError("feature embeddings are not unit normalized")
    return embedding


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bridge-dir", type=Path,
                        help="Directory produced by export_metdna3_official_bridge.R")
    source.add_argument("--feature-cache-dir", type=Path,
                        help="Existing outcome-blind BioAware feature-MS2 cache")
    parser.add_argument("--official-checkpoint", type=Path,
                        default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path,
                        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--shared-encoder-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for checkpoint in (args.official_checkpoint, args.architecture_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output directory is non-empty: {output}")

    if args.bridge_dir is not None:
        mgf = args.bridge_dir / "feature_spectra.mgf"
        manifest = args.bridge_dir / "feature_spectra.csv"
        if not mgf.is_file() or not manifest.is_file():
            raise FileNotFoundError(f"incomplete official bridge: {args.bridge_dir}")
        names, tensors = parse_mgf(mgf, args.n_highest_peaks)
        expected = pd.read_csv(manifest)["feature_name"].astype(str).tolist()
        if names != expected:
            raise RuntimeError("MGF/manifest feature order mismatch")
        source_provenance = {"mgf_sha256": sha256_file(mgf),
                             "manifest_sha256": sha256_file(manifest)}
        source_kind = "official_metdna3_bridge"
    else:
        names, tensors, source_provenance = load_feature_cache(args.feature_cache_dir)
        source_kind = "bioaware_outcome_blind_feature_cache"

    device = torch.device(args.device)
    model, model_metadata = load_inference_model(
        args.official_checkpoint, args.architecture_checkpoint, device,
        args.n_highest_peaks, args.shared_encoder_checkpoint,
    )
    embedding = encode(model, tensors, device, args.batch_size)
    artifact_path = output / "feature_embeddings.npz"
    np.savez_compressed(artifact_path, feature_name=np.asarray(names, dtype=str),
                        embedding=embedding)
    payload = {
        "status": "metdna3_feature_dreams_embeddings_complete",
        "formal": True,
        "features": len(names),
        "dimension": int(embedding.shape[1]),
        "source_kind": source_kind,
        "model": model_metadata,
        "contracts": {
            "one_shared_query_reference_encoder": True,
            "identity_labels_used": False,
            "candidate_labels_used": False,
            "annotation_outcomes_used": False,
            "P2b_used": False,
        },
        "provenance": {**source_provenance,
                       "artifact_sha256": sha256_file(artifact_path),
                       "script_sha256": sha256_file(Path(__file__))},
        "claim_limit": "Execution cache only; no MetDNA3 propagation or annotation gain.",
    }
    atomic_json(output / "report.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
