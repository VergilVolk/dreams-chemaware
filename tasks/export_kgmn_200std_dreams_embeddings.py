#!/usr/bin/env python3
"""Encode every MetDNA2 200STD feature spectrum with frozen shared DreaMS."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from dreams.utils.io import read_msp  # noqa: E402
from step5_gate_eval import embed  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


OFFICIAL_SHA256 = "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_msp_frame(frame: pd.DataFrame) -> None:
    required = {"name", "precursor_mz", "spectrum"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"MSP parser did not recover required fields: {missing}")
    if frame.empty:
        raise RuntimeError("200STD MSP is empty")
    names = frame["name"].astype(str)
    if names.str.strip().eq("").any() or names.duplicated().any():
        raise RuntimeError("200STD feature names must be non-empty and unique")
    precursor = frame["precursor_mz"].to_numpy(dtype=float)
    if np.any(~np.isfinite(precursor)) or np.any(precursor <= 0):
        raise RuntimeError("200STD precursor masses must be finite and positive")
    for index, spectrum in enumerate(frame["spectrum"]):
        array = np.asarray(spectrum, dtype=float)
        if array.ndim != 2 or array.shape[0] != 2 or array.shape[1] < 1:
            raise RuntimeError(f"invalid 200STD spectrum at row {index}: shape={array.shape}")
        if np.any(~np.isfinite(array)):
            raise RuntimeError(f"non-finite 200STD spectrum at row {index}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msp", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument(
        "--architecture-checkpoint",
        type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.msp, args.official_checkpoint, args.architecture_checkpoint):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {args.output_dir}")
    if sha256(args.official_checkpoint) != OFFICIAL_SHA256:
        raise RuntimeError("official DreaMS checkpoint hash mismatch")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    frame = read_msp(args.msp)
    validate_msp_frame(frame)
    spectra = [
        preprocess_spectrum(np.asarray(row.spectrum), float(row.precursor_mz), args.n_highest_peaks)
        for row in frame.itertuples(index=False)
    ]
    device = torch.device(args.device)
    model, kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks
    )
    if kind not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError(f"unexpected official checkpoint format: {kind}")
    model.eval()
    vectors = embed(model, spectra, device, args.batch_size).numpy().astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise RuntimeError("official DreaMS produced invalid 200STD embeddings")
    vectors /= norms

    args.output_dir.mkdir(parents=True)
    columns = [f"z_{index:04d}" for index in range(vectors.shape[1])]
    exported = pd.DataFrame(vectors, columns=columns)
    exported.insert(0, "feature_name", frame["name"].astype(str).to_numpy())
    embeddings_path = args.output_dir / "official_200std_embeddings.csv.gz"
    exported.to_csv(embeddings_path, index=False, compression="gzip", float_format="%.9g")

    # Re-read the deployment representation.  The R hook consumes this exact
    # text artifact, so its quantisation—not the in-memory tensor—is the formal
    # object whose unit norms must be verified.
    replay = pd.read_csv(embeddings_path)
    replay_vectors = replay.drop(columns=["feature_name"]).to_numpy(dtype=float)
    replay_norms = np.linalg.norm(replay_vectors, axis=1)
    if not np.array_equal(replay["feature_name"].astype(str).to_numpy(), frame["name"].astype(str).to_numpy()):
        raise RuntimeError("embedding export changed the 200STD feature order")
    if float(np.max(np.abs(replay_norms - 1.0))) > 2e-6:
        raise RuntimeError("quantised deployment embeddings are not unit-normalised")

    report = {
        "status": "kgmn_200std_official_dreams_embeddings_frozen",
        "formal": True,
        "spectra": int(len(frame)),
        "unique_feature_names": int(frame["name"].nunique()),
        "embedding_dimension": int(vectors.shape[1]),
        "maximum_exported_norm_error": float(np.max(np.abs(replay_norms - 1.0))),
        "contracts": {
            "shared_encoder": True,
            "model_eval_mode": True,
            "identity_labels_used": False,
            "phenotype_used": False,
            "P2b_used": False,
            "feature_order_preserved": True,
        },
        "provenance": {
            "msp_sha256": sha256(args.msp),
            "official_checkpoint_sha256": sha256(args.official_checkpoint),
            "architecture_checkpoint_sha256": sha256(args.architecture_checkpoint),
            "embeddings_sha256": sha256(embeddings_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "Frozen execution artifact for KGMN dynamic-edge scoring; no annotation result.",
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
