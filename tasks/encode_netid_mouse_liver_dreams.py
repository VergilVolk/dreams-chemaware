#!/usr/bin/env python
"""Encode the public NetID mouse-liver targeted MS2 with official DreaMS."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OFFICIAL_SHA256 = "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_records(cache: np.lib.npyio.NpzFile, minimum_peaks: int) -> tuple[list[dict], np.ndarray]:
    offsets = np.asarray(cache["peak_offsets"], dtype=np.int64)
    if len(offsets) != len(cache["precursor_mz"]) + 1:
        raise RuntimeError("invalid packed spectrum offsets")
    counts = np.diff(offsets)
    selected = np.flatnonzero(counts >= minimum_peaks)
    records: list[dict] = []
    for index in selected:
        start, end = int(offsets[index]), int(offsets[index + 1])
        peaks = np.stack(
            [cache["fragment_mz"][start:end], cache["fragment_intensity"][start:end]],
            axis=0,
        ).astype(np.float32, copy=False)
        records.append(
            {"peaks": peaks, "precursor_mz": float(cache["precursor_mz"][index])}
        )
    return records, selected


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("data/validation/netid_public_release_audit_v2_20260831"),
    )
    parser.add_argument(
        "--official-checkpoint",
        type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--architecture-checkpoint",
        type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/netid_mouse_liver_dreams_20260831"),
    )
    parser.add_argument("--minimum-peaks", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-status", default="netid_public_release_audit_complete")
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    embeddings_path = args.output_dir / "official_dreams_embeddings.npz"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "netid_mouse_liver_dreams_embeddings_frozen":
            raise RuntimeError("invalid existing DreaMS report")
        if not embeddings_path.is_file() or sha256(embeddings_path) != report["provenance"]["embeddings_sha256"]:
            raise RuntimeError("existing DreaMS embeddings changed")
        print(f"[reuse] verified {report_path}", flush=True)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {args.output_dir}")
    audit_report_path = args.audit_dir / "report.json"
    audit = json.loads(audit_report_path.read_text(encoding="utf-8"))
    if audit.get("status") != args.audit_status:
        raise RuntimeError("NetID public audit has not passed")
    if audit.get("gates", {}).get("pass_to_component_isolated_ms2_edge_stage") is not True:
        raise RuntimeError("NetID public audit blocked the DreaMS edge stage")
    cache_path = args.audit_dir / audit["artifacts"]["mouse_liver_ms2_spectra"]["relative_path"]
    if sha256(cache_path) != audit["artifacts"]["mouse_liver_ms2_spectra"]["sha256"]:
        raise RuntimeError("mouse-liver spectrum cache hash mismatch")
    if sha256(args.official_checkpoint) != OFFICIAL_SHA256:
        raise RuntimeError("official DreaMS checkpoint hash mismatch")
    if not args.architecture_checkpoint.is_file():
        raise FileNotFoundError(args.architecture_checkpoint)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    from annotation.embed import embed_records, load_embedder

    with np.load(cache_path, allow_pickle=False) as cache:
        records, selected = unpack_records(cache, args.minimum_peaks)
        metadata = {
            name: np.asarray(cache[name])[selected]
            for name in (
                "source_file",
                "source_sheet",
                "feature_group_id",
                "netid_peak_id",
                "precursor_mz",
                "raw_rt_min",
                "collision_energy",
                "netid_class",
                "netid_formula",
                "netid_annotation",
            )
        }
        counts = np.diff(np.asarray(cache["peak_offsets"], dtype=np.int64))[selected]
    if len(records) < 750:
        raise RuntimeError(f"too few >=minimum-peak spectra: {len(records)}")
    model, weight, bias = load_embedder(
        device=args.device,
        raw_path=args.architecture_checkpoint,
        official_path=args.official_checkpoint,
        n_highest_peaks=100,
    )
    vectors = embed_records(
        records,
        model,
        weight,
        bias,
        device=args.device,
        n_highest_peaks=100,
        batch_size=args.batch_size,
    )
    norms = np.linalg.norm(vectors, axis=1)
    if vectors.ndim != 2 or len(vectors) != len(records):
        raise RuntimeError("DreaMS embedding shape mismatch")
    if not np.isfinite(vectors).all() or float(np.max(np.abs(norms - 1.0))) > 2e-5:
        raise RuntimeError("invalid DreaMS embedding norms")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_npz(
        embeddings_path,
        embeddings=vectors.astype(np.float32),
        n_fragment_peaks=counts.astype(np.int32),
        **metadata,
    )
    report = {
        "status": "netid_mouse_liver_dreams_embeddings_frozen",
        "formal": True,
        "spectra": len(records),
        "unique_feature_group_ids": int(np.unique(metadata["feature_group_id"]).size),
        "unique_netid_peak_ids": int(np.unique(metadata["netid_peak_id"]).size),
        "minimum_fragment_peaks": args.minimum_peaks,
        "embedding_dimension": int(vectors.shape[1]),
        "maximum_norm_error": float(np.max(np.abs(norms - 1.0))),
        "contracts": {
            "shared_official_encoder": True,
            "model_eval_mode": True,
            "identity_labels_used": False,
            "author_netid_predictions_used_by_encoder": False,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "provenance": {
            "audit_report_sha256": sha256(audit_report_path),
            "spectrum_cache_sha256": sha256(cache_path),
            "official_checkpoint_sha256": sha256(args.official_checkpoint),
            "architecture_checkpoint_sha256": sha256(args.architecture_checkpoint),
            "embeddings_sha256": sha256(embeddings_path),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_limit": "Feature-edge execution cache only; no annotation or performance result.",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
