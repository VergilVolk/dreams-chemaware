"""Materialize the once-only test pair table for the frozen ChemAware gate.

This script does not fit a model or read the frozen gate.  It validates the
previously untouched formula split, checks official embeddings, computes the
same symmetric raw-spectrum features used by discovery/confirmation, and
writes an immutable-by-default test input ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-dir", type=Path,
        default=ROOT / "data/validation/large_observability_embeddings_test_frozen_gate_20260902",
    )
    parser.add_argument(
        "--discovery-manifest", type=Path,
        default=ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv",
    )
    parser.add_argument(
        "--confirmation-manifest", type=Path,
        default=ROOT / "data/validation/large_observability_embeddings_confirmation/manifest.csv",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/chemaware_frozen_gate_test_inputs_20260902",
    )
    args = parser.parse_args()
    from audit_large_observability_residual import build_pairs

    required = [
        args.embedding_dir / "manifest.csv",
        args.embedding_dir / "official_embeddings.npy",
        args.discovery_manifest,
        args.confirmation_manifest,
        args.data,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite consumed test inputs: {args.output_dir}")
    test = pd.read_csv(args.embedding_dir / "manifest.csv")
    discovery = pd.read_csv(args.discovery_manifest)
    confirmation = pd.read_csv(args.confirmation_manifest)
    if set(test["audit_split"].astype(str)) != {"test"}:
        raise RuntimeError("embedding manifest is not test-only")
    test_formula = set(test["formula"].astype(str))
    discovery_formula = set(discovery["formula"].astype(str))
    confirmation_formula = set(confirmation["formula"].astype(str))
    if test_formula & (discovery_formula | confirmation_formula):
        raise RuntimeError("test formula leakage into discovery/confirmation")
    embeddings = np.load(args.embedding_dir / "official_embeddings.npy").astype(np.float32)
    if embeddings.shape != (len(test), 1024) or not np.isfinite(embeddings).all():
        raise RuntimeError(f"invalid official test embedding array: {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    if max_norm_error > 1e-5:
        raise RuntimeError(f"test embeddings are not L2 normalized: {max_norm_error}")
    pair_table = build_pairs(test, embeddings, args.data, args.tolerance, args.ppm)
    if pair_table.empty or set(pair_table["split"].astype(str)) != {"test"}:
        raise RuntimeError("empty or mislabeled test pair table")
    args.output_dir.mkdir(parents=True)
    pair_path = args.output_dir / "test_pair_features.csv.gz"
    manifest_path = args.output_dir / "test_manifest.csv"
    pair_table.to_csv(pair_path, index=False)
    test.to_csv(manifest_path, index=False)
    report = {
        "status": "chemaware_frozen_gate_test_inputs_built",
        "test_split_consumed": True,
        "model_or_gate_fit": False,
        "spectra": len(test),
        "identities": int(test["ik14"].nunique()),
        "formulas": int(test["formula"].nunique()),
        "pair_edges": len(pair_table),
        "formula_overlap_with_discovery": len(test_formula & discovery_formula),
        "formula_overlap_with_confirmation": len(test_formula & confirmation_formula),
        "embedding_shape": list(embeddings.shape),
        "embedding_max_l2_norm_error": max_norm_error,
        "parameters": {"fragment_tolerance_da": args.tolerance, "precursor_ppm": args.ppm},
        "provenance": {
            "test_embedding_manifest_sha256": sha256(args.embedding_dir / "manifest.csv"),
            "test_embedding_array_sha256": sha256(args.embedding_dir / "official_embeddings.npy"),
            "discovery_manifest_sha256": sha256(args.discovery_manifest),
            "confirmation_manifest_sha256": sha256(args.confirmation_manifest),
            "hdf5_sha256": sha256(args.data),
            "test_pair_features_sha256": sha256(pair_path),
            "test_manifest_copy_sha256": sha256(manifest_path),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
