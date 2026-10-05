#!/usr/bin/env python
"""Fail-closed semantic validation of the frozen BioAware B4 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest_dir / "manifest.npz"
    report_path = args.manifest_dir / "report.json"
    if not manifest.is_file() or not report_path.is_file():
        raise FileNotFoundError(args.manifest_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b4_direct_manifest_frozen":
        raise RuntimeError("wrong B4 manifest status")
    if report["provenance"].get("manifest_sha256") != sha256_file(manifest):
        raise RuntimeError("manifest hash mismatch")
    with np.load(manifest, allow_pickle=False) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    n = len(body["query_id"])
    if n != 548 or len(np.unique(body["query_id"])) != n:
        raise RuntimeError("B4 must contain 548 unique queries")
    if not np.isfinite(body["query_tensor"]).all():
        raise RuntimeError("query tensors contain non-finite values")
    query_embedding = body["query_official_embedding"]
    if query_embedding.shape != (n, 1024) or not np.isfinite(query_embedding).all():
        raise RuntimeError("frozen query embeddings are invalid")
    if float(np.max(np.abs(np.linalg.norm(query_embedding, axis=1) - 1.0))) > 1e-5:
        raise RuntimeError("frozen query embeddings are not unit-normalized")
    reference_tensor_rows = body["reference_tensor_rows"].astype(np.int64)
    reference_tensor = body["reference_tensor"]
    reference_embedding = body["reference_official_embedding"]
    if (reference_tensor_rows.ndim != 1
            or len(reference_tensor_rows) != len(reference_tensor)
            or len(reference_tensor_rows) != len(reference_embedding)
            or np.any(np.diff(reference_tensor_rows) <= 0)):
        raise RuntimeError("MoNA reference tensor index is invalid")
    if reference_tensor.shape[1:] != (101, 2):
        raise RuntimeError(f"unexpected MoNA reference tensor shape: {reference_tensor.shape}")
    if reference_embedding.shape[1:] != (1024,):
        raise RuntimeError(f"unexpected MoNA embedding shape: {reference_embedding.shape}")
    if not np.isfinite(reference_tensor).all() or not np.isfinite(reference_embedding).all():
        raise RuntimeError("MoNA reference cache contains non-finite values")
    if set(map(int, body["reference_rows"])) - set(map(int, reference_tensor_rows)):
        raise RuntimeError("candidate references are absent from the MoNA tensor cache")
    if float(np.max(np.abs(np.linalg.norm(reference_embedding, axis=1) - 1.0))) > 1e-5:
        raise RuntimeError("frozen MoNA reference embeddings are not unit-normalized")
    if len(body["query_ptr"]) != n + 1 or int(body["query_ptr"][-1]) != len(body["molecule_id"]):
        raise RuntimeError("query pointers are invalid")
    if len(body["reference_ptr"]) != len(body["molecule_id"]) + 1:
        raise RuntimeError("reference pointers are invalid")
    if np.any(np.diff(body["query_ptr"]) < 2) or np.any(np.diff(body["reference_ptr"]) < 1):
        raise RuntimeError("a query/candidate lacks a competitor or reference")
    for query in range(n):
        left = int(body["query_ptr"][query])
        if str(body["molecule_id"][left]) != str(body["query_ik14"][query]):
            raise RuntimeError(f"query {query}: truth molecule is not first")
        if str(body["molecule_formula"][left]) != str(body["query_formula"][query]):
            raise RuntimeError(f"query {query}: truth formula is inconsistent")
    fold = body["formula_fold"].astype(int)
    if set(fold) != set(range(5)):
        raise RuntimeError("formula folds are incomplete")
    formula_fold = {}
    for formula, value in zip(body["query_formula"].astype(str), fold, strict=True):
        if formula in formula_fold and formula_fold[formula] != int(value):
            raise RuntimeError(f"formula {formula} occurs in multiple folds")
        formula_fold[formula] = int(value)
    if int(np.sum(body["baseline_rank"] != 1)) != 163:
        raise RuntimeError("official error count is no longer 163")
    if int(np.sum(body["safe_corrected"])) != 17 or int(np.sum(body["safe_introduced"])) != 0:
        raise RuntimeError("archived safe compatibility gate is not 17/0")
    if int(np.sum(body["recall_corrected"])) != 24 or int(np.sum(body["recall_introduced"])) != 2:
        raise RuntimeError("archived high-recall compatibility gate is not 24/2")
    for prefix in ("safe", "recall"):
        corrected = body[f"{prefix}_corrected_by_outer"].astype(bool)
        introduced = body[f"{prefix}_introduced_by_outer"].astype(bool)
        if corrected.shape != (5, n) or introduced.shape != (5, n):
            raise RuntimeError(f"{prefix}: nested route shape is invalid")
        for outer in range(5):
            held = fold == outer
            if corrected[outer, held].any() or introduced[outer, held].any():
                raise RuntimeError(f"{prefix}/{outer}: held formula entered training router")
            minimum = 8 if prefix == "safe" else 12
            if int(corrected[outer].sum()) < minimum:
                raise RuntimeError(f"{prefix}/{outer}: fewer than {minimum} corrective routes")
    forbidden = [key for key in body if "p2b" in key.lower() or "phenotype" in key.lower()]
    if forbidden:
        raise RuntimeError(f"forbidden manifest fields: {forbidden}")
    print(json.dumps({
        "status": "bioaware_b4_direct_manifest_validation_passed",
        "queries": n,
        "identities": int(len(np.unique(body["query_ik14"]))),
        "formulas": int(len(np.unique(body["query_formula"]))),
        "candidate_molecules": int(len(body["molecule_id"])),
        "candidate_references": int(len(body["reference_rows"])),
        "unique_candidate_reference_spectra": int(len(reference_tensor_rows)),
        "manifest_sha256": sha256_file(manifest),
    }, indent=2))


if __name__ == "__main__":
    main()
