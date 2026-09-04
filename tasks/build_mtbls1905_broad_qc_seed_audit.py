#!/usr/bin/env python
"""Build a phenotype-blind MTBLS1905 QC seed audit against a broad library.

The top candidate is selected from every strict-mass candidate before checking
whether that identity is represented in Rhea.  This prevents graph membership
from leaking into spectral identification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_REFERENCE_DIRS = (
    Path("data/validation/large_observability_embeddings_discovery"),
    Path("data/validation/large_observability_embeddings_confirmation"),
    Path("data/validation/large_observability_embeddings_test"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-dir",
        type=Path,
        default=Path("data/external/MTBLS1905/qc_ms2/dreams_official_full"),
    )
    parser.add_argument(
        "--reference-dirs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_REFERENCE_DIRS),
    )
    parser.add_argument(
        "--participants",
        type=Path,
        default=Path(
            "data/reference/bioaware_rhea_offline_20260827/"
            "rhea_participants.csv.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/mtbls1905_broad_qc_seed_audit_20260830"),
    )
    parser.add_argument("--ppm-tolerance", type=float, default=10.0)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    query_manifest_path = args.query_dir / "manifest.csv"
    query_embedding_path = args.query_dir / "official_embeddings.npy"
    query_manifest = pd.read_csv(query_manifest_path)
    query_embeddings = np.load(query_embedding_path, mmap_mode="r")
    if len(query_manifest) != len(query_embeddings):
        raise RuntimeError("query manifest/embedding row mismatch")

    reference_frames: list[pd.DataFrame] = []
    reference_embeddings: list[np.ndarray] = []
    provenance: dict[str, str] = {
        str(query_manifest_path): sha256(query_manifest_path),
        str(query_embedding_path): sha256(query_embedding_path),
        str(args.participants): sha256(args.participants),
    }
    seen_rows: set[int] = set()
    for directory in args.reference_dirs:
        manifest_path = directory / "manifest.csv"
        embedding_path = directory / "official_embeddings.npy"
        manifest = pd.read_csv(manifest_path)
        embeddings = np.load(embedding_path, mmap_mode="r")
        if len(manifest) != len(embeddings):
            raise RuntimeError(f"reference row mismatch: {directory}")
        required = {"ik14", "precursor_mz", "hdf5_row"}
        missing = required - set(manifest)
        if missing:
            raise RuntimeError(f"{manifest_path} missing columns: {sorted(missing)}")
        rows = manifest["hdf5_row"].astype(int).tolist()
        overlap = seen_rows.intersection(rows)
        if overlap:
            raise RuntimeError(
                f"reference partitions overlap in {len(overlap)} HDF5 rows"
            )
        seen_rows.update(rows)
        reference_frames.append(manifest.reset_index(drop=True))
        reference_embeddings.append(np.asarray(embeddings, dtype=np.float32))
        provenance[str(manifest_path)] = sha256(manifest_path)
        provenance[str(embedding_path)] = sha256(embedding_path)

    references = pd.concat(reference_frames, ignore_index=True)
    embeddings = np.concatenate(reference_embeddings, axis=0)
    if len(references) != len(embeddings):
        raise RuntimeError("combined reference row mismatch")
    precursor = pd.to_numeric(references["precursor_mz"], errors="coerce").to_numpy()
    if not np.isfinite(precursor).all():
        raise RuntimeError("reference precursor_mz contains non-finite values")
    order = np.argsort(precursor, kind="stable")
    sorted_precursor = precursor[order]
    graph = set(
        pd.read_csv(args.participants, usecols=["compound_id"])["compound_id"]
        .dropna()
        .astype(str)
    )

    audit_rows: list[dict] = []
    for qidx, query in query_manifest.reset_index(drop=True).iterrows():
        query_id = f"{query.source_file}|{query.spectrum_id}"
        query_mz = float(query.precursor_mz)
        if not np.isfinite(query_mz) or query_mz <= 0:
            audit_rows.append(
                {
                    "seed_query_id": query_id,
                    "top_candidate_id": "",
                    "top_score": np.nan,
                    "top_margin": np.nan,
                    "candidate_count": 0,
                    "candidate_spectrum_count": 0,
                    "top_candidate_in_graph": False,
                    "rejection_reason": "invalid_precursor_mz",
                }
            )
            continue
        delta = query_mz * args.ppm_tolerance * 1e-6
        left = int(np.searchsorted(sorted_precursor, query_mz - delta, side="left"))
        right = int(np.searchsorted(sorted_precursor, query_mz + delta, side="right"))
        candidate_rows = order[left:right]
        if not len(candidate_rows):
            audit_rows.append(
                {
                    "seed_query_id": query_id,
                    "top_candidate_id": "",
                    "top_score": np.nan,
                    "top_margin": np.nan,
                    "candidate_count": 0,
                    "candidate_spectrum_count": 0,
                    "top_candidate_in_graph": False,
                    "rejection_reason": "no_strict10ppm_candidate",
                }
            )
            continue
        scores = embeddings[candidate_rows] @ np.asarray(
            query_embeddings[int(qidx)], dtype=np.float32
        )
        collapsed = (
            pd.DataFrame(
                {
                    "candidate_id": references.iloc[candidate_rows]["ik14"]
                    .astype(str)
                    .to_numpy(),
                    "spectral_score": np.asarray(scores, dtype=float),
                }
            )
            .groupby("candidate_id", as_index=False)["spectral_score"]
            .max()
            .sort_values(
                ["spectral_score", "candidate_id"], ascending=[False, True]
            )
            .reset_index(drop=True)
        )
        top = collapsed.iloc[0]
        second = float(collapsed.iloc[1].spectral_score) if len(collapsed) > 1 else -1.0
        top_id = str(top.candidate_id)
        audit_rows.append(
            {
                "seed_query_id": query_id,
                "top_candidate_id": top_id,
                "top_score": float(top.spectral_score),
                "top_margin": float(top.spectral_score) - second,
                "candidate_count": int(len(collapsed)),
                "candidate_spectrum_count": int(len(candidate_rows)),
                "top_candidate_in_graph": top_id in graph,
                "rejection_reason": (
                    "evaluable_graph_top" if top_id in graph else "top_outside_graph"
                ),
            }
        )
        if (qidx + 1) % 1000 == 0 or qidx + 1 == len(query_manifest):
            print(f"[broad-seed] {qidx + 1:,}/{len(query_manifest):,}", flush=True)

    audit = pd.DataFrame(audit_rows)
    audit_path = args.output_dir / "auto_seed_audit.csv.gz"
    report_path = args.output_dir / "report.json"
    audit.to_csv(audit_path, index=False)
    evaluable = audit["top_score"].notna()
    graph_top = evaluable & audit["top_candidate_in_graph"].astype(bool)
    report = {
        "status": "mtbls1905_broad_qc_seed_audit_complete",
        "formal": True,
        "queries": int(len(audit)),
        "reference_spectra": int(len(references)),
        "reference_identities": int(references["ik14"].astype(str).nunique()),
        "reference_graph_identities": int(
            references.loc[references["ik14"].astype(str).isin(graph), "ik14"]
            .astype(str)
            .nunique()
        ),
        "evaluable_queries": int(evaluable.sum()),
        "queries_with_graph_top1": int(graph_top.sum()),
        "graph_top1_identities": int(
            audit.loc[graph_top, "top_candidate_id"].astype(str).nunique()
        ),
        "protocol": (
            "strict-10ppm; all mass-compatible library identities compete; "
            "max score per IK14; graph membership checked only after Top-1"
        ),
        "phenotype_labels_used": False,
        "truth_labels_used": False,
        "parameters": {"ppm_tolerance": float(args.ppm_tolerance)},
        "provenance": provenance,
        "audit_sha256": sha256(audit_path),
        "claim_limit": (
            "Candidate-coverage and seed-input audit only; no seed reliability "
            "or annotation improvement is established."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
