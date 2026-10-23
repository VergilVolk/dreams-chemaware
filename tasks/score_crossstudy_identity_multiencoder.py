#!/usr/bin/env python3
"""Fully score mass-compatible cross-study identity candidates.

Unlike the earlier application handoff, candidate admission does not depend on
a WSE threshold.  Every candidate within the fixed precursor-ppm identity
window is scored by WSE and every supplied frozen encoder.  Chemical-relation
edges outside that window are deliberately not called identities here.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_msv000095331_chemprop_open_retrieval import CHEMPROP_MGF, NEW_STUDY
from encode_pan_redu_panel_spectra import load_mgf
from noise_gnps_article_spectral_scores import (
    prepare_spectrum,
    weighted_entropy_backend,
    weighted_entropy_similarity,
)


NEW_MGF = Path("data/reference/msv000095331_drug_microbiome_20261023/consensus.mgf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_embedding(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        ids = np.asarray(body["entity_id"]).astype("U")
        matrix = np.asarray(body["embeddings"], dtype=np.float32)
    if matrix.ndim != 2 or len(ids) != len(matrix) or len(ids) != len(set(ids)):
        raise RuntimeError(f"invalid embedding cache: {path}")
    return dict(zip(ids.tolist(), matrix, strict=True))


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo, hi = float(np.min(values)), float(np.max(values))
    return np.zeros_like(values) if hi <= lo else (values - lo) / (hi - lo)


def matched_peaks(left: np.ndarray, right: np.ndarray, tolerance: float) -> int:
    a = np.sort(np.asarray(left, dtype=np.float64)[:, 0])
    b = np.sort(np.asarray(right, dtype=np.float64)[:, 0])
    i = j = total = 0
    while i < len(a) and j < len(b):
        delta = a[i] - b[j]
        if abs(delta) <= tolerance:
            total += 1; i += 1; j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", action="append", required=True,
                        help="Repeated METHOD=NPZ input")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    parser.add_argument("--fragment-da", type=float, default=0.02)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    methods: list[str] = []
    caches: dict[str, dict[str, np.ndarray]] = {}
    cache_paths: dict[str, Path] = {}
    for item in args.embedding:
        if "=" not in item:
            raise ValueError(f"invalid --embedding value: {item}")
        method, raw_path = item.split("=", 1)
        if not method or method in caches:
            raise ValueError(f"duplicate or empty method: {method!r}")
        path = Path(raw_path)
        caches[method] = load_embedding(path)
        cache_paths[method] = path
        methods.append(method)

    mgfs = {NEW_STUDY: NEW_MGF, **CHEMPROP_MGF}
    spectra: dict[str, dict[str, tuple[float, np.ndarray]]] = {}
    for study, path in mgfs.items():
        ids, records = load_mgf(path, minimum_peaks=0, skip_ineligible=False)
        spectra[study] = {
            str(entity): (float(record["precursor_mz"]), np.asarray(record["peaks"], dtype=np.float32).T)
            for entity, record in zip(ids, records, strict=True)
        }

    rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    directions = [(NEW_STUDY, study) for study in CHEMPROP_MGF]
    directions += [(study, NEW_STUDY) for study in CHEMPROP_MGF]
    for query_study, candidate_study in directions:
        candidate_items = sorted(spectra[candidate_study].items())
        candidate_mz = np.asarray([item[1][0] for item in candidate_items], dtype=float)
        order = np.argsort(candidate_mz, kind="mergesort")
        candidate_mz = candidate_mz[order]
        candidate_items = [candidate_items[index] for index in order]
        for raw_query, (query_mz, query_peaks) in sorted(spectra[query_study].items()):
            query_id = f"{query_study}::{raw_query}"
            delta = query_mz * args.precursor_ppm * 1e-6
            left = int(np.searchsorted(candidate_mz, query_mz - delta, side="left"))
            right = int(np.searchsorted(candidate_mz, query_mz + delta, side="right"))
            block: list[dict[str, object]] = []
            for raw_candidate, (cand_mz, cand_peaks) in candidate_items[left:right]:
                candidate_id = f"{candidate_study}::{raw_candidate}"
                if any(query_id not in caches[name] or candidate_id not in caches[name]
                       for name in methods):
                    continue
                wse = weighted_entropy_similarity(
                    prepare_spectrum(query_peaks.T),
                    prepare_spectrum(cand_peaks.T),
                    tolerance=args.fragment_da,
                )
                text = f"{query_id}|{candidate_id}"
                record: dict[str, object] = {
                    "candidate_pair_id": hashlib.sha256(text.encode()).hexdigest(),
                    "query_id": query_id, "candidate_id": candidate_id,
                    "query_study_id": query_study, "candidate_study_id": candidate_study,
                    "query_precursor_mz": query_mz, "candidate_precursor_mz": cand_mz,
                    "precursor_ppm_difference": abs(query_mz - cand_mz) / ((query_mz + cand_mz) / 2) * 1e6,
                    "WSE_score": wse,
                    "WSE_matched_fragments": matched_peaks(query_peaks, cand_peaks, args.fragment_da),
                    "edge_track": "precursor_compatible_identity_hypothesis",
                }
                for name in methods:
                    record[f"{name}_score"] = float(np.dot(caches[name][query_id], caches[name][candidate_id]))
                block.append(record)
            if block:
                components = [np.asarray([float(row["WSE_score"]) for row in block])]
                components.extend(np.asarray([float(row[f"{name}_score"]) for row in block]) for name in methods)
                simple = np.mean(np.stack([minmax(value) for value in components]), axis=0)
                for record, value in zip(block, simple, strict=True):
                    record["simple_fusion_score"] = float(value)
                rows.extend(block)
            query_rows.append({
                "query_id": query_id, "query_study_id": query_study,
                "candidate_study_id": candidate_study, "candidate_count": len(block),
                "query_status": "identity_candidates" if block else "no_identity_candidate",
            })

    pairs = pd.DataFrame(rows)
    queries = pd.DataFrame(query_rows)
    if not pairs.empty and pairs["candidate_pair_id"].duplicated().any():
        raise RuntimeError("duplicate directed identity pair")
    args.output_dir.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    pair_path = args.output_dir / "identity_candidate_scores.tsv.gz"
    long_path = args.output_dir / "method_scores_long.tsv.gz"
    query_path = args.output_dir / "directional_query_registry.tsv.gz"
    pairs.to_csv(pair_path, sep="\t", index=False, compression=compression)
    queries.to_csv(query_path, sep="\t", index=False, compression=compression)
    long_rows: list[pd.DataFrame] = []
    score_columns = {
        "weighted_spectral_entropy": "WSE_score",
        **{name: f"{name}_score" for name in methods},
        "simple_query_minmax_mean": "simple_fusion_score",
    }
    for method, column in score_columns.items():
        block = pairs[["candidate_pair_id", "query_id", "candidate_id"]].copy()
        block["method_name"] = method
        block["method_score"] = pd.to_numeric(pairs[column], errors="coerce")
        block["evidence_available"] = block["method_score"].notna()
        long_rows.append(block)
    long_table = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame(
        columns=["candidate_pair_id", "query_id", "candidate_id", "method_name",
                 "method_score", "evidence_available"]
    )
    long_table.to_csv(long_path, sep="\t", index=False, compression=compression)
    report = {
        "status": "CROSSSTUDY_IDENTITY_MULTIENCODER_COMPLETE",
        "labels_or_responses_accessed": False,
        "candidate_admission": "all precursor-compatible candidates; no spectral-score threshold",
        "methods": ["weighted_spectral_entropy", *methods, "simple_query_minmax_mean"],
        "weighted_entropy_backend": weighted_entropy_backend(),
        "parameters": {"precursor_ppm": args.precursor_ppm, "fragment_da": args.fragment_da},
        "denominators": {
            "directional_queries": int(len(queries)),
            "queries_with_identity_candidates": int(queries["candidate_count"].gt(0).sum()),
            "identity_candidate_pairs": int(len(pairs)),
        },
        "inputs": {
            "embeddings": {name: {"path": str(cache_paths[name]), "sha256": sha256(cache_paths[name])}
                           for name in methods},
            "mgfs": {study: {"path": str(path), "sha256": sha256(path)} for study, path in mgfs.items()},
        },
        "outputs": {
            "candidate_scores": {"path": str(pair_path), "sha256": sha256(pair_path)},
            "method_scores_long": {"path": str(long_path), "sha256": sha256(long_path)},
            "query_registry": {"path": str(query_path), "sha256": sha256(query_path)},
        },
        "claim_boundary": (
            "Identity-compatible candidate scoring only. Out-of-window chemical relationships, "
            "BioAware context and structure-dependent V2 evidence are separate tracks."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
