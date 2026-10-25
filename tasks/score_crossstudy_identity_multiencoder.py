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
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from encode_pan_redu_panel_spectra import load_mgf
from massspecgym_embedding_readout_core import (
    FrozenEmbeddingReadout, ReadoutSettings, build_independent_scalar_features,
)
from noise_gnps_article_spectral_scores import (
    P2B_WEIGHTS, frozen_p2b_pair_features,
)


NEW_STUDY = "MSV000095331"
NEW_MGF = Path("data/reference/msv000095331_drug_microbiome_20261023/consensus.mgf")
CHEMPROP_MGF = {
    "MSV000094899": Path("data/reference/chemprop2_m1_20261011/consensus.mgf"),
    "MSV000096724": Path(
        "data/reference/chemprop_cross_design_20261011/chemprop_23drug/spectra/specs_ms.mgf"
    ),
    "MSV000093571": Path(
        "data/reference/chemprop_cross_design_20261011/chemprop_45drug/spectra/specs_ms.mgf"
    ),
}


def prepare_spectrum(spectrum: np.ndarray, n_peaks: int = 100) -> np.ndarray:
    values = np.asarray(spectrum, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("spectrum must have shape (2, peaks)")
    keep = (
        np.isfinite(values[0]) & np.isfinite(values[1])
        & (values[0] > 0) & (values[1] > 0)
    )
    values = values[:, keep]
    if values.shape[1] > n_peaks:
        selected = np.argpartition(values[1], -n_peaks)[-n_peaks:]
        values = values[:, selected]
    return values[:, np.argsort(values[0], kind="stable")]


def _entropy_weight(intensity: np.ndarray) -> np.ndarray:
    values = np.asarray(intensity, dtype=np.float64)
    values /= max(float(values.sum()), 1e-15)
    nonzero = values > 0
    entropy = -float(np.sum(values[nonzero] * np.log(values[nonzero])))
    if entropy < 3.0:
        values = values ** (0.25 + 0.25 * entropy)
        values /= max(float(values.sum()), 1e-15)
    return values


def weighted_entropy_similarity(
    spectrum_a: np.ndarray, spectrum_b: np.ndarray, tolerance: float = 0.02,
) -> float:
    """Dependency-free weighted spectral entropy with greedy peak matching."""
    mz_a, mz_b = spectrum_a[0], spectrum_b[0]
    intensity_a = _entropy_weight(spectrum_a[1])
    intensity_b = _entropy_weight(spectrum_b[1])
    candidates: list[tuple[float, int, int]] = []
    for i, value in enumerate(mz_a):
        left = int(np.searchsorted(mz_b, value - tolerance, side="left"))
        right = int(np.searchsorted(mz_b, value + tolerance, side="right"))
        candidates.extend((abs(float(value - mz_b[j])), i, j) for j in range(left, right))
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, i, j in sorted(candidates):
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); matches.append((i, j))
    pa: list[float] = []
    pb: list[float] = []
    for i, j in matches:
        pa.append(float(intensity_a[i])); pb.append(float(intensity_b[j]))
    for i in set(range(len(intensity_a))) - used_a:
        pa.append(float(intensity_a[i])); pb.append(0.0)
    for j in set(range(len(intensity_b))) - used_b:
        pa.append(0.0); pb.append(float(intensity_b[j]))
    left = np.asarray(pa, dtype=np.float64)
    right = np.asarray(pb, dtype=np.float64)
    mean = 0.5 * (left + right)
    nz_left, nz_right = left > 0, right > 0
    js = 0.5 * np.sum(left[nz_left] * np.log(left[nz_left] / mean[nz_left]))
    js += 0.5 * np.sum(right[nz_right] * np.log(right[nz_right] / mean[nz_right]))
    return float(np.clip(1.0 - js / math.log(2.0), 0.0, 1.0))


def weighted_entropy_backend() -> str:
    return "dependency_free_weighted_entropy_v1"


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


def frozen_p2b_block(
    noise: np.ndarray, sqrt_cosine: np.ndarray, entropy: np.ndarray,
    neutral_loss: np.ndarray,
) -> np.ndarray:
    """Apply the registered one-reference P2b support gate to one query."""
    noise = np.asarray(noise, dtype=np.float64)
    raw = np.stack((sqrt_cosine, entropy, neutral_loss), axis=1).clip(0, 1)
    channels = np.column_stack((np.clip((noise + 1) / 2, 0, 1), raw))
    fused = channels @ P2B_WEIGHTS

    def unique_top(values: np.ndarray) -> int | None:
        winners = np.flatnonzero(values >= float(np.max(values)) - 1e-12)
        return int(winners[0]) if len(winners) == 1 else None

    winner = unique_top(fused)
    if winner is None:
        return noise.astype(np.float32)
    support = sum(unique_top(raw[:, column]) == winner for column in range(3))
    baseline = unique_top(noise)
    advantage = math.inf if baseline is None else (
        0.0 if winner == baseline else float(fused[winner] - fused[baseline])
    )
    return (fused if support >= 1 and advantage >= -1e-12 else noise).astype(np.float32)


def apply_unified_model(
    pairs: pd.DataFrame, caches: dict[str, dict[str, np.ndarray]], model_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the selected MassSpecGym ranker without biological outcomes."""
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only keyword.
        payload = torch.load(model_path, map_location="cpu")
    settings = ReadoutSettings(**payload["settings"])
    encoder_names = tuple(payload["encoder_names"])
    if encoder_names != settings.encoder_names:
        raise RuntimeError("unified model encoder registry drifted")
    missing = sorted(set(encoder_names) - set(caches))
    if missing:
        raise RuntimeError(f"unified model requires missing encoder caches: {missing}")
    model = FrozenEmbeddingReadout(str(payload["arm"]), settings)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()

    required_columns = ["WSE_score", "P2b_score", *[f"{name}_score" for name in encoder_names]]
    row_complete = pairs[required_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    query_complete = row_complete.groupby(pairs["query_id"], sort=False).transform("all")
    selected_indices = np.flatnonzero(query_complete.to_numpy(bool))
    full_output = np.full(len(pairs), np.nan, dtype=np.float32)
    pairs = pairs.iloc[selected_indices].copy().reset_index(drop=True)
    if pairs.empty:
        return full_output, {
            "model_path": str(model_path), "model_sha256": sha256(model_path),
            "arm": str(payload["arm"]), "encoder_names": list(encoder_names),
            "scored_pairs": 0, "scored_queries": 0,
            "chemaware_v2": "unavailable_without_candidate_structures_exact_zero",
        }

    ordered_queries = pairs["query_id"].drop_duplicates().astype(str).tolist()
    counts = pairs.groupby("query_id", sort=False).size().reindex(ordered_queries).to_numpy(np.int64)
    query_ptr = np.concatenate(([0], np.cumsum(counts)))
    if int(query_ptr[-1]) != len(pairs):
        raise RuntimeError("application candidate rows are not query-contiguous")
    graph = SimpleNamespace(
        query_ptr=query_ptr,
        molecule_ptr=np.arange(len(pairs) + 1, dtype=np.int64),
        molecule_label=np.zeros(len(pairs), dtype=np.int8),
    )
    encoder_scores = {
        name: pd.to_numeric(pairs[f"{name}_score"], errors="raise").to_numpy(np.float64)
        for name in encoder_names
    }
    evidence = {
        "v1_cosine": encoder_scores["noise_v1"],
        "weighted_entropy": pd.to_numeric(pairs["WSE_score"], errors="raise").to_numpy(float),
        "p2b_fused": pd.to_numeric(pairs["P2b_score"], errors="raise").to_numpy(float),
    }
    unavailable_chemistry = {
        "chem_valid": np.zeros(len(pairs), dtype=bool),
        "candidate_utility_correct": np.zeros(len(pairs), dtype=np.float64),
        "candidate_utility_zero": np.zeros(len(pairs), dtype=np.float64),
        "candidate_utility_reversed": np.zeros(len(pairs), dtype=np.float64),
        "candidate_utility_rotated": np.zeros(len(pairs), dtype=np.float64),
    }
    scalar, names = build_independent_scalar_features(
        evidence, graph, encoder_scores, "noise_v1", unavailable_chemistry,
    )
    if list(names) != list(payload["scalar_feature_names"]):
        raise RuntimeError("application scalar feature contract differs from selected model")
    mean = np.asarray(payload["scalar_mean"], dtype=np.float32)
    scale = np.asarray(payload["scalar_scale"], dtype=np.float32)

    entity_ids = sorted(set(pairs["query_id"].astype(str)) | set(pairs["candidate_id"].astype(str)))
    entity_index = {entity: index for index, entity in enumerate(entity_ids)}
    banks = {
        name: torch.as_tensor(
            np.stack([caches[name][entity] for entity in entity_ids]).astype(np.float32),
            device=device,
        )
        for name in encoder_names
    }
    query_rows = np.asarray([entity_index[value] for value in pairs["query_id"].astype(str)], dtype=np.int64)
    candidate_rows = np.asarray([entity_index[value] for value in pairs["candidate_id"].astype(str)], dtype=np.int64)
    reference_rows = np.repeat(candidate_rows[:, None], len(encoder_names), axis=1)
    output = np.empty(len(pairs), dtype=np.float32)
    with torch.inference_mode():
        for left in range(0, len(pairs), 8192):
            right = min(left + 8192, len(pairs))
            scalar_tensor = torch.as_tensor((scalar[left:right] - mean) / scale, device=device)
            output[left:right] = model(
                scalar_tensor,
                torch.as_tensor(query_rows[left:right], device=device),
                torch.as_tensor(reference_rows[left:right], device=device),
                banks,
            ).float().cpu().numpy()
    full_output[selected_indices] = output
    return full_output, {
        "model_path": str(model_path), "model_sha256": sha256(model_path),
        "arm": str(payload["arm"]), "encoder_names": list(encoder_names),
        "scored_pairs": int(len(pairs)), "scored_queries": int(len(ordered_queries)),
        "missing_rule": "score only complete query candidate blocks; never shrink a query selectively",
        "chemaware_v2": "unavailable_without_candidate_structures_exact_zero",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", action="append", required=True,
                        help="Repeated METHOD=NPZ input")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    parser.add_argument("--fragment-da", type=float, default=0.02)
    parser.add_argument("--unified-model", type=Path)
    parser.add_argument("--device", default="cuda")
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
                    if query_id in caches[name] and candidate_id in caches[name]:
                        record[f"{name}_score"] = float(
                            np.dot(caches[name][query_id], caches[name][candidate_id])
                        )
                    else:
                        record[f"{name}_score"] = np.nan
                sqrt_cosine, entropy, neutral_loss = frozen_p2b_pair_features(
                    prepare_spectrum(query_peaks.T), query_mz,
                    prepare_spectrum(cand_peaks.T), cand_mz,
                    tolerance=args.fragment_da,
                )
                record["P2b_sqrt_cosine"] = sqrt_cosine
                record["P2b_entropy"] = entropy
                record["P2b_neutral_loss"] = neutral_loss
                block.append(record)
            if block:
                noise_method = "noise_v1"
                if noise_method not in methods:
                    raise RuntimeError("cross-study unified evidence requires noise_v1 embeddings")
                noise = np.asarray([float(row[f"{noise_method}_score"]) for row in block])
                p2b = np.full(len(block), np.nan, dtype=np.float32)
                available = np.isfinite(noise)
                if available.any():
                    p2b[available] = frozen_p2b_block(
                        noise[available],
                        np.asarray([float(row["P2b_sqrt_cosine"]) for row in block])[available],
                        np.asarray([float(row["P2b_entropy"]) for row in block])[available],
                        np.asarray([float(row["P2b_neutral_loss"]) for row in block])[available],
                    )
                for record, value in zip(block, p2b, strict=True):
                    record["P2b_score"] = float(value)
                components = [np.asarray([float(row["WSE_score"]) for row in block])]
                components.extend(np.asarray([float(row[f"{name}_score"]) for row in block]) for name in methods)
                component_matrix = np.stack(components)
                simple = np.full(len(block), np.nan, dtype=np.float64)
                if np.isfinite(component_matrix).all():
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
    unified_provenance = None
    if args.unified_model is not None and not pairs.empty:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("unified application inference requested CUDA without an allocated GPU")
        pairs["unified_candidate_ranker_score"], unified_provenance = apply_unified_model(
            pairs, caches, args.unified_model, device,
        )
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
        "p2b_noise_v1_frozen": "P2b_score",
        **{name: f"{name}_score" for name in methods},
        "simple_query_minmax_mean": "simple_fusion_score",
    }
    if args.unified_model is not None:
        score_columns["unified_candidate_ranker"] = "unified_candidate_ranker_score"
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
        "methods": list(score_columns),
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
            "unified_model": unified_provenance,
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
