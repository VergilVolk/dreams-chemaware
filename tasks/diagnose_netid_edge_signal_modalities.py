#!/usr/bin/env python
"""Diagnose why raw DreaMS cosine does not identify public NetID edges.

The positive edges and matched nonedges are frozen by the preceding audit.
This script adds label-free peak similarities only; it does not refit matches,
select a subgroup, or authorize a downstream overlay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".csv.gz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalized_sqrt_intensity(peaks: np.ndarray) -> np.ndarray:
    intensity = np.sqrt(np.maximum(np.asarray(peaks[:, 1], dtype=float), 0.0))
    norm = float(np.linalg.norm(intensity))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("spectrum has no positive finite intensity")
    return intensity / norm


def peak_cosine(
    peaks_a: np.ndarray,
    precursor_a: float,
    peaks_b: np.ndarray,
    precursor_b: float,
    tolerance: float,
    mode: str,
) -> float:
    """Maximum one-to-one sqrt-intensity peak matching cosine."""

    mz_a = np.asarray(peaks_a[:, 0], dtype=float)
    mz_b = np.asarray(peaks_b[:, 0], dtype=float)
    intensity_a = _normalized_sqrt_intensity(peaks_a)
    intensity_b = _normalized_sqrt_intensity(peaks_b)
    if mode == "direct":
        allowed = np.abs(mz_a[:, None] - mz_b[None, :]) <= tolerance
    elif mode == "neutral_loss":
        loss_a = precursor_a - mz_a
        loss_b = precursor_b - mz_b
        allowed = np.abs(loss_a[:, None] - loss_b[None, :]) <= tolerance
    elif mode == "modified":
        direct = np.abs(mz_a[:, None] - mz_b[None, :]) <= tolerance
        shifted = (
            np.abs(
                (mz_a[:, None] - mz_b[None, :])
                - (float(precursor_a) - float(precursor_b))
            )
            <= tolerance
        )
        allowed = direct | shifted
    else:
        raise ValueError(f"unknown peak matching mode: {mode}")
    weight = intensity_a[:, None] * intensity_b[None, :]
    weight = np.where(allowed, weight, 0.0)
    rows, columns = linear_sum_assignment(-weight)
    return float(weight[rows, columns].sum())


def unpack_feature_spectra(cache_path: Path, minimum_peaks: int) -> dict[int, list[tuple[np.ndarray, float]]]:
    with np.load(cache_path, allow_pickle=False) as cache:
        offsets = np.asarray(cache["peak_offsets"], dtype=np.int64)
        counts = np.diff(offsets)
        feature_ids = np.asarray(cache["netid_peak_id"], dtype=np.int64)
        precursors = np.asarray(cache["precursor_mz"], dtype=float)
        fragment_mz = np.asarray(cache["fragment_mz"], dtype=float)
        fragment_intensity = np.asarray(cache["fragment_intensity"], dtype=float)
        result: dict[int, list[tuple[np.ndarray, float]]] = {}
        for index in np.flatnonzero(counts >= minimum_peaks):
            start, end = int(offsets[index]), int(offsets[index + 1])
            peaks = np.stack(
                [fragment_mz[start:end], fragment_intensity[start:end]], axis=1
            )
            result.setdefault(int(feature_ids[index]), []).append(
                (peaks, float(precursors[index]))
            )
    return result


def feature_pair_score(
    spectra: dict[int, list[tuple[np.ndarray, float]]],
    left: int,
    right: int,
    tolerance: float,
    mode: str,
) -> float:
    scores = [
        peak_cosine(peaks_a, precursor_a, peaks_b, precursor_b, tolerance, mode)
        for peaks_a, precursor_a in spectra[int(left)]
        for peaks_b, precursor_b in spectra[int(right)]
    ]
    return float(max(scores))


def cluster_bootstrap(
    delta: np.ndarray, clusters: np.ndarray, repeats: int, seed: int
) -> dict[str, Any]:
    unique = np.unique(clusters)
    grouped = {value: delta[clusters == value] for value in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        selected = rng.choice(unique, size=len(unique), replace=True)
        draws[repeat] = float(np.mean(np.concatenate([grouped[value] for value in selected])))
    return {
        "mean": float(np.mean(delta)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(unique)),
        "resamples": int(repeats),
    }


def summarize(frame: pd.DataFrame, score: str, repeats: int, seed: int) -> dict[str, Any]:
    decoy_columns = [column for column in frame if column.startswith(f"decoy_{score}_")]
    positive = frame[score].to_numpy(float)
    decoy = frame[decoy_columns].to_numpy(float)
    delta = positive - decoy.mean(axis=1)
    labels = np.concatenate([np.ones(len(positive)), np.zeros(decoy.size)])
    scores = np.concatenate([positive, decoy.ravel()])
    return {
        "edges": int(len(frame)),
        "positive_mean": float(np.mean(positive)),
        "matched_nonedge_mean": float(np.mean(decoy)),
        "paired_mean_delta": float(np.mean(delta)),
        "matched_auc": float(roc_auc_score(labels, scores)),
        "component_cluster_bootstrap": cluster_bootstrap(
            delta, frame["component"].to_numpy(int), repeats, seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge-dir",
        type=Path,
        default=Path("data/validation/netid_dreams_edge_signal_20260831"),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("data/validation/netid_public_release_audit_v2_20260831"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/netid_edge_signal_modalities_20260901"),
    )
    parser.add_argument("--minimum-peaks", type=int, default=3)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    edge_report_path = args.edge_dir / "report.json"
    edge_table_path = args.edge_dir / "edge_matched_nonedges.csv.gz"
    report_path = args.output_dir / "report.json"
    table_path = args.output_dir / "edge_modality_diagnostics.csv.gz"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "netid_edge_signal_modalities_complete":
            raise RuntimeError("invalid existing modality diagnostic")
        if sha256(table_path) != report["provenance"]["diagnostic_table_sha256"]:
            raise RuntimeError("existing modality diagnostic table changed")
        print(f"[reuse] verified {report_path}", flush=True)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {args.output_dir}")

    edge_report = json.loads(edge_report_path.read_text(encoding="utf-8"))
    if edge_report.get("status") != "netid_dreams_edge_signal_failed":
        raise RuntimeError("this diagnostic is locked to the failed DreaMS edge gate")
    if sha256(edge_table_path) != edge_report["provenance"]["pair_table_sha256"]:
        raise RuntimeError("frozen matched edge/nonedge table hash mismatch")
    audit_report_path = args.audit_dir / "report.json"
    audit = json.loads(audit_report_path.read_text(encoding="utf-8"))
    spectrum_path = args.audit_dir / audit["artifacts"]["mouse_liver_ms2_spectra"]["relative_path"]
    if sha256(spectrum_path) != audit["artifacts"]["mouse_liver_ms2_spectra"]["sha256"]:
        raise RuntimeError("public spectrum cache hash mismatch")

    frame = pd.read_csv(edge_table_path)
    spectra = unpack_feature_spectra(spectrum_path, args.minimum_peaks)
    modes = ("direct", "neutral_loss", "modified")
    for mode in modes:
        frame[mode] = [
            feature_pair_score(
                spectra, left, right, args.fragment_tolerance, mode
            )
            for left, right in frame[["feature1", "feature2"]].to_numpy(int)
        ]
        for control in range(edge_report["controls_per_edge"]):
            frame[f"decoy_{mode}_{control}"] = [
                feature_pair_score(
                    spectra, left, right, args.fragment_tolerance, mode
                )
                for left, right in frame[
                    [f"decoy_feature1_{control}", f"decoy_feature2_{control}"]
                ].to_numpy(int)
            ]

    families = {
        "overall": frame,
        "biotransform": frame[frame["family"].eq("biotransform")],
        "ion_phenomenon": frame[frame["family"].eq("ion_phenomenon")],
    }
    summaries = {
        family: {
            mode: summarize(subset, mode, args.bootstrap_resamples, args.seed + 10 * i + j)
            for j, mode in enumerate(modes)
        }
        for i, (family, subset) in enumerate(families.items())
    }
    author_ms2_edges = int(frame["score_MS2_similarity"].notna().sum())
    primary = {
        "biotransform_modified": summaries["biotransform"]["modified"],
        "ion_phenomenon_direct": summaries["ion_phenomenon"]["direct"],
    }
    diagnosis = (
        "raw_relation_specific_signal_present_but_dreams_hook_misaligned"
        if any(value["component_cluster_bootstrap"]["ci_low"] > 0 for value in primary.values())
        else "public_post_solution_edges_not_identifiable_by_preregistered_raw_ms2_scores"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_csv(table_path, frame)
    report = {
        "status": "netid_edge_signal_modalities_complete",
        "formal": True,
        "protocol": "same frozen positives and matched nonedges; no rematching, fitting or subgroup selection",
        "edges": int(len(frame)),
        "author_edges_with_explicit_ms2_similarity_score": author_ms2_edges,
        "author_edges_without_explicit_ms2_similarity_score": int(len(frame) - author_ms2_edges),
        "modalities": summaries,
        "preregistered_primary_diagnostics": primary,
        "diagnosis": diagnosis,
        "contracts": {
            "edge_gate_changed": False,
            "matched_nonedges_changed": False,
            "outcome_used_for_matching": False,
            "posthoc_subgroup_promoted_to_result": False,
            "author_graph_is_independent_truth": False,
            "overlay_authorized": False,
        },
        "provenance": {
            "edge_report_sha256": sha256(edge_report_path),
            "edge_table_sha256": sha256(edge_table_path),
            "spectrum_cache_sha256": sha256(spectrum_path),
            "diagnostic_table_sha256": sha256(table_path),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "claim_limit": (
            "Failure-mechanism diagnostic only. The post-solution NetID graph is not "
            "independent edge truth, and no modality result establishes annotation gain."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
