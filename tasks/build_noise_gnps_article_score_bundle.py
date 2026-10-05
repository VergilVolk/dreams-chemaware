#!/usr/bin/env python
"""Build classical, Noise and frozen-P2b scores on the sealed GNPS graph."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from encode_unified_library_for_p2b import iter_mgf
from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel, load_embeddings
from noise_corrected_fullgraph_evaluation import expanded_indices, score_embeddings
from noise_gnps_article_spectral_scores import (
    apply_frozen_p2b,
    cosine_greedy,
    frozen_p2b_pair_features,
    modified_cosine,
    prepare_spectrum,
    weighted_entropy_similarity,
    weighted_entropy_backend,
)


ROOT = Path(__file__).resolve().parents[1]
PANELS = ("identity_disjoint", "formula_disjoint")
METHODS = (
    "official_dreams",
    "noise_v1",
    "cosine_greedy",
    "modified_cosine",
    "weighted_spectral_entropy",
    "p2b_sqrt_cosine",
    "p2b_unweighted_entropy",
    "neutral_loss_sqrt_cosine",
    "p2b_official_frozen",
    "p2b_noise_v1_frozen",
)

_SPECTRA: dict[int, np.ndarray] | None = None
_PRECURSOR: np.ndarray | None = None
_TOLERANCE: float | None = None


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path,
        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1",
    )
    parser.add_argument("--official-embeddings", type=Path, required=True)
    parser.add_argument("--noise-v1-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def _init_worker(spectra, precursor, tolerance) -> None:
    global _SPECTRA, _PRECURSOR, _TOLERANCE
    _SPECTRA, _PRECURSOR, _TOLERANCE = spectra, precursor, float(tolerance)


def _score_pair(pair: tuple[int, int]) -> tuple[tuple[int, int], tuple[float, ...]]:
    if _SPECTRA is None or _PRECURSOR is None or _TOLERANCE is None:
        raise RuntimeError("pair-score worker was not initialized")
    left, right = pair
    spectrum_left, spectrum_right = _SPECTRA[left], _SPECTRA[right]
    precursor_left, precursor_right = float(_PRECURSOR[left]), float(_PRECURSOR[right])
    sqrt_cosine, entropy, neutral_loss = frozen_p2b_pair_features(
        spectrum_left, precursor_left, spectrum_right, precursor_right, _TOLERANCE,
    )
    return pair, (
        cosine_greedy(spectrum_left, spectrum_right, _TOLERANCE),
        modified_cosine(
            spectrum_left, precursor_left, spectrum_right, precursor_right, _TOLERANCE,
        ),
        weighted_entropy_similarity(spectrum_left, spectrum_right, _TOLERANCE),
        sqrt_cosine,
        entropy,
        neutral_loss,
    )


def load_spectra(path: Path, used_rows: set[int], n_peaks: int) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for row, (_, _, spectrum) in enumerate(iter_mgf(path)):
        if row in used_rows:
            output[row] = prepare_spectrum(spectrum, n_peaks)
        if len(output) == len(used_rows):
            break
    missing = sorted(used_rows - set(output))
    if missing:
        raise RuntimeError(f"MGF misses {len(missing)} registered rows; first={missing[:5]}")
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.workers < 1 or args.fragment_tolerance <= 0 or args.n_highest_peaks < 1:
        raise ValueError("invalid score-builder configuration")
    if sys.platform == "win32" and args.workers != 1:
        raise RuntimeError("Windows local validation must use --workers 1")
    entropy_backend = weighted_entropy_backend()
    if entropy_backend != "ms_entropy_1.5.2":
        raise RuntimeError(
            "formal article scoring requires pinned official ms_entropy 1.5.2; "
            f"observed {entropy_backend}"
        )
    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row", "precursor_mz"])
    if not np.array_equal(manifest["row"].to_numpy(np.int64), np.arange(len(manifest))):
        raise RuntimeError("GNPS manifest row registry is not contiguous")
    official_rows, official_embeddings = load_embeddings(args.official_embeddings, len(manifest))
    v1_rows, v1_embeddings = load_embeddings(args.noise_v1_embeddings, len(manifest))
    if not np.array_equal(official_rows, v1_rows):
        raise RuntimeError("official and Noise V1 embedding registries differ")

    graphs = {
        name: graph_from_panel(args.benchmark / f"panel_{name}.npz") for name in PANELS
    }
    used_rows: set[int] = set()
    unique_pairs: set[tuple[int, int]] = set()
    panel_pairs: dict[str, list[tuple[int, int]]] = {}
    for name, graph in graphs.items():
        _, pair_query = expanded_indices(graph)
        pairs = [
            tuple(sorted((int(graph.query_row[query]), int(candidate))))
            for query, candidate in zip(pair_query, graph.pair_candidate_row)
        ]
        panel_pairs[name] = pairs
        unique_pairs.update(pairs)
        used_rows.update(row for pair in pairs for row in pair)
    print(
        f"[article scores] used spectra={len(used_rows):,}; unique pairs={len(unique_pairs):,}",
        flush=True,
    )
    spectra = load_spectra(
        args.benchmark / "spectra.mgf", used_rows, args.n_highest_peaks,
    )
    precursor = manifest["precursor_mz"].to_numpy(np.float64)
    ordered_pairs = sorted(unique_pairs)
    lookup: dict[tuple[int, int], tuple[float, ...]] = {}
    if args.workers == 1:
        _init_worker(spectra, precursor, args.fragment_tolerance)
        iterator = map(_score_pair, ordered_pairs)
        executor = None
    else:
        context = mp.get_context("fork")
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(spectra, precursor, args.fragment_tolerance),
        )
        iterator = executor.map(_score_pair, ordered_pairs, chunksize=64)
    try:
        for index, (pair, scores) in enumerate(iterator, start=1):
            lookup[pair] = scores
            if index % 5000 == 0 or index == len(ordered_pairs):
                print(f"[article scores] {index:,}/{len(ordered_pairs):,} pairs", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    arrays: dict[str, np.ndarray] = {
        "method_names": np.asarray(METHODS, dtype="U64"),
    }
    for name, graph in graphs.items():
        official = score_embeddings(graph, official_rows, official_embeddings)
        v1 = score_embeddings(graph, v1_rows, v1_embeddings)
        classical = np.asarray([lookup[pair] for pair in panel_pairs[name]], dtype=np.float32).T
        cosine, modified, weighted_entropy, sqrt_cosine, entropy, neutral_loss = classical
        p2b_official = apply_frozen_p2b(
            graph, official.pair, sqrt_cosine, entropy, neutral_loss,
        )
        p2b_v1 = apply_frozen_p2b(
            graph, v1.pair, sqrt_cosine, entropy, neutral_loss,
        )
        scores = np.stack([
            official.pair, v1.pair, cosine, modified, weighted_entropy,
            sqrt_cosine, entropy, neutral_loss,
            p2b_official.pair, p2b_v1.pair,
        ]).astype(np.float32)
        if scores.shape != (len(METHODS), len(graph.pair_candidate_row)):
            raise RuntimeError(f"score bundle shape drift for {name}: {scores.shape}")
        if not np.all(np.isfinite(scores)):
            raise RuntimeError(f"non-finite method score for {name}")
        arrays[f"scores_{name}"] = scores
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    metadata = {
        "status": "noise_gnps_article_score_bundle_complete",
        "methods": list(METHODS),
        "score_rows": {name: int(arrays[f"scores_{name}"].shape[1]) for name in PANELS},
        "fragment_tolerance_da": args.fragment_tolerance,
        "n_highest_peaks": args.n_highest_peaks,
        "weighted_spectral_entropy_backend": entropy_backend,
        "p2b": {
            "weights": [0.10, 0.00, 0.10, 0.80],
            "normalization": "absolute",
            "minimum_support": 1,
            "minimum_advantage": 0.0,
            "configuration_selection_on_gnps": False,
        },
        "information_levels": {
            "spectrum_only": list(METHODS[:8]),
            "frozen_candidate_reranker": list(METHODS[8:]),
        },
        "claim_limit": (
            "One sealed GNPS graph, no fitting or method selection. Weighted entropy uses the "
            "pinned official MSEntropy implementation; P2b entropy remains its frozen historical feature."
        ),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
