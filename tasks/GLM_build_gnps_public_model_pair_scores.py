#!/usr/bin/env python
"""Score the sealed GNPS panels with pinned public pretrained models.

Two literature-standard public models are supported, both pinned by Zenodo
record and md5, both scored on the exact frozen panel edge order and written
through the shared gnps_pair_score_cache_v1 contract:

- spec2vec_gnps_2019: the pretrained Spec2Vec Word2Vec model of the PLoS
  Computational Biology article (Zenodo 3978054, trained 50 iterations on the
  GNPS UniqueInchikey positive-mode subset).
- ms2deepscore_dual_2024: the dual ionisation-mode MS2DeepScore model
  (Zenodo 13897744 v2).  Its settings require precursor_mz and ionmode
  metadata; the benchmark is entirely [M+H]+ positive mode.

No method selection, no tuning, no truth contact.  Heavy dependencies
(matchms, gensim, ms2deepscore) are imported lazily inside the scoring
functions so the module imports anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel
from gnps_pair_score_cache import PANELS, write_pair_score_cache
from noise_corrected_fullgraph_evaluation import expanded_indices


ROOT = Path(__file__).resolve().parents[1]
PROTON_MASS_DA = 1.007276466621

MODEL_PINNING = {
    "spec2vec_gnps_2019": {
        "zenodo": "10.5281/zenodo.3978054",
        "files_md5": {
            "spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model":
                "a7333c19ace863c5a0a04abc89600f49",
            "spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model.trainables.syn1neg.npy":
                "cd774bf75d00b1715334a4fbfe2b711a",
            "spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model.wv.vectors.npy":
                "0962044c1020073d487ae2e07d195f0b",
        },
        "main_file": "spec2vec_UniqueInchikeys_ratio05_filtered_iter_50.model",
        "note": "newer zenodo versions exist; this pinned release is the article model",
    },
    "ms2deepscore_dual_2024": {
        "zenodo": "10.5281/zenodo.13897744",
        "files_md5": {
            "ms2deepscore_model.pt": "d5cbf4694a1c476ae59e0c810f56c320",
            "settings.json": "5d6007c453692b44cd73941a9a434d84",
        },
        "main_file": "ms2deepscore_model.pt",
        "note": "v2 dual-mode model; newer record exists; pinned for reproducibility",
    },
}


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_dir(method: str, model_dir: Path) -> dict[str, str]:
    pinning = MODEL_PINNING[method]
    observed: dict[str, str] = {}
    for name, pinned in pinning["files_md5"].items():
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"{method} model file missing: {path}")
        actual = md5_of(path)
        if actual != pinned:
            raise RuntimeError(
                f"{method} model md5 mismatch for {name}: "
                f"pinned={pinned} observed={actual}"
            )
        observed[name] = actual
    return observed


def panel_edge_rows(benchmark: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Directed (query_row, candidate_row) edges in the frozen panel order."""
    edges: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in PANELS:
        graph = graph_from_panel(benchmark / f"panel_{name}.npz")
        _, pair_query = expanded_indices(graph)
        queries = np.asarray(graph.query_row, dtype=np.int64)[pair_query]
        candidates = np.asarray(graph.pair_candidate_row, dtype=np.int64)
        if len(queries) != len(candidates):
            raise RuntimeError(f"panel {name} edge arrays are not aligned")
        edges[name] = (queries, candidates)
    return edges


def load_matchms_spectra(
    benchmark: Path, used_rows: set[int],
) -> dict[int, object]:
    """Build matchms Spectrum objects for the registered benchmark rows.

    The dual-mode MS2DeepScore settings require precursor_mz and ionmode
    metadata; the whole benchmark is validated [M+H]+ positive mode.  The
    canonical MGF iterator is imported lazily because its home module pulls
    the full torch stack at import time.
    """
    from encode_unified_library_for_p2b import iter_mgf  # lazy heavy import
    from matchms import Spectrum  # lazy heavy import

    spectra: dict[int, object] = {}
    for row, (fields, precursor, peaks) in enumerate(iter_mgf(benchmark / "spectra.mgf")):
        if row in used_rows:
            mz = np.asarray(peaks[0], dtype=np.float64)
            intensity = np.asarray(peaks[1], dtype=np.float64)
            keep = (mz > 0) & np.isfinite(mz) & (intensity > 0)
            clean_intensity = intensity[keep]
            clean_intensity = clean_intensity / float(np.max(clean_intensity))
            spectra[row] = Spectrum(
                mz=mz[keep],
                intensities=clean_intensity,
                metadata={
                    "precursor_mz": float(precursor),
                    # The sealed benchmark is validated [M+H]+ only.  The
                    # Spec2Vec article preprocessing scales its peak cap by
                    # neutral parent mass, not precursor m/z.
                    "parent_mass": float(precursor) - PROTON_MASS_DA,
                    "ionmode": "positive",
                    "charge": 1,
                    "adduct": "[M+H]+",
                },
            )
        if len(spectra) == len(used_rows):
            break
    missing = sorted(used_rows - set(spectra))
    if missing:
        raise RuntimeError(f"MGF misses {len(missing)} registered rows; first={missing[:5]}")
    return spectra


def package_version(name: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001 - record, never fail provenance on this
        return "unknown"


def make_scorer(method: str, model_dir: Path) -> tuple[object, dict[str, object]]:
    """Lazy-import the public scorer with literature-standard parameters."""
    if method == "spec2vec_gnps_2019":
        from gensim.models import Word2Vec
        from spec2vec import Spec2Vec

        model = Word2Vec.load(str(model_dir / MODEL_PINNING[method]["main_file"]))
        scorer = Spec2Vec(
            model=model, intensity_weighting_power=0.5, allowed_missing_percentage=10.0,
        )
        parameters = {
            "intensity_weighting_power": 0.5,
            "allowed_missing_percentage": 10.0,
            "preprocessing": {
                "normalize_intensities": True,
                "mz_range": [0.0, 1000.0],
                "minimum_peaks": 10,
                "reduce_to_number_of_peaks_ratio_desired": 0.5,
                "relative_intensity_floor_if_ten_remain": 0.001,
                "neutral_loss_range_da": [5.0, 200.0],
            },
            "packages": {
                "matchms": package_version("matchms"),
                "spec2vec": package_version("spec2vec"),
                "gensim": package_version("gensim"),
            },
        }
        return scorer, parameters
    if method == "ms2deepscore_dual_2024":
        from ms2deepscore import MS2DeepScore
        from ms2deepscore.models import load_model

        # The pinned model was deposited on 2024-10-07 and is paired with the
        # contemporaneous 2.4 loader.  Newer loaders add ``allow_legacy``;
        # retain a narrow compatibility fallback so the adapter records which
        # serialization path was taken.
        model_path = str(model_dir / MODEL_PINNING[method]["main_file"])
        try:
            model = load_model(model_path, allow_legacy=True)
            loader = "safe_first_allow_legacy"
        except TypeError:
            model = load_model(model_path)
            loader = "ms2deepscore_2_4_native"
        scorer = MS2DeepScore(model)
        parameters = {
            "loader": loader,
            "packages": {
                "ms2deepscore": package_version("ms2deepscore"),
                "matchms": package_version("matchms"),
            },
        }
        return scorer, parameters
    raise RuntimeError(f"unsupported public model method: {method}")


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise RuntimeError("public-model embeddings are malformed or non-finite")
    norm = np.linalg.norm(values, axis=1)
    output = np.zeros_like(values)
    keep = norm > 1e-12
    output[keep] = values[keep] / norm[keep, None]
    return output


def prepare_spec2vec_document(spectrum: object, n_decimals: int) -> object | None:
    """Reproduce the preprocessing used to train the pinned article model.

    The model was trained after normalization, 0--1000 m/z selection, a
    minimum of ten peaks, mass-scaled peak reduction (ratio_desired=0.5), a
    0.001 relative-intensity floor when ten peaks remain, and neutral losses
    restricted to 5--200 Da. matchms 0.27 computes every possible loss
    lazily, so the final loss-word restriction is applied explicitly to the
    SpectrumDocument.
    """
    from matchms.filtering import (
        normalize_intensities,
        reduce_to_number_of_peaks,
        require_minimum_number_of_peaks,
        select_by_mz,
        select_by_relative_intensity,
    )
    from spec2vec import SpectrumDocument

    processed = normalize_intensities(spectrum)
    processed = select_by_mz(processed, mz_from=0.0, mz_to=1000.0)
    processed = require_minimum_number_of_peaks(processed, n_required=10)
    if processed is None:
        return None
    processed = reduce_to_number_of_peaks(
        processed, n_required=10, ratio_desired=0.5,
    )
    intensity_filtered = select_by_relative_intensity(
        processed, intensity_from=0.001,
    )
    if len(intensity_filtered.peaks) >= 10:
        processed = intensity_filtered
    document = SpectrumDocument(processed, n_decimals=n_decimals)
    peak_count = len(processed.peaks)
    losses = processed.compute_losses(loss_mz_from=5.0, loss_mz_to=200.0)
    document.words = document.words[:peak_count] + [
        f"loss@{mz:.{n_decimals}f}" for mz in losses.mz
    ]
    document.weights = document.weights[:peak_count] + losses.intensities.tolist()
    return document


def embed_registered_spectra(
    method: str,
    scorer: object,
    spectra: dict[int, object],
    ordered_rows: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Encode each registered spectrum once; never forward once per edge."""
    ordered = [spectra[int(row)] for row in ordered_rows]
    if method == "spec2vec_gnps_2019":
        vectors = []
        for spectrum in ordered:
            document = prepare_spec2vec_document(spectrum, scorer.n_decimals)
            if document is None:
                vectors.append(np.zeros(scorer.vector_size, dtype=np.float32))
            else:
                # This is the exact embedding path used by Spec2Vec.pair after
                # constructing the article-compatible SpectrumDocument.
                vectors.append(scorer._calculate_embedding(document))
        vectors = np.stack(vectors).astype(np.float32)
        return unit_rows(vectors)
    if method == "ms2deepscore_dual_2024":
        vectors: list[np.ndarray] = []
        for left in range(0, len(ordered), batch_size):
            right = min(left + batch_size, len(ordered))
            block = scorer.get_embedding_array(ordered[left:right])
            vectors.append(np.asarray(block, dtype=np.float32))
            if right % 5000 < batch_size or right == len(ordered):
                print(f"[{method}] encoded {right:,}/{len(ordered):,} spectra", flush=True)
        return unit_rows(np.concatenate(vectors, axis=0))
    raise RuntimeError(f"unsupported public model method: {method}")


def pair_scores_from_embeddings(
    embeddings: np.ndarray,
    row_positions: dict[int, int],
    queries: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    try:
        query_position = np.asarray([row_positions[int(row)] for row in queries], dtype=np.int64)
        candidate_position = np.asarray(
            [row_positions[int(row)] for row in candidates], dtype=np.int64,
        )
    except KeyError as error:
        raise RuntimeError(f"public embedding registry misses panel row: {error}") from error
    output = np.einsum(
        "ij,ij->i", embeddings[query_position], embeddings[candidate_position], optimize=True,
    ).astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("public-model pair scores are non-finite")
    return output


def score_panel_edges(
    scorer: object,
    spectra: dict[int, object],
    queries: np.ndarray,
    candidates: np.ndarray,
    label: str,
) -> np.ndarray:
    """Score every directed edge in the frozen order via matchms .pair()."""
    scores = np.empty(len(queries), dtype=np.float32)
    for index, (query_row, candidate_row) in enumerate(zip(queries, candidates)):
        value = scorer.pair(
            spectra[int(query_row)], spectra[int(candidate_row)],
        )
        if isinstance(value, tuple):
            value = value[0]
        number = float(value)
        if not np.isfinite(number):
            raise RuntimeError(
                f"non-finite {label} score at edge {index}: "
                f"{int(query_row)}->{int(candidate_row)}"
            )
        scores[index] = number
        if (index + 1) % 5000 == 0 or index + 1 == len(queries):
            print(f"[{label}] {index + 1:,}/{len(queries):,} edges", flush=True)
    return scores


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path,
        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1",
    )
    parser.add_argument(
        "--method", action="append", required=True,
        choices=tuple(MODEL_PINNING),
    )
    parser.add_argument("--spec2vec-model-dir", type=Path)
    parser.add_argument("--ms2deepscore-model-dir", type=Path)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    methods = list(dict.fromkeys(args.method))
    if len(methods) != len(args.method):
        raise RuntimeError("duplicate --method entries")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if args.embedding_batch_size < 1:
        raise ValueError("embedding-batch-size must be positive")
    model_dirs = {
        "spec2vec_gnps_2019": args.spec2vec_model_dir,
        "ms2deepscore_dual_2024": args.ms2deepscore_model_dir,
    }
    for method in methods:
        if model_dirs[method] is None:
            raise RuntimeError(f"a model directory for {method} was not provided")
    verified = {
        method: verify_model_dir(method, model_dirs[method]) for method in methods
    }
    edges = panel_edge_rows(args.benchmark)
    used_rows: set[int] = set()
    for queries, candidates in edges.values():
        used_rows.update(map(int, queries))
        used_rows.update(map(int, candidates))
    print(f"[public models] scoring methods={methods}; spectra={len(used_rows):,}", flush=True)
    spectra = load_matchms_spectra(args.benchmark, used_rows)
    ordered_rows = np.asarray(sorted(used_rows), dtype=np.int64)
    row_positions = {int(row): index for index, row in enumerate(ordered_rows)}

    args.output_root.mkdir(parents=True)
    for method in methods:
        scorer, parameters = make_scorer(method, model_dirs[method])
        embeddings = embed_registered_spectra(
            method, scorer, spectra, ordered_rows, args.embedding_batch_size,
        )
        scores = {
            name: pair_scores_from_embeddings(
                embeddings, row_positions, edges[name][0], edges[name][1],
            )
            for name in PANELS
        }
        report = write_pair_score_cache(
            args.output_root / f"{method}_pair_cache",
            args.benchmark,
            {
                "name": method,
                "kind": "public_pretrained_model",
                "zenodo": MODEL_PINNING[method]["zenodo"],
                "model_files_md5": verified[method],
                "parameters": parameters,
                "inference": {
                    "unique_spectra_encoded_once": int(len(ordered_rows)),
                    "embedding_dimension": int(embeddings.shape[1]),
                    "zero_embedding_spectra": int(np.sum(
                        np.linalg.norm(embeddings, axis=1) <= 1e-12
                    )),
                    "pair_scores": "cosine of cached unit embeddings in frozen edge order",
                    "embedding_batch_size": int(args.embedding_batch_size),
                },
                "claim_limit": MODEL_PINNING[method]["note"],
                "gnps_used_for_model_selection_or_tuning": False,
            },
            scores,
        )
        print(json.dumps(report, indent=2), flush=True)
    print(
        json.dumps(
            {"status": "GLM_GNPS_PUBLIC_MODEL_PAIR_SCORES_COMPLETE",
             "methods": methods}, indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
