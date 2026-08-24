"""Build the real-only, P3-disjoint P2 molecule-listwise training cache.

The cache preserves complete strict-10ppm candidate groups inside the sealed
P2 allow-list.  Scores are stored per query-spectrum/candidate-spectrum pair;
training and evaluation aggregate them by candidate molecule with max(), the
same operation used by deployment retrieval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from audit_large_observability_residual import symmetric_features  # noqa: E402
from g8r_p2_listwise_core import FEATURE_NAMES, RAW_FEATURES  # noqa: E402
from step5_gate_eval import embed  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_PAIRS = ROOT / "tasks/massspecgym_isomers/pairs.json"
DEFAULT_P3 = ROOT / "data/validation/g8r_p3_test"
DEFAULT_PREFLIGHT = ROOT / "data/validation/g8r_p2_preflight.json"
DEFAULT_BASE = ROOT / "data/e1/official_embedding_slim.pt"
DEFAULT_ARCH = ROOT / "dreams/models/pretrained/ssl_model_server.pt"
DEFAULT_EMBED = ROOT / "data/validation/g8r_p2_official_embeddings.npz"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_p2_listwise_cache.npz"

_WORKER_SPECTRA = None
_WORKER_PRECURSOR = None
_WORKER_TOLERANCE = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    p.add_argument("--p3-dir", type=Path, default=DEFAULT_P3)
    p.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    p.add_argument("--base-ckpt", type=Path, default=DEFAULT_BASE)
    p.add_argument("--architecture-ckpt", type=Path, default=DEFAULT_ARCH)
    p.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBED)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-highest-peaks", type=int, default=100)
    p.add_argument("--ppm-tol", type=float, default=10.0)
    p.add_argument("--peak-tolerance", type=float, default=0.02)
    p.add_argument("--max-queries", type=int, default=20000)
    p.add_argument("--queries-per-identity", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel CPU workers for RAW peak-pair features.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def text_values(dataset, rows: np.ndarray) -> np.ndarray:
    order = np.argsort(rows, kind="mergesort")
    values = dataset[rows[order]]
    inverse = np.argsort(order, kind="mergesort")
    return np.asarray([
        value.decode("utf-8", "ignore") if isinstance(value, bytes) else str(value)
        for value in values[inverse]
    ], dtype=object)


def numeric_values(dataset, rows: np.ndarray) -> np.ndarray:
    order = np.argsort(rows, kind="mergesort")
    values = np.asarray(dataset[rows[order]])
    return values[np.argsort(order, kind="mergesort")]


def stable_key(seed: int, identity: str, row: int) -> int:
    value = f"{seed}|{identity}|{row}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "little")


def _init_raw_worker(spectra: np.ndarray, precursor_mz: np.ndarray, tolerance: float) -> None:
    global _WORKER_SPECTRA, _WORKER_PRECURSOR, _WORKER_TOLERANCE
    _WORKER_SPECTRA = spectra
    _WORKER_PRECURSOR = precursor_mz
    _WORKER_TOLERANCE = float(tolerance)


def _compute_raw_pair(pair: tuple[int, int]) -> tuple[tuple[int, int], np.ndarray]:
    if _WORKER_SPECTRA is None or _WORKER_PRECURSOR is None or _WORKER_TOLERANCE is None:
        raise RuntimeError("RAW feature worker was not initialized")
    left, right = pair
    values = symmetric_features(
        _WORKER_SPECTRA[left], float(_WORKER_PRECURSOR[left]),
        _WORKER_SPECTRA[right], float(_WORKER_PRECURSOR[right]),
        _WORKER_TOLERANCE,
    )
    raw = np.asarray([values[name] for name in RAW_FEATURES], dtype=np.float32)
    if np.any(~np.isfinite(raw)):
        raise RuntimeError(f"non-finite RAW feature for local rows {left}/{right}")
    return pair, raw


def build_raw_feature_cache(
    pairs: list[tuple[int, int]],
    spectra: np.ndarray,
    precursor_mz: np.ndarray,
    tolerance: float,
    workers: int,
) -> dict[tuple[int, int], np.ndarray]:
    print(f"[raw] computing {len(pairs):,} unique spectrum pairs with {workers} worker(s)", flush=True)
    output: dict[tuple[int, int], np.ndarray] = {}
    if workers == 1:
        _init_raw_worker(spectra, precursor_mz, tolerance)
        iterator = map(_compute_raw_pair, pairs)
        executor = None
    else:
        if sys.platform == "win32":
            raise RuntimeError("--workers >1 is supported on the Linux server only")
        context = mp.get_context("fork")
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_raw_worker,
            initargs=(spectra, precursor_mz, tolerance),
        )
        iterator = executor.map(_compute_raw_pair, pairs, chunksize=64)
    try:
        for position, (key, raw) in enumerate(iterator, start=1):
            output[key] = raw
            if position % 5000 == 0 or position == len(pairs):
                print(f"[raw] {position:,}/{len(pairs):,} unique pairs", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return output


def load_grade_lookup(path: Path) -> dict[tuple[str, str], int]:
    body = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[tuple[str, str], int] = {}
    for grade, name in enumerate(("near", "mid", "far")):
        for row in body.get(name, []):
            key = tuple(sorted((str(row["ik_a"]), str(row["ik_b"]))))
            previous = mapping.get(key)
            if previous is not None and previous != grade:
                raise RuntimeError(f"conflicting MCES grade for {key}")
            mapping[key] = grade
    return mapping


def load_or_build_embeddings(a: argparse.Namespace, rows: np.ndarray, spectra: np.ndarray,
                             precursor_mz: np.ndarray) -> np.ndarray:
    metadata_path = a.embedding_cache.with_suffix(".json")
    if a.embedding_cache.is_file() and metadata_path.is_file():
        with np.load(a.embedding_cache) as cache:
            cached_rows = cache["rows"].astype(np.int64)
            embeddings = cache["embeddings"].astype(np.float32)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not np.array_equal(cached_rows, rows):
            raise RuntimeError("existing P2 embedding cache belongs to different rows")
        if metadata.get("base_checkpoint_sha256") != sha256_file(a.base_ckpt):
            raise RuntimeError("existing P2 embedding cache belongs to different official weights")
        if embeddings.shape[0] != len(rows) or np.any(~np.isfinite(embeddings)):
            raise RuntimeError("existing P2 embedding cache is corrupt")
        return embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(a.device)
    model, _ = load_base_model(a.base_ckpt, a.architecture_ckpt, device, a.n_highest_peaks)
    model.eval()
    prepared = [
        preprocess_spectrum(spectra[index], float(precursor_mz[index]), a.n_highest_peaks)
        for index in range(len(rows))
    ]
    embeddings = embed(model, prepared, device, a.batch_size).numpy().astype(np.float32)
    embeddings /= np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    a.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.embedding_cache.with_name(a.embedding_cache.stem + ".tmp.npz")
    np.savez_compressed(temporary, rows=rows, embeddings=embeddings)
    temporary.replace(a.embedding_cache)
    metadata_path.write_text(json.dumps({
        "format": "g8r_p2_official_embeddings_v1",
        "n_rows": len(rows),
        "dimension": int(embeddings.shape[1]),
        "base_checkpoint_sha256": sha256_file(a.base_ckpt),
        "architecture_checkpoint_sha256": sha256_file(a.architecture_ckpt),
    }, indent=2), encoding="utf-8")
    return embeddings


def main() -> None:
    a = parse_args()
    if (a.ppm_tol <= 0 or a.peak_tolerance <= 0 or a.max_queries <= 0
            or a.queries_per_identity <= 0 or a.workers <= 0):
        raise ValueError("invalid cache-builder parameter")
    if a.output.exists() and not a.overwrite:
        raise FileExistsError(f"refusing to overwrite {a.output}")
    print("[cache] validating P2/P3 provenance", flush=True)
    preflight = json.loads(a.preflight.read_text(encoding="utf-8"))
    if preflight.get("status") != "g8r_p2_preflight_passed":
        raise RuntimeError("P2 fail-closed preflight has not passed")
    allow = json.loads((a.p3_dir / "p3_p2_allowed_training_ik14.json").read_text(encoding="utf-8"))
    rows = np.asarray(allow["real_train_primary"]["rows"], dtype=np.int64)
    allowed_ik = {str(value) for value in allow["real_train_primary"]["ik14"]}

    print(f"[cache] loading {len(rows):,} allowed real spectra from HDF5", flush=True)
    with h5py.File(a.data, "r") as h:
        precursor_mz = numeric_values(h["precursor_mz"], rows).astype(np.float64)
        spectra = numeric_values(h["spectrum"], rows)
        inchikey = text_values(h["INCHIKEY"], rows)
        ik14 = np.asarray([value[:14] for value in inchikey], dtype=object)
        formula = text_values(h["FORMULA"], rows)
        adduct = text_values(h["adduct"], rows)
    if set(ik14) != allowed_ik:
        raise RuntimeError("P2 row/identity allow-list mismatch")
    print("[cache] loading/reusing official DreaMS embeddings", flush=True)
    embeddings = load_or_build_embeddings(a, rows, spectra, precursor_mz)
    print(f"[cache] embeddings ready: {embeddings.shape}", flush=True)
    grade_lookup = load_grade_lookup(a.pairs)

    mass_order = np.argsort(precursor_mz, kind="mergesort")
    sorted_mass = precursor_mz[mass_order]
    valid_by_ik: dict[str, list[int]] = defaultdict(list)
    candidate_by_query: dict[int, np.ndarray] = {}
    for query in range(len(rows)):
        mass = precursor_mz[query]
        tolerance = a.ppm_tol * 1e-6 * mass
        left = np.searchsorted(sorted_mass, mass - tolerance, side="left")
        right = np.searchsorted(sorted_mass, mass + tolerance, side="right")
        candidate = mass_order[left:right]
        candidate = candidate[(candidate != query) & (adduct[candidate] == adduct[query])]
        if len(candidate) == 0:
            continue
        candidate_identities = set(map(str, ik14[candidate]))
        qik = str(ik14[query])
        if qik not in candidate_identities or not (candidate_identities - {qik}):
            continue
        candidate_by_query[query] = candidate
        valid_by_ik[qik].append(query)
        if (query + 1) % 5000 == 0 or query + 1 == len(rows):
            print(
                f"[candidates] {query + 1:,}/{len(rows):,}; valid={len(candidate_by_query):,}",
                flush=True,
            )

    selected = []
    for identity, identity_rows in sorted(valid_by_ik.items()):
        identity_rows.sort(key=lambda row: stable_key(a.seed, identity, row))
        selected.extend(identity_rows[:a.queries_per_identity])
    selected.sort(key=lambda row: stable_key(a.seed + 1, str(ik14[row]), row))
    selected = selected[:a.max_queries]
    if len(selected) < 1000:
        raise RuntimeError(f"too few listwise queries after filtering: {len(selected)}")
    print(
        f"[cache] selected {len(selected):,} queries across "
        f"{len(set(str(ik14[row]) for row in selected)):,} identities",
        flush=True,
    )

    unique_pairs = sorted({
        tuple(sorted((int(query), int(candidate_row))))
        for query in selected
        for candidate_row in candidate_by_query[query]
    })
    raw_cache = build_raw_feature_cache(
        unique_pairs, spectra, precursor_mz, a.peak_tolerance, a.workers,
    )

    pair_features: list[np.ndarray] = []
    pair_candidate_row: list[int] = []
    molecule_ptr = [0]
    molecule_query: list[int] = []
    molecule_label: list[int] = []
    molecule_ik: list[str] = []
    molecule_formula: list[str] = []
    molecule_grade: list[int] = []
    query_ptr = [0]
    query_row: list[int] = []
    query_ik: list[str] = []
    query_formula: list[str] = []
    query_has_near: list[bool] = []
    for query_index, query in enumerate(selected):
        candidate = candidate_by_query[query]
        similarities = embeddings[candidate] @ embeddings[query]
        grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for score, candidate_row in zip(similarities, candidate):
            grouped[str(ik14[candidate_row])].append((float(score), int(candidate_row)))
        qik = str(ik14[query])
        ordered_identities = [qik] + sorted(
            (identity for identity in grouped if identity != qik),
            key=lambda identity: (-max(score for score, _ in grouped[identity]), identity),
        )
        has_near = False
        for identity in ordered_identities:
            spectra_for_molecule = sorted(grouped[identity], key=lambda item: (-item[0], item[1]))
            label = int(identity == qik)
            if label:
                grade = -2
            else:
                grade = grade_lookup.get(tuple(sorted((qik, identity))), -1)
                has_near |= grade == 0
            for baseline_score, candidate_row in spectra_for_molecule:
                cache_key = tuple(sorted((query, candidate_row)))
                raw = raw_cache[cache_key]
                pair_features.append(np.r_[np.float32(baseline_score), raw].astype(np.float32))
                pair_candidate_row.append(int(rows[candidate_row]))
            molecule_ptr.append(len(pair_features))
            molecule_query.append(query_index)
            molecule_label.append(label)
            molecule_ik.append(identity)
            molecule_formula.append(str(formula[spectra_for_molecule[0][1]]))
            molecule_grade.append(int(grade))
        query_ptr.append(len(molecule_label))
        query_row.append(int(rows[query]))
        query_ik.append(qik)
        query_formula.append(str(formula[query]))
        query_has_near.append(bool(has_near))
        if (query_index + 1) % 250 == 0 or query_index + 1 == len(selected):
            print(f"[features] {query_index + 1:,}/{len(selected):,} queries; "
                  f"{len(molecule_label):,} molecules; {len(pair_features):,} pairs", flush=True)

    features_array = np.asarray(pair_features, dtype=np.float32)
    if features_array.ndim != 2 or features_array.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("P2 feature matrix has the wrong shape")
    query_ptr_array = np.asarray(query_ptr, dtype=np.int64)
    molecule_ptr_array = np.asarray(molecule_ptr, dtype=np.int64)
    if query_ptr_array[-1] != len(molecule_label) or molecule_ptr_array[-1] != len(pair_features):
        raise RuntimeError("P2 pointer arrays do not span their values")
    for left, right in zip(query_ptr_array[:-1], query_ptr_array[1:]):
        labels = np.asarray(molecule_label[left:right], dtype=np.int8)
        if labels.sum() != 1 or labels[0] != 1 or len(labels) < 2:
            raise RuntimeError("every P2 query must contain exactly one first-position positive")

    arrays = {
        "feature_names": np.asarray(FEATURE_NAMES, dtype=object),
        "features": features_array,
        "pair_candidate_row": np.asarray(pair_candidate_row, dtype=np.int64),
        "query_ptr": query_ptr_array,
        "molecule_ptr": molecule_ptr_array,
        "molecule_query": np.asarray(molecule_query, dtype=np.int64),
        "molecule_label": np.asarray(molecule_label, dtype=np.int8),
        "molecule_ik14": np.asarray(molecule_ik, dtype=object),
        "molecule_formula": np.asarray(molecule_formula, dtype=object),
        "molecule_mces_grade": np.asarray(molecule_grade, dtype=np.int8),
        "query_row": np.asarray(query_row, dtype=np.int64),
        "query_ik14": np.asarray(query_ik, dtype=object),
        "query_formula": np.asarray(query_formula, dtype=object),
        "query_has_near": np.asarray(query_has_near, dtype=bool),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_name(a.output.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(a.output)
    audit = {
        "status": "g8r_p2_listwise_cache_built",
        "n_queries": len(query_row),
        "n_query_identities": len(set(query_ik)),
        "n_query_formulas": len(set(query_formula)),
        "n_near_queries": int(sum(query_has_near)),
        "n_candidate_molecules": len(molecule_label),
        "n_spectrum_pairs": len(pair_features),
        "mean_molecules_per_query": float(len(molecule_label) / len(query_row)),
        "feature_names": FEATURE_NAMES,
        "candidate_protocol": "strict-10ppm same-adduct, self-row excluded, all P2-real spectra, max per molecule",
        "query_protocol": f"at most {a.queries_per_identity} deterministic real spectra per IK14",
        "p3_query_overlap": 0,
        "preflight_sha256": sha256_file(a.preflight),
        "base_checkpoint_sha256": sha256_file(a.base_ckpt),
        "cache_sha256": sha256_file(a.output),
        "parameters": vars(a) | {"data": str(a.data), "pairs": str(a.pairs), "p3_dir": str(a.p3_dir),
                                  "preflight": str(a.preflight), "base_ckpt": str(a.base_ckpt),
                                  "architecture_ckpt": str(a.architecture_ckpt),
                                  "embedding_cache": str(a.embedding_cache), "output": str(a.output)},
    }
    a.output.with_suffix(".json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
