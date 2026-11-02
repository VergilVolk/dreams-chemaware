#!/usr/bin/env python
"""Adapt public SpecEmbedding/MSBERT checkpoints to the common identity axis.

This module deliberately separates expensive spectrum encoding from cheap
candidate-graph scoring:

``inventory``
    Inspect the two official repositories and their checkpoints.  Missing
    checkpoints and Git-LFS pointer stubs are reported explicitly.

``encode``
    Read spectra in MGF record order and write the repository-standard
    ``rows``/``embeddings`` NPZ cache.  Embeddings are L2 normalised because
    both publications use cosine retrieval.

``score``
    Score a standard panel NPZ and max-pool reference spectra within each
    candidate molecule.  The output follows the mega evaluator's preferred
    ``method_names``/``molecule_scores`` schema.

``merge``
    Concatenate independently generated score caches after verifying a common
    candidate-molecule axis length and unique method names.

No checkpoint is downloaded, substituted, or randomly initialised.  A model
whose official weights are absent fails before the MGF is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = ROOT / "data/external/sota_identity_methods_20261102"
METHODS = ("specembedding", "msbert")
SPECEMBEDDING_SHA256 = "0ca0aa002a0d061a95410f7a4055e82c7fcb428d0ba04b5714ac3a4e7f0f5cca"
SPECEMBEDDING_BYTES = 31_572_706
SPECEMBEDDING_SPACE_COMMIT = "26665447238100728da3640675927f2c8bfb10cd"
MSBERT_SHA256 = "f6a50e1a5650504370e563a9daf9117b4f4ecd60f5ba508dce2ee9e81f064605"
MSBERT_BYTES = 486_361_781


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(directory: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _lfs_pointer(path: Path) -> dict[str, object] | None:
    if not path.is_file() or path.stat().st_size > 4096:
        return None
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.startswith("version https://git-lfs.github.com/spec/v1\n"):
        return None
    body: dict[str, object] = {"pointer": True}
    for line in text.splitlines()[1:]:
        key, _, value = line.partition(" ")
        if key == "oid" and value.startswith("sha256:"):
            body["sha256"] = value.removeprefix("sha256:")
        elif key == "size":
            body["bytes"] = int(value)
    return body


def default_repo(method: str) -> Path:
    return METHOD_ROOT / ("SpecEmbedding" if method == "specembedding" else "MSBERT")


def default_checkpoint(method: str, repo: Path) -> Path:
    if method == "specembedding":
        return repo / "model/transformer-SupConWithTanimotoLoss-Augmentation/model.ckpt"
    return repo / "model/MSBERT.pkl"


def inspect_asset(method: str, repo: Path, checkpoint: Path) -> dict[str, object]:
    repo = repo.resolve()
    checkpoint = checkpoint.resolve()
    item: dict[str, object] = {
        "method": method,
        "repository": str(repo),
        "repository_present": repo.is_dir(),
        "repository_revision": _git_revision(repo),
        "checkpoint": str(checkpoint),
        "checkpoint_present": checkpoint.is_file(),
    }
    if method == "specembedding":
        item["official_distribution"] = {
            "article": "https://doi.org/10.1002/cem.70065",
            "figshare_article": "https://doi.org/10.6084/m9.figshare.28876751.v2",
            "figshare_file": "SpecEmbedding.zip",
            "figshare_file_id": 53998028,
            "figshare_archive_bytes": 549_196_238,
            "huggingface_space": "xp113280/SpecEmbedding",
            "huggingface_space_commit": SPECEMBEDDING_SPACE_COMMIT,
            "huggingface_path": "src/model.ckpt",
            "checkpoint_sha256": SPECEMBEDDING_SHA256,
            "checkpoint_bytes": SPECEMBEDDING_BYTES,
            "note": "GitHub has no weights; the authors' public Space supplies the exact demo checkpoint",
        }
    else:
        item["official_distribution"] = {
            "repository": "https://github.com/zhanghailiangcsu/MSBERT",
            "checkpoint_sha256": MSBERT_SHA256,
            "checkpoint_bytes": MSBERT_BYTES,
        }
    pointer = _lfs_pointer(checkpoint)
    if pointer is not None:
        item["status"] = "git_lfs_pointer_not_materialized"
        item["git_lfs_pointer"] = pointer
        return item
    if not checkpoint.is_file():
        item["status"] = "checkpoint_missing"
        return item
    item["checkpoint_bytes"] = checkpoint.stat().st_size
    item["checkpoint_sha256"] = sha256_file(checkpoint)
    pinned_bytes = SPECEMBEDDING_BYTES if method == "specembedding" else MSBERT_BYTES
    pinned_sha256 = SPECEMBEDDING_SHA256 if method == "specembedding" else MSBERT_SHA256
    if (
        item["checkpoint_bytes"] != pinned_bytes
        or item["checkpoint_sha256"] != pinned_sha256
    ):
        item["status"] = "checkpoint_identity_mismatch"
    else:
        item["status"] = "ready"
    return item


def require_asset(
    method: str, repo: Path, checkpoint: Path, expected_sha256: str | None = None,
) -> dict[str, object]:
    item = inspect_asset(method, repo, checkpoint)
    if item["status"] != "ready":
        raise RuntimeError(
            f"PUBLIC_SOTA_CHECKPOINT_NOT_READY: method={method} "
            f"status={item['status']} checkpoint={checkpoint}"
        )
    pinned = (
        MSBERT_SHA256 if method == "msbert"
        else (expected_sha256 or SPECEMBEDDING_SHA256)
    )
    if item["checkpoint_sha256"] != pinned:
        raise RuntimeError(
            f"PUBLIC_SOTA_CHECKPOINT_SHA256_MISMATCH: method={method} "
            f"expected={pinned} observed={item['checkpoint_sha256']}"
        )
    item["expected_checkpoint_sha256"] = pinned
    return item


def _manifest_rows(path: Path | None, row_column: str) -> np.ndarray | None:
    if path is None:
        return None
    import pandas as pd

    frame = pd.read_csv(path, usecols=[row_column])
    rows = frame[row_column].to_numpy(np.int64)
    if rows.ndim != 1 or not len(rows) or np.any(rows < 0) or len(np.unique(rows)) != len(rows):
        raise RuntimeError("manifest row registry is empty, negative, or duplicated")
    if not np.array_equal(rows, np.sort(rows)):
        raise RuntimeError("manifest rows must be in increasing MGF record order")
    return rows


def _panel_rows(paths: Sequence[Path]) -> np.ndarray:
    selected: list[np.ndarray] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as body:
            missing = {"query_row", "candidate_row"} - set(body.files)
            if missing:
                raise RuntimeError(f"row-selection panel misses {sorted(missing)}: {path}")
            query = np.asarray(body["query_row"], np.int64)
            candidate = np.asarray(body["candidate_row"], np.int64)
        if query.ndim != 1 or candidate.ndim != 1 or not len(query) or not len(candidate):
            raise RuntimeError(f"row-selection panel has malformed row arrays: {path}")
        if np.any(query < 0) or np.any(candidate < 0):
            raise RuntimeError(f"row-selection panel contains negative rows: {path}")
        selected.extend((query, candidate))
    if not selected:
        raise RuntimeError("at least one row-selection panel is required")
    return np.unique(np.concatenate(selected)).astype(np.int64, copy=False)


def _progress_records(
    records: Iterable[tuple[int, float, np.ndarray]], every: int, method: str,
) -> Iterable[tuple[int, float, np.ndarray]]:
    started = time.monotonic()
    count = 0
    for record in records:
        count += 1
        yield record
        if every and count % every == 0:
            elapsed = max(time.monotonic() - started, 1e-9)
            print(json.dumps({
                "status": "PUBLIC_SOTA_ENCODING_PROGRESS",
                "method": method,
                "selected_records": count,
                "elapsed_seconds": round(elapsed, 3),
                "selected_records_per_second": round(count / elapsed, 3),
            }), flush=True)


def _selected_records(mgf: Path, rows: np.ndarray | None) -> Iterable[tuple[int, float, np.ndarray]]:
    # Reuse the project's canonical MGF order/parser contract.
    from encode_unified_library_for_p2b import iter_mgf

    wanted = None if rows is None else set(map(int, rows))
    observed: list[int] = []
    for row, (_fields, precursor, peaks) in enumerate(iter_mgf(mgf)):
        if wanted is None or row in wanted:
            observed.append(row)
            yield row, float(precursor), np.asarray(peaks, dtype=np.float32)
    if rows is not None and not np.array_equal(np.asarray(observed, np.int64), rows):
        missing = sorted(set(map(int, rows)) - set(observed))
        raise RuntimeError(f"MGF misses {len(missing)} manifest rows; first={missing[:5]}")


def _valid_peaks(peaks: np.ndarray) -> np.ndarray:
    if peaks.ndim != 2 or peaks.shape[0] != 2:
        raise RuntimeError("canonical MGF parser returned a malformed peak array")
    keep = (
        np.isfinite(peaks).all(axis=0)
        & (peaks[0] > 0)
        & (peaks[1] > 0)
    )
    output = peaks[:, keep].T
    if not len(output):
        raise RuntimeError("spectrum has no finite positive peaks")
    return output


def _top_peaks(peaks: np.ndarray, count: int) -> np.ndarray:
    if len(peaks) <= count:
        return peaks
    # Match both official repositories: select by intensity, then restore the
    # original peak order rather than sorting by intensity.
    indices = np.sort(np.argsort(peaks[:, 1])[::-1][:count])
    return peaks[indices]


def _load_state(path: Path, device: str) -> dict[str, object]:
    import torch

    try:
        body = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        body = torch.load(path, map_location=device)
    if isinstance(body, dict) and "state_dict" in body and isinstance(body["state_dict"], dict):
        body = body["state_dict"]
    if not isinstance(body, dict):
        raise RuntimeError("checkpoint is not a PyTorch state dictionary")
    return body


def _specembedding_batches(
    records: Iterable[tuple[int, float, np.ndarray]], batch_size: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    rows: list[int] = []
    mzs: list[np.ndarray] = []
    intensities: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for row, precursor, raw in records:
        peaks = _top_peaks(_valid_peaks(raw), 99)
        intensity = peaks[:, 1].astype(np.float32)
        intensity /= float(np.max(intensity))
        mz = np.concatenate(([precursor], peaks[:, 0])).astype(np.float32)
        intensity = np.concatenate(([2.0], intensity)).astype(np.float32)
        mask = np.zeros(100, dtype=bool)
        mask[len(mz):] = True
        mz = np.pad(mz, (0, 100 - len(mz)))
        intensity = np.pad(intensity, (0, 100 - len(intensity)))
        rows.append(row)
        mzs.append(mz)
        intensities.append(intensity)
        masks.append(mask)
        if len(rows) == batch_size:
            yield np.asarray(rows, np.int64), np.stack(mzs), np.stack(intensities), np.stack(masks)
            rows, mzs, intensities, masks = [], [], [], []
    if rows:
        yield np.asarray(rows, np.int64), np.stack(mzs), np.stack(intensities), np.stack(masks)


def _msbert_token(mz: float) -> int:
    # Official vocabulary is 0.00, 0.01, ..., 999.99 plus PAD/MASK.
    rounded = float(f"{float(mz):.2f}")
    index = int(round(rounded * 100))
    if index < 0 or index >= 100_000:
        raise RuntimeError(f"MSBERT m/z outside official [0,1000) vocabulary: {mz}")
    return index + 2


def _msbert_batches(
    records: Iterable[tuple[int, float, np.ndarray]], batch_size: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows: list[int] = []
    tokens: list[np.ndarray] = []
    intensities: list[np.ndarray] = []
    for row, precursor, raw in records:
        # The official GNPS preparation calls PrecursorFilter before tokenising:
        # spectra whose precursor is >=1000 Da are outside the checkpoint's
        # 0.00--999.99 vocabulary and are omitted, not clipped or remapped.
        if not np.isfinite(precursor) or precursor < 0 or precursor >= 1000:
            continue
        # Official MinimalProcessing applies select_by_mz(10, 1000) and
        # normalize_intensities before ProDataset selects the top 99 peaks.
        peaks = _valid_peaks(raw)
        peaks = peaks[(peaks[:, 0] >= 10.0) & (peaks[:, 0] < 1000.0)]
        if not len(peaks):
            continue
        peaks = _top_peaks(peaks, 99)
        peak_intensity = peaks[:, 1].astype(np.float32)
        peak_intensity /= float(np.max(peak_intensity))
        ids = np.asarray([_msbert_token(precursor), *(_msbert_token(x) for x in peaks[:, 0])], np.int64)
        weight = np.concatenate(([2.0], peak_intensity)).astype(np.float32)
        weight /= float(np.max(weight))
        ids = np.pad(ids, (0, 100 - len(ids)))
        weight = np.pad(weight, (0, 100 - len(weight)))[None, :]
        rows.append(row)
        tokens.append(ids)
        intensities.append(weight)
        if len(rows) == batch_size:
            yield np.asarray(rows, np.int64), np.stack(tokens), np.stack(intensities)
            rows, tokens, intensities = [], [], []
    if rows:
        yield np.asarray(rows, np.int64), np.stack(tokens), np.stack(intensities)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise RuntimeError("public-model embedding is malformed or non-finite")
    norm = np.linalg.norm(values, axis=1)
    if np.any(norm < 1e-12):
        raise RuntimeError("public-model embedding is malformed, non-finite, or zero")
    return values / norm[:, None]


def encode_specembedding(
    repo: Path, checkpoint: Path, records: Iterable[tuple[int, float, np.ndarray]],
    device: str, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    sys.path.insert(0, str(repo))
    try:
        from SpecEmbedding.models import SiameseModel
    finally:
        sys.path.pop(0)
    model = SiameseModel(
        embedding_dim=512, n_head=16, n_layer=4, dim_feedward=512,
        dim_target=512, feedward_activation="selu",
    )
    model.load_state_dict(_load_state(checkpoint, device), strict=True)
    model = model.to(device).eval()
    all_rows: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for rows, mz, intensity, mask in _specembedding_batches(records, batch_size):
            result = model(
                torch.as_tensor(mz, device=device),
                torch.as_tensor(intensity, device=device),
                torch.as_tensor(mask, device=device),
            ).float().cpu().numpy()
            all_rows.append(rows)
            all_embeddings.append(result)
    return np.concatenate(all_rows), _unit_rows(np.concatenate(all_embeddings))


def encode_msbert(
    repo: Path, checkpoint: Path, records: Iterable[tuple[int, float, np.ndarray]],
    device: str, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    sys.path.insert(0, str(repo))
    try:
        from model.MSBERTModel import MSBERT
    finally:
        sys.path.pop(0)
    model = MSBERT(100002, 512, 6, 16, 0, 100, 3)
    model.load_state_dict(_load_state(checkpoint, device), strict=True)
    model = model.to(device).eval()
    all_rows: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for rows, ids, intensity in _msbert_batches(records, batch_size):
            result = model.predict(
                torch.as_tensor(ids, device=device),
                torch.as_tensor(intensity, device=device),
            ).float().cpu().numpy()
            all_rows.append(rows)
            all_embeddings.append(result.reshape(len(rows), -1))
    return np.concatenate(all_rows), _unit_rows(np.concatenate(all_embeddings))


def load_embedding_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        if set(body.files) != {"rows", "embeddings"}:
            raise RuntimeError(f"embedding cache has unexpected members: {body.files}")
        rows = np.asarray(body["rows"], np.int64)
        embeddings = _unit_rows(np.asarray(body["embeddings"], np.float32))
    if len(rows) != len(embeddings) or len(np.unique(rows)) != len(rows) or np.any(rows < 0):
        raise RuntimeError("embedding row registry is malformed")
    return rows, embeddings


def embedding_cache_rows(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as body:
        if "rows" not in body.files:
            raise RuntimeError(f"embedding cache lacks rows: {path}")
        rows = np.asarray(body["rows"], np.int64)
    if (
        rows.ndim != 1 or not len(rows) or np.any(rows < 0)
        or len(np.unique(rows)) != len(rows)
        or not np.array_equal(rows, np.sort(rows))
    ):
        raise RuntimeError("embedding cache row registry is malformed or unsorted")
    return rows


def subset_panel_to_supported_rows(
    panel: dict[str, np.ndarray], supported_rows: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    required = {"query_row", "query_ptr", "molecule_ptr", "candidate_row"}
    missing = sorted(required - set(panel))
    if missing:
        raise RuntimeError(f"panel misses required arrays: {missing}")
    query_row = np.asarray(panel["query_row"], np.int64)
    query_ptr = np.asarray(panel["query_ptr"], np.int64)
    molecule_ptr = np.asarray(panel["molecule_ptr"], np.int64)
    candidate_row = np.asarray(panel["candidate_row"], np.int64)
    n_q = len(query_row)
    n_e = len(molecule_ptr) - 1
    n_p = len(candidate_row)
    if (
        query_row.ndim != 1 or query_ptr.ndim != 1 or molecule_ptr.ndim != 1
        or candidate_row.ndim != 1 or len(query_ptr) != n_q + 1
        or query_ptr[0] != 0 or query_ptr[-1] != n_e
        or molecule_ptr[0] != 0 or molecule_ptr[-1] != n_p
        or np.any(np.diff(query_ptr) < 1) or np.any(np.diff(molecule_ptr) < 1)
    ):
        raise RuntimeError("panel pointer contract is malformed or contains empty candidates")
    supported_rows = np.asarray(supported_rows, np.int64)
    if (
        supported_rows.ndim != 1 or not len(supported_rows) or np.any(supported_rows < 0)
        or len(np.unique(supported_rows)) != len(supported_rows)
    ):
        raise RuntimeError("supported row registry is malformed")
    query_supported = np.isin(query_row, supported_rows)
    pair_supported = np.isin(candidate_row, supported_rows)
    keep = np.zeros(n_q, dtype=bool)
    for query in range(n_q):
        molecule_start = int(query_ptr[query])
        molecule_stop = int(query_ptr[query + 1])
        pair_start = int(molecule_ptr[molecule_start])
        pair_stop = int(molecule_ptr[molecule_stop])
        keep[query] = bool(query_supported[query] and np.all(pair_supported[pair_start:pair_stop]))
    query_indices = np.flatnonzero(keep).astype(np.int64)
    if not len(query_indices):
        raise RuntimeError("embedding cache supports no complete query block")
    molecule_indices = np.concatenate([
        np.arange(query_ptr[q], query_ptr[q + 1], dtype=np.int64) for q in query_indices
    ])
    pair_indices = np.concatenate([
        np.arange(molecule_ptr[m], molecule_ptr[m + 1], dtype=np.int64)
        for m in molecule_indices
    ])
    molecule_counts = np.asarray(
        [query_ptr[q + 1] - query_ptr[q] for q in query_indices], np.int64,
    )
    pair_counts = np.asarray(
        [molecule_ptr[m + 1] - molecule_ptr[m] for m in molecule_indices], np.int64,
    )
    subset: dict[str, np.ndarray] = {}
    for key, value in panel.items():
        array = np.asarray(value)
        if key == "query_ptr":
            subset[key] = np.concatenate((np.asarray([0], np.int64), np.cumsum(molecule_counts)))
        elif key == "molecule_ptr":
            subset[key] = np.concatenate((np.asarray([0], np.int64), np.cumsum(pair_counts)))
        elif array.ndim == 0:
            subset[key] = array
        elif array.shape[0] == n_q and key not in {"molecule_ptr", "candidate_row"}:
            subset[key] = array[query_indices]
        elif array.shape[0] == n_e and key != "candidate_row":
            subset[key] = array[molecule_indices]
        elif array.shape[0] == n_p:
            subset[key] = array[pair_indices]
        else:
            raise RuntimeError(
                f"cannot assign panel member to query/molecule/pair axis: {key} shape={array.shape}"
            )
    for key in ("source_query_index", "source_molecule_index", "source_pair_index"):
        if key in subset:
            raise RuntimeError(f"source panel already reserves common-domain index member: {key}")
    subset["source_query_index"] = query_indices
    subset["source_molecule_index"] = molecule_indices
    subset["source_pair_index"] = pair_indices
    counts = {
        "original_N_Q": int(n_q),
        "original_N_E": int(n_e),
        "original_N_P": int(n_p),
        "retained_N_Q": int(len(query_indices)),
        "retained_N_E": int(len(molecule_indices)),
        "retained_N_P": int(len(pair_indices)),
        "dropped_N_Q": int(n_q - len(query_indices)),
        "dropped_query_row_unavailable": int(np.sum(~query_supported)),
        "dropped_incomplete_candidate_block": int(np.sum(query_supported & ~keep)),
    }
    return subset, counts


def score_molecules(panel: dict[str, np.ndarray], rows: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    required = {"query_row", "query_ptr", "molecule_ptr", "candidate_row"}
    missing = sorted(required - set(panel))
    if missing:
        raise RuntimeError(f"panel misses required arrays: {missing}")
    query_row = np.asarray(panel["query_row"], np.int64)
    query_ptr = np.asarray(panel["query_ptr"], np.int64)
    molecule_ptr = np.asarray(panel["molecule_ptr"], np.int64)
    candidate_row = np.asarray(panel["candidate_row"], np.int64)
    if (
        len(query_ptr) != len(query_row) + 1 or query_ptr[0] != 0
        or query_ptr[-1] != len(molecule_ptr) - 1 or molecule_ptr[0] != 0
        or molecule_ptr[-1] != len(candidate_row)
        or np.any(np.diff(query_ptr) < 1) or np.any(np.diff(molecule_ptr) < 1)
    ):
        raise RuntimeError("panel pointer contract is malformed or contains empty candidates")
    positions = {int(row): index for index, row in enumerate(rows)}
    used = np.unique(np.concatenate((query_row, candidate_row)))
    absent = [int(row) for row in used if int(row) not in positions]
    if absent:
        raise RuntimeError(f"embedding cache misses {len(absent)} panel rows; first={absent[:5]}")
    molecule_query = np.repeat(np.arange(len(query_row)), np.diff(query_ptr))
    pair_query = np.repeat(molecule_query, np.diff(molecule_ptr))
    query_position = np.asarray([positions[int(row)] for row in query_row[pair_query]], np.int64)
    candidate_position = np.asarray([positions[int(row)] for row in candidate_row], np.int64)
    pair = np.einsum(
        "ij,ij->i", embeddings[query_position], embeddings[candidate_position], optimize=True,
    ).astype(np.float32)
    molecule = np.maximum.reduceat(pair, molecule_ptr[:-1]).astype(np.float32)
    if not np.all(np.isfinite(molecule)):
        raise RuntimeError("molecule aggregation produced non-finite values")
    return molecule


def _write_json(path: Path, body: dict[str, object]) -> None:
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")


def inventory_command(args: argparse.Namespace) -> None:
    items = []
    for method in METHODS:
        repo = getattr(args, f"{method}_repo") or default_repo(method)
        checkpoint = getattr(args, f"{method}_checkpoint") or default_checkpoint(method, repo)
        items.append(inspect_asset(method, repo, checkpoint))
    report = {"schema": "public_sota_embedding_asset_inventory_v1", "methods": items}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


def encode_command(args: argparse.Namespace) -> None:
    if args.benchmark is not None:
        mgf = args.benchmark / "spectra.mgf"
        manifest = args.benchmark / "manifest.csv.gz"
    else:
        mgf = args.mgf
        manifest = args.manifest
    if mgf is None or not mgf.is_file():
        raise FileNotFoundError(mgf)
    if manifest is not None and not manifest.is_file():
        raise FileNotFoundError(manifest)
    repo = (args.repo or default_repo(args.method)).resolve()
    checkpoint = (args.checkpoint or default_checkpoint(args.method, repo)).resolve()
    asset = require_asset(
        args.method, repo, checkpoint, args.expected_checkpoint_sha256,
    )
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite an embedding cache or report")
    manifest_rows = _manifest_rows(manifest, args.row_column)
    row_panels = list(args.row_panel or [])
    if row_panels:
        rows = _panel_rows(row_panels)
        if manifest_rows is not None:
            absent = np.setdiff1d(rows, manifest_rows, assume_unique=True)
            if len(absent):
                raise RuntimeError(
                    f"row-selection panels contain {len(absent)} rows absent from manifest; "
                    f"first={absent[:5].tolist()}"
                )
    else:
        rows = manifest_rows
    records = _progress_records(
        _selected_records(mgf, rows), args.progress_every, args.method,
    )
    if args.method == "specembedding":
        encoded_rows, embeddings = encode_specembedding(
            repo, checkpoint, records, args.device, args.batch_size,
        )
        preprocessing = "official top-99 peaks plus precursor token; peak max normalisation"
    else:
        encoded_rows, embeddings = encode_msbert(
            repo, checkpoint, records, args.device, args.batch_size,
        )
        preprocessing = (
            "official precursor<1000 domain; 10<=fragment_mz<1000; peak max normalisation; "
            "top-99 peaks plus precursor token; precursor/peak weights jointly normalised"
        )
    if not len(encoded_rows):
        raise RuntimeError("no MGF records were encoded")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, rows=encoded_rows, embeddings=embeddings)
    report = {
        "schema": "public_sota_spectrum_embedding_cache_v1",
        "status": "complete",
        "method": args.method,
        "asset": asset,
        "mgf": str(mgf.resolve()),
        "mgf_sha256": sha256_file(mgf),
        "manifest": None if manifest is None else str(manifest.resolve()),
        "row_selection": "all_manifest_rows" if not row_panels else "union_of_panel_query_and_candidate_rows",
        "row_selection_panels": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in row_panels
        ],
        "rows": int(len(encoded_rows)),
        "requested_rows": None if rows is None else int(len(rows)),
        "official_domain_excluded_rows": (
            None if rows is None else np.setdiff1d(rows, encoded_rows, assume_unique=True).tolist()
        ),
        "dimension": int(embeddings.shape[1]),
        "maximum_norm_error": float(np.max(np.abs(np.linalg.norm(embeddings, axis=1) - 1))),
        "preprocessing": preprocessing,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


def score_command(args: argparse.Namespace) -> None:
    panel_path = (
        args.benchmark / f"panel_{args.panel_name}.npz"
        if args.benchmark is not None else args.panel
    )
    if panel_path is None or not panel_path.is_file():
        raise FileNotFoundError(panel_path)
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite a score cache or report")
    with np.load(panel_path, allow_pickle=False) as body:
        panel = {key: np.asarray(body[key]) for key in body.files}
    rows, embeddings = load_embedding_cache(args.embeddings)
    molecule = score_molecules(panel, rows, embeddings)
    panel_sha256 = sha256_file(panel_path)
    n_q = int(len(panel["query_row"]))
    n_e = int(len(molecule))
    n_p = int(len(panel["candidate_row"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray("mega_identity_method_scores_v1"),
        panel_sha256=np.asarray(panel_sha256),
        N_Q=np.asarray(n_q, dtype=np.int64),
        N_E=np.asarray(n_e, dtype=np.int64),
        N_P=np.asarray(n_p, dtype=np.int64),
        method_names=np.asarray([args.method_name]),
        molecule_scores=molecule[None, :],
    )
    report = {
        "schema": "public_sota_molecule_score_cache_v1",
        "status": "complete",
        "method": args.method_name,
        "panel": str(panel_path.resolve()),
        "panel_sha256": panel_sha256,
        "embeddings": str(args.embeddings.resolve()),
        "embeddings_sha256": sha256_file(args.embeddings),
        "N_Q": n_q,
        "N_E": n_e,
        "N_P": n_p,
        "queries": n_q,
        "candidate_molecules": n_e,
        "reference_spectrum_edges": n_p,
        "aggregation": "maximum cosine over reference spectra within candidate molecule",
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


def subset_panel_command(args: argparse.Namespace) -> None:
    if not args.panel.is_file():
        raise FileNotFoundError(args.panel)
    if not args.embeddings.is_file():
        raise FileNotFoundError(args.embeddings)
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite a common-domain panel or report")
    with np.load(args.panel, allow_pickle=False) as body:
        panel = {key: np.asarray(body[key]) for key in body.files}
    supported_rows = embedding_cache_rows(args.embeddings)
    subset, counts = subset_panel_to_supported_rows(panel, supported_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **subset)
    report = {
        "schema": "public_sota_common_domain_panel_v1",
        "status": "complete",
        "selection_rule": (
            "retain a query iff query_row and every candidate_row in its complete query block "
            "occur in the embedding cache"
        ),
        "source_panel": str(args.panel.resolve()),
        "source_panel_sha256": sha256_file(args.panel),
        "embedding_cache": str(args.embeddings.resolve()),
        "embedding_cache_sha256": sha256_file(args.embeddings),
        "supported_spectrum_rows": int(len(supported_rows)),
        **counts,
        "retained_query_fraction": counts["retained_N_Q"] / counts["original_N_Q"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


def _score_cache_provenance(body: np.lib.npyio.NpzFile, path: Path) -> dict[str, object]:
    for key in ("N_Q", "N_E", "N_P"):
        if key not in body.files:
            raise RuntimeError(f"score cache lacks required provenance {key}: {path}")
    has_panel = "panel_sha256" in body.files
    has_axis = "axis_digest" in body.files
    if not has_panel and not has_axis:
        raise RuntimeError(f"score cache lacks panel_sha256/axis_digest: {path}")
    provenance: dict[str, object] = {
        key: int(np.asarray(body[key]).item()) for key in ("N_Q", "N_E", "N_P")
    }
    for key in ("panel_sha256", "axis_digest"):
        if key in body.files:
            value = str(np.asarray(body[key]).item()).strip().lower()
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise RuntimeError(f"score cache has malformed {key}: {path}")
            provenance[key] = value
    if min(provenance[key] for key in ("N_Q", "N_E", "N_P")) < 1:
        raise RuntimeError(f"score cache has non-positive benchmark counts: {path}")
    return provenance


def merge_command(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    names: list[str] = []
    matrices: list[np.ndarray] = []
    width: int | None = None
    inputs: list[dict[str, object]] = []
    shared_provenance: dict[str, object] | None = None
    for path in args.score_cache:
        with np.load(path, allow_pickle=False) as body:
            required = {"method_names", "molecule_scores"}
            if not required.issubset(body.files):
                raise RuntimeError(f"score cache misses required members: {path}: {body.files}")
            provenance = _score_cache_provenance(body, path)
            local_names = [str(value) for value in body["method_names"]]
            local = np.asarray(body["molecule_scores"], np.float32)
        if local.ndim != 2 or local.shape[0] != len(local_names) or not np.all(np.isfinite(local)):
            raise RuntimeError(f"score cache is malformed: {path}")
        if local.shape[1] != int(provenance["N_E"]):
            raise RuntimeError(f"score cache width does not equal N_E: {path}")
        if shared_provenance is None:
            shared_provenance = provenance
        elif provenance != shared_provenance:
            raise RuntimeError(
                f"score cache benchmark provenance/count mismatch: {path}; "
                f"expected={shared_provenance} observed={provenance}"
            )
        if width is None:
            width = int(local.shape[1])
        elif local.shape[1] != width:
            raise RuntimeError(f"candidate-molecule axis length differs: {path}")
        names.extend(local_names)
        matrices.append(local)
        inputs.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    if not matrices or shared_provenance is None or len(set(names)) != len(names):
        raise RuntimeError("no score cache supplied or method names are duplicated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        key: np.asarray(value, dtype=np.int64) if key in {"N_Q", "N_E", "N_P"}
        else np.asarray(value)
        for key, value in shared_provenance.items()
    }
    np.savez_compressed(
        args.output,
        schema_version=np.asarray("mega_identity_method_scores_v1"),
        **metadata,
        method_names=np.asarray(names),
        molecule_scores=np.concatenate(matrices, axis=0),
    )
    print(json.dumps({
        "schema": "public_sota_merged_molecule_scores_v1",
        "status": "complete",
        "methods": names,
        "candidate_molecules": width,
        "benchmark_provenance": shared_provenance,
        "inputs": inputs,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(description=__doc__)
    sub = main.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory", help="Inspect official repositories/checkpoints.")
    inventory.add_argument("--specembedding-repo", type=Path)
    inventory.add_argument("--specembedding-checkpoint", type=Path)
    inventory.add_argument("--msbert-repo", type=Path)
    inventory.add_argument("--msbert-checkpoint", type=Path)
    inventory.add_argument("--output", type=Path)
    inventory.set_defaults(function=inventory_command)

    encode = sub.add_parser("encode", help="Encode MGF spectra with one official checkpoint.")
    encode.add_argument("--method", choices=METHODS, required=True)
    encode.add_argument("--repo", type=Path)
    encode.add_argument("--checkpoint", type=Path)
    encode.add_argument("--expected-checkpoint-sha256")
    encode_source = encode.add_mutually_exclusive_group(required=True)
    encode_source.add_argument("--mgf", type=Path)
    encode_source.add_argument(
        "--benchmark", type=Path,
        help="Directory containing spectra.mgf and manifest.csv.gz.",
    )
    encode.add_argument("--manifest", type=Path)
    encode.add_argument("--row-column", default="row")
    encode.add_argument(
        "--row-panel", type=Path, action="append",
        help=(
            "Encode only the sorted union of query_row/candidate_row from this panel NPZ. "
            "Repeat for multiple panels; scores on those panels are exactly unchanged."
        ),
    )
    encode.add_argument(
        "--progress-every", type=int, default=10_000,
        help="Emit one JSON progress line per N selected spectra; 0 disables progress.",
    )
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--batch-size", type=int, default=128)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--report", type=Path, required=True)
    encode.set_defaults(function=encode_command)

    score = sub.add_parser("score", help="Aggregate embedding cosine onto a standard panel.")
    score_source = score.add_mutually_exclusive_group(required=True)
    score_source.add_argument("--panel", type=Path)
    score_source.add_argument("--benchmark", type=Path)
    score.add_argument(
        "--panel-name", default="identity_disjoint",
        choices=("identity_disjoint", "formula_disjoint"),
    )
    score.add_argument("--embeddings", type=Path, required=True)
    score.add_argument("--method-name", required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--report", type=Path, required=True)
    score.set_defaults(function=score_command)

    subset_panel = sub.add_parser(
        "subset-panel",
        help="Build a complete-query panel on the row domain supported by an embedding cache.",
    )
    subset_panel.add_argument("--panel", type=Path, required=True)
    subset_panel.add_argument("--embeddings", type=Path, required=True)
    subset_panel.add_argument("--output", type=Path, required=True)
    subset_panel.add_argument("--report", type=Path, required=True)
    subset_panel.set_defaults(function=subset_panel_command)

    merge = sub.add_parser("merge", help="Merge standard score caches for the mega evaluator.")
    merge.add_argument("--score-cache", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(function=merge_command)
    return main


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if getattr(args, "batch_size", 1) < 1:
        raise ValueError("batch-size must be positive")
    if getattr(args, "progress_every", 0) < 0:
        raise ValueError("progress-every must be non-negative")
    args.function(args)


if __name__ == "__main__":
    main()
