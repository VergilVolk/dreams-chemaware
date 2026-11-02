#!/usr/bin/env python
"""Strict, reusable evaluator for a very large labelled identity benchmark.

The formal panel is a three-axis graph:

* ``N_Q`` queries, delimited by ``query_ptr``;
* ``N_E`` query-candidate molecule edges, carrying molecule labels/identities;
* ``N_P`` query-candidate spectrum-pair edges, delimited by ``molecule_ptr``.

Required panel arrays are ``query_ptr``, ``molecule_ptr``, ``candidate_row``
(or ``pair_candidate_row``), ``molecule_label``, ``query_ik14``,
``molecule_ik14``, ``query_formula``, ``molecule_formula``, and ``near_query``.
``query_row`` is also required to bind repeated queries from the same identity.
Each query has exactly one identity-matched positive and unique candidate
identities. Open-set evaluation is deliberately outside this closed-set
identity contract.

The score NPZ must bind itself to the panel with the exact ``panel_sha256`` or
the canonical ``axis_digest`` and must record scalar ``N_Q``, ``N_E``, and
``N_P``. This rejects a same-length score vector on the wrong candidate axis.
The preferred score schema is ``method_names`` (M) plus ``molecule_scores``
(M, N_E); named one-dimensional score arrays are also accepted.

NPZ members are extracted to temporary ``.npy`` files and memory-mapped. One
method is evaluated at a time. Exact pooled ROC/PR reduction retains only an
N_E sort index in memory, while per-query rows and gzip curve CSVs are streamed.
Retrieval ties are conservative: every tied negative is placed ahead of the
positive. ROC uses the standard half-credit convention for tied pairs.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import re
import shutil
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator, Sequence

import numpy as np


TOP_K = (1, 2, 3, 5, 10, 20)
SCHEMA_VERSION = "mega_identity_benchmark_evaluation_v2"
AXIS_DIGEST_VERSION = "mega_identity_axis_v1"
PAIRED_METRIC_NAMES = (
    *(f"recall@{cutoff}" for cutoff in TOP_K),
    "mrr",
    "mean_rank_improvement",
    "macro_query_auroc",
    "macro_query_average_precision",
)
SUMMARY_FIELDS = [
    "method", "N_Q", "N_E", "N_P", "N_I", "N_F",
    *(f"recall@{k}" for k in TOP_K),
    "mrr", "mean_rank", "median_rank", "macro_query_auroc",
    "macro_query_average_precision", "macro_query_historical_auprc",
    "micro_candidate_auroc", "micro_candidate_average_precision",
    "micro_candidate_historical_auprc", "near_queries",
    *(f"near_recall@{k}" for k in TOP_K),
    "near_mrr", "near_mean_rank", "near_median_rank",
    "near_macro_query_auroc", "near_macro_query_average_precision",
    "near_macro_query_historical_auprc",
]


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True, help="Formal identity panel NPZ.")
    parser.add_argument("--scores", type=Path, required=True, help="Bound multi-method scores NPZ.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory.")
    parser.add_argument("--method", action="append", default=[], help="Select a method; repeatable.")
    parser.add_argument(
        "--reference-method", default=None,
        help="Paired-comparison reference (default: first selected method).",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261102)
    parser.add_argument(
        "--evaluation-role", default="unspecified",
        choices=("development", "meta_development", "stress", "external_confirmation", "unspecified"),
        help="Scientific role written into the machine-readable report.",
    )
    parser.add_argument(
        "--truth-status", default="unspecified",
        choices=("consumed", "unopened", "independent", "unspecified"),
        help="Truth-use status written into the machine-readable report.",
    )
    parser.add_argument(
        "--allow-final-claim", action="store_true",
        help="Mark the run as eligible for a final claim; false unless explicitly supplied.",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=None,
        help="Temporary extraction/sort directory (defaults to output parent).",
    )
    parser.add_argument("--sort-chunk-size", type=int, default=250_000)
    parser.add_argument("--validation-chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--progress-every", type=int, default=100_000,
        help="Print progress every this many queries (0 disables).",
    )
    return parser.parse_args(argv)


def _as_path(path: Path) -> Path:
    return path.expanduser().resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_text(value: object) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            raise RuntimeError("formal text metadata must be valid UTF-8") from error
    return str(value)


class NpzMemmapStore:
    """Extract selected NPZ members safely and return read-only memmaps."""

    def __init__(self, path: Path, temporary_root: Path, prefix: str):
        self.path = _as_path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if not zipfile.is_zipfile(self.path):
            raise RuntimeError(f"input is not an NPZ/ZIP archive: {self.path}")
        self.directory = temporary_root / prefix
        self.directory.mkdir(parents=True, exist_ok=False)
        self._zip = zipfile.ZipFile(self.path, "r")
        self._members = {
            name[:-4]: name for name in self._zip.namelist()
            if name.endswith(".npy") and not name.endswith("/")
        }
        if not self._members:
            raise RuntimeError(f"NPZ has no arrays: {self.path}")
        self._cache: dict[str, np.ndarray] = {}

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._members)

    def has(self, key: str) -> bool:
        return key in self._members

    def array(self, key: str) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]
        try:
            member = self._members[key]
        except KeyError as error:
            raise RuntimeError(f"{self.path.name} lacks required array {key!r}") from error
        target = self.directory / f"{len(self._cache):04d}.npy"
        with self._zip.open(member, "r") as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=1 << 20)
        try:
            array = np.load(target, mmap_mode="r", allow_pickle=False)
        except ValueError as error:
            raise RuntimeError(f"array {key!r} must be non-object") from error
        self._cache[key] = array
        return array

    def scalar(self, key: str) -> object:
        array = self.array(key)
        if array.ndim == 0:
            return array.item()
        if array.size == 1:
            return array.reshape(-1)[0].item()
        raise RuntimeError(f"metadata field {key!r} must be scalar")

    def close(self) -> None:
        self._zip.close()
        for array in self._cache.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                try:
                    mapping.close()
                except (BufferError, OSError):
                    pass
        self._cache.clear()


@dataclass(frozen=True)
class Panel:
    query_ptr: np.ndarray
    molecule_ptr: np.ndarray
    pair_candidate_row: np.ndarray
    molecule_label: np.ndarray
    query_ik14: np.ndarray
    molecule_ik14: np.ndarray
    query_formula: np.ndarray
    molecule_formula: np.ndarray
    near_query: np.ndarray
    query_row: np.ndarray
    positive_index: np.ndarray
    axis_digest: str
    n_identities: int
    n_formulas: int

    @property
    def n_queries(self) -> int:
        return len(self.query_ptr) - 1

    @property
    def n_molecule_edges(self) -> int:
        return len(self.molecule_label)

    @property
    def n_pair_edges(self) -> int:
        return len(self.pair_candidate_row)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    key: str
    row: int | None


@dataclass(frozen=True)
class QueryOutcomes:
    rank: np.ndarray
    macro_auroc: np.ndarray


@dataclass(frozen=True)
class ClusterIndex:
    name: str
    inverse: np.ndarray
    counts: np.ndarray

    @property
    def n_clusters(self) -> int:
        return len(self.counts)


def _require_vector(name: str, array: np.ndarray) -> None:
    if array.ndim != 1:
        raise RuntimeError(f"{name} must be one-dimensional, got {array.shape}")


def _canonical_numeric(
    digest: object, name: str, array: np.ndarray, dtype: str, chunk: int = 1_000_000
) -> None:
    digest.update(name.encode("ascii") + b"\0N\0")
    digest.update(struct.pack("<Q", len(array)))
    for left in range(0, len(array), chunk):
        block = np.ascontiguousarray(array[left : left + chunk], dtype=np.dtype(dtype))
        digest.update(memoryview(block).cast("B"))


def _canonical_text(digest: object, name: str, array: np.ndarray) -> None:
    digest.update(name.encode("ascii") + b"\0T\0")
    digest.update(struct.pack("<Q", len(array)))
    for value in array:
        encoded = _decode_text(value).encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)


def canonical_axis_digest(panel: Panel) -> str:
    """Return the canonical closed-set axis digest recorded by score archives."""
    digest = hashlib.sha256()
    digest.update(AXIS_DIGEST_VERSION.encode("ascii") + b"\0")
    _canonical_numeric(digest, "query_ptr", panel.query_ptr, "<i8")
    _canonical_numeric(digest, "molecule_ptr", panel.molecule_ptr, "<i8")
    _canonical_numeric(digest, "pair_candidate_row", panel.pair_candidate_row, "<i8")
    _canonical_numeric(digest, "query_row", panel.query_row, "<i8")
    _canonical_numeric(digest, "molecule_label", panel.molecule_label, "u1")
    _canonical_numeric(digest, "near_query", panel.near_query, "u1")
    _canonical_text(digest, "query_ik14", panel.query_ik14)
    _canonical_text(digest, "molecule_ik14", panel.molecule_ik14)
    _canonical_text(digest, "query_formula", panel.query_formula)
    _canonical_text(digest, "molecule_formula", panel.molecule_formula)
    return digest.hexdigest()


def _validate_text_vector(name: str, array: np.ndarray, expected: int) -> None:
    _require_vector(name, array)
    if len(array) != expected or array.dtype.kind not in "SU":
        raise RuntimeError(f"{name} must be a length-{expected} text vector")


def load_and_validate_panel(store: NpzMemmapStore) -> Panel:
    query_ptr = store.array("query_ptr")
    molecule_ptr = store.array("molecule_ptr")
    if store.has("candidate_row"):
        pair_candidate_row = store.array("candidate_row")
        if store.has("pair_candidate_row"):
            alias = store.array("pair_candidate_row")
            if not np.array_equal(pair_candidate_row, alias):
                raise RuntimeError("candidate_row and pair_candidate_row disagree")
    elif store.has("pair_candidate_row"):
        pair_candidate_row = store.array("pair_candidate_row")
    else:
        raise RuntimeError("formal panel requires candidate_row or pair_candidate_row")
    labels = store.array("molecule_label")
    query_ik14 = store.array("query_ik14")
    molecule_ik14 = store.array("molecule_ik14")
    query_formula = store.array("query_formula")
    molecule_formula = store.array("molecule_formula")
    near_query = store.array("near_query")
    for name, array in (
        ("query_ptr", query_ptr), ("molecule_ptr", molecule_ptr),
        ("pair_candidate_row", pair_candidate_row), ("molecule_label", labels),
        ("near_query", near_query),
    ):
        _require_vector(name, array)
    if query_ptr.dtype.kind not in "iu" or molecule_ptr.dtype.kind not in "iu":
        raise RuntimeError("query_ptr and molecule_ptr must be integer vectors")
    if pair_candidate_row.dtype.kind not in "iu":
        raise RuntimeError("candidate rows must be integers")
    int64_max = np.iinfo(np.int64).max
    if np.any(np.asarray(pair_candidate_row) < 0) or np.any(
        np.asarray(pair_candidate_row) > int64_max
    ):
        raise RuntimeError("candidate rows must lie in the non-negative int64 range")
    if len(query_ptr) < 2 or int(query_ptr[0]) != 0:
        raise RuntimeError("query_ptr must start at zero and contain at least one query")
    if int(query_ptr[-1]) != len(labels):
        raise RuntimeError("query_ptr[-1] must equal N_E=len(molecule_label)")
    if len(molecule_ptr) != len(labels) + 1 or int(molecule_ptr[0]) != 0:
        raise RuntimeError("molecule_ptr must have N_E+1 entries and start at zero")
    if int(molecule_ptr[-1]) != len(pair_candidate_row):
        raise RuntimeError("molecule_ptr[-1] must equal N_P=len(pair candidate rows)")
    if np.any(np.asarray(query_ptr) < 0) or np.any(np.asarray(query_ptr) > len(labels)):
        raise RuntimeError("query_ptr entries are outside [0, N_E]")
    if np.any(np.asarray(molecule_ptr) < 0) or np.any(
        np.asarray(molecule_ptr) > len(pair_candidate_row)
    ):
        raise RuntimeError("molecule_ptr entries are outside [0, N_P]")
    query_ptr_i64 = np.asarray(query_ptr, dtype=np.int64)
    molecule_ptr_i64 = np.asarray(molecule_ptr, dtype=np.int64)
    if np.any(np.diff(query_ptr_i64) < 2):
        raise RuntimeError("every query needs one positive and at least one negative molecule")
    if np.any(np.diff(molecule_ptr_i64) < 1):
        raise RuntimeError("every candidate molecule needs at least one spectrum-pair edge")

    n_queries = len(query_ptr) - 1
    n_edges = len(labels)
    _validate_text_vector("query_ik14", query_ik14, n_queries)
    _validate_text_vector("molecule_ik14", molecule_ik14, n_edges)
    _validate_text_vector("query_formula", query_formula, n_queries)
    _validate_text_vector("molecule_formula", molecule_formula, n_edges)
    if labels.dtype.kind not in "biu" or near_query.dtype.kind not in "biu":
        raise RuntimeError("molecule_label and near_query must be boolean/binary")
    if len(near_query) != n_queries:
        raise RuntimeError("near_query must align to queries")
    if not np.all(np.isin(np.unique(labels), (0, 1))):
        raise RuntimeError("molecule_label must contain only zero/one")
    if not np.all(np.isin(np.unique(near_query), (0, 1))):
        raise RuntimeError("near_query must contain only zero/one")

    positive_index = np.empty(n_queries, dtype=np.int64)
    for query in range(n_queries):
        query_identity_raw = _decode_text(query_ik14[query])
        query_formula_raw = _decode_text(query_formula[query])
        if query_identity_raw != query_identity_raw.strip() or query_formula_raw != query_formula_raw.strip():
            raise RuntimeError(f"query {query} IK14/formula contains surrounding whitespace")
        query_identity = query_identity_raw
        query_formula_value = query_formula_raw
        if re.fullmatch(r"[A-Z]{14}", query_identity) is None or not query_formula_value:
            raise RuntimeError(f"query {query} has invalid IK14/formula")
        left, right = map(int, query_ptr[query : query + 2])
        raw_identities = [_decode_text(value) for value in molecule_ik14[left:right]]
        raw_formulas = [_decode_text(value) for value in molecule_formula[left:right]]
        if any(value != value.strip() for value in (*raw_identities, *raw_formulas)):
            raise RuntimeError(f"query {query} candidate IK14/formula contains surrounding whitespace")
        identities = raw_identities
        formulas = raw_formulas
        if any(re.fullmatch(r"[A-Z]{14}", identity) is None for identity in identities) or any(
            not value for value in formulas
        ):
            raise RuntimeError(f"query {query} has invalid candidate IK14/formula")
        if len(set(identities)) != len(identities):
            raise RuntimeError(f"query {query} has duplicate candidate identities")
        local_labels = np.asarray(labels[left:right], dtype=bool)
        identity_match = np.asarray([value == query_identity for value in identities], dtype=bool)
        if not np.array_equal(local_labels, identity_match) or int(local_labels.sum()) != 1:
            raise RuntimeError(
                f"query {query} labels must exactly equal molecule_ik14==query_ik14"
            )
        local_positive = int(np.flatnonzero(local_labels)[0])
        positive_index[query] = left + local_positive
        if formulas[local_positive] != query_formula_value:
            raise RuntimeError(f"query {query} positive molecule formula disagrees with query")
        expected_near = any(
            (not bool(label)) and formula == query_formula_value
            for label, formula in zip(local_labels, formulas)
        )
        if bool(near_query[query]) != expected_near:
            raise RuntimeError(f"query {query} near_query disagrees with same-formula negatives")

    query_row = store.array("query_row")
    _require_vector("query_row", query_row)
    if len(query_row) != n_queries or query_row.dtype.kind not in "iu":
        raise RuntimeError("query_row must be a length-N_Q integer vector")
    if np.any(np.asarray(query_row) < 0) or np.any(np.asarray(query_row) > int64_max):
        raise RuntimeError("query_row values must lie in the non-negative int64 range")
    if len(np.unique(query_row)) != n_queries:
        raise RuntimeError("query_row values must be unique")
    for query in range(n_queries):
        molecule_left, molecule_right = map(int, query_ptr[query : query + 2])
        pair_left = int(molecule_ptr[molecule_left])
        pair_right = int(molecule_ptr[molecule_right])
        local_rows = pair_candidate_row[pair_left:pair_right]
        if np.any(local_rows == query_row[query]):
            raise RuntimeError(f"query {query} contains an exact self-match candidate row")
        if len(np.unique(local_rows)) != len(local_rows):
            raise RuntimeError(f"query {query} contains duplicate candidate spectrum rows")
    provisional = Panel(
        query_ptr=query_ptr, molecule_ptr=molecule_ptr,
        pair_candidate_row=pair_candidate_row, molecule_label=labels,
        query_ik14=query_ik14, molecule_ik14=molecule_ik14,
        query_formula=query_formula, molecule_formula=molecule_formula,
        near_query=near_query, query_row=query_row, positive_index=positive_index,
        axis_digest="", n_identities=len(np.unique(query_ik14)),
        n_formulas=len(np.unique(query_formula)),
    )
    return Panel(**{**provisional.__dict__, "axis_digest": canonical_axis_digest(provisional)})


def validate_score_contract(
    store: NpzMemmapStore, panel: Panel, panel_sha256: str
) -> dict[str, object]:
    expected = {"N_Q": panel.n_queries, "N_E": panel.n_molecule_edges, "N_P": panel.n_pair_edges}
    observed: dict[str, int] = {}
    for key, value in expected.items():
        if not store.has(key):
            raise RuntimeError(f"score NPZ lacks required scalar {key}")
        metadata_array = store.array(key)
        if metadata_array.dtype.kind not in "iu":
            raise RuntimeError(f"score metadata {key} must use an integer dtype")
        raw = store.scalar(key)
        if isinstance(raw, (bool, np.bool_)):
            raise RuntimeError(f"score metadata {key} must be an integer")
        try:
            parsed = int(raw)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"score metadata {key} must be an integer") from error
        if isinstance(raw, (float, np.floating)) and float(raw) != parsed:
            raise RuntimeError(f"score metadata {key} must be an exact integer")
        if parsed != value:
            raise RuntimeError(f"score metadata {key}={parsed} does not match panel {value}")
        observed[key] = parsed
    has_file_hash = store.has("panel_sha256")
    has_axis_digest = store.has("axis_digest")
    if not has_file_hash and not has_axis_digest:
        raise RuntimeError("score NPZ requires panel_sha256 or canonical axis_digest")
    provenance: dict[str, object] = {**observed}
    if has_file_hash:
        value = _decode_text(store.scalar("panel_sha256")).strip().lower()
        if value != panel_sha256:
            raise RuntimeError("score NPZ panel_sha256 does not match the supplied panel")
        provenance["panel_sha256"] = value
    if has_axis_digest:
        value = _decode_text(store.scalar("axis_digest")).strip().lower()
        if value != panel.axis_digest:
            raise RuntimeError("score NPZ axis_digest does not match the canonical panel axis")
        provenance["axis_digest"] = value
    return provenance


def discover_methods(store: NpzMemmapStore, n_edges: int) -> list[MethodSpec]:
    if store.has("molecule_scores"):
        if not store.has("method_names"):
            raise RuntimeError("matrix score schema requires method_names")
        names_array = store.array("method_names")
        scores = store.array("molecule_scores")
        _require_vector("method_names", names_array)
        names = [_decode_text(value).strip() for value in names_array]
        if scores.ndim != 2 or scores.shape != (len(names), n_edges):
            raise RuntimeError(
                "molecule_scores must have shape (len(method_names), N_E); "
                f"got {scores.shape}"
            )
        specs = [MethodSpec(name, "molecule_scores", row) for row, name in enumerate(names)]
    else:
        ignored = {
            "method_names", "schema_version", "panel_sha256", "axis_digest",
            "N_Q", "N_E", "N_P",
        }
        specs = []
        for key in store.keys:
            if key in ignored:
                continue
            array = store.array(key)
            if array.ndim != 1 or len(array) != n_edges or array.dtype.kind not in "fiu":
                continue
            name = key.removeprefix("score__").removesuffix("_molecule_score")
            specs.append(MethodSpec(name, key, None))
        if not specs:
            raise RuntimeError(
                "score NPZ needs method_names+molecule_scores or numeric length-N_E arrays"
            )
    names = [spec.name for spec in specs]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise RuntimeError("method names must be non-empty and unique")
    return specs


def method_scores(store: NpzMemmapStore, spec: MethodSpec) -> np.ndarray:
    array = store.array(spec.key)
    return array if spec.row is None else array[spec.row]


def validate_scores(scores: np.ndarray, n_edges: int, chunk_size: int) -> None:
    if scores.ndim != 1 or len(scores) != n_edges:
        raise RuntimeError(f"score vector has shape {scores.shape}; expected ({n_edges},)")
    if scores.dtype.kind not in "fiu":
        raise RuntimeError("method scores must be numeric")
    for left in range(0, len(scores), chunk_size):
        block = np.asarray(scores[left : left + chunk_size])
        if not np.all(np.isfinite(block)):
            bad = left + int(np.flatnonzero(~np.isfinite(block))[0])
            raise RuntimeError(f"method score vector has a non-finite value at edge {bad}")


def metric_summary(
    outcomes: QueryOutcomes, mask: np.ndarray | None = None
) -> dict[str, float | int] | None:
    rank = outcomes.rank if mask is None else outcomes.rank[mask]
    auroc = outcomes.macro_auroc if mask is None else outcomes.macro_auroc[mask]
    if not len(rank):
        return None
    reciprocal = 1.0 / rank
    result: dict[str, float | int] = {
        "queries": int(len(rank)), "mrr": float(np.mean(reciprocal)),
        "mean_rank": float(np.mean(rank)), "median_rank": float(np.median(rank)),
        "macro_query_auroc": float(np.mean(auroc)),
        "macro_query_average_precision": float(np.mean(reciprocal)),
        "macro_query_historical_auprc": float(np.mean(reciprocal)),
    }
    for cutoff in TOP_K:
        result[f"recall@{cutoff}"] = float(np.mean(rank <= cutoff))
    return result


QUERY_FIELDS = [
    "method", "query_index", "query_row", "query_ik14", "query_formula", "near",
    "candidate_molecule_edges", "candidate_pair_edges", "positive_molecule_edge",
    "positive_molecule_ik14", "positive_molecule_formula", "positive_score",
    "best_negative_score", "positive_vs_best_negative_margin", "rank",
    "reciprocal_rank", "macro_query_auroc", "macro_query_average_precision",
    "macro_query_historical_auprc", "tied_negatives",
]


def evaluate_queries(
    panel: Panel, method: str, scores: np.ndarray, writer: csv.DictWriter,
    progress_every: int,
) -> tuple[dict[str, float | int], dict[str, float | int] | None, QueryOutcomes]:
    ranks = np.empty(panel.n_queries, dtype=np.int64)
    query_aurocs = np.empty(panel.n_queries, dtype=np.float64)
    for query in range(panel.n_queries):
        left, right = map(int, panel.query_ptr[query : query + 2])
        positive_index = int(panel.positive_index[query])
        values = np.asarray(scores[left:right], dtype=np.float64)
        local_positive = positive_index - left
        positive_score = float(values[local_positive])
        negative_mask = np.ones(len(values), dtype=bool)
        negative_mask[local_positive] = False
        negatives = values[negative_mask]
        tied = int(np.count_nonzero(negatives == positive_score))
        rank = 1 + int(np.count_nonzero(negatives > positive_score)) + tied
        wins = int(np.count_nonzero(positive_score > negatives))
        query_auroc = (wins + 0.5 * tied) / len(negatives)
        best_negative = float(np.max(negatives))
        ranks[query] = rank
        query_aurocs[query] = query_auroc
        pair_left = int(panel.molecule_ptr[left])
        pair_right = int(panel.molecule_ptr[right])
        writer.writerow({
            "method": method, "query_index": query,
            "query_row": int(panel.query_row[query]),
            "query_ik14": _decode_text(panel.query_ik14[query]),
            "query_formula": _decode_text(panel.query_formula[query]),
            "near": int(bool(panel.near_query[query])),
            "candidate_molecule_edges": right - left,
            "candidate_pair_edges": pair_right - pair_left,
            "positive_molecule_edge": positive_index,
            "positive_molecule_ik14": _decode_text(panel.molecule_ik14[positive_index]),
            "positive_molecule_formula": _decode_text(panel.molecule_formula[positive_index]),
            "positive_score": format(positive_score, ".17g"),
            "best_negative_score": format(best_negative, ".17g"),
            "positive_vs_best_negative_margin": format(positive_score - best_negative, ".17g"),
            "rank": rank, "reciprocal_rank": format(1.0 / rank, ".17g"),
            "macro_query_auroc": format(query_auroc, ".17g"),
            "macro_query_average_precision": format(1.0 / rank, ".17g"),
            "macro_query_historical_auprc": format(1.0 / rank, ".17g"),
            "tied_negatives": tied,
        })
        if progress_every and (query + 1) % progress_every == 0:
            print(f"[{method}] evaluated {query + 1:,}/{panel.n_queries:,} queries", flush=True)
    outcomes = QueryOutcomes(rank=ranks, macro_auroc=query_aurocs)
    overall = metric_summary(outcomes)
    assert overall is not None
    near = metric_summary(outcomes, np.asarray(panel.near_query, dtype=bool))
    return overall, near, outcomes


def _safe_slug(name: str, index: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "method"
    return f"{index:03d}_{slug[:80]}"


def _curve_rows(
    handle: IO[str], thresholds: np.ndarray, first: np.ndarray, second: np.ndarray,
    true_positive: np.ndarray, false_positive: np.ndarray, batch: int = 10_000,
) -> None:
    for left in range(0, len(thresholds), batch):
        right = min(left + batch, len(thresholds))
        handle.write("".join(
            f"{thresholds[i]:.17g},{first[i]:.17g},{second[i]:.17g},"
            f"{int(true_positive[i])},{int(false_positive[i])}\n"
            for i in range(left, right)
        ))


def exact_micro_curves(
    scores: np.ndarray, labels: np.ndarray, roc_path: Path, pr_path: Path, chunk_size: int,
) -> dict[str, float | int]:
    binary = np.asarray(labels, dtype=bool)
    positives = int(np.count_nonzero(binary))
    negatives = len(binary) - positives
    if positives == 0 or negatives == 0:
        raise RuntimeError("micro metrics require both positive and negative molecule edges")
    order = np.argsort(scores, kind="quicksort")
    tp = fp = 0
    auroc = average_precision = 0.0
    points = 1
    pending: tuple[float, int, int] | None = None
    roc_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(roc_path, "wt", encoding="utf-8", newline="") as roc, gzip.open(
        pr_path, "wt", encoding="utf-8", newline=""
    ) as pr:
        roc.write("threshold,false_positive_rate,true_positive_rate,true_positives,false_positives\n")
        roc.write("inf,0,0,0,0\n")
        pr.write("threshold,recall,precision,true_positives,false_positives\n")
        pr.write("inf,0,1,0,0\n")

        def consume(group_score: np.ndarray, group_pos: np.ndarray, group_neg: np.ndarray) -> None:
            nonlocal tp, fp, auroc, average_precision, points
            if not len(group_score):
                return
            old_tp, old_fp = tp, fp
            cumulative_pos = old_tp + np.cumsum(group_pos, dtype=np.int64)
            cumulative_neg = old_fp + np.cumsum(group_neg, dtype=np.int64)
            prev_tpr = np.concatenate(([old_tp / positives], cumulative_pos[:-1] / positives))
            prev_fpr = np.concatenate(([old_fp / negatives], cumulative_neg[:-1] / negatives))
            tpr = cumulative_pos / positives
            fpr = cumulative_neg / negatives
            auroc += float(np.sum((fpr - prev_fpr) * (tpr + prev_tpr) * 0.5))
            precision = cumulative_pos / (cumulative_pos + cumulative_neg)
            increments = cumulative_pos - np.concatenate(([old_tp], cumulative_pos[:-1]))
            average_precision += float(np.sum(increments / positives * precision))
            _curve_rows(roc, group_score, fpr, tpr, cumulative_pos, cumulative_neg)
            _curve_rows(pr, group_score, tpr, precision, cumulative_pos, cumulative_neg)
            tp, fp = int(cumulative_pos[-1]), int(cumulative_neg[-1])
            points += len(group_score)

        end = len(order)
        while end > 0:
            start = max(0, end - chunk_size)
            indices = order[start:end][::-1]
            block_scores = np.asarray(scores[indices], dtype=np.float64)
            block_labels = np.asarray(binary[indices], dtype=np.int8)
            boundaries = np.concatenate((
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(block_scores[1:] != block_scores[:-1]) + 1,
                np.asarray([len(block_scores)], dtype=np.int64),
            ))
            starts = boundaries[:-1]
            group_score = block_scores[starts]
            group_count = np.diff(boundaries)
            group_pos = np.add.reduceat(block_labels, starts).astype(np.int64, copy=False)
            group_neg = group_count - group_pos
            if pending is not None:
                pending_score, pending_pos, pending_neg = pending
                if group_score[0] == pending_score:
                    group_pos[0] += pending_pos
                    group_neg[0] += pending_neg
                else:
                    group_score = np.concatenate(([pending_score], group_score))
                    group_pos = np.concatenate(([pending_pos], group_pos))
                    group_neg = np.concatenate(([pending_neg], group_neg))
            pending = (float(group_score[-1]), int(group_pos[-1]), int(group_neg[-1]))
            consume(group_score[:-1], group_pos[:-1], group_neg[:-1])
            end = start
        assert pending is not None
        consume(np.asarray([pending[0]]), np.asarray([pending[1]]), np.asarray([pending[2]]))
    del order
    if tp != positives or fp != negatives:
        raise AssertionError("curve reduction did not consume every molecule edge")
    return {
        "molecule_edges": len(labels), "positive_molecule_edges": positives,
        "negative_molecule_edges": negatives, "auroc": auroc,
        "average_precision": average_precision,
        "historical_auprc": average_precision, "curve_points": points,
    }


def make_cluster_index(name: str, values: np.ndarray) -> ClusterIndex:
    _, inverse = np.unique(np.asarray(values).astype(str), return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.int64)
    counts = np.bincount(inverse).astype(np.int64)
    if len(counts) < 2:
        raise RuntimeError(f"paired cluster bootstrap requires at least two {name} clusters")
    return ClusterIndex(name=name, inverse=inverse, counts=counts)


def paired_metric_arrays(
    reference: QueryOutcomes, candidate: QueryOutcomes
) -> Iterator[tuple[str, np.ndarray]]:
    for cutoff in TOP_K:
        yield f"recall@{cutoff}", (
            (candidate.rank <= cutoff).astype(np.float64)
            - (reference.rank <= cutoff).astype(np.float64)
        )
    yield "mrr", 1.0 / candidate.rank - 1.0 / reference.rank
    yield "mean_rank_improvement", reference.rank.astype(np.float64) - candidate.rank
    yield "macro_query_auroc", candidate.macro_auroc - reference.macro_auroc
    yield "macro_query_average_precision", 1.0 / candidate.rank - 1.0 / reference.rank


def cluster_bootstrap(
    cluster: ClusterIndex, reference: QueryOutcomes, candidate: QueryOutcomes,
    repeats: int, seed: int, familywise_hypotheses: int,
) -> dict[str, object]:
    metric_names: list[str] = []
    cluster_sums: list[np.ndarray] = []
    points: dict[str, float] = {}
    for name, delta in paired_metric_arrays(reference, candidate):
        metric_names.append(name)
        points[name] = float(np.mean(delta))
        cluster_sums.append(np.bincount(cluster.inverse, weights=delta, minlength=cluster.n_clusters))
    sums = np.column_stack(cluster_sums)
    bootstrap = np.empty((repeats, len(metric_names)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        take = rng.integers(0, cluster.n_clusters, size=cluster.n_clusters)
        bootstrap[repeat] = np.sum(sums[take], axis=0) / int(np.sum(cluster.counts[take]))
    tail = 0.05 / (2 * familywise_hypotheses)
    metrics: dict[str, object] = {}
    for column, name in enumerate(metric_names):
        metrics[name] = {
            "delta": points[name],
            "ci_low": float(np.quantile(bootstrap[:, column], tail)),
            "ci_high": float(np.quantile(bootstrap[:, column], 1 - tail)),
        }
    return {
        "cluster_unit": cluster.name, "clusters": cluster.n_clusters,
        "bootstrap_resamples": repeats, "bootstrap_seed": seed,
        "familywise_confidence_level": 0.95,
        "per_interval_nominal_level": 1 - 0.05 / familywise_hypotheses,
        "familywise_hypotheses": familywise_hypotheses,
        "tail_probability": tail, "metrics": metrics,
    }


def paired_comparison(
    reference_name: str, candidate_name: str, reference: QueryOutcomes,
    candidate: QueryOutcomes, near: np.ndarray, formula_clusters: ClusterIndex,
    identity_clusters: ClusterIndex, repeats: int, seed: int,
) -> dict[str, object]:
    corrected = (reference.rank != 1) & (candidate.rank == 1)
    introduced = (reference.rank == 1) & (candidate.rank != 1)
    n_corrected = int(np.count_nonzero(corrected))
    n_introduced = int(np.count_nonzero(introduced))
    familywise = len(PAIRED_METRIC_NAMES) * 2
    return {
        "reference_method": reference_name, "candidate_method": candidate_name,
        "queries": len(reference.rank), "corrected": n_corrected,
        "introduced": n_introduced,
        "net_corrected_minus_introduced": n_corrected - n_introduced,
        "risk_net_lambda2": n_corrected - 2 * n_introduced,
        "near_queries": int(np.count_nonzero(near)),
        "near_corrected": int(np.count_nonzero(corrected & near)),
        "near_introduced": int(np.count_nonzero(introduced & near)),
        "formula_cluster_bootstrap": cluster_bootstrap(
            formula_clusters, reference, candidate, repeats, seed, familywise
        ),
        "identity_cluster_bootstrap": cluster_bootstrap(
            identity_clusters, reference, candidate, repeats, seed + 1, familywise
        ),
    }


def _summary_row(
    method: str, panel: Panel, retrieval: dict[str, float | int],
    near: dict[str, float | int] | None, micro: dict[str, float | int],
) -> dict[str, object]:
    row: dict[str, object] = {
        "method": method, "N_Q": panel.n_queries, "N_E": panel.n_molecule_edges,
        "N_P": panel.n_pair_edges, "N_I": panel.n_identities, "N_F": panel.n_formulas,
        **{f"recall@{k}": retrieval[f"recall@{k}"] for k in TOP_K},
        "mrr": retrieval["mrr"], "mean_rank": retrieval["mean_rank"],
        "median_rank": retrieval["median_rank"],
        "macro_query_auroc": retrieval["macro_query_auroc"],
        "macro_query_average_precision": retrieval["macro_query_average_precision"],
        "macro_query_historical_auprc": retrieval["macro_query_historical_auprc"],
        "micro_candidate_auroc": micro["auroc"],
        "micro_candidate_average_precision": micro["average_precision"],
        "micro_candidate_historical_auprc": micro["historical_auprc"],
        "near_queries": 0 if near is None else near["queries"],
    }
    for cutoff in TOP_K:
        row[f"near_recall@{cutoff}"] = "" if near is None else near[f"recall@{cutoff}"]
    for key in (
        "mrr", "mean_rank", "median_rank", "macro_query_auroc",
        "macro_query_average_precision", "macro_query_historical_auprc",
    ):
        row[f"near_{key}"] = "" if near is None else near[key]
    return row


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    panel_path = _as_path(args.panel)
    scores_path = _as_path(args.scores)
    output_dir = _as_path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if args.sort_chunk_size < 1 or args.validation_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    if args.progress_every < 0:
        raise ValueError("progress-every cannot be negative")
    if args.bootstrap_resamples < 10_000:
        raise ValueError("formal bootstrap-resamples must be at least 10000")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_parent = _as_path(args.work_dir) if args.work_dir is not None else output_dir.parent
    work_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent))
    extraction = Path(tempfile.mkdtemp(prefix="mega_eval_arrays.", dir=work_parent))
    panel_store: NpzMemmapStore | None = None
    score_store: NpzMemmapStore | None = None
    panel: Panel | None = None
    active_scores: np.ndarray | None = None
    reference_outcomes: QueryOutcomes | None = None
    try:
        panel_sha256 = sha256_file(panel_path)
        panel_store = NpzMemmapStore(panel_path, extraction, "panel")
        score_store = NpzMemmapStore(scores_path, extraction, "scores")
        panel = load_and_validate_panel(panel_store)
        score_contract = validate_score_contract(score_store, panel, panel_sha256)
        specs = discover_methods(score_store, panel.n_molecule_edges)
        requested = list(dict.fromkeys(args.method))
        if requested:
            by_name = {spec.name: spec for spec in specs}
            missing = [name for name in requested if name not in by_name]
            if missing:
                raise RuntimeError(f"requested methods absent from score NPZ: {missing}")
            specs = [by_name[name] for name in requested]
        if not specs:
            raise RuntimeError("no methods selected")
        reference_method = args.reference_method or specs[0].name
        by_name = {spec.name: spec for spec in specs}
        if reference_method not in by_name:
            raise RuntimeError("reference-method must be among selected methods")
        specs = [by_name[reference_method], *(spec for spec in specs if spec.name != reference_method)]

        (staging / "curves").mkdir()
        report: dict[str, object] = {
            "schema_version": SCHEMA_VERSION, "status": "complete",
            "closed_set_identity_contract": True, "open_set_claim": False,
            "evaluation_role": getattr(args, "evaluation_role", "unspecified"),
            "truth_status": getattr(args, "truth_status", "unspecified"),
            "allow_final_claim": bool(getattr(args, "allow_final_claim", False)),
            "panel": {
                "path": str(panel_path), "sha256": panel_sha256,
                "axis_digest_version": AXIS_DIGEST_VERSION, "axis_digest": panel.axis_digest,
                "N_Q": panel.n_queries, "N_E": panel.n_molecule_edges,
                "N_P": panel.n_pair_edges, "N_I": panel.n_identities,
                "N_F": panel.n_formulas,
                "count_semantics": {
                    "N_Q": "queries",
                    "N_E": "directed query-candidate molecule edges",
                    "N_P": "directed query-candidate spectrum-pair edges",
                    "N_I": "unique query IK14 identities",
                    "N_F": "unique query formula clusters",
                },
                "near_queries": int(np.count_nonzero(panel.near_query)),
            },
            "scores": {
                "path": str(scores_path), "sha256": sha256_file(scores_path),
                "validated_contract": score_contract,
            },
            "reference_method": reference_method,
            "paired_delta_definitions": {
                "recall/mrr/auroc/average_precision": "candidate minus reference; positive is better",
                "mean_rank_improvement": "reference mean rank minus candidate mean rank; positive is better",
                "corrected": "reference rank != 1 and candidate rank == 1",
                "introduced": "reference rank == 1 and candidate rank != 1",
            },
            "tie_policy": (
                "conservative rank: every negative with score >= positive is ahead; "
                "curves aggregate equal-score edges"
            ),
            "metric_definitions": {
                "macro_query_auroc": "mean one-positive per-query AUROC; pair ties receive 0.5",
                "macro_query_average_precision": "mean non-interpolated AP; equals 1/rank",
                "macro_query_historical_auprc": "historical AUPRC label; exact AP alias",
                "micro_candidate_auroc": "pooled over all N_E molecule edges",
                "micro_candidate_average_precision": "pooled non-interpolated AP",
                "micro_candidate_historical_auprc": "historical AUPRC label; exact AP alias",
            },
            "higher_score_is_better": True, "top_k": list(TOP_K),
            "methods": {}, "paired_comparisons": {},
            "artifacts": {
                "summary_csv": "summary_metrics.csv",
                "query_table_csv_gz": "query_metrics.csv.gz",
                "paired_comparisons_json": "paired_comparisons.json",
            },
        }
        summary_rows: list[dict[str, object]] = []
        formula_clusters: ClusterIndex | None = None
        identity_clusters: ClusterIndex | None = None
        near_mask = np.asarray(panel.near_query, dtype=bool)
        with gzip.open(staging / "query_metrics.csv.gz", "wt", encoding="utf-8", newline="") as handle:
            query_writer = csv.DictWriter(handle, fieldnames=QUERY_FIELDS)
            query_writer.writeheader()
            for method_index, spec in enumerate(specs):
                print(f"[{spec.name}] validating {panel.n_molecule_edges:,} molecule scores", flush=True)
                active_scores = method_scores(score_store, spec)
                validate_scores(active_scores, panel.n_molecule_edges, args.validation_chunk_size)
                retrieval, near, outcomes = evaluate_queries(
                    panel, spec.name, active_scores, query_writer, args.progress_every
                )
                slug = _safe_slug(spec.name, method_index)
                roc_relative = Path("curves") / f"{slug}_roc.csv.gz"
                pr_relative = Path("curves") / f"{slug}_pr.csv.gz"
                print(f"[{spec.name}] sorting exact pooled ROC/PR thresholds", flush=True)
                micro = exact_micro_curves(
                    active_scores, panel.molecule_label,
                    staging / roc_relative, staging / pr_relative, args.sort_chunk_size,
                )
                report["methods"][spec.name] = {  # type: ignore[index]
                    "retrieval": retrieval, "near_subset": near,
                    "micro_candidate": micro,
                    "curves": {
                        "roc_csv_gz": roc_relative.as_posix(),
                        "pr_csv_gz": pr_relative.as_posix(),
                    },
                }
                summary_rows.append(_summary_row(spec.name, panel, retrieval, near, micro))
                if spec.name == reference_method:
                    reference_outcomes = outcomes
                else:
                    assert reference_outcomes is not None
                    if formula_clusters is None:
                        formula_clusters = make_cluster_index("query_formula", panel.query_formula)
                        identity_clusters = make_cluster_index("query_ik14", panel.query_ik14)
                    assert identity_clusters is not None
                    report["paired_comparisons"][spec.name] = paired_comparison(  # type: ignore[index]
                        reference_method, spec.name, reference_outcomes, outcomes, near_mask,
                        formula_clusters, identity_clusters, args.bootstrap_resamples,
                        args.bootstrap_seed + method_index * 1000,
                    )
                active_scores = None
                gc.collect()

        with (staging / "summary_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summary_rows)
        with (staging / "paired_comparisons.json").open("w", encoding="utf-8") as handle:
            json.dump(report["paired_comparisons"], handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        with (staging / "report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        staging.replace(output_dir)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        active_scores = None
        reference_outcomes = None
        panel = None
        gc.collect()
        if score_store is not None:
            score_store.close()
        if panel_store is not None:
            panel_store.close()
        gc.collect()
        shutil.rmtree(extraction, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> None:
    evaluate(arguments(argv))


if __name__ == "__main__":
    main()
