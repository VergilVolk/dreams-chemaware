"""Pure helpers for the full-MassSpecGym native Noise triplet library."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np


def decode_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8").astype(str)
    return values.astype(str)


def stable_u64(seed: int, *parts: object) -> int:
    payload = "|".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def ordered_unique(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for raw in values:
        value = int(raw)
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def diverse_by_similarity(
    rows: np.ndarray,
    similarities: np.ndarray,
    maximum: int,
    *,
    hard_first: bool,
) -> np.ndarray:
    """Keep the whole difficulty span rather than an unstable hardest-only tail."""
    rows = np.asarray(rows, dtype=np.int64)
    similarities = np.asarray(similarities, dtype=np.float64)
    if maximum < 1 or rows.ndim != 1 or similarities.shape != rows.shape:
        raise ValueError("invalid diverse-similarity selection")
    if len(rows) == 0:
        return rows.copy()
    if not np.all(np.isfinite(similarities)):
        raise RuntimeError("non-finite similarity reached triplet selection")
    order = np.argsort(-similarities if hard_first else similarities, kind="stable")
    if len(order) <= maximum:
        return rows[order]
    positions = np.unique(np.rint(np.linspace(0, len(order) - 1, maximum)).astype(int))
    return rows[order[positions]]


def strict_ppm_rows(
    sorted_rows: np.ndarray,
    sorted_mz: np.ndarray,
    precursor_mz: float,
    ppm: float,
) -> np.ndarray:
    if precursor_mz <= 0 or ppm <= 0:
        raise ValueError("precursor m/z and ppm must be positive")
    tolerance = float(precursor_mz) * float(ppm) * 1e-6
    left = int(np.searchsorted(sorted_mz, precursor_mz - tolerance, side="left"))
    right = int(np.searchsorted(sorted_mz, precursor_mz + tolerance, side="right"))
    return np.asarray(sorted_rows[left:right], dtype=np.int64)


def nearest_different_identity_rows(
    sorted_rows: np.ndarray,
    sorted_mz: np.ndarray,
    identities: np.ndarray,
    precursor_mz: float,
    query_identity: str,
    maximum: int,
) -> np.ndarray:
    """Deterministic nearest-mass fallback; never crosses adduct groups."""
    if maximum < 1:
        raise ValueError("nearest fallback maximum must be positive")
    insertion = int(np.searchsorted(sorted_mz, precursor_mz, side="left"))
    left, right = insertion - 1, insertion
    output: list[int] = []
    while len(output) < maximum and (left >= 0 or right < len(sorted_rows)):
        choose_left = right >= len(sorted_rows)
        if left >= 0 and right < len(sorted_rows):
            choose_left = abs(float(sorted_mz[left]) - precursor_mz) <= abs(
                float(sorted_mz[right]) - precursor_mz
            )
        position = left if choose_left else right
        left -= int(choose_left)
        right += int(not choose_left)
        row = int(sorted_rows[position])
        if str(identities[row]) != str(query_identity):
            output.append(row)
    return np.asarray(ordered_unique(output), dtype=np.int64)


def query_rotation_positions(
    event_queries: np.ndarray,
    epoch: int,
    seed: int,
) -> np.ndarray:
    """Choose exactly one stored triplet event per query and rotate by epoch."""
    event_queries = np.asarray(event_queries, dtype=np.int64)
    if event_queries.ndim != 1 or len(event_queries) == 0 or np.any(event_queries < 0):
        raise ValueError("invalid event-query ledger")
    positions: list[int] = []
    for query in np.unique(event_queries):
        members = np.flatnonzero(event_queries == query)
        start = stable_u64(seed, int(query), "rotation") % len(members)
        positions.append(int(members[(start + int(epoch)) % len(members)]))
    output = np.asarray(positions, dtype=np.int64)
    if len(output) != len(np.unique(event_queries)):
        raise RuntimeError("query rotation did not produce one event per query")
    return output


def query_disjoint_batches(
    selected_positions: np.ndarray,
    event_queries: np.ndarray,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    selected_positions = np.asarray(selected_positions, dtype=np.int64)
    event_queries = np.asarray(event_queries, dtype=np.int64)
    if batch_size < 2 or len(selected_positions) == 0:
        raise ValueError("invalid native batch configuration")
    if np.any((selected_positions < 0) | (selected_positions >= len(event_queries))):
        raise RuntimeError("selected event position is out of range")
    selected_queries = event_queries[selected_positions]
    if len(np.unique(selected_queries)) != len(selected_queries):
        raise RuntimeError("query-balanced selection duplicated a query")
    rng = np.random.default_rng(stable_u64(seed, epoch, "batch-order"))
    order = selected_positions[rng.permutation(len(selected_positions))]
    return [
        list(map(int, order[left:left + batch_size]))
        for left in range(0, len(order), batch_size)
    ]
