#!/usr/bin/env python
"""Immutable pair-score cache contract for the GNPS Gold/Silver benchmark.

The cache stores one score for every directed query-reference spectrum edge in
the frozen panel order.  It is deliberately agnostic to how a score was
produced: a classical similarity, a public neural model, or a project model can
all be evaluated by the same downstream code.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


SCHEMA = "gnps_gold_silver_10ppm_pair_score_cache_v1"
PANELS = ("identity_disjoint", "formula_disjoint")


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def benchmark_fingerprint(benchmark: Path) -> dict[str, str]:
    report = benchmark / "report.json"
    manifest = benchmark / "manifest.csv.gz"
    required = [report, manifest]
    required.extend(benchmark / f"panel_{name}.npz" for name in PANELS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"benchmark files are missing: {missing}")
    return {
        "report_sha256": sha256_file(report),
        "manifest_sha256": sha256_file(manifest),
        **{
            f"panel_{name}_sha256": sha256_file(benchmark / f"panel_{name}.npz")
            for name in PANELS
        },
    }


def expected_pair_count(panel: Path) -> int:
    with np.load(panel, allow_pickle=False) as body:
        return int(len(body["candidate_row"]))


def write_pair_score_cache(
    output: Path,
    benchmark: Path,
    method: dict[str, object],
    scores: dict[str, np.ndarray],
) -> dict[str, object]:
    """Atomically write a complete two-panel score cache."""
    if output.exists():
        raise FileExistsError(output)
    if set(scores) != set(PANELS):
        raise RuntimeError(f"score panels must be exactly {PANELS}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    report: dict[str, object] = {
        "schema": SCHEMA,
        "method": method,
        "benchmark": benchmark_fingerprint(benchmark),
        "panels": {},
    }
    try:
        for name in PANELS:
            value = np.asarray(scores[name], dtype=np.float32)
            expected = expected_pair_count(benchmark / f"panel_{name}.npz")
            if value.ndim != 1 or len(value) != expected:
                raise RuntimeError(
                    f"{name} score shape {value.shape} does not match {expected} edges"
                )
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"{name} pair scores are non-finite")
            filename = f"pair_scores_{name}.npy"
            np.save(staging / filename, np.ascontiguousarray(value), allow_pickle=False)
            report["panels"][name] = {
                "file": filename,
                "pairs": expected,
                "sha256": sha256_file(staging / filename),
                "minimum": float(value.min()),
                "maximum": float(value.max()),
            }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def load_pair_score_cache(
    cache: Path,
    benchmark: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load a cache only after byte-level benchmark and score validation."""
    report = json.loads((cache / "report.json").read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA:
        raise RuntimeError(f"unsupported pair-score cache schema: {cache}")
    observed_benchmark = benchmark_fingerprint(benchmark)
    if report.get("benchmark") != observed_benchmark:
        raise RuntimeError(f"pair-score cache benchmark provenance mismatch: {cache}")
    scores: dict[str, np.ndarray] = {}
    if set(report.get("panels", {})) != set(PANELS):
        raise RuntimeError(f"pair-score cache does not contain both frozen panels: {cache}")
    for name in PANELS:
        entry = report["panels"][name]
        path = cache / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"pair-score cache checksum mismatch: {path}")
        # Pair-score panels are small (< 1 MB for the current benchmark).  Load
        # them eagerly so Windows does not retain mmap file handles and so the
        # validated bytes cannot change underneath a running evaluation.
        value = np.array(np.load(path, allow_pickle=False), dtype=np.float32, copy=True)
        expected = expected_pair_count(benchmark / f"panel_{name}.npz")
        if value.ndim != 1 or len(value) != expected or int(entry["pairs"]) != expected:
            raise RuntimeError(f"pair-score cache length mismatch: {path}")
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"pair-score cache is non-finite: {path}")
        scores[name] = value
    return report, scores
