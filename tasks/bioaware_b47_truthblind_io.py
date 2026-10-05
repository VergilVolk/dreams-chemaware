#!/usr/bin/env python
"""Small, dependency-light I/O helpers for the B47 truth-blind benchmark."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_mgf_records(path: Path) -> tuple[list[str], list[dict[str, object]]]:
    """Read MGF spectra while preserving TITLE as the only alignment key.

    Records contain ``precursor_mz`` and a float32 ``(2, n)`` peak array.  The
    function deliberately does not infer ordering from any external manifest.
    """
    titles: list[str] = []
    records: list[dict[str, object]] = []
    header: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    in_record = False

    def finish() -> None:
        nonlocal header, peaks, in_record
        if not in_record:
            raise RuntimeError(f"MGF END without BEGIN: {path}")
        title = header.get("TITLE", "").strip()
        precursor_text = header.get("PEPMASS", "").replace(",", " ").split()
        if not title or not precursor_text or not peaks:
            raise RuntimeError(f"incomplete MGF record in {path}: title={title!r}")
        precursor = float(precursor_text[0])
        peak_array = np.asarray(peaks, dtype=np.float32).T
        if not np.isfinite(precursor) or precursor <= 0:
            raise RuntimeError(f"invalid precursor for {title!r}: {precursor}")
        if peak_array.shape[0] != 2 or not np.isfinite(peak_array).all():
            raise RuntimeError(f"invalid peaks for {title!r}")
        if np.any(peak_array[0] <= 0) or np.any(peak_array[1] < 0):
            raise RuntimeError(f"nonphysical peak values for {title!r}")
        titles.append(title)
        records.append({"precursor_mz": precursor, "peaks": peak_array})
        header, peaks, in_record = {}, [], False

    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper == "BEGIN IONS":
                if in_record:
                    raise RuntimeError(f"nested MGF BEGIN: {path}")
                in_record, header, peaks = True, {}, []
            elif upper == "END IONS":
                finish()
            elif in_record and "=" in line:
                key, value = line.split("=", 1)
                header[key.strip().upper()] = value.strip()
            elif in_record:
                fields = line.split()
                if len(fields) < 2:
                    raise RuntimeError(f"invalid MGF peak line: {line!r}")
                peaks.append((float(fields[0]), float(fields[1])))
            else:
                raise RuntimeError(f"content outside MGF record: {line!r}")
    if in_record:
        raise RuntimeError(f"unterminated MGF record: {path}")
    if not titles:
        raise RuntimeError(f"empty MGF: {path}")
    if len(set(titles)) != len(titles):
        raise RuntimeError(f"duplicate MGF TITLE values: {path}")
    return titles, records

