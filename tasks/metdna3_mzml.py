#!/usr/bin/env python
"""Small dependency-light mzML metadata reader for the MetDNA3 benchmark.

The development workstation does not necessarily have the optional ``psims``
dependency required by pyteomics.  This module streams only spectrum metadata
with lxml, so it neither materialises nor decodes the large binary arrays.
"""
from __future__ import annotations

import base64
import math
import zlib
from pathlib import Path
from typing import Iterator

import numpy as np
from lxml import etree


NS = "{http://psi.hupo.org/ms/mzml}"


def _number(value: str | None) -> float:
    try:
        return float(value) if value is not None else math.nan
    except (TypeError, ValueError):
        return math.nan


def iter_spectrum_metadata(path: Path) -> Iterator[dict]:
    """Yield compact metadata for every spectrum in *path*.

    Retention time is normalised to seconds.  ``defaultArrayLength`` is used as
    the peak count; binary payloads are deliberately not decoded in preflight.
    """
    context = etree.iterparse(
        str(path), events=("end",), tag=f"{NS}spectrum", huge_tree=True
    )
    try:
        for _, spectrum in context:
            row = {
                "spectrum_id": str(spectrum.get("id", "")),
                "ms_level": 0,
                "rt_sec": math.nan,
                "precursor_mz": math.nan,
                "charge": math.nan,
                "collision_energy": math.nan,
                "n_peaks": int(spectrum.get("defaultArrayLength", "0") or 0),
            }
            for cv in spectrum.iterfind(f".//{NS}cvParam"):
                accession = cv.get("accession")
                value = cv.get("value")
                if accession == "MS:1000511":
                    row["ms_level"] = int(_number(value))
                elif accession == "MS:1000016":
                    rt = _number(value)
                    unit = cv.get("unitAccession")
                    if unit == "UO:0000031":  # minute
                        rt *= 60.0
                    elif unit not in (None, "", "UO:0000010"):  # second
                        raise RuntimeError(f"unsupported retention-time unit {unit}")
                    row["rt_sec"] = rt
                elif accession == "MS:1000744":
                    row["precursor_mz"] = _number(value)
                elif accession == "MS:1000041":
                    row["charge"] = _number(value)
                elif accession == "MS:1000045":
                    row["collision_energy"] = _number(value)
            yield row
            spectrum.clear()
            while spectrum.getprevious() is not None:
                del spectrum.getparent()[0]
    finally:
        del context


def _binary_array(element: etree._Element) -> tuple[str, np.ndarray]:
    accessions = {
        cv.get("accession") for cv in element.iterfind(f".//{NS}cvParam")
    }
    if "MS:1000514" in accessions:
        kind = "mz"
    elif "MS:1000515" in accessions:
        kind = "intensity"
    else:
        raise RuntimeError("unsupported mzML binary-array kind")
    if "MS:1000521" in accessions:
        dtype = np.dtype("<f4")
    elif "MS:1000523" in accessions:
        dtype = np.dtype("<f8")
    else:
        raise RuntimeError("mzML array has no supported float precision")
    unsupported = accessions & {"MS:1002312", "MS:1002313", "MS:1002314"}
    if unsupported:
        raise RuntimeError(f"Numpress-compressed mzML is unsupported: {unsupported}")
    node = element.find(f"{NS}binary")
    encoded = "" if node is None or node.text is None else node.text.strip()
    raw = base64.b64decode(encoded)
    if "MS:1000574" in accessions:
        raw = zlib.decompress(raw)
    elif "MS:1000576" not in accessions:
        raise RuntimeError("mzML array has unknown compression")
    return kind, np.frombuffer(raw, dtype=dtype).astype(np.float32, copy=False)


def iter_ms2_spectra(path: Path) -> Iterator[dict]:
    """Yield decoded MS2 spectra using the same metadata semantics as preflight."""
    context = etree.iterparse(
        str(path), events=("end",), tag=f"{NS}spectrum", huge_tree=True
    )
    try:
        for _, spectrum in context:
            level = 0
            rt_sec = math.nan
            precursor_mz = math.nan
            charge = math.nan
            collision_energy = math.nan
            for cv in spectrum.iterfind(f".//{NS}cvParam"):
                accession = cv.get("accession")
                value = cv.get("value")
                if accession == "MS:1000511":
                    level = int(_number(value))
                elif accession == "MS:1000016":
                    rt_sec = _number(value)
                    if cv.get("unitAccession") == "UO:0000031":
                        rt_sec *= 60.0
                elif accession == "MS:1000744":
                    precursor_mz = _number(value)
                elif accession == "MS:1000041":
                    charge = _number(value)
                elif accession == "MS:1000045":
                    collision_energy = _number(value)
            if level == 2:
                arrays = dict(
                    _binary_array(item)
                    for item in spectrum.iterfind(f".//{NS}binaryDataArray")
                )
                mz = arrays.get("mz", np.empty(0, dtype=np.float32))
                intensity = arrays.get("intensity", np.empty(0, dtype=np.float32))
                if len(mz) != len(intensity):
                    raise RuntimeError(
                        f"m/z and intensity lengths differ in {path}:{spectrum.get('id')}"
                    )
                yield {
                    "spectrum_id": str(spectrum.get("id", "")),
                    "rt_sec": rt_sec,
                    "precursor_mz": precursor_mz,
                    "charge": charge,
                    "collision_energy": collision_energy,
                    "mz": mz,
                    "intensity": intensity,
                }
            spectrum.clear()
            while spectrum.getprevious() is not None:
                del spectrum.getparent()[0]
    finally:
        del context
