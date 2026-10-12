#!/usr/bin/env python
"""Stream pooled-QC mzXML and build structure-hidden feature--MS2 links."""
from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


DATASET = "MSV000096214"
BASE = "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?"
FILES = {
    "SJ-Mutant-Flavo-CQ2-NEG.mzXML": 369678418,
    "SJ-Mutant-Flavo-CQ2-POS.mzXML": 283126077,
    "SJ-Mutant-Flavo-CQ3-NEG.mzXML": 366876719,
    "SJ-Mutant-Flavo-CQ3-POS.mzXML": 269519504,
    "SJ-Mutant-Flavo-CQ4-NEG.mzXML": 361823461,
    "SJ-Mutant-Flavo-CQ4-POS.mzXML": 263572893,
}
PRIMARY_PPM = 10.0
PRIMARY_RT_SEC = 6.0


class HashingReader:
    def __init__(self, response):
        self.response = response
        self.hasher = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        value = self.response.read(size)
        self.hasher.update(value)
        self.bytes_read += len(value)
        return value

    def close(self) -> None:
        self.response.close()


def open_remote(filename: str):
    url = BASE + urllib.parse.urlencode({"file": f"f.{DATASET}/peak/{filename}"})
    for attempt in range(6):
        request = urllib.request.Request(url, headers={"User-Agent": "DreaMS-biology/1.0"})
        try:
            return urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
            retry_after = error.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.isdigit() else 5.0 * (attempt + 1)
            time.sleep(min(delay, 30.0))
    raise RuntimeError(filename)


def rt_seconds(value: str) -> float:
    match = re.fullmatch(r"PT(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?", value)
    if match is None:
        raise ValueError(f"unsupported retention time: {value}")
    hours, minutes, seconds = (float(item or 0.0) for item in match.groups())
    return hours * 3600.0 + minutes * 60.0 + seconds


def decode_peaks(node: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    encoded = (node.text or "").strip()
    if not encoded:
        return np.empty(0), np.empty(0)
    raw = base64.b64decode(encoded)
    if node.attrib.get("compressionType", "none").lower() == "zlib":
        raw = zlib.decompress(raw)
    precision = int(node.attrib.get("precision", "32"))
    byte_order = node.attrib.get("byteOrder", "network").lower()
    endian = ">" if byte_order in {"network", "big"} else "<"
    dtype = np.dtype(endian + ("f8" if precision == 64 else "f4"))
    values = np.frombuffer(raw, dtype=dtype)
    if len(values) % 2:
        raise ValueError("odd mzXML peak array")
    pairs = values.reshape(-1, 2).astype(np.float64)
    mz, intensity = pairs[:, 0], pairs[:, 1]
    keep = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    order = np.argsort(mz[keep])
    return mz[keep][order], intensity[keep][order]


def peak_hash(mz: np.ndarray, intensity: np.ndarray) -> str:
    normalized = intensity / np.max(intensity)
    text = "\n".join(
        f"{mass:.6f}\t{height:.6f}" for mass, height in zip(mz, normalized, strict=True)
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def candidate_matches(
    precursor: float,
    scan_rt: float,
    feature_mz: np.ndarray,
    feature_rt: np.ndarray,
    ppm_limit: float,
    rt_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ppm = np.abs(feature_mz - precursor) / feature_mz * 1e6
    delta_rt = np.abs(feature_rt - scan_rt)
    matched = np.flatnonzero((ppm <= ppm_limit) & (delta_rt <= rt_limit))
    if len(matched) == 0:
        return matched, ppm, delta_rt
    distance = np.sqrt((ppm[matched] / ppm_limit) ** 2 + (delta_rt[matched] / rt_limit) ** 2)
    return matched[np.argsort(distance)], ppm, delta_rt


def write_mgf_block(handle, record: dict, mz: np.ndarray, intensity: np.ndarray) -> None:
    handle.write("BEGIN IONS\n")
    handle.write(f"TITLE={record['spectrum_id']}\n")
    handle.write(f"SCANS={record['scan']}\n")
    handle.write(f"PEPMASS={record['precursor_mz']:.8f}\n")
    handle.write(f"RTINSECONDS={record['rt_sec']:.6f}\n")
    handle.write(f"FEATURE_ID={record['feature_id']}\n")
    handle.write(f"ION_MODE={record['ion_mode']}\n")
    handle.write(f"COLLISION_ENERGY={record['collision_energy']}\n")
    for mass, height in zip(mz, intensity, strict=True):
        handle.write(f"{mass:.8f} {height:.8g}\n")
    handle.write("END IONS\n\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    table = pd.read_csv(args.feature_table)
    required = {"feature_id", "ion_mode", "mz", "rt_min"}
    if not required.issubset(table.columns):
        raise ValueError(f"missing structure-hidden columns: {required - set(table.columns)}")
    feature = {}
    for ion_mode in ("ESI+", "ESI-"):
        part = table.loc[table["ion_mode"] == ion_mode].reset_index(drop=True)
        feature[ion_mode] = {
            "ids": part["feature_id"].astype(str).to_numpy(),
            "mz": part["mz"].to_numpy(float),
            "rt_sec": part["rt_min"].to_numpy(float) * 60.0,
        }

    links: list[dict] = []
    source_records = []
    coverage = {
        "5ppm_3sec": set(),
        "10ppm_6sec": set(),
        "20ppm_12sec": set(),
    }
    mgf_path = args.output / "linked_spectra.mgf"
    with mgf_path.open("w", encoding="utf-8", newline="\n") as mgf:
        for filename, expected_size in FILES.items():
            ion_mode = "ESI+" if "-POS." in filename else "ESI-"
            arrays = feature[ion_mode]
            response = open_remote(filename)
            reader = HashingReader(response)
            total_ms2 = linked_ms2 = invalid_ms2 = 0
            try:
                for _, node in ET.iterparse(reader, events=("end",)):
                    if not node.tag.endswith("scan"):
                        continue
                    if int(node.attrib.get("msLevel", "1")) != 2:
                        node.clear()
                        continue
                    total_ms2 += 1
                    precursor_node = next((child for child in node if child.tag.endswith("precursorMz")), None)
                    peaks_node = next((child for child in node if child.tag.endswith("peaks")), None)
                    if precursor_node is None or peaks_node is None or not (precursor_node.text or "").strip():
                        invalid_ms2 += 1
                        node.clear()
                        continue
                    precursor = float(precursor_node.text)
                    scan_rt = rt_seconds(node.attrib["retentionTime"])
                    sensitivity = (
                        ("5ppm_3sec", 5.0, 3.0),
                        ("10ppm_6sec", 10.0, 6.0),
                        ("20ppm_12sec", 20.0, 12.0),
                    )
                    primary_order = np.empty(0, dtype=int)
                    primary_ppm = primary_rt = None
                    for name, ppm_limit, rt_limit in sensitivity:
                        order, ppm, delta_rt = candidate_matches(
                            precursor, scan_rt, arrays["mz"], arrays["rt_sec"], ppm_limit, rt_limit
                        )
                        if len(order):
                            coverage[name].update(arrays["ids"][order].tolist())
                        if name == "10ppm_6sec":
                            primary_order, primary_ppm, primary_rt = order, ppm, delta_rt
                    if len(primary_order) == 0:
                        node.clear()
                        continue
                    mz, intensity = decode_peaks(peaks_node)
                    if len(mz) < 5:
                        invalid_ms2 += 1
                        node.clear()
                        continue
                    linked_ms2 += 1
                    chosen = int(primary_order[0])
                    spectrum_id = f"{filename}::scan={node.attrib.get('num', total_ms2)}"
                    normalized_hash = peak_hash(mz, intensity)
                    second_distance = None
                    if len(primary_order) > 1:
                        second = int(primary_order[1])
                        second_distance = math.sqrt(
                            (float(primary_ppm[second]) / PRIMARY_PPM) ** 2
                            + (float(primary_rt[second]) / PRIMARY_RT_SEC) ** 2
                        )
                    record = {
                        "spectrum_id": spectrum_id,
                        "source_file": filename,
                        "scan": int(node.attrib.get("num", total_ms2)),
                        "ion_mode": ion_mode,
                        "feature_id": str(arrays["ids"][chosen]),
                        "precursor_mz": precursor,
                        "rt_sec": scan_rt,
                        "feature_mz": float(arrays["mz"][chosen]),
                        "feature_rt_sec": float(arrays["rt_sec"][chosen]),
                        "abs_ppm": float(primary_ppm[chosen]),
                        "abs_rt_sec": float(primary_rt[chosen]),
                        "candidate_features_in_window": int(len(primary_order)),
                        "second_best_normalized_distance": second_distance,
                        "peak_count": int(len(mz)),
                        "tic": float(np.sum(intensity)),
                        "collision_energy": node.attrib.get("collisionEnergy", ""),
                        "normalized_peak_hash": normalized_hash,
                    }
                    links.append(record)
                    write_mgf_block(mgf, record, mz, intensity)
                    node.clear()
            finally:
                reader.close()
            if reader.bytes_read != expected_size:
                raise IOError(f"{filename}: expected {expected_size} bytes, streamed {reader.bytes_read}")
            source_records.append({
                "filename": filename,
                "bytes": reader.bytes_read,
                "sha256": reader.hasher.hexdigest(),
                "ion_mode": ion_mode,
                "ms2_scans": total_ms2,
                "linked_ms2_scans": linked_ms2,
                "invalid_ms2_scans": invalid_ms2,
            })
            print(json.dumps(source_records[-1]), flush=True)
            if linked_ms2 == 0:
                failure = {
                    "status": "MSV000096214_POOLEDQC_FEATURE_MS2_LINKAGE_FAILED",
                    "formal": True,
                    "dataset": DATASET,
                    "data_role": "development/consumed",
                    "failed_source": source_records[-1],
                    "frozen_rule": {"precursor_ppm": PRIMARY_PPM, "rt_half_window_sec": PRIMARY_RT_SEC},
                    "pass_to_frozen_spectral_increment": False,
                    "reason": (
                        "Zero links under the frozen rule. Stop rather than widening tolerances after "
                        "observing coverage; use the independently frozen raw-feature reconstruction route."
                    ),
                }
                (args.output / "FAILED_report.json").write_text(
                    json.dumps(failure, indent=2), encoding="utf-8"
                )
                raise RuntimeError(f"zero frozen links in {filename}")

    frame = pd.DataFrame(links)
    with (args.output / "feature_spectrum_links.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(frame.to_csv(index=False, quoting=csv.QUOTE_MINIMAL).encode("utf-8"))
    support = frame.groupby("feature_id").agg(
        linked_spectra=("spectrum_id", "nunique"),
        pooled_qc_files=("source_file", "nunique"),
        unique_peak_hashes=("normalized_peak_hash", "nunique"),
        min_abs_ppm=("abs_ppm", "min"),
        min_abs_rt_sec=("abs_rt_sec", "min"),
        ambiguous_scans=("candidate_features_in_window", lambda values: int((values > 1).sum())),
    ).reset_index()
    support = table[["feature_id", "ion_mode"]].merge(support, on="feature_id", how="left").fillna(0)
    with (args.output / "feature_support.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(support.to_csv(index=False, quoting=csv.QUOTE_MINIMAL).encode("utf-8"))

    covered = support["linked_spectra"] > 0
    covered_count = int(covered.sum())
    replicate_supported = int((support.loc[covered, "pooled_qc_files"] >= 2).sum())
    positive_covered = int(((support["ion_mode"] == "ESI+") & covered).sum())
    negative_covered = int(((support["ion_mode"] == "ESI-") & covered).sum())
    gates = {
        "at_least_300_of_920_features_covered": covered_count >= 300,
        "at_least_100_positive_features_covered": positive_covered >= 100,
        "at_least_100_negative_features_covered": negative_covered >= 100,
        "at_least_70pct_covered_features_supported_by_two_qc_files": (
            replicate_supported / covered_count >= 0.70 if covered_count else False
        ),
        "coverage_monotonic_across_frozen_windows": (
            len(coverage["5ppm_3sec"]) <= len(coverage["10ppm_6sec"])
            <= len(coverage["20ppm_12sec"])
        ),
    }
    report = {
        "status": "MSV000096214_POOLEDQC_FEATURE_MS2_LINKAGE_COMPLETE",
        "formal": True,
        "dataset": DATASET,
        "data_role": "development/consumed",
        "sealed": False,
        "structure_annotation_or_genotype_effect_used": False,
        "feature_universe": {
            "features": int(len(table)),
            "positive": int((table["ion_mode"] == "ESI+").sum()),
            "negative": int((table["ion_mode"] == "ESI-").sum()),
            "author_processed_table": True,
        },
        "source_files": source_records,
        "primary_linkage_10ppm_6sec": {
            "linked_ms2_scans": int(len(frame)),
            "covered_features": covered_count,
            "positive_covered_features": positive_covered,
            "negative_covered_features": negative_covered,
            "features_supported_by_at_least_two_qc_files": replicate_supported,
            "fraction_covered_supported_by_at_least_two_qc_files": (
                replicate_supported / covered_count if covered_count else 0.0
            ),
            "ambiguous_scans": int((frame["candidate_features_in_window"] > 1).sum()),
            "unique_peak_hashes": int(frame["normalized_peak_hash"].nunique()),
        },
        "coverage_sensitivity": {name: len(values) for name, values in coverage.items()},
        "gates": gates,
        "pass_to_frozen_spectral_increment": all(gates.values()),
        "provenance": {
            "feature_table_sha256": hashlib.sha256(args.feature_table.read_bytes()).hexdigest(),
            "feature_spectrum_links_sha256": hashlib.sha256(
                (args.output / "feature_spectrum_links.csv.gz").read_bytes()
            ).hexdigest(),
            "linked_spectra_mgf_sha256": hashlib.sha256(mgf_path.read_bytes()).hexdigest(),
        },
        "claim_limit": (
            "Truth-blind pooled-QC feature--MS2 linkage only. Coverage is not algorithmic or biological "
            "increment, and the author-processed 920-feature universe is not an end-to-end unbiased raw feature table."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
