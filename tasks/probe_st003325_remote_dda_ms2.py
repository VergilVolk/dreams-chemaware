#!/usr/bin/env python
"""Range-extract only the six ST003325 pooled-QC DDA mzML files and census MS2.

The public Workbench archive is 22.5 GB.  Its HTTP endpoint supports byte-range
requests, so this gate reads the ZIP directory and the selected compressed
members without downloading unrelated study files.  It does not inspect
perturbation effects or fit a model.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

import requests


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str, chunk_size: int = 8 << 20):
        self.url = url
        self.chunk_size = chunk_size
        self.position = 0
        self.session = requests.Session()
        response = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        response.raise_for_status()
        match = re.search(r"/(\d+)$", response.headers.get("Content-Range", ""))
        if response.status_code != 206 or not match:
            raise RuntimeError("remote archive does not provide a valid byte-range response")
        self.length = int(match.group(1))
        self.cache_start = 0
        self.cache = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = min(position, self.length)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        if size == 0:
            return b""
        cache_end = self.cache_start + len(self.cache)
        if not (self.cache_start <= self.position and self.position + size <= cache_end):
            start = self.position
            end = min(self.length - 1, start + max(size, self.chunk_size) - 1)
            response = self.session.get(
                self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=180
            )
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"range request returned HTTP {response.status_code}")
            self.cache_start = start
            self.cache = response.content
        offset = self.position - self.cache_start
        result = self.cache[offset:offset + size]
        if len(result) != size:
            raise EOFError(f"short range read: requested {size}, received {len(result)}")
        self.position += size
        return result

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_name(member: str) -> Path:
    parts = PurePosixPath(member).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe ZIP member: {member}")
    return Path(*parts)


def spectrum_census(path: Path) -> dict[str, int]:
    totals = {"spectra": 0, "ms1": 0, "ms2": 0, "positive_ms2": 0,
              "negative_ms2": 0, "ms2_with_selected_ion_mz": 0}
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.tag.rsplit("}", 1)[-1] != "spectrum":
            continue
        totals["spectra"] += 1
        params = {
            child.attrib.get("accession"): child.attrib.get("value", "")
            for child in element.iter()
            if child.tag.rsplit("}", 1)[-1] == "cvParam"
        }
        level = params.get("MS:1000511")
        if level == "1":
            totals["ms1"] += 1
        elif level == "2":
            totals["ms2"] += 1
            totals["positive_ms2"] += int("MS:1000130" in params)
            totals["negative_ms2"] += int("MS:1000129" in params)
            totals["ms2_with_selected_ion_mz"] += int("MS:1000744" in params)
        element.clear()
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--url", default="https://www.metabolomicsworkbench.org/studydownload/ST003325_Rawdata.zip"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    inventory = json.loads(args.inventory.read_text(encoding="utf-8-sig"))
    members = inventory["compressed_file_content"]["ST003325_Rawdata.zip"]
    expected = {
        str(row["name"]): int(row["size"])
        for row in members if re.search(r"Pool_QC_DDA_.*\.mzML$", str(row["name"]), re.I)
    }
    if len(expected) != 6:
        raise RuntimeError(f"expected six frozen DDA members, found {len(expected)}")

    remote = HTTPRangeReader(args.url)
    rows = []
    with zipfile.ZipFile(remote) as archive:
        names = set(archive.namelist())
        if set(expected) - names:
            raise RuntimeError(f"DDA members missing from live ZIP: {sorted(set(expected)-names)}")
        for member in sorted(expected):
            info = archive.getinfo(member)
            if info.file_size != expected[member]:
                raise RuntimeError(f"live ZIP size changed for {member}")
            destination = args.output / "mzml" / local_name(member)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1 << 20)
            if destination.stat().st_size != expected[member]:
                raise RuntimeError(f"extracted size mismatch for {member}")
            census = spectrum_census(destination)
            polarity = "positive" if "RPpos" in member else "negative"
            rows.append({
                "member": member,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "expected_polarity_from_name": polarity,
                **census,
            })
            print(json.dumps(rows[-1]), flush=True)

    positive = [row for row in rows if row["expected_polarity_from_name"] == "positive"]
    negative = [row for row in rows if row["expected_polarity_from_name"] == "negative"]
    gates = {
        "six_dda_files_extracted": len(rows) == 6,
        "three_positive_and_three_negative": len(positive) == len(negative) == 3,
        "all_files_have_ms2": all(row["ms2"] > 0 for row in rows),
        "all_ms2_have_selected_ion_mz": all(
            row["ms2_with_selected_ion_mz"] == row["ms2"] for row in rows),
        "polarity_matches_filename": all(
            (row["positive_ms2"] == row["ms2"] if row["expected_polarity_from_name"] == "positive"
             else row["negative_ms2"] == row["ms2"])
            for row in rows),
    }
    report = {
        "status": "ST003325_TARGETED_DDA_PROBE_COMPLETE",
        "formal": True,
        "dataset": "Metabolomics Workbench ST003325 / PR001844",
        "data_role": "development/consumed after metadata and raw DDA inspection",
        "sealed": False,
        "archive_url": args.url,
        "archive_bytes": remote.length,
        "archive_md5_from_repository": "c36568f353b09042ffa614db5af381a1",
        "files": rows,
        "totals": {
            "files": len(rows),
            "bytes": sum(row["bytes"] for row in rows),
            "spectra": sum(row["spectra"] for row in rows),
            "ms2": sum(row["ms2"] for row in rows),
            "positive_ms2": sum(row["positive_ms2"] for row in rows),
            "negative_ms2": sum(row["negative_ms2"] for row in rows),
        },
        "gates": gates,
        "pass_to_feature_spectrum_linkage": all(gates.values()),
        "provenance": {"inventory_sha256": sha256(args.inventory)},
        "claim_limit": (
            "Pooled-QC DDA scan availability only. No sample-specific MS2, feature-spectrum linkage, "
            "molecular identity, perturbation effect, public transfer, or model performance."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
