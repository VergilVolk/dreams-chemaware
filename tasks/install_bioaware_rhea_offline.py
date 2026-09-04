#!/usr/bin/env python
"""Atomically install the frozen BioAware Rhea cache without network access."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED = {
    "rhea_participants.csv.gz": "ab8ecb5515c35c042d055bf0cac7035b9f7c81771e3214cec4959d6f7001b556",
    "rhea_reactions.csv.gz": "93aefc03df8791e51b8954920ea16b9805a08ad16679d703542277f506ed567c",
}
REQUIRED = set(EXPECTED) | {
    "report.json",
    "rhea-directions.tsv",
    "rhea-reaction-smiles.tsv",
    "rhea2reactome.tsv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("bioaware_rhea_cache_20260827.zip"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reference/bioaware_rhea_offline_20260827"),
    )
    args = parser.parse_args()
    archive = args.archive.resolve()
    output = args.output_dir.resolve()
    if not archive.exists():
        raise FileNotFoundError(archive)
    if output.exists():
        raise RuntimeError(f"fail-closed: output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".bioaware_rhea_", dir=output.parent))
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            for member in members:
                name = PurePosixPath(member.filename)
                if name.is_absolute() or ".." in name.parts:
                    raise RuntimeError(f"unsafe archive member: {member.filename}")
                if member.is_dir():
                    continue
                target = temporary / name.name
                if target.name not in REQUIRED:
                    raise RuntimeError(f"unexpected archive member: {member.filename}")
                if target.exists():
                    raise RuntimeError(f"duplicate archive member: {target.name}")
                with bundle.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        present = {path.name for path in temporary.iterdir() if path.is_file()}
        if present != REQUIRED:
            raise RuntimeError(f"offline cache members mismatch: missing={sorted(REQUIRED-present)} extra={sorted(present-REQUIRED)}")
        for name, expected in EXPECTED.items():
            observed = sha256(temporary / name)
            if observed != expected:
                raise RuntimeError(f"hash mismatch for {name}: {observed} != {expected}")
        report = json.loads((temporary / "report.json").read_text(encoding="utf-8"))
        if report.get("status") != "bioaware_rhea_cache_complete":
            raise RuntimeError("unexpected bundled Rhea report status")
        os.replace(temporary, output)
        print(
            json.dumps(
                {
                    "status": "bioaware_rhea_offline_install_complete",
                    "archive": str(archive),
                    "archive_sha256": sha256(archive),
                    "output_dir": str(output),
                    "participants_sha256": EXPECTED["rhea_participants.csv.gz"],
                    "reactions_sha256": EXPECTED["rhea_reactions.csv.gz"],
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
