"""Acquire pinned public fragmentation resources used by ChemAware.

The first release intentionally acquires only the versioned MS2LDA MotifDB
artifact.  It is small enough to checksum exactly and contains curated
fragment/neutral-loss co-occurrence motifs.  Larger MassBank releases are
handled by a separate record-level importer so that per-record licences are
not erased by a bulk download step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MS2LDA_COMMIT = "cc4bfd16e81698a89f9bc7e378813b4faacec7aa"
MOTIFDB_URL = (
    "https://raw.githubusercontent.com/vdhooftcompmet/MS2LDA/"
    f"{MS2LDA_COMMIT}/MS2LDA/MotifDB/motifDB.json"
)
MOTIFDB_SHA256 = "e4941fc7cbc0834d1461e16085d0bf8dd81533a42965852a62c4723e921bce94"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "data/validation/chemaware_public_fragmentation_sources_v1_20260928"
        ),
    )
    parser.add_argument(
        "--motifdb-source",
        type=Path,
        help="Optional already-downloaded motifDB.json; the pinned hash is still required.",
    )
    return parser.parse_args()


def acquire(source: Path | None, destination: Path) -> None:
    if source is not None:
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, destination)
        return
    request = urllib.request.Request(
        MOTIFDB_URL,
        headers={"User-Agent": "ChemAware-public-resource-acquisition/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_public_sources_", dir=args.output.parent))
    try:
        motif_dir = temporary / "ms2lda_motifdb"
        motif_dir.mkdir()
        motif_path = motif_dir / "motifDB.json"
        acquire(args.motifdb_source, motif_path)
        observed_hash = sha256_file(motif_path)
        if observed_hash != MOTIFDB_SHA256:
            raise RuntimeError(
                f"MotifDB checksum mismatch: expected={MOTIFDB_SHA256} observed={observed_hash}"
            )
        body = json.loads(motif_path.read_text(encoding="utf-8"))
        if not isinstance(body, dict) or not isinstance(body.get("ms2"), list):
            raise RuntimeError("pinned MotifDB schema is not recognized")
        manifest = {
            "status": "CHEMAWARE_PUBLIC_FRAGMENTATION_SOURCES_COMPLETE",
            "sources": {
                "ms2lda_motifdb": {
                    "repository": "https://github.com/vdhooftcompmet/MS2LDA",
                    "commit": MS2LDA_COMMIT,
                    "relative_path": "MS2LDA/MotifDB/motifDB.json",
                    "download_url": MOTIFDB_URL,
                    "sha256": observed_hash,
                    "bytes": motif_path.stat().st_size,
                    "ms2_motifs": len(body["ms2"]),
                    "license": "MIT for the MS2LDA code/repository; retain motif-level provenance",
                    "scientific_role": (
                        "curated fragment and neutral-loss co-occurrence hypotheses; "
                        "not identity truth and not a training target"
                    ),
                }
            },
        }
        (temporary / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
