"""Fail-closed deployment audit for local Showspace assets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from inference_utils import _sha256, available_model_types


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    database = Path(os.environ["DREAMS_MOLECULE_DB"])
    if not database.is_file():
        raise FileNotFoundError(database)
    database_hash = sha256(database)
    registry = available_model_types()
    reports = {}
    for model_type, entry in registry.items():
        if not entry["ready"]:
            reports[model_type] = {"ready": False, "reason": "checkpoint_not_configured"}
            continue
        prefix = {"official_dreams": "DREAMS_OFFICIAL", "e4a_shared": "DREAMS_E4", "e8_shared": "DREAMS_E8"}[model_type]
        index_path = Path(os.environ.get(f"{prefix}_EMBEDDING_INDEX", ""))
        manifest_path = Path(os.environ.get(f"{prefix}_EMBEDDING_INDEX_MANIFEST", ""))
        if not index_path.is_file() or not manifest_path.is_file():
            reports[model_type] = {"ready": False, "reason": "index_not_configured"}
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_hash = _sha256(Path(entry["checkpoint"]))
        index_shape = list(np.load(index_path, mmap_mode="r").shape)
        gates = {
            "status": manifest.get("status") == "showspace_model_aligned_embedding_index",
            "complete": manifest.get("complete") is True,
            "model_type": manifest.get("model_type") == model_type,
            "checkpoint_fingerprint": manifest.get("model_fingerprint") == checkpoint_hash,
            "hdf5_fingerprint": manifest.get("source_hdf5_sha256") == database_hash,
            "shape": manifest.get("shape") == [manifest.get("source_rows"), 1024],
            "index_file": index_shape == manifest.get("shape"),
        }
        reports[model_type] = {"ready": all(gates.values()), "gates": gates, "manifest": str(manifest_path)}
    if not reports.get("official_dreams", {}).get("ready"):
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        raise RuntimeError("official_dreams deployment contract failed")
    print(json.dumps({"status": "showspace_deployment_preflight_passed", "models": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
