#!/usr/bin/env python
"""Fail-closed validator for the frozen E2 matrix manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    manifest_path = directory / "e2_manifest.json"
    cells_path = directory / "e2_preregistered_cells.csv"
    for path in (manifest_path, cells_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = pd.read_csv(cells_path)
    if manifest.get("status") != "noise_final_e2_matrix_manifest_frozen" or not manifest.get("formal"):
        raise RuntimeError("E2 manifest is not formal")
    if len(cells) != manifest["cells"] or cells["cell_id"].nunique() != len(cells):
        raise RuntimeError("E2 cell cardinality mismatch")
    if sha256(cells_path) != manifest["provenance"]["cells_sha256"]:
        raise RuntimeError("E2 cell hash mismatch")
    if cells["p2b_allowed"].astype(bool).any() or cells["selection_uses_action_outcome"].astype(bool).any():
        raise RuntimeError("E2 contains forbidden information")
    if not {"corrective", "robustness", "negative_control"}.issubset(set(cells["arm"])):
        raise RuntimeError("E2 is missing a required arm")
    if cells.loc[cells["arm"].eq("corrective"), "selector"].nunique() < 4:
        raise RuntimeError("E2 corrective selector space is incomplete")
    if len(manifest["eligible_acquisition_relations"]) < 2:
        raise RuntimeError("E2 empirical condition coverage is insufficient")
    print(json.dumps({
        "status": "noise_final_e2_matrix_manifest_validation_passed",
        "cells": len(cells),
        "corrective": int(cells["arm"].eq("corrective").sum()),
        "robustness": int(cells["arm"].eq("robustness").sum()),
        "negative_controls": int(cells["arm"].eq("negative_control").sum()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
