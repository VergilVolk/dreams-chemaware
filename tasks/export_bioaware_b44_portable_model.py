#!/usr/bin/env python
"""Export the frozen B44 sklearn expert to a version-independent NumPy tree ensemble."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]
from bioaware_portable_hgb import PortableBinaryHGB, export_binary_hist_gradient_boosting  # noqa: E402

FEATURES = [
    "spectral_score", "independent_member_count", "independent_member_intersection",
    "independent_log_degree_mean", "independent_log_degree_min",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=ROOT / "data/validation/bioaware_b44_catalogue_v1_localcheck_20260913_v1/model.joblib",
    )
    parser.add_argument(
        "--catalogue-lookup", type=Path,
        default=ROOT / "data/validation/bioaware_b44_catalogue_v1_localcheck_20260913_v1/catalogue_lookup.csv.gz",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/bioaware_b44_portable_model_20260913_v1.npz",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "data/validation/bioaware_b44_portable_model_20260913_v1.json",
    )
    args = parser.parse_args()
    for path in (args.model, args.catalogue_lookup):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    sklearn_model = joblib.load(args.model)
    source_hash = sha256(args.model)
    export_binary_hist_gradient_boosting(sklearn_model, args.output, FEATURES, source_hash)
    portable = PortableBinaryHGB(args.output, FEATURES)

    lookup = pd.read_csv(args.catalogue_lookup)
    spectral_grid = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    topology = lookup[FEATURES[1:]].to_numpy(np.float64)
    values = np.column_stack((
        np.tile(spectral_grid, len(topology)),
        np.repeat(topology, len(spectral_grid), axis=0),
    ))
    expected = sklearn_model.predict_proba(values)
    observed = portable.predict_proba(values)
    maximum_error = float(np.max(np.abs(expected - observed)))
    predicted_class_mismatches = int(np.sum(np.argmax(expected, axis=1) != np.argmax(observed, axis=1)))
    if maximum_error > 1e-14 or predicted_class_mismatches:
        raise RuntimeError(
            f"portable replay failed: error={maximum_error} class_mismatches={predicted_class_mismatches}"
        )
    report = {
        "status": "bioaware_b44_portable_model_exported",
        "source_joblib_sha256": source_hash,
        "portable_model_sha256": sha256(args.output),
        "features": FEATURES,
        "trees": int(len(portable.tree_offsets) - 1),
        "nodes": int(len(portable.value)),
        "replay_rows": int(len(values)),
        "maximum_probability_error": maximum_error,
        "predicted_class_mismatches": predicted_class_mismatches,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.report.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, args.report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
