#!/usr/bin/env python
"""Freeze the sealed B44 panel, expert, and only required spectra into a portable bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b11_catalog_interaction_action import atomic_json, sha256  # noqa: E402
from bioaware_portable_hgb import PortableBinaryHGB  # noqa: E402
from pilot_paired_layer_cka import preprocess_spectrum  # noqa: E402

FEATURES = [
    "spectral_score", "independent_member_count", "independent_member_intersection",
    "independent_log_degree_mean", "independent_log_degree_min",
]


PANEL_FILES = ("report.json", "queries.csv.gz", "candidate_references.csv.gz", "reference_library.csv.gz")
SOURCE_ARTIFACT_FILES = ("report.json", "model.joblib", "catalogue_lookup.csv.gz", "oof_gate_ledger.csv.gz")
PORTABLE_ARTIFACT_FILES = ("report.json", "catalogue_lookup.csv.gz", "oof_gate_ledger.csv.gz")


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b44_massbank_panel_localcheck_20260913_v4",
    )
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b44_catalogue_v1_localcheck_20260913_v1",
    )
    parser.add_argument(
        "--massbank-hdf5", type=Path,
        default=ROOT / "data/massbank/massbank_full.hdf5",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b44_portable_bundle_20260913_v3",
    )
    parser.add_argument(
        "--portable-model", type=Path,
        default=ROOT / "data/validation/bioaware_b44_portable_model_20260913_v1.npz",
    )
    parser.add_argument(
        "--portable-model-report", type=Path,
        default=ROOT / "data/validation/bioaware_b44_portable_model_20260913_v1.json",
    )
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    for directory, files in ((args.panel_dir, PANEL_FILES), (args.artifact_dir, SOURCE_ARTIFACT_FILES)):
        for name in files:
            path = directory / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
    if not args.massbank_hdf5.is_file() or args.massbank_hdf5.stat().st_size == 0:
        raise FileNotFoundError(args.massbank_hdf5)
    for path in (args.portable_model, args.portable_model_report):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    panel_report = json.loads((args.panel_dir / "report.json").read_text(encoding="utf-8"))
    artifact_report = json.loads((args.artifact_dir / "report.json").read_text(encoding="utf-8"))
    if panel_report.get("pass_to_one_time_evaluation") is not True:
        raise RuntimeError("source B44 panel is not sealed")
    if artifact_report.get("status") != "bioaware_catalogue_v1_frozen":
        raise RuntimeError("source BioAware catalogue artefact is not frozen")
    if sha256(args.massbank_hdf5) != panel_report["provenance"]["massbank_hdf5"]:
        raise RuntimeError("source MassBank HDF5 does not match the sealed panel")

    queries = pd.read_csv(args.panel_dir / "queries.csv.gz")
    references = pd.read_csv(args.panel_dir / "candidate_references.csv.gz")
    rows = np.sort(np.unique(np.concatenate((
        queries["query_hdf5_row"].to_numpy(np.int64),
        references["reference_hdf5_row"].to_numpy(np.int64),
    ))))
    # DreaMS preprocessing appends one precursor token to the selected fragment peaks.
    spectrum_length = args.n_highest_peaks + 1
    spectra = np.empty((len(rows), spectrum_length, 2), dtype=np.float32)
    with h5py.File(args.massbank_hdf5, "r") as handle:
        for index, row in enumerate(rows):
            payload = json.loads(handle["data"][int(row)])
            if len(payload) != 7:
                raise RuntimeError(f"MassBank payload length drift at row {row}")
            _, _, mz, _, intensity, _, precursor = payload
            if len(mz) == 0 or len(mz) != len(intensity):
                raise RuntimeError(f"invalid sealed MassBank spectrum at row {row}")
            raw = np.vstack((np.asarray(mz, np.float32), np.asarray(intensity, np.float32)))
            processed = preprocess_spectrum(raw, float(precursor), args.n_highest_peaks)
            spectra[index] = np.asarray(processed, dtype=np.float32)
            if (index + 1) % 1000 == 0 or index + 1 == len(rows):
                print(f"[portable] {index + 1:,}/{len(rows):,}", flush=True)
    if not np.isfinite(spectra).all():
        raise RuntimeError("portable preprocessed spectra contain non-finite values")

    panel_out = args.output_dir / "panel"
    artifact_out = args.output_dir / "artifact"
    panel_out.mkdir(parents=True, exist_ok=False)
    artifact_out.mkdir(parents=True, exist_ok=False)
    for name in PANEL_FILES:
        shutil.copy2(args.panel_dir / name, panel_out / name)
    for name in PORTABLE_ARTIFACT_FILES:
        shutil.copy2(args.artifact_dir / name, artifact_out / name)
    source_model = args.artifact_dir / "model.joblib"
    portable_model_path = artifact_out / "portable_model.npz"
    portable_export_report = json.loads(args.portable_model_report.read_text(encoding="utf-8"))
    if portable_export_report.get("status") != "bioaware_b44_portable_model_exported":
        raise RuntimeError("portable model export report status mismatch")
    if portable_export_report["source_joblib_sha256"] != sha256(source_model):
        raise RuntimeError("portable model was not exported from the frozen source model")
    if portable_export_report["portable_model_sha256"] != sha256(args.portable_model):
        raise RuntimeError("portable model export hash mismatch")
    if portable_export_report["maximum_probability_error"] > 1e-14:
        raise RuntimeError("portable model export replay tolerance failed")
    shutil.copy2(args.portable_model, portable_model_path)
    portable_model = PortableBinaryHGB(portable_model_path, FEATURES)
    if portable_model.source_joblib_sha256 != sha256(source_model):
        raise RuntimeError("portable model internal source hash mismatch")
    cache_path = args.output_dir / "preprocessed_spectra.npz"
    atomic_npz(
        cache_path,
        hdf5_rows=rows,
        spectra=spectra,
        n_highest_peaks=np.asarray(args.n_highest_peaks, dtype=np.int64),
        source_massbank_sha256=np.asarray(panel_report["provenance"]["massbank_hdf5"]),
    )
    copied = {
        f"panel/{name}": sha256(panel_out / name) for name in PANEL_FILES
    } | {
        f"artifact/{name}": sha256(artifact_out / name) for name in PORTABLE_ARTIFACT_FILES
    }
    copied["artifact/portable_model.npz"] = sha256(portable_model_path)
    report = {
        "status": "bioaware_b44_portable_bundle_frozen",
        "formal": True,
        "rows": int(len(rows)),
        "queries": int(len(queries)),
        "n_highest_peaks": int(args.n_highest_peaks),
        "preprocessed_spectrum_length": int(spectrum_length),
        "server_requires_raw_massbank": False,
        "server_loads_joblib_model": False,
        "source_massbank_hdf5_sha256": panel_report["provenance"]["massbank_hdf5"],
        "portable_model_replay": {
            "rows": int(portable_export_report["replay_rows"]),
            "maximum_probability_error": float(portable_export_report["maximum_probability_error"]),
            "predicted_class_mismatches": int(portable_export_report["predicted_class_mismatches"]),
            "source_joblib_sha256": sha256(source_model),
        },
        "files": {**copied, "preprocessed_spectra.npz": sha256(cache_path)},
        "contracts": {
            "panel_unchanged": True,
            "frozen_expert_unchanged": True,
            "spectrum_operation": "deterministic DreaMS preprocessing only",
            "embeddings_computed": False,
            "outcomes_computed": False,
        },
        "claim_limit": "Execution-only packaging of an already sealed panel and frozen expert; it contains no B44 performance outcome.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
