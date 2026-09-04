#!/usr/bin/env python3
"""Encode a KGMN MSP/MGF feature network with the frozen shared DreaMS encoder.

The feature identifiers are reconciled against the MS1 table.  Ambiguous or
unmatched identifiers fail closed because MetDNA2's recursive hook addresses
embeddings by feature name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from dreams.utils.io import read_mgf, read_msp  # noqa: E402
from step5_gate_eval import embed  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


OFFICIAL_SHA256 = "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_feature_spectra(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".msp":
        frame = read_msp(path)
    elif suffix == ".mgf":
        frame = read_mgf(
            path,
            name_name=["NAME", "TITLE"],
            feature_id_name=["FEATURE_ID", "FEATUREID"],
            scan_number_name=["SCAN_NUMBER", "SCANS"],
        )
    else:
        raise RuntimeError(
            f"DreaMS external edge scoring currently requires MSP or MGF, found {suffix}; "
            "convert the author-supported source without changing feature identifiers"
        )
    if frame.empty:
        raise RuntimeError("feature spectrum file is empty")
    return frame


def select_feature_names(frame: pd.DataFrame, ms1_names: set[str]) -> tuple[pd.Series, str]:
    candidates = [column for column in ("name", "feature_id", "scan_number") if column in frame.columns]
    valid: list[tuple[str, pd.Series]] = []
    diagnostics: dict[str, dict[str, int]] = {}
    for column in candidates:
        values = frame[column].fillna("").astype(str).str.strip()
        nonempty = values.ne("")
        matched = nonempty & values.isin(ms1_names)
        diagnostics[column] = {
            "nonempty": int(nonempty.sum()),
            "matched": int(matched.sum()),
            "duplicates": int(values[nonempty].duplicated().sum()),
        }
        if nonempty.all() and matched.all() and not values.duplicated().any():
            valid.append((column, values))
    if len(valid) != 1:
        raise RuntimeError(
            "cannot identify exactly one unique spectrum identifier mapped to MS1 names; "
            f"diagnostics={diagnostics}"
        )
    column, values = valid[0]
    return values, column


def validate_frame(frame: pd.DataFrame, feature_names: pd.Series) -> None:
    required = {"precursor_mz", "spectrum"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"spectrum parser did not recover required fields: {missing}")
    if feature_names.eq("").any() or feature_names.duplicated().any():
        raise RuntimeError("feature names must be non-empty and unique")
    precursor = pd.to_numeric(frame["precursor_mz"], errors="coerce").to_numpy(float)
    if np.any(~np.isfinite(precursor)) or np.any(precursor <= 0):
        raise RuntimeError("precursor masses must be finite and positive")
    for index, spectrum in enumerate(frame["spectrum"]):
        array = np.asarray(spectrum, dtype=float)
        if array.ndim != 2 or array.shape[0] != 2 or array.shape[1] < 1:
            raise RuntimeError(f"invalid spectrum at row {index}: shape={array.shape}")
        if np.any(~np.isfinite(array)):
            raise RuntimeError(f"non-finite spectrum at row {index}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra", type=Path, required=True)
    parser.add_argument("--ms1-table", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument(
        "--architecture-checkpoint",
        type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.spectra, args.ms1_table, args.official_checkpoint, args.architecture_checkpoint):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {args.output_dir}")
    if sha256(args.official_checkpoint) != OFFICIAL_SHA256:
        raise RuntimeError("official DreaMS checkpoint hash mismatch")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    ms1 = pd.read_csv(args.ms1_table, nrows=0)
    if [str(value).strip().lower() for value in ms1.columns[:3]] != ["name", "mz", "rt"]:
        raise RuntimeError("MS1 table must start with name,mz,rt")
    ms1_names_frame = pd.read_csv(args.ms1_table, usecols=[0], dtype=str)
    ms1_names = set(ms1_names_frame.iloc[:, 0].fillna("").str.strip())
    if "" in ms1_names or len(ms1_names) != len(ms1_names_frame):
        raise RuntimeError("MS1 feature names must be non-empty and unique")

    frame = read_feature_spectra(args.spectra)
    feature_names, identifier_column = select_feature_names(frame, ms1_names)
    validate_frame(frame, feature_names)
    precursor = pd.to_numeric(frame["precursor_mz"], errors="raise").to_numpy(float)
    spectra = [
        preprocess_spectrum(np.asarray(spectrum), float(mz), args.n_highest_peaks)
        for spectrum, mz in zip(frame["spectrum"], precursor, strict=True)
    ]

    device = torch.device(args.device)
    model, kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks
    )
    if kind not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError(f"unexpected official checkpoint format: {kind}")
    model.eval()
    vectors = embed(model, spectra, device, args.batch_size).numpy().astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise RuntimeError("official DreaMS produced invalid feature embeddings")
    vectors /= norms

    args.output_dir.mkdir(parents=True)
    columns = [f"z_{index:04d}" for index in range(vectors.shape[1])]
    exported = pd.DataFrame(vectors, columns=columns)
    exported.insert(0, "feature_name", feature_names.to_numpy())
    embedding_path = args.output_dir / "official_feature_embeddings.csv.gz"
    exported.to_csv(embedding_path, index=False, compression="gzip", float_format="%.9g")

    replay = pd.read_csv(embedding_path)
    replay_vectors = replay.drop(columns=["feature_name"]).to_numpy(float)
    replay_norm_error = float(np.max(np.abs(np.linalg.norm(replay_vectors, axis=1) - 1.0)))
    if replay_norm_error > 2e-6:
        raise RuntimeError("quantised deployment embeddings are not unit-normalised")
    if not np.array_equal(replay["feature_name"].astype(str).to_numpy(), feature_names.to_numpy()):
        raise RuntimeError("embedding export changed feature order")

    report = {
        "status": "kgmn_official_dreams_feature_embeddings_frozen",
        "formal": True,
        "spectra": int(len(frame)),
        "ms1_features": int(len(ms1_names)),
        "identifier_column": identifier_column,
        "embedding_dimension": int(vectors.shape[1]),
        "maximum_exported_norm_error": replay_norm_error,
        "contracts": {
            "shared_encoder": True,
            "model_eval_mode": True,
            "identity_labels_used": False,
            "phenotype_used": False,
            "P2b_used": False,
            "all_spectrum_identifiers_match_ms1": True,
        },
        "provenance": {
            "spectra_sha256": sha256(args.spectra),
            "ms1_table_sha256": sha256(args.ms1_table),
            "official_checkpoint_sha256": sha256(args.official_checkpoint),
            "architecture_checkpoint_sha256": sha256(args.architecture_checkpoint),
            "embeddings_sha256": sha256(embedding_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "Frozen KGMN feature-edge execution artifact; no annotation result.",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
