#!/usr/bin/env python3
"""Score and calibrate KGMN reaction edges against exact-formula decoys."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from kgmn_edge_calibration_core import crossfit_edge_scores  # noqa: E402
from step5_gate_eval import embed, load_trained  # noqa: E402
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


OFFICIAL_SHA256 = "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245"
METDNA2_COMMIT = "5685ab219269c2f35cd5087655b0470b2da4d93c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def take(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    order = np.argsort(rows, kind="mergesort")
    ordered = np.asarray(dataset[rows[order]])
    return ordered[np.argsort(order, kind="mergesort")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--author-baseline", type=Path, required=True)
    parser.add_argument("--author-scores", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--noise-checkpoint", type=Path)
    parser.add_argument("--noise-eligibility-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        args.manifest_dir / "report.json",
        args.manifest_dir / "paired_reaction_decoy_triples.csv.gz",
        args.author_baseline,
        args.author_scores,
        args.data,
        args.official_checkpoint,
        args.architecture_checkpoint,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if sha256(args.official_checkpoint) != OFFICIAL_SHA256:
        raise RuntimeError("official DreaMS checkpoint hash mismatch")

    baseline = json.loads(args.author_baseline.read_text(encoding="utf-8"))
    if baseline.get("status") != "kgmn_metdna2_200std_author_baseline_frozen":
        raise RuntimeError("untouched KGMN/MetDNA2 author baseline has not been frozen")
    if baseline.get("source_commit") != METDNA2_COMMIT:
        raise RuntimeError("author baseline belongs to an unexpected MetDNA2 commit")
    contracts = baseline.get("contracts", {})
    if contracts.get("full_annotation_credential") is not True or contracts.get("rt_calibration") is not False:
        raise RuntimeError("author baseline did not use the frozen full-credential 200STD protocol")

    manifest_report = json.loads((args.manifest_dir / "report.json").read_text(encoding="utf-8"))
    triples_path = args.manifest_dir / "paired_reaction_decoy_triples.csv.gz"
    if manifest_report.get("status") != "kgmn_dreams_edge_calibration_manifest_frozen":
        raise RuntimeError("edge manifest status mismatch")
    if manifest_report["provenance"].get("triples_sha256") != sha256(triples_path):
        raise RuntimeError("edge manifest triples hash mismatch")
    triples = pd.read_csv(triples_path)
    forbidden = [column for column in triples.columns if any(token in column.lower() for token in ("outcome", "correct", "truth"))]
    if forbidden:
        raise RuntimeError(f"outcome-like columns are forbidden in edge calibration input: {forbidden}")
    exact_author_report_path = args.author_scores.parent / "report.json"
    if not exact_author_report_path.is_file():
        raise FileNotFoundError(exact_author_report_path)
    exact_author_report = json.loads(exact_author_report_path.read_text(encoding="utf-8"))
    if exact_author_report.get("status") != "kgmn_edge_calibration_exact_author_dp_complete":
        raise RuntimeError("exact MetDNA2 author edge score report status mismatch")
    if exact_author_report.get("triples") != len(triples):
        raise RuntimeError("exact MetDNA2 author edge score report count mismatch")
    exact_contracts = exact_author_report.get("contracts", {})
    if (
        exact_contracts.get("python_similarity_proxy_used") is not False
        or exact_contracts.get("triple_order_preserved") is not True
    ):
        raise RuntimeError("exact MetDNA2 author edge score contract mismatch")
    author_scores = pd.read_csv(args.author_scores)
    if exact_author_report.get("provenance", {}).get("scores_sha256") != sha256(args.author_scores):
        raise RuntimeError("exact MetDNA2 author score file hash mismatch")
    expected_author_columns = {
        "triple_index", "source_row", "positive_row", "decoy_row", "author_positive", "author_decoy"
    }
    if not expected_author_columns.issubset(author_scores.columns):
        raise RuntimeError("exact author edge score table is incomplete")
    if len(author_scores) != len(triples):
        raise RuntimeError("exact author score and edge manifest row counts differ")
    if not np.array_equal(author_scores["triple_index"].to_numpy(dtype=np.int64), np.arange(len(triples))):
        raise RuntimeError("exact author score triple order mismatch")
    for column in ("source_row", "positive_row", "decoy_row"):
        if not np.array_equal(
            author_scores[column].to_numpy(dtype=np.int64), triples[column].to_numpy(dtype=np.int64)
        ):
            raise RuntimeError(f"exact author score row mapping mismatch: {column}")
    author_positive = author_scores["author_positive"].to_numpy(dtype=float)
    author_decoy = author_scores["author_decoy"].to_numpy(dtype=float)
    if (
        np.any(~np.isfinite(author_positive)) or np.any(~np.isfinite(author_decoy))
        or np.any((author_positive < 0) | (author_positive > 1))
        or np.any((author_decoy < 0) | (author_decoy > 1))
    ):
        raise RuntimeError("exact author edge scores are invalid")

    noise_requested = args.noise_checkpoint is not None or args.noise_eligibility_report is not None
    if noise_requested:
        if args.noise_checkpoint is None or args.noise_eligibility_report is None:
            raise RuntimeError("noise checkpoint and eligibility report must be supplied together")
        eligibility = json.loads(args.noise_eligibility_report.read_text(encoding="utf-8"))
        if eligibility.get("eligible_for_network_edge_calibration") is not True:
            raise RuntimeError("noise-tuned checkpoint lacks an explicit shared-embedding eligibility gate")
    else:
        eligibility = None

    row_columns = ["source_row", "positive_row", "decoy_row"]
    unique_rows = np.unique(triples[row_columns].to_numpy(dtype=np.int64).ravel())
    with h5py.File(args.data, "r") as handle:
        if unique_rows.min() < 0 or unique_rows.max() >= len(handle["spectrum"]):
            raise RuntimeError("edge manifest contains invalid HDF5 rows")
        spectra = take(handle["spectrum"], unique_rows)
        precursor = take(handle["precursor_mz"], unique_rows).astype(float)
    row_position = {int(row): index for index, row in enumerate(unique_rows)}
    dreams_spectra = [
        preprocess_spectrum(spectra[index], float(precursor[index]), args.n_highest_peaks)
        for index in range(len(unique_rows))
    ]

    print(f"[edge calibration] encoding {len(unique_rows):,} unique spectra", flush=True)
    device = torch.device(args.device)
    official_model, official_kind = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks
    )
    if official_kind not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError("unexpected official checkpoint format")
    official_model.eval()
    official_embedding = embed(official_model, dreams_spectra, device, args.batch_size).numpy().astype(np.float32)
    official_embedding /= np.clip(np.linalg.norm(official_embedding, axis=1, keepdims=True), 1e-12, None)
    del official_model
    torch.cuda.empty_cache()

    noise_embedding = None
    if noise_requested:
        noise_model, _ = load_trained(
            args.official_checkpoint,
            args.architecture_checkpoint,
            device,
            args.n_highest_peaks,
            args.noise_checkpoint,
        )
        noise_embedding = embed(noise_model, dreams_spectra, device, args.batch_size).numpy().astype(np.float32)
        noise_embedding /= np.clip(np.linalg.norm(noise_embedding, axis=1, keepdims=True), 1e-12, None)
        del noise_model
        torch.cuda.empty_cache()

    def positions(column: str) -> np.ndarray:
        return triples[column].map(row_position).to_numpy(dtype=np.int64)

    source = positions("source_row")
    positive = positions("positive_row")
    decoy = positions("decoy_row")
    official_positive = np.sum(official_embedding[source] * official_embedding[positive], axis=1)
    official_decoy = np.sum(official_embedding[source] * official_embedding[decoy], axis=1)
    raw_scores = {
        "author_dp": (author_positive, author_decoy),
        "official_dreams": (official_positive, official_decoy),
    }
    if noise_embedding is not None:
        raw_scores["noise_tuned_dreams"] = (
            np.sum(noise_embedding[source] * noise_embedding[positive], axis=1),
            np.sum(noise_embedding[source] * noise_embedding[decoy], axis=1),
        )

    metrics, probabilities, artifact = crossfit_edge_scores(
        raw_scores,
        triples["component_fold"].to_numpy(dtype=int),
        triples["edge_equal_weight"].to_numpy(dtype=float),
        triples["component_id"].astype(str).to_numpy(),
        triples["positive_formula"].astype(str).to_numpy(),
        bootstrap_resamples=args.bootstrap_resamples,
    )
    official_delta = metrics["official_dreams"]["component_bootstrap_accuracy_delta_vs_author"]
    official_formula_delta = metrics["official_dreams"]["formula_bootstrap_accuracy_delta_vs_author"]
    official_eligible = (
        official_delta["ci_low"] > 0
        and official_formula_delta["ci_low"] > 0
        and metrics["official_dreams"]["edge_weighted_corrected_vs_author"]
        > metrics["official_dreams"]["edge_weighted_introduced_vs_author"]
    )

    args.output_dir.mkdir(parents=True)
    scored = triples.copy()
    for name, (positive_raw, negative_raw) in raw_scores.items():
        scored[f"{name}_positive_raw"] = positive_raw
        scored[f"{name}_decoy_raw"] = negative_raw
    for name, (positive_probability, negative_probability) in probabilities.items():
        scored[f"{name}_positive_oof_probability"] = positive_probability
        scored[f"{name}_decoy_oof_probability"] = negative_probability
    scored_path = args.output_dir / "scored_paired_triples.csv.gz"
    scored.to_csv(scored_path, index=False)
    artifact_path = args.output_dir / "calibration_artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = {
        "status": "kgmn_dreams_edge_calibration_complete",
        "formal": True,
        "protocol": "nested component-fold scalar calibration with inner-OOF target-decoy thresholds",
        "counts": {
            "triples": int(len(triples)),
            "unique_rows": int(len(unique_rows)),
            "components": int(triples["component_id"].nunique()),
            "formulas": int(triples["positive_formula"].nunique()),
        },
        "arms": metrics,
        "gates": {
            "official_dreams_eligible_for_dynamic_propagation_test": bool(official_eligible),
            "noise_arm_present": bool(noise_requested),
            "noise_arm_requires_prior_shared_embedding_gate": True,
        },
        "contracts": {
            "author_score": "exact MetDNA2 1.2.10 R runSpecMatch scoreReverse at 25 ppm",
            "candidate_generation_changed": False,
            "network_changed": False,
            "outcome_used_as_model_feature": False,
            "outer_fold_used_for_calibration_or_threshold": False,
            "P2b_used": False,
        },
        "provenance": {
            "manifest_report_sha256": sha256(args.manifest_dir / "report.json"),
            "triples_sha256": sha256(triples_path),
            "author_baseline_sha256": sha256(args.author_baseline),
            "author_scores_sha256": sha256(args.author_scores),
            "author_scores_report_sha256": sha256(exact_author_report_path),
            "hdf5_sha256": sha256(args.data),
            "official_checkpoint_sha256": sha256(args.official_checkpoint),
            "architecture_checkpoint_sha256": sha256(args.architecture_checkpoint),
            "noise_checkpoint_sha256": sha256(args.noise_checkpoint) if args.noise_checkpoint else None,
            "noise_eligibility_report_sha256": sha256(args.noise_eligibility_report) if args.noise_eligibility_report else None,
            "scored_triples_sha256": sha256(scored_path),
            "calibration_artifact_sha256": sha256(artifact_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "This is an acquisition-matched reaction-edge mechanism test. Eligibility only permits a dynamic "
            "KGMN propagation experiment; it does not establish improved metabolite annotation or SOTA performance."
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
