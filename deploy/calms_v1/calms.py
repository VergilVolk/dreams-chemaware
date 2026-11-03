#!/usr/bin/env python3
"""Run the frozen CALMS v1 metabolite-annotation scoring engine.

CALMS consumes a fixed candidate axis, aligned candidate evidence, and four
frozen embedding banks.  This file is deliberately a thin I/O layer over the
byte-identical v1.3.2 runtime; it does not reimplement features or model math.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
RUNTIME = PACKAGE_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME))

from full23_tensor_set_readout_core import ENCODER_NAMES  # noqa: E402
from full23_v132_release import FrozenFull23V132  # noqa: E402
from unified_additive_release import (  # noqa: E402
    CHEMICAL_KEYS,
    CORE_SHA256,
    EVIDENCE_SCHEMA,
    EXPECTED_UPSTREAM_SHA256,
    REQUIRED_SCORE_NAMES,
    V2_POLICY_SHA256,
    V2_UNAVAILABLE_CONTRACT,
    _load_evidence,
    _query_registry,
    sha256_file,
)


METHOD_NAME = "CALMS"
EXPANDED_NAME = (
    "Context-Aware Learning for Metabolite Annotation from Mass Spectra"
)
PUBLIC_VERSION = "1.0"
FROZEN_IMPLEMENTATION = "Full23 v1.3.2"
REPRESENTATION_SCHEMA = "calms_aligned_representation_v1"
BANK_SCHEMA = "calms_embedding_bank_v1"
PARENT_CHECKPOINT = PACKAGE_ROOT / "assets" / "calms_candidate_evidence.pt"
READOUT_CHECKPOINT = PACKAGE_ROOT / "assets" / "calms_embedding_corrector.pt"
EXPECTED_ENCODER_CHECKPOINT_SHA256 = {
    "noise_v1": "01ac8c7bfa75d89862e0b1eb2b5ffb297955cffae471fd483ad6c38d13f33beb",
    "chemaware_stage1": "09c419dcabf2ff424fe38bb33eaefcd25ccab0b6d125daf03b7959c25fff5838",
    "chemaware_phasea": "a8428329ca1d12bfe735f3f8b848ed020a8db18d0a654d4f62cdbeb006e61135",
    "noise_chemaware_fullmsg": "4f61a4efefe734b2c6bc66c06b315c37164b67e0f424208c3b8c7f81e6618130",
}


def _scalar_text(value: object, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise RuntimeError(f"{name} must be a scalar string")
    return str(array.item())


def _valid_sha256(value: object, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"{name} SHA256 is malformed")
    return digest


def _parse_named_paths(items: list[str], option: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{option} must use NAME=PATH")
        name, raw_path = item.split("=", 1)
        if not name or name in output:
            raise ValueError(f"duplicate or empty name in {option}: {name!r}")
        output[name] = Path(raw_path)
    return output


def _validate_evidence_manifest(
    evidence_path: Path, manifest_path: Path,
) -> tuple[dict, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != EVIDENCE_SCHEMA:
        raise RuntimeError("candidate-evidence manifest schema mismatch")
    if tuple(manifest.get("required_score_names", ())) != REQUIRED_SCORE_NAMES:
        raise RuntimeError("candidate-evidence family registry drifted")
    evidence_sha = sha256_file(evidence_path)
    if manifest.get("evidence_sha256") != evidence_sha:
        raise RuntimeError("candidate-evidence file hash mismatch")
    upstream = manifest.get("upstream_artifacts")
    if not isinstance(upstream, dict):
        raise RuntimeError("candidate-evidence manifest lacks upstream identities")
    expected = {"candidate_axis", *REQUIRED_SCORE_NAMES, "chemaware_v2_continuous_null"}
    if set(upstream) != expected:
        raise RuntimeError("candidate-evidence upstream registry is incomplete or ambiguous")
    for name, identity in upstream.items():
        if not isinstance(identity, dict):
            raise RuntimeError(f"upstream artifact identity is malformed: {name}")
        digest = _valid_sha256(identity.get("sha256", ""), name)
        if name in EXPECTED_UPSTREAM_SHA256 and digest != EXPECTED_UPSTREAM_SHA256[name]:
            raise RuntimeError(f"upstream frozen artifact identity drifted: {name}")
    return manifest, evidence_sha


def _validate_v2_mode(manifest: dict, chemical: Mapping[str, object] | None) -> str:
    identity = manifest["upstream_artifacts"]["chemaware_v2_continuous_null"]
    mode = identity.get("mode")
    if mode == "real":
        if chemical is None:
            raise RuntimeError("real ChemAware V2 mode requires all five arrays")
        if identity.get("policy_sha256") != V2_POLICY_SHA256:
            raise RuntimeError("real ChemAware V2 policy identity drifted")
    elif mode == "unavailable":
        if chemical is not None:
            raise RuntimeError("unavailable ChemAware V2 mode forbids chemical arrays")
        if (
            identity.get("contract") != V2_UNAVAILABLE_CONTRACT
            or identity.get("sha256") != CORE_SHA256
        ):
            raise RuntimeError("ChemAware V2 unavailable contract drifted")
    else:
        raise RuntimeError("ChemAware V2 mode must be exactly real or unavailable")
    return str(mode)


def _load_bank(path: Path, encoder_name: str) -> tuple[np.ndarray, str]:
    digest = sha256_file(path)
    with np.load(path, allow_pickle=False) as body:
        if "embeddings" not in body.files:
            raise RuntimeError(f"{encoder_name} bank lacks embeddings")
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
        if "schema" in body.files and _scalar_text(body["schema"], "bank schema") != BANK_SCHEMA:
            raise RuntimeError(f"{encoder_name} embedding-bank schema mismatch")
        if "encoder_name" in body.files and _scalar_text(
            body["encoder_name"], "encoder_name"
        ) != encoder_name:
            raise RuntimeError(f"{encoder_name} embedding-bank name mismatch")
        if "rows" in body.files:
            rows = np.asarray(body["rows"], dtype=np.int64)
            if rows.shape != (len(embeddings),) or len(np.unique(rows)) != len(rows):
                raise RuntimeError(f"{encoder_name} embedding row registry is malformed")
    if embeddings.ndim != 2 or embeddings.shape[1] != 1024:
        raise RuntimeError(f"{encoder_name} embedding bank must have shape (n, 1024)")
    if not np.all(np.isfinite(embeddings)):
        raise RuntimeError(f"{encoder_name} embedding bank contains non-finite values")
    return embeddings, digest


def _load_representation(
    path: Path,
    manifest_path: Path,
    bank_paths: Mapping[str, Path],
    evidence_sha: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != REPRESENTATION_SCHEMA:
        raise RuntimeError("representation manifest schema mismatch")
    representation_sha = sha256_file(path)
    if manifest.get("representation_sha256") != representation_sha:
        raise RuntimeError("representation file hash mismatch")
    if manifest.get("candidate_evidence_sha256") != evidence_sha:
        raise RuntimeError("representation is not bound to this candidate-evidence file")
    registry = manifest.get("embedding_banks")
    if not isinstance(registry, dict) or set(registry) != set(ENCODER_NAMES):
        raise RuntimeError("representation embedding-bank registry drifted")
    if set(bank_paths) != set(ENCODER_NAMES):
        raise RuntimeError("exactly four named embedding banks are required")

    banks: dict[str, np.ndarray] = {}
    bank_hashes: dict[str, str] = {}
    for name in ENCODER_NAMES:
        identity = registry[name]
        if not isinstance(identity, dict):
            raise RuntimeError(f"embedding-bank identity is malformed: {name}")
        expected_file_sha = _valid_sha256(identity.get("sha256", ""), name)
        checkpoint_sha = _valid_sha256(
            identity.get("encoder_checkpoint_sha256", ""), f"{name} encoder checkpoint"
        )
        if checkpoint_sha != EXPECTED_ENCODER_CHECKPOINT_SHA256[name]:
            raise RuntimeError(f"frozen encoder checkpoint identity drifted: {name}")
        embeddings, observed_sha = _load_bank(bank_paths[name], name)
        if observed_sha != expected_file_sha:
            raise RuntimeError(f"embedding-bank file hash mismatch: {name}")
        banks[name] = embeddings
        bank_hashes[name] = observed_sha

    with np.load(path, allow_pickle=False) as body:
        required = {
            "candidate_pair_ids",
            "query_positions",
            "reference_positions",
            "reference_mask",
        }
        missing = sorted(required - set(body.files))
        if missing:
            raise RuntimeError(f"representation arrays are missing: {missing}")
        arrays = {name: np.asarray(body[name]) for name in required}
    return arrays, banks, bank_hashes, manifest


class CALMS(FrozenFull23V132):
    """Public frozen CALMS v1 API; model mathematics are inherited unchanged."""

    def __init__(self, device: str = "cpu", batch_queries: int = 128):
        super().__init__(
            parent_checkpoint=PARENT_CHECKPOINT,
            readout_checkpoint=READOUT_CHECKPOINT,
            device=device,
            batch_queries=batch_queries,
        )

    def score(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        scores, internal = super().score(*args, **kwargs)
        report = dict(internal)
        report.update({
            "schema": "calms_metabolite_annotation_prediction_v1",
            "method_name": METHOD_NAME,
            "expanded_name": EXPANDED_NAME,
            "public_version": PUBLIC_VERSION,
            "frozen_implementation": FROZEN_IMPLEMENTATION,
            "implementation_report_schema": internal["schema"],
        })
        return scores, report


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    evidence_manifest, evidence_sha = _validate_evidence_manifest(
        args.evidence, args.evidence_manifest
    )
    axis, evidence, chemical = _load_evidence(args.evidence)
    v2_mode = _validate_v2_mode(evidence_manifest, chemical)
    bank_paths = _parse_named_paths(args.embedding_bank, "--embedding-bank")
    representation, banks, bank_hashes, representation_manifest = _load_representation(
        args.representation,
        args.representation_manifest,
        bank_paths,
        evidence_sha,
    )
    scorer = CALMS(device=args.device, batch_queries=args.batch_queries)
    scores, report = scorer.score(
        axis,
        evidence,
        representation["candidate_pair_ids"],
        representation["query_positions"],
        representation["reference_positions"],
        representation["reference_mask"],
        banks,
        bank_hashes,
        chemical,
    )
    registry = _query_registry(axis, scores)

    args.output.mkdir(parents=True)
    score_path = args.output / "candidate_scores.npz"
    registry_path = args.output / "query_registry.npz"
    np.savez_compressed(
        score_path,
        query_ptr=axis.query_ptr,
        query_ids=axis.query_ids,
        candidate_query_ids=axis.candidate_query_ids,
        candidate_ids=axis.candidate_ids,
        candidate_pair_ids=axis.candidate_pair_ids,
        calms_score=scores,
        evidence_available=np.isfinite(scores),
    )
    np.savez_compressed(registry_path, **registry)
    report.update({
        "candidate_evidence": str(args.evidence),
        "candidate_evidence_sha256": evidence_sha,
        "candidate_evidence_manifest": str(args.evidence_manifest),
        "candidate_evidence_manifest_sha256": sha256_file(args.evidence_manifest),
        "representation": str(args.representation),
        "representation_sha256": sha256_file(args.representation),
        "representation_manifest": str(args.representation_manifest),
        "representation_manifest_sha256": sha256_file(args.representation_manifest),
        "embedding_banks": {
            name: {"path": str(bank_paths[name]), "sha256": bank_hashes[name]}
            for name in ENCODER_NAMES
        },
        "v2_mode": v2_mode,
        "candidate_scores": score_path.name,
        "query_registry": registry_path.name,
        "input_claim_boundary": representation_manifest.get("claim_boundary", ""),
    })
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    checksums_path = args.output / "SHA256SUMS.txt"
    output_files = (score_path, registry_path, report_path)
    checksums_path.write_text(
        "\n".join(f"{sha256_file(item)}  {item.name}" for item in output_files) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "CALMS_INFERENCE_COMPLETE",
        "output": str(args.output),
        "scored_queries": report["parent"]["scored_queries"],
        "scored_candidates": report["parent"]["scored_candidates"],
        "residual_scored_queries": report["residual_scored_queries"],
        "v2_mode": v2_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--representation", type=Path, required=True)
    parser.add_argument("--representation-manifest", type=Path, required=True)
    parser.add_argument(
        "--embedding-bank",
        action="append",
        required=True,
        help="Repeat NAME=PATH for the four frozen encoder embedding banks",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-queries", type=int, default=128)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments), indent=2))


if __name__ == "__main__":
    main()
