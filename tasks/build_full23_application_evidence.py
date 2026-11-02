#!/usr/bin/env python3
"""Build fixed-axis Full23 evidence from cross-study identity scoring output.

This is a provenance and layout adapter only.  It does not recompute a score,
construct a candidate, or invoke a model.  The complete composite-query axis
comes from ``directional_query_registry.tsv.gz`` and every candidate score is
copied from ``identity_candidate_scores.tsv.gz``.  ChemAware V2 is deliberately
absent: the emitted manifest registers the frozen release's exact unavailable
contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Mapping

import numpy as np
import pandas as pd

from unified_additive_release import (
    CORE_SHA256,
    EVIDENCE_SCHEMA,
    EXPECTED_UPSTREAM_SHA256,
    MSG_FAMILY,
    REQUIRED_SCORE_NAMES,
    V2_UNAVAILABLE_CONTRACT,
    sha256_file,
    validate_axis,
)


SCORER_STATUS = "CROSSSTUDY_IDENTITY_MULTIENCODER_COMPLETE"
SCORER_CANDIDATE_ADMISSION = (
    "frozen complete-identity handoff only; no application-side candidate generation"
)
SCORER_WSE_BACKEND = "canonical_handoff_wse_score_no_recomputation"
PRODUCTION_QUERY_COUNT = 38_628
PRODUCTION_CANDIDATE_COUNT = 22_046
PRODUCTION_QUERY_CARDINALITIES = {
    "zero": 27_815,
    "singleton": 6_477,
    "multiple": 4_336,
}
ENCODER_NAMES = (
    "noise_v1",
    "chemaware_stage1",
    "chemaware_phasea",
    "noise_chemaware_fullmsg",
)
ENCODER_NAME_ALIASES = {"chemaware_phase_a": "chemaware_phasea"}
SCORE_COLUMNS = {
    "noise_v1": "noise_v1_score",
    "weighted_spectral_entropy": "WSE_score",
    "p2b_noise_v1_frozen": "P2b_score",
    MSG_FAMILY: f"auxiliary__{MSG_FAMILY}_score",
    "chemaware_stage1": "chemaware_stage1_score",
    "chemaware_phasea": "chemaware_phasea_score",
    "noise_chemaware_fullmsg": "noise_chemaware_fullmsg_score",
}
OUTPUT_EVIDENCE = "candidate_evidence.npz"
OUTPUT_MANIFEST = "candidate_evidence.json"


def _valid_sha256(value: object, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError(f"malformed SHA256 for {label}")
    return digest


def _required_text(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise RuntimeError(f"required text column is absent: {column}")
    values = frame[column].astype("string")
    if values.isna().any() or values.str.strip().eq("").any():
        raise RuntimeError(f"required text column contains missing/empty values: {column}")
    return values.astype(str).to_numpy(dtype="U")


def _strict_boolean(series: pd.Series, label: str) -> np.ndarray:
    if getattr(series.dtype, "kind", "") == "b":
        return series.to_numpy(bool)
    normalised = series.astype("string").str.strip().str.lower()
    invalid = normalised.isna() | ~normalised.isin({"true", "false", "1", "0"})
    if invalid.any():
        raise RuntimeError(f"{label} must contain only true/false or 1/0")
    return normalised.isin({"true", "1"}).to_numpy(bool)


def _numeric_vector(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise RuntimeError(f"required score column is absent: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)


def _semantic_array_hash(arrays: Mapping[str, np.ndarray], schema: str) -> str:
    """Hash named arrays without relying on NPZ container serialization."""
    digest = hashlib.sha256()
    digest.update(schema.encode("utf-8") + b"\0")
    for name, raw in arrays.items():
        array = np.asarray(raw)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.shape).encode("ascii") + b"\0")
        if array.dtype.kind in "USO":
            digest.update(b"utf8-length-prefixed\0")
            for item in array.astype("U").flat:
                encoded = str(item).encode("utf-8")
                digest.update(struct.pack("<Q", len(encoded)))
                digest.update(encoded)
        else:
            canonical = np.ascontiguousarray(array)
            if canonical.dtype.byteorder == ">" or (
                canonical.dtype.byteorder == "=" and not np.little_endian
            ):
                canonical = canonical.byteswap().view(canonical.dtype.newbyteorder("<"))
            digest.update(canonical.dtype.str.encode("ascii") + b"\0")
            digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _parse_sidecars(items: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--encoder-sidecar must use NAME=PATH")
        name, raw_path = item.split("=", 1)
        name = ENCODER_NAME_ALIASES.get(name, name)
        if name not in ENCODER_NAMES or name in paths or not raw_path:
            raise ValueError(f"invalid or duplicate encoder sidecar: {name!r}")
        paths[name] = Path(raw_path)
    missing = sorted(set(ENCODER_NAMES) - set(paths))
    if missing:
        raise RuntimeError(f"encoder sidecars are incomplete: {missing}")
    return paths


def _reported_output(
    report: Mapping[str, object], key: str, actual_path: Path,
) -> dict[str, str]:
    outputs = report.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get(key), dict):
        raise RuntimeError(f"scorer report lacks output identity: {key}")
    identity = outputs[key]
    reported = _valid_sha256(identity.get("sha256"), f"report.outputs.{key}")
    observed = sha256_file(actual_path)
    if reported != observed:
        raise RuntimeError(f"scorer output SHA256 mismatch: {key}")
    return {"path": str(actual_path.resolve()), "sha256": observed}


def _load_encoder_provenance(
    report: Mapping[str, object], sidecar_paths: Mapping[str, Path],
) -> dict[str, dict[str, object]]:
    inputs = report.get("inputs")
    embeddings = inputs.get("embeddings") if isinstance(inputs, dict) else None
    if not isinstance(embeddings, dict) or set(embeddings) != set(ENCODER_NAMES):
        raise RuntimeError("scorer report encoder registry is incomplete or ambiguous")

    provenance: dict[str, dict[str, object]] = {}
    for name in ENCODER_NAMES:
        report_identity = embeddings[name]
        if not isinstance(report_identity, dict):
            raise RuntimeError(f"malformed scorer encoder identity: {name}")
        cache_sha = _valid_sha256(report_identity.get("sha256"), f"report.inputs.embeddings.{name}")
        sidecar_path = sidecar_paths[name]
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        observed_name = ENCODER_NAME_ALIASES.get(str(sidecar.get("encoder_name")), str(sidecar.get("encoder_name")))
        if (
            sidecar.get("status") != "CROSSSTUDY_FROZEN_ENCODER_EMBEDDINGS_COMPLETE"
            or observed_name != name
            or sidecar.get("labels_or_responses_accessed") is not False
            or int(sidecar.get("embedding_dimension", -1)) != 1024
        ):
            raise RuntimeError(f"encoder cache sidecar contract drifted: {name}")
        output = sidecar.get("output")
        checkpoint = sidecar.get("inputs", {}).get("checkpoint") if isinstance(sidecar.get("inputs"), dict) else None
        if not isinstance(output, dict) or not isinstance(checkpoint, dict):
            raise RuntimeError(f"encoder cache sidecar lacks output/checkpoint identity: {name}")
        sidecar_cache_sha = _valid_sha256(output.get("sha256"), f"{name} sidecar output")
        checkpoint_sha = _valid_sha256(checkpoint.get("sha256"), f"{name} sidecar checkpoint")
        if sidecar_cache_sha != cache_sha:
            raise RuntimeError(f"encoder cache SHA256 differs between scorer report and sidecar: {name}")
        canonical_sha = EXPECTED_UPSTREAM_SHA256[name]
        provenance[name] = {
            # This is the identity consumed by the frozen Full23 release.
            "sha256": canonical_sha,
            "canonical_checkpoint_sha256": canonical_sha,
            # Slim application containers are separately visible; they need
            # not have the same container hash as their canonical checkpoint.
            "application_checkpoint": {
                "path": str(checkpoint.get("path", "")),
                "sha256": checkpoint_sha,
                "matches_canonical_container": checkpoint_sha == canonical_sha,
            },
            "application_cache": {
                "path": str(report_identity.get("path", output.get("path", ""))),
                "sha256": cache_sha,
            },
            "cache_sidecar": {
                "path": str(sidecar_path.resolve()),
                "sha256": sha256_file(sidecar_path),
            },
        }
    return provenance


def build_evidence(
    candidate_path: Path,
    query_path: Path,
    report_path: Path,
    sidecar_paths: Mapping[str, Path],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Validate one scorer output set and return release-ready arrays/manifest body."""
    for path in (candidate_path, query_path, report_path, *sidecar_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != SCORER_STATUS:
        raise RuntimeError("cross-study scorer status drifted")
    if report.get("labels_or_responses_accessed") is not False:
        raise RuntimeError("application evidence must be built without labels or responses")
    if report.get("candidate_admission") != SCORER_CANDIDATE_ADMISSION:
        raise RuntimeError("application candidate-admission contract drifted")
    if report.get("weighted_entropy_backend") != SCORER_WSE_BACKEND:
        raise RuntimeError("application WSE source is not the canonical frozen handoff")
    methods = report.get("methods")
    if not isinstance(methods, list) or not set(REQUIRED_SCORE_NAMES).issubset(methods):
        raise RuntimeError("scorer output lacks one or more Full23 score families")

    candidate_identity = _reported_output(report, "candidate_scores", candidate_path)
    query_identity = _reported_output(report, "query_registry", query_path)
    encoder_provenance = _load_encoder_provenance(report, sidecar_paths)
    inputs = report.get("inputs")
    msg = inputs.get("noise_msg_composite") if isinstance(inputs, dict) else None
    if not isinstance(msg, dict):
        raise RuntimeError("scorer report lacks frozen Noise-MSG provenance")
    msg_sha = _valid_sha256(msg.get("model_sha256"), "Noise-MSG model")
    if msg_sha != EXPECTED_UPSTREAM_SHA256[MSG_FAMILY]:
        raise RuntimeError("Noise-MSG frozen artifact identity drifted")

    candidates = pd.read_csv(
        candidate_path,
        sep="\t",
        keep_default_na=False,
        dtype={
            "candidate_pair_id": "string",
            "query_block_id": "string",
            "query_id": "string",
            "candidate_id": "string",
        },
    )
    queries = pd.read_csv(
        query_path,
        sep="\t",
        keep_default_na=False,
        dtype={"query_block_id": "string", "query_id": "string"},
    )
    query_ids = _required_text(queries, "query_block_id")
    if len(set(query_ids.tolist())) != len(query_ids):
        raise RuntimeError("directional query registry has duplicate query_block_id")
    if "candidate_count" not in queries:
        raise RuntimeError("directional query registry lacks candidate_count")
    counts_numeric = pd.to_numeric(queries["candidate_count"], errors="raise").to_numpy(np.float64)
    if not np.all(np.isfinite(counts_numeric)) or np.any(counts_numeric < 0) or np.any(counts_numeric != np.floor(counts_numeric)):
        raise RuntimeError("candidate_count must contain finite nonnegative integers")
    counts = counts_numeric.astype(np.int64)
    query_ptr = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(counts)))
    if int(query_ptr[-1]) != len(candidates):
        raise RuntimeError("query registry candidate counts do not match candidate-score rows")

    candidate_query_ids = _required_text(candidates, "query_block_id")
    candidate_ids = _required_text(candidates, "candidate_id")
    candidate_pair_ids = _required_text(candidates, "candidate_pair_id")
    axis = validate_axis(
        query_ptr, query_ids, candidate_query_ids, candidate_ids, candidate_pair_ids,
    )
    if axis.queries != PRODUCTION_QUERY_COUNT or axis.candidates != PRODUCTION_CANDIDATE_COUNT:
        raise RuntimeError(
            "production Full23 application axis drifted: "
            f"queries={axis.queries} candidates={axis.candidates}"
        )
    observed_cardinalities = {
        "zero": int(np.sum(counts == 0)),
        "singleton": int(np.sum(counts == 1)),
        "multiple": int(np.sum(counts > 1)),
    }
    if observed_cardinalities != PRODUCTION_QUERY_CARDINALITIES:
        raise RuntimeError(
            "production Full23 query-cardinality distribution drifted: "
            f"{observed_cardinalities}"
        )

    denominators = report.get("denominators")
    expected_denominators = {
        "directional_queries": axis.queries,
        "queries_with_identity_candidates": int(np.sum(counts > 0)),
        "queries_without_identity_candidates": int(np.sum(counts == 0)),
        "queries_with_one_identity_candidate": int(np.sum(counts == 1)),
        "queries_with_multiple_identity_candidates": int(np.sum(counts > 1)),
        "identity_candidate_pairs": axis.candidates,
    }
    if not isinstance(denominators, dict) or any(
        int(denominators.get(name, -1)) != value
        for name, value in expected_denominators.items()
    ):
        raise RuntimeError("scorer report denominators differ from the exact TSV axis")

    selected_columns = {name: SCORE_COLUMNS[name] for name in REQUIRED_SCORE_NAMES}
    evidence = {
        name: _numeric_vector(candidates, selected_columns[name])
        for name in REQUIRED_SCORE_NAMES
    }
    if "WSE_available" not in candidates:
        raise RuntimeError("candidate scores lack canonical WSE availability")
    wse_available = _strict_boolean(candidates["WSE_available"], "WSE_available")
    evidence["weighted_spectral_entropy"] = evidence["weighted_spectral_entropy"].copy()
    evidence["weighted_spectral_entropy"][~wse_available] = np.nan

    for name in ENCODER_NAMES:
        encoder_provenance[name]["score_source"] = {
            "path": candidate_identity["path"],
            "sha256": candidate_identity["sha256"],
            "column": selected_columns[name],
            "score_vector_sha256": _semantic_array_hash(
                {name: evidence[name]}, "full23_application_score_vector_v1",
            ),
        }

    arrays = {
        "query_ptr": axis.query_ptr,
        "query_ids": axis.query_ids,
        "candidate_query_ids": axis.candidate_query_ids,
        "candidate_ids": axis.candidate_ids,
        "candidate_pair_ids": axis.candidate_pair_ids,
        **evidence,
    }
    axis_arrays = {
        name: arrays[name]
        for name in (
            "query_ptr", "query_ids", "candidate_query_ids", "candidate_ids",
            "candidate_pair_ids",
        )
    }
    axis_sha = _semantic_array_hash(axis_arrays, "full23_application_candidate_axis_v1")

    score_artifacts: dict[str, dict[str, object]] = {}
    for name in ("weighted_spectral_entropy", "p2b_noise_v1_frozen"):
        score_artifacts[name] = {
            "sha256": candidate_identity["sha256"],
            "path": candidate_identity["path"],
            "column": selected_columns[name],
            "score_vector_sha256": _semantic_array_hash(
                {name: arrays[name]}, "full23_application_score_vector_v1",
            ),
        }
    upstream: dict[str, dict[str, object]] = {
        "candidate_axis": {
            "sha256": axis_sha,
            "semantic_hash_schema": "full23_application_candidate_axis_v1",
            "query_registry": query_identity,
            "candidate_scores": candidate_identity,
            **expected_denominators,
        },
        **encoder_provenance,
        **score_artifacts,
        MSG_FAMILY: {
            "sha256": msg_sha,
            "model_path": str(msg.get("model_path", "")),
            "report_path": str(msg.get("report_path", "")),
            "report_sha256": _valid_sha256(msg.get("report_sha256"), "Noise-MSG report"),
            "score_source": {
                "path": candidate_identity["path"],
                "sha256": candidate_identity["sha256"],
                "column": selected_columns[MSG_FAMILY],
                "score_vector_sha256": _semantic_array_hash(
                    {MSG_FAMILY: arrays[MSG_FAMILY]}, "full23_application_score_vector_v1",
                ),
            },
        },
        "chemaware_v2_continuous_null": {
            "mode": "unavailable",
            "sha256": CORE_SHA256,
            "contract": V2_UNAVAILABLE_CONTRACT,
        },
    }
    if set(upstream) != {"candidate_axis", *REQUIRED_SCORE_NAMES, "chemaware_v2_continuous_null"}:
        raise AssertionError("internal Full23 upstream registry construction drifted")
    manifest = {
        "schema": EVIDENCE_SCHEMA,
        "evidence_sha256": None,
        "required_score_names": list(REQUIRED_SCORE_NAMES),
        "v2_mode": "unavailable",
        "axis": {
            "queries": axis.queries,
            "candidates": axis.candidates,
            "zero_candidate_queries": int(np.sum(counts == 0)),
            "singleton_queries": int(np.sum(counts == 1)),
            "multiple_candidate_queries": int(np.sum(counts > 1)),
            "semantic_sha256": axis_sha,
        },
        "source_artifacts": {
            "candidate_scores": candidate_identity,
            "query_registry": query_identity,
            "scorer_report": {
                "path": str(report_path.resolve()),
                "sha256": sha256_file(report_path),
            },
        },
        "upstream_artifacts": upstream,
        "claim_boundary": (
            "Same-observed-precursor identity hypotheses on the frozen scorer axis only. "
            "ChemAware V2 is unavailable, not imputed; no labels, responses, BioAware "
            "context, candidate generation, or score recomputation enters this adapter."
        ),
    }
    return arrays, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scoring-dir", type=Path, required=True,
        help="Directory containing scorer candidate/query TSVs and report.json",
    )
    parser.add_argument(
        "--encoder-sidecar", action="append", required=True,
        help="Repeated NAME=PATH for the four application embedding-cache JSON sidecars",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    sidecars = _parse_sidecars(args.encoder_sidecar)
    arrays, manifest = build_evidence(
        args.scoring_dir / "identity_candidate_scores.tsv.gz",
        args.scoring_dir / "directional_query_registry.tsv.gz",
        args.scoring_dir / "report.json",
        sidecars,
    )

    args.output_dir.mkdir(parents=True)
    evidence_path = args.output_dir / OUTPUT_EVIDENCE
    manifest_path = args.output_dir / OUTPUT_MANIFEST
    np.savez_compressed(evidence_path, **arrays)
    manifest["evidence_sha256"] = sha256_file(evidence_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": "FULL23_APPLICATION_EVIDENCE_COMPLETE",
        "evidence": str(evidence_path),
        "manifest": str(manifest_path),
        "queries": manifest["axis"]["queries"],
        "candidates": manifest["axis"]["candidates"],
        "v2_mode": "unavailable",
    }, indent=2))


if __name__ == "__main__":
    main()
