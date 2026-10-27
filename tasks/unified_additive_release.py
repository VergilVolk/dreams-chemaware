#!/usr/bin/env python3
"""Portable, fail-closed wrapper for the frozen Full23 Additive ranker.

This module does not reimplement the learned model or feature mathematics.  It
delegates feature construction and checkpoint inference to the exact modules
used by the frozen Enveda/GNPS runs.  The wrapper only validates a candidate
axis, enforces whole-query missingness, and writes fixed-axis predictions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch

from enveda_unified_candidate_integrator_core import (
    CHECKPOINT_SCHEMA,
    CONTROL_ARM,
    NO_QUERY_CONSTANT_CONTEXT_POLICY,
    predict_from_payload,
    unavailable_chemaware,
)
from massspecgym_embedding_readout_core import (
    REQUIRED_ENCODERS,
    build_independent_scalar_features,
)


RELEASE_SCHEMA = "dreams_unified_additive_full23_v1"
EVIDENCE_SCHEMA = "unified_additive_candidate_evidence_v1"
CHECKPOINT_SHA256 = "7f1309e2a89ef3640d9b6871a821df6866f2528e629547075d5305ab6707d717"
CORE_SHA256 = "b8c86591b1314321ad113440d8796b3deefefa96a874474a520335e3eab46a79"
V2_POLICY_SHA256 = "dc9d76dc439fca40db88e3a95ca33e6b59438a92e6a80080d8df77e4615e5a4a"
V2_UNAVAILABLE_CONTRACT = (
    "enveda_unified_candidate_integrator_core.unavailable_chemaware"
)
MSG_FAMILY = "noise_msg_fusion_stack_v1"
CHEMICAL_KEYS = (
    "chem_valid",
    "candidate_utility_correct",
    "candidate_utility_zero",
    "candidate_utility_reversed",
    "candidate_utility_rotated",
)
REQUIRED_SCORE_NAMES = (
    "noise_v1",
    "weighted_spectral_entropy",
    "p2b_noise_v1_frozen",
    MSG_FAMILY,
    "chemaware_stage1",
    "chemaware_phasea",
    "noise_chemaware_fullmsg",
)
EXPECTED_FEATURE_NAMES = (
    "noise_v1",
    "noise_v1_percentile",
    "weighted_spectral_entropy",
    "weighted_spectral_entropy_percentile",
    "p2b_noise_v1_frozen",
    "p2b_noise_v1_frozen_percentile",
    "noise_msg_fusion_stack_v1",
    "noise_msg_fusion_stack_v1_percentile",
    "chemaware_stage1_cosine",
    "chemaware_stage1_percentile",
    "chemaware_phasea_cosine",
    "chemaware_phasea_percentile",
    "noise_chemaware_fullmsg_cosine",
    "noise_chemaware_fullmsg_percentile",
    "reliability_top1_vote_fraction",
    "reliability_mean_percentile",
    "reliability_percentile_std",
    "chemaware_v2_available",
    "chemaware_v2_correct_minus_zero",
    "chemaware_v2_correct_minus_worst_null",
    "chemaware_v2_null_spread",
    "chemaware_v2_delta_percentile",
    "chemaware_v2_delta_top_gap",
)
_PACKAGE_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "assets" / "unified_candidate_integrator.pt"
)
_REPOSITORY_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "unified_additive_full23_v1"
    / "assets"
    / "unified_candidate_integrator.pt"
)
DEFAULT_CHECKPOINT = (
    _PACKAGE_CHECKPOINT if _PACKAGE_CHECKPOINT.is_file() else _REPOSITORY_CHECKPOINT
)
EXPECTED_UPSTREAM_SHA256 = {
    "noise_v1": "01ac8c7bfa75d89862e0b1eb2b5ffb297955cffae471fd483ad6c38d13f33beb",
    "chemaware_stage1": "09c419dcabf2ff424fe38bb33eaefcd25ccab0b6d125daf03b7959c25fff5838",
    "chemaware_phasea": "a8428329ca1d12bfe735f3f8b848ed020a8db18d0a654d4f62cdbeb006e61135",
    "noise_chemaware_fullmsg": "4f61a4efefe734b2c6bc66c06b315c37164b67e0f424208c3b8c7f81e6618130",
    MSG_FAMILY: "d257c5b03185953e3098bc82a7d268ec4037df5dee15d4f4ee6d415c23466062",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was added.
        return torch.load(path, map_location="cpu")


def _finite_vector(values: object, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,):
        raise RuntimeError(f"{name} must have shape ({size},)")
    return vector


def validate_checkpoint(path: Path) -> dict:
    """Load the one frozen checkpoint and reject any contract drift."""
    path = Path(path)
    if not path.is_file() or sha256_file(path) != CHECKPOINT_SHA256:
        raise RuntimeError("Full23 Additive checkpoint SHA256 mismatch")
    payload = _load_torch(path)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("Full23 Additive checkpoint schema mismatch")
    if payload.get("selection_class") != "candidate_evidence_integration":
        raise RuntimeError("checkpoint is not a candidate-evidence integrator")
    if payload.get("control_arm") != CONTROL_ARM or CONTROL_ARM not in payload.get("arms", {}):
        raise RuntimeError("checkpoint lacks the frozen Additive arm")
    if tuple(payload.get("scalar_feature_names", ())) != EXPECTED_FEATURE_NAMES:
        raise RuntimeError("Full23 feature names or order drifted")
    if payload.get("feature_selection_policy") != NO_QUERY_CONSTANT_CONTEXT_POLICY:
        raise RuntimeError("Full23 feature-selection policy drifted")
    if tuple(payload.get("encoder_names", ())) != tuple(REQUIRED_ENCODERS):
        raise RuntimeError("Full23 encoder registry drifted")
    expected_required = (
        "noise_v1",
        "chemaware_stage1",
        "chemaware_phasea",
        "noise_chemaware_fullmsg",
        "weighted_spectral_entropy",
        "p2b_noise_v1_frozen",
        MSG_FAMILY,
    )
    if tuple(payload.get("required_families", ())) != expected_required:
        raise RuntimeError("Full23 required-family registry drifted")
    if tuple(payload.get("optional_families", ())) != ("chemaware_v2_continuous_null",):
        raise RuntimeError("Full23 optional-family registry drifted")
    if payload.get("optional_v2_noop_helper") != (
        "enveda_unified_candidate_integrator_core.unavailable_chemaware"
    ):
        raise RuntimeError("Full23 V2-unavailable semantics drifted")
    if payload.get("anchor_feature") != "weighted_spectral_entropy" or int(
        payload.get("anchor_index", -1)
    ) != 2:
        raise RuntimeError("Full23 WSE anchor contract drifted")
    mean = np.asarray(payload.get("scalar_mean"), dtype=np.float32)
    scale = np.asarray(payload.get("scalar_scale"), dtype=np.float32)
    if (
        mean.shape != (23,)
        or scale.shape != (23,)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0)
    ):
        raise RuntimeError("Full23 normalization contract is malformed")
    state = payload["arms"][CONTROL_ARM]
    weight = np.asarray(state["network.weight"].detach().cpu())
    bias = np.asarray(state["network.bias"].detach().cpu())
    if weight.shape != (1, 23) or bias.shape != (1,) or not (
        np.all(np.isfinite(weight)) and np.all(np.isfinite(bias))
    ):
        raise RuntimeError("Full23 Additive state is malformed")
    return payload


@dataclass(frozen=True)
class CandidateAxis:
    query_ptr: np.ndarray
    query_ids: np.ndarray
    candidate_query_ids: np.ndarray
    candidate_ids: np.ndarray
    candidate_pair_ids: np.ndarray

    @property
    def queries(self) -> int:
        return len(self.query_ids)

    @property
    def candidates(self) -> int:
        return len(self.candidate_ids)


def validate_axis(
    query_ptr: object,
    query_ids: object,
    candidate_query_ids: object,
    candidate_ids: object,
    candidate_pair_ids: object,
) -> CandidateAxis:
    ptr = np.asarray(query_ptr, dtype=np.int64)
    queries = np.asarray(query_ids).astype("U")
    row_queries = np.asarray(candidate_query_ids).astype("U")
    candidates = np.asarray(candidate_ids).astype("U")
    pairs = np.asarray(candidate_pair_ids).astype("U")
    if ptr.ndim != 1 or len(ptr) != len(queries) + 1:
        raise RuntimeError("query_ptr must have one more entry than query_ids")
    if len(ptr) == 0 or ptr[0] != 0 or np.any(np.diff(ptr) < 0):
        raise RuntimeError("query_ptr must be zero-based and nondecreasing")
    size = int(ptr[-1])
    if any(values.shape != (size,) for values in (row_queries, candidates, pairs)):
        raise RuntimeError("candidate identifiers do not align to query_ptr")
    if len(set(queries.tolist())) != len(queries) or np.any(queries == ""):
        raise RuntimeError("query_ids must be non-empty and unique")
    if len(set(pairs.tolist())) != len(pairs) or np.any(pairs == ""):
        raise RuntimeError("candidate_pair_ids must be non-empty and unique")
    if np.any(candidates == ""):
        raise RuntimeError("candidate_ids must be non-empty")
    if not np.array_equal(row_queries, np.repeat(queries, np.diff(ptr))):
        raise RuntimeError("candidate rows are not contiguous on the declared query axis")
    if len(set(zip(row_queries.tolist(), candidates.tolist()))) != size:
        raise RuntimeError("a query cannot contain the same candidate identity twice")
    return CandidateAxis(ptr, queries, row_queries, candidates, pairs)


def _chemical_evidence(
    chemical: Mapping[str, object] | None, size: int,
) -> dict[str, np.ndarray]:
    if chemical is None:
        return unavailable_chemaware(size)
    if set(chemical) != set(CHEMICAL_KEYS):
        raise RuntimeError(
            "real ChemAware V2 evidence must contain exactly " + ", ".join(CHEMICAL_KEYS)
        )
    validity = np.asarray(chemical["chem_valid"])
    if validity.dtype.kind == "b":
        validity = validity.astype(bool, copy=False)
    elif validity.dtype.kind in "iu" and np.all((validity == 0) | (validity == 1)):
        validity = validity.astype(bool)
    else:
        raise RuntimeError("chem_valid must contain only booleans or integer 0/1")
    output = {"chem_valid": validity}
    if output["chem_valid"].shape != (size,):
        raise RuntimeError("chem_valid does not align to the candidate axis")
    for name in CHEMICAL_KEYS[1:]:
        output[name] = _finite_vector(chemical[name], size, name)
        if not np.all(np.isfinite(output[name])):
            raise RuntimeError(f"non-finite ChemAware V2 evidence: {name}")
    return output


def _selected_query_rows(
    query_ptr: np.ndarray, row_complete: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries: list[int] = []
    parts: list[np.ndarray] = []
    counts: list[int] = []
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        left, right = int(left), int(right)
        if right > left and bool(np.all(row_complete[left:right])):
            queries.append(query)
            parts.append(np.arange(left, right, dtype=np.int64))
            counts.append(right - left)
    rows = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    packed = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return rows, np.asarray(queries, dtype=np.int64), packed


class FrozenUnifiedAdditive:
    """Thin facade over the frozen feature builder and Additive checkpoint."""

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT, device: str = "cpu"):
        self.checkpoint = Path(checkpoint)
        self.payload = validate_checkpoint(self.checkpoint)
        self.device = device

    def score(
        self,
        axis: CandidateAxis,
        evidence: Mapping[str, object],
        chemical: Mapping[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        missing = sorted(set(REQUIRED_SCORE_NAMES) - set(evidence))
        if missing:
            raise RuntimeError(f"missing required candidate evidence: {missing}")
        values = {
            name: _finite_vector(evidence[name], axis.candidates, name)
            for name in REQUIRED_SCORE_NAMES
        }
        row_complete = np.column_stack([
            np.isfinite(values[name]) for name in REQUIRED_SCORE_NAMES
        ]).all(axis=1)
        chemistry = _chemical_evidence(chemical, axis.candidates)
        rows, queries, packed_ptr = _selected_query_rows(axis.query_ptr, row_complete)
        scores = np.full(axis.candidates, np.nan, dtype=np.float32)
        if len(rows):
            graph = SimpleNamespace(
                query_ptr=packed_ptr,
                molecule_ptr=np.arange(len(rows) + 1, dtype=np.int64),
                molecule_label=np.zeros(len(rows), dtype=np.int8),
            )
            encoder_scores = {
                name: values[name][rows]
                for name in REQUIRED_ENCODERS
            }
            spectral = {
                "v1_cosine": encoder_scores["noise_v1"],
                "weighted_entropy": values["weighted_spectral_entropy"][rows],
                "p2b_fused": values["p2b_noise_v1_frozen"][rows],
            }
            selected_chemistry = {
                name: np.asarray(body)[rows] for name, body in chemistry.items()
            }
            scalar, names = build_independent_scalar_features(
                spectral,
                graph,
                encoder_scores,
                "noise_v1",
                selected_chemistry,
                auxiliary_scores={MSG_FAMILY: values[MSG_FAMILY][rows]},
                include_reliability_context=True,
            )
            scores[rows] = predict_from_payload(
                self.payload,
                names,
                scalar,
                arm=CONTROL_ARM,
                device=self.device,
            )
        widths = np.diff(axis.query_ptr)
        query_scored = np.zeros(axis.queries, dtype=bool)
        query_scored[queries] = True
        return scores, {
            "schema": RELEASE_SCHEMA,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "arm": CONTROL_ARM,
            "feature_selection_policy": NO_QUERY_CONSTANT_CONTEXT_POLICY,
            "scalar_feature_names": list(EXPECTED_FEATURE_NAMES),
            "queries": axis.queries,
            "candidates": axis.candidates,
            "zero_candidate_queries": int(np.sum(widths == 0)),
            "singleton_queries": int(np.sum(widths == 1)),
            "multiple_candidate_queries": int(np.sum(widths > 1)),
            "scored_queries": int(query_scored.sum()),
            "abstained_nonempty_queries": int(np.sum(widths > 0) - query_scored.sum()),
            "scored_candidates": int(np.sum(np.isfinite(scores))),
            "v2_mode": "unavailable" if chemical is None else "real_or_mixed",
            "v2_available_candidates": int(np.sum(chemistry["chem_valid"])),
            "v2_available_queries": int(sum(
                bool(np.any(chemistry["chem_valid"][int(left):int(right)]))
                for left, right in zip(axis.query_ptr[:-1], axis.query_ptr[1:])
            )),
            "bioaware": "not_in_full23_requires_separate_real_sample_context",
        }


def _load_evidence(path: Path) -> tuple[CandidateAxis, dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    with np.load(path, allow_pickle=False) as body:
        axis = validate_axis(
            body["query_ptr"],
            body["query_ids"],
            body["candidate_query_ids"],
            body["candidate_ids"],
            body["candidate_pair_ids"],
        )
        evidence = {name: np.asarray(body[name]) for name in REQUIRED_SCORE_NAMES}
        present = [name in body.files for name in CHEMICAL_KEYS]
        if any(present) and not all(present):
            raise RuntimeError("ChemAware V2 arrays are partially present")
        chemical = (
            {name: np.asarray(body[name]) for name in CHEMICAL_KEYS}
            if all(present) else None
        )
    return axis, evidence, chemical


def _query_registry(
    axis: CandidateAxis, scores: np.ndarray,
) -> dict[str, np.ndarray]:
    widths = np.diff(axis.query_ptr)
    status = np.empty(axis.queries, dtype="U32")
    top_candidate = np.full(axis.queries, "", dtype="U256")
    top_pair = np.full(axis.queries, "", dtype="U256")
    top_score = np.full(axis.queries, np.nan, dtype=np.float32)
    top_ties = np.zeros(axis.queries, dtype=np.int32)
    for query, (left, right) in enumerate(zip(axis.query_ptr[:-1], axis.query_ptr[1:])):
        left, right = int(left), int(right)
        if right == left:
            status[query] = "no_candidates"
            continue
        block = scores[left:right]
        if not np.all(np.isfinite(block)):
            status[query] = "missing_required_evidence"
            continue
        maximum = float(np.max(block))
        winners = np.flatnonzero(block == maximum)
        top_ties[query] = len(winners)
        top_score[query] = maximum
        if len(winners) == 1:
            row = left + int(winners[0])
            top_candidate[query] = axis.candidate_ids[row]
            top_pair[query] = axis.candidate_pair_ids[row]
        status[query] = (
            "singleton_hypothesis" if right - left == 1
            else ("scored" if len(winners) == 1 else "top_score_tie")
        )
    return {
        "query_ids": axis.query_ids,
        "candidate_count": widths.astype(np.int64),
        "query_status": status,
        "top_candidate_id": top_candidate,
        "top_candidate_pair_id": top_pair,
        "top_score": top_score,
        "top_tie_count": top_ties,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.evidence_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != EVIDENCE_SCHEMA:
        raise RuntimeError("candidate-evidence manifest schema mismatch")
    if tuple(manifest.get("required_score_names", ())) != REQUIRED_SCORE_NAMES:
        raise RuntimeError("candidate-evidence family registry drifted")
    if manifest.get("evidence_sha256") != sha256_file(args.evidence):
        raise RuntimeError("candidate-evidence file hash mismatch")
    if not isinstance(manifest.get("upstream_artifacts"), dict):
        raise RuntimeError("candidate-evidence manifest lacks upstream artifact identities")
    upstream = manifest["upstream_artifacts"]
    expected_upstream = {"candidate_axis", *REQUIRED_SCORE_NAMES, "chemaware_v2_continuous_null"}
    if set(upstream) != expected_upstream:
        raise RuntimeError("candidate-evidence upstream artifact registry is incomplete or ambiguous")
    for name, identity in upstream.items():
        if not isinstance(identity, dict):
            raise RuntimeError(f"upstream artifact identity is malformed: {name}")
        digest = str(identity.get("sha256", "")).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"upstream artifact SHA256 is malformed: {name}")
        if name in EXPECTED_UPSTREAM_SHA256 and digest != EXPECTED_UPSTREAM_SHA256[name]:
            raise RuntimeError(f"upstream frozen artifact identity drifted: {name}")

    axis, evidence, chemical = _load_evidence(args.evidence)
    v2_identity = upstream["chemaware_v2_continuous_null"]
    v2_mode = v2_identity.get("mode")
    if v2_mode == "real":
        if chemical is None:
            raise RuntimeError("real ChemAware V2 mode requires all five evidence arrays")
        if v2_identity.get("policy_sha256") != V2_POLICY_SHA256:
            raise RuntimeError("real ChemAware V2 policy identity drifted")
    elif v2_mode == "unavailable":
        if chemical is not None:
            raise RuntimeError("unavailable ChemAware V2 mode forbids evidence arrays")
        if (
            v2_identity.get("contract") != V2_UNAVAILABLE_CONTRACT
            or v2_identity.get("sha256") != CORE_SHA256
        ):
            raise RuntimeError("ChemAware V2 unavailable contract drifted")
    else:
        raise RuntimeError("ChemAware V2 mode must be exactly real or unavailable")
    ranker = FrozenUnifiedAdditive(args.checkpoint, args.device)
    scores, report = ranker.score(axis, evidence, chemical)
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
        unified_candidate_ranker_score=scores,
        evidence_available=np.isfinite(scores),
    )
    np.savez_compressed(registry_path, **registry)
    report.update({
        "evidence": str(args.evidence),
        "evidence_sha256": sha256_file(args.evidence),
        "evidence_manifest": str(args.evidence_manifest),
        "evidence_manifest_sha256": sha256_file(args.evidence_manifest),
        "upstream_artifacts": manifest["upstream_artifacts"],
        "candidate_scores": score_path.name,
        "candidate_scores_sha256": sha256_file(score_path),
        "query_registry": registry_path.name,
        "query_registry_sha256": sha256_file(registry_path),
    })
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    checksums = args.output / "SHA256SUMS.txt"
    checksums.write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in (score_path, registry_path, report_path)
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "UNIFIED_ADDITIVE_FULL23_INFERENCE_COMPLETE",
        "output": str(args.output),
        "scored_queries": report["scored_queries"],
        "scored_candidates": report["scored_candidates"],
    }, indent=2))


if __name__ == "__main__":
    main()
