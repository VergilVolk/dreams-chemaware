#!/usr/bin/env python3
"""Thin executable composition wrapper for Full23 v1.3.1.

The scalar Full23 v1 parent remains the candidate-evidence integrator. The
selected four-encoder tensor-set readout supplies one bounded residual on
queries for which its representation inputs are complete. Missing tensor
inputs fall back to the valid parent score; they are never replaced by zeros.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from full23_tensor_set_readout_core import ENCODER_NAMES, TOP_K, model_from_payload
from unified_additive_release import CandidateAxis, FrozenUnifiedAdditive


READOUT_SHA256 = (
    "6603013a04fdc26946445a6ba1ed566d768bac49423046ce3922daac79fe637e"
)
_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = _ROOT / "assets" / "full23_v1_parent.pt"
DEFAULT_READOUT = (
    _ROOT / "assets" / "full23_v131_query_centered_raw_penalty_readout.pt"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_readout(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or _sha256(path) != READOUT_SHA256:
        raise RuntimeError("Full23 v1.3.1 readout SHA256 mismatch")
    payload = _load(path)
    settings = payload.get("settings", {})
    expected = {
        "encoder_names": list(ENCODER_NAMES),
        "embedding_dimension": 1024,
        "pair_dimension": 64,
        "top_k": TOP_K,
        "include_reference_count": False,
        "query_center_raw_residual": True,
        "include_anchor_context": False,
        "include_signed_difference": False,
    }
    for field, value in expected.items():
        observed = settings.get(field)
        if field == "encoder_names":
            observed = list(observed or ())
        if observed != value:
            raise RuntimeError(f"v1.3.1 readout setting drifted: {field}")
    if payload.get("model_role") != "Full23_v13_cumulative_01":
        raise RuntimeError("unexpected v1.3.1 model role")
    if payload.get("training", {}).get("anchor_correct_multiplier") != 1.0:
        raise RuntimeError("v1.3.1 training multiplier drifted")
    return payload


class FrozenFull23V131:
    """Compose the frozen scalar parent and selected embedding residual."""

    def __init__(
        self,
        parent_checkpoint: Path = DEFAULT_PARENT,
        readout_checkpoint: Path = DEFAULT_READOUT,
        device: str = "cpu",
        batch_queries: int = 128,
    ):
        if batch_queries < 1:
            raise ValueError("batch_queries must be positive")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.parent = FrozenUnifiedAdditive(Path(parent_checkpoint), device=device)
        self.readout_path = Path(readout_checkpoint)
        self.readout = model_from_payload(
            validate_readout(self.readout_path), self.device
        )
        self.batch_queries = int(batch_queries)

    @staticmethod
    def _representation_contract(
        axis: CandidateAxis,
        representation_pair_ids: object,
        query_positions: object,
        reference_positions: object,
        reference_mask: object,
        embedding_banks: Mapping[str, object],
        embedding_bank_sha256: Mapping[str, str],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, np.ndarray],
        np.ndarray,
        dict[str, str],
    ]:
        size = axis.candidates
        encoder_count = len(ENCODER_NAMES)
        pair_ids = np.asarray(representation_pair_ids).astype("U")
        if not np.array_equal(pair_ids, axis.candidate_pair_ids):
            raise RuntimeError("representation rows do not match candidate_pair_ids")
        query = np.asarray(query_positions, dtype=np.int64)
        references = np.asarray(reference_positions, dtype=np.int64)
        mask = np.asarray(reference_mask, dtype=bool)
        if query.shape != (size, encoder_count):
            raise RuntimeError("query_positions axis mismatch")
        expected = (size, encoder_count, TOP_K)
        if references.shape != expected or mask.shape != expected:
            raise RuntimeError("reference position or mask axis mismatch")
        if set(embedding_banks) != set(ENCODER_NAMES):
            raise RuntimeError("embedding bank registry drifted")
        if set(embedding_bank_sha256) != set(ENCODER_NAMES):
            raise RuntimeError("embedding bank provenance registry drifted")
        provenance = {}
        for name, value in embedding_bank_sha256.items():
            value = str(value).lower()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise RuntimeError(f"{name} embedding bank SHA256 is malformed")
            provenance[name] = value

        multiplicity = mask.sum(axis=2)
        if np.any(multiplicity != multiplicity[:, :1]):
            raise RuntimeError("reference multiplicity differs across encoders")

        banks = {}
        row_valid = np.ones(size, dtype=bool)
        for encoder_index, name in enumerate(ENCODER_NAMES):
            bank = np.asarray(embedding_banks[name])
            if bank.ndim != 2 or bank.shape[1] != 1024:
                raise RuntimeError(f"{name} embedding bank must have shape (n, 1024)")
            if not np.all(np.isfinite(bank)):
                raise RuntimeError(f"{name} embedding bank contains non-finite values")
            banks[name] = bank
            row_valid &= (query[:, encoder_index] >= 0) & (
                query[:, encoder_index] < len(bank)
            )
            active = mask[:, encoder_index]
            valid_reference = (references[:, encoder_index] >= 0) & (
                references[:, encoder_index] < len(bank)
            )
            row_valid &= active[:, 0]
            row_valid &= np.all(~active | valid_reference, axis=1)

        # A single query must point to the same query spectrum on all of its
        # candidate rows. Different values indicate candidate-axis drift.
        for left, right in zip(axis.query_ptr[:-1], axis.query_ptr[1:]):
            left, right = int(left), int(right)
            if right > left and not np.all(query[left:right] == query[left]):
                raise RuntimeError("query embedding position changes within a query")
        return query, references, mask, banks, row_valid, provenance

    @torch.inference_mode()
    def score(
        self,
        axis: CandidateAxis,
        evidence: Mapping[str, object],
        representation_pair_ids: object,
        query_positions: object,
        reference_positions: object,
        reference_mask: object,
        embedding_banks: Mapping[str, object],
        embedding_bank_sha256: Mapping[str, str],
        chemical: Mapping[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        parent_score, parent_report = self.parent.score(axis, evidence, chemical)
        query, references, mask, cpu_banks, row_valid, provenance = self._representation_contract(
            axis,
            representation_pair_ids,
            query_positions,
            reference_positions,
            reference_mask,
            embedding_banks,
            embedding_bank_sha256,
        )
        output = parent_score.copy()
        widths = np.diff(axis.query_ptr)
        eligible_queries = []
        for query_index, (left, right) in enumerate(
            zip(axis.query_ptr[:-1], axis.query_ptr[1:])
        ):
            left, right = int(left), int(right)
            if (
                right - left >= 2
                and np.all(np.isfinite(parent_score[left:right]))
                and np.all(row_valid[left:right])
            ):
                eligible_queries.append(query_index)

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        banks = {
            name: torch.as_tensor(values, dtype=dtype, device=self.device)
            for name, values in cpu_banks.items()
        }
        residual_queries = 0
        residual_candidates = 0
        for start in range(0, len(eligible_queries), self.batch_queries):
            chosen = eligible_queries[start : start + self.batch_queries]
            parts = [
                np.arange(
                    int(axis.query_ptr[index]),
                    int(axis.query_ptr[index + 1]),
                    dtype=np.int64,
                )
                for index in chosen
            ]
            rows = np.concatenate(parts)
            ptr = np.concatenate(
                ([0], np.cumsum([len(part) for part in parts]))
            ).astype(np.int64)
            safe_references = references[rows].copy()
            safe_references[~mask[rows]] = 0
            values = self.readout(
                torch.as_tensor(parent_score[rows], device=self.device),
                torch.as_tensor(query[rows], device=self.device),
                torch.as_tensor(safe_references, device=self.device),
                torch.as_tensor(mask[rows], device=self.device),
                banks,
                query_ptr=ptr,
            )
            output[rows] = values.float().cpu().numpy()
            residual_queries += len(chosen)
            residual_candidates += len(rows)

        return output, {
            "schema": "dreams_unified_additive_full23_v131_prediction_v1",
            "parent": parent_report,
            "readout_sha256": READOUT_SHA256,
            "embedding_bank_sha256": provenance,
            "queries": axis.queries,
            "candidates": axis.candidates,
            "residual_scored_queries": residual_queries,
            "residual_scored_candidates": residual_candidates,
            "parent_fallback_nonempty_queries": int(
                np.sum(widths > 0) - residual_queries
            ),
            "missing_representation_semantics": "keep_parent_score",
            "bioaware": "not_in_full23_requires_separate_real_sample_context",
        }
