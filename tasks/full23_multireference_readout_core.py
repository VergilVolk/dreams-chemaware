"""Full23-anchored multi-reference embedding readout.

The frozen Full23 Additive score remains the complete scalar-evidence model.
This module adds one narrowly scoped representation residual over the four
registered project encoders.  It never accepts the official DreaMS embedding
as a representation input and never retrains an encoder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from massspecgym_embedding_readout_core import REQUIRED_ENCODERS
from noise_corrected_fullgraph_evaluation import expanded_indices, score_embeddings


SCHEMA = "full23_anchored_multireference_readout_v1"
TOP1_ARM = "top1_embedding_residual_control"
TOPK_ARM = "top3_multireference_embedding_residual"
ARMS = (TOP1_ARM, TOPK_ARM)
ENCODER_NAMES = tuple(REQUIRED_ENCODERS)
TOP_K = 3


def _positions(rows: np.ndarray, requested: np.ndarray, label: str) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    requested = np.asarray(requested, dtype=np.int64)
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
        raise RuntimeError(f"{label} embedding rows must be one-dimensional and unique")
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    at = np.searchsorted(sorted_rows, requested)
    if np.any(at == len(sorted_rows)) or not np.array_equal(sorted_rows[at], requested):
        raise RuntimeError(f"embedding cache misses at least one {label} row")
    return order[at]


@dataclass(frozen=True)
class TopKReferenceIndex:
    """Bank-local and source-row identities for one encoder and panel."""

    query_positions: np.ndarray
    reference_positions: np.ndarray
    reference_rows: np.ndarray
    reference_mask: np.ndarray
    molecule_scores: np.ndarray


def stable_topk_references(
    graph: object,
    rows: np.ndarray,
    embeddings: np.ndarray,
    k: int = TOP_K,
) -> TopKReferenceIndex:
    """Select truth-blind Top-K references by frozen raw cosine.

    Ties are resolved by source row and then by original edge position.  A
    source row can contribute at most once to one candidate.  Short candidate
    reference sets are padded and masked; a spectrum is never duplicated to
    manufacture K observations.
    """
    if k != TOP_K:
        raise ValueError(f"registered multi-reference contract requires k={TOP_K}")
    rows = np.asarray(rows, dtype=np.int64)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    scored = score_embeddings(graph, rows, embeddings)
    pair_rows = np.asarray(graph.pair_candidate_row, dtype=np.int64)
    pair_positions = _positions(rows, pair_rows, "candidate")
    molecule_query, _ = expanded_indices(graph)
    query_by_query = _positions(rows, np.asarray(graph.query_row), "query")
    query_positions = query_by_query[molecule_query]
    count = len(np.asarray(graph.molecule_label))
    reference_positions = np.zeros((count, k), dtype=np.int64)
    reference_rows = np.full((count, k), -1, dtype=np.int64)
    reference_mask = np.zeros((count, k), dtype=bool)
    ptr = np.asarray(graph.molecule_ptr, dtype=np.int64)
    if len(ptr) != count + 1 or np.any(np.diff(ptr) < 1):
        raise RuntimeError("candidate reference blocks must be non-empty")
    for molecule, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        left, right = int(left), int(right)
        local_rows = pair_rows[left:right]
        if len(np.unique(local_rows)) > k:
            raise RuntimeError(
                f"candidate {molecule} has more than the registered {k} unique references"
            )
        if np.any(local_rows == int(graph.query_row[molecule_query[molecule]])):
            raise RuntimeError("query spectrum appears in its own candidate reference block")
        local_scores = scored.pair[left:right]
        local_order = np.lexsort((
            np.arange(right - left, dtype=np.int64), local_rows, -local_scores,
        ))
        selected: list[int] = []
        observed: set[int] = set()
        for offset in local_order:
            row = int(local_rows[int(offset)])
            if row in observed:
                continue
            observed.add(row)
            selected.append(left + int(offset))
            if len(selected) == k:
                break
        width = len(selected)
        reference_positions[molecule, :width] = pair_positions[selected]
        reference_rows[molecule, :width] = pair_rows[selected]
        reference_mask[molecule, :width] = True
    if not np.all(reference_mask[:, 0]):
        raise RuntimeError("every candidate must retain at least one reference")
    return TopKReferenceIndex(
        query_positions=query_positions,
        reference_positions=reference_positions,
        reference_rows=reference_rows,
        reference_mask=reference_mask,
        molecule_scores=np.asarray(scored.molecule, dtype=np.float32),
    )


@dataclass(frozen=True)
class MultiReferenceSettings:
    encoder_names: tuple[str, ...] = ENCODER_NAMES
    embedding_dimension: int = 1024
    projection_dimension: int = 16
    fusion_hidden: int = 32
    top_k: int = TOP_K


class Full23AnchoredMultiReferenceReadout(nn.Module):
    """One residual over four frozen-encoder multi-reference sets."""

    def __init__(self, settings: MultiReferenceSettings):
        super().__init__()
        if tuple(settings.encoder_names) != ENCODER_NAMES:
            raise ValueError(
                "representation registry must be exactly Noise, Stage-1, Phase-A, FullMSG"
            )
        if settings.top_k != TOP_K or settings.embedding_dimension < 1:
            raise ValueError("invalid registered multi-reference settings")
        if settings.projection_dimension < 1 or settings.fusion_hidden < 1:
            raise ValueError("invalid readout width")
        self.settings = settings
        self.projections = nn.ModuleDict({
            name: nn.Linear(
                settings.embedding_dimension, settings.projection_dimension, bias=False,
            )
            for name in settings.encoder_names
        })
        self.pair_tokens = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(2 * settings.projection_dimension, settings.projection_dimension),
                nn.GELU(),
            )
            for name in settings.encoder_names
        })
        pooled = len(settings.encoder_names) * 2 * settings.projection_dimension
        self.residual_head = nn.Sequential(
            nn.Linear(pooled, settings.fusion_hidden),
            nn.GELU(),
            nn.Linear(settings.fusion_hidden, 1),
        )
        # At initialization the model is exactly Full23 candidate by candidate.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def residual(
        self,
        query_positions: torch.Tensor,
        reference_positions: torch.Tensor,
        reference_mask: torch.Tensor,
        banks: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if set(banks) != set(self.settings.encoder_names):
            raise ValueError("embedding banks differ from the four registered project encoders")
        count = query_positions.shape[0]
        expected_query = (count, len(self.settings.encoder_names))
        expected_reference = (count, len(self.settings.encoder_names), self.settings.top_k)
        if query_positions.shape != expected_query:
            raise ValueError("query-position axis mismatch")
        if reference_positions.shape != expected_reference or reference_mask.shape != expected_reference:
            raise ValueError("multi-reference axis mismatch")
        if not torch.all(reference_mask[:, :, 0]):
            raise ValueError("every encoder/candidate needs at least one reference")
        pooled: list[torch.Tensor] = []
        for encoder_index, name in enumerate(self.settings.encoder_names):
            bank = banks[name]
            dtype = self.projections[name].weight.dtype
            query = bank[query_positions[:, encoder_index]].to(dtype=dtype)
            refs = bank[reference_positions[:, encoder_index]].to(dtype=dtype)
            q = F.normalize(self.projections[name](query), dim=-1)
            r = F.normalize(self.projections[name](refs), dim=-1)
            q = q[:, None, :].expand_as(r)
            token = self.pair_tokens[name](torch.cat((torch.abs(q - r), q * r), dim=-1))
            mask = reference_mask[:, encoder_index, :, None]
            valid = mask.sum(dim=1).clamp_min(1)
            mean = (token * mask).sum(dim=1) / valid
            maximum = token.masked_fill(~mask, torch.finfo(token.dtype).min).max(dim=1).values
            pooled.extend((mean, maximum))
        joined = torch.cat(pooled, dim=1)
        return self.residual_head(joined).squeeze(-1)

    def forward(
        self,
        anchor: torch.Tensor,
        query_positions: torch.Tensor,
        reference_positions: torch.Tensor,
        reference_mask: torch.Tensor,
        banks: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if anchor.shape != (query_positions.shape[0],):
            raise ValueError("Full23 anchor axis mismatch")
        return anchor + self.residual(
            query_positions, reference_positions, reference_mask, banks,
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def model_from_payload(
    payload: dict, arm: str, device: torch.device | str = "cpu",
) -> Full23AnchoredMultiReferenceReadout:
    if payload.get("schema") != SCHEMA:
        raise RuntimeError("unsupported multi-reference checkpoint schema")
    if tuple(payload.get("encoder_names", ())) != ENCODER_NAMES:
        raise RuntimeError("multi-reference encoder registry drifted")
    if payload.get("official_embedding_direct_input") is not False:
        raise RuntimeError("official DreaMS entered the representation head")
    if payload.get("full23_anchor_has_indirect_official_dependency") is not True:
        raise RuntimeError("Full23 indirect official dependency is not declared")
    if int(payload.get("top_k", -1)) != TOP_K:
        raise RuntimeError("multi-reference K drifted")
    if arm not in payload.get("arms", {}):
        raise RuntimeError(f"checkpoint lacks arm {arm}")
    raw = dict(payload.get("settings", {}))
    raw["encoder_names"] = tuple(raw.get("encoder_names", ENCODER_NAMES))
    settings = MultiReferenceSettings(**raw)
    if settings != MultiReferenceSettings():
        raise RuntimeError("multi-reference production settings drifted")
    if set(payload.get("arms", {})) != set(ARMS):
        raise RuntimeError("multi-reference arm registry drifted")
    model = Full23AnchoredMultiReferenceReadout(settings)
    model.load_state_dict(payload["arms"][arm])
    return model.to(device).eval()


def arm_mask(mask: np.ndarray, arm: str) -> np.ndarray:
    """Use identical parameters for Top-1 and Top-3 mechanism arms."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 3 or values.shape[2] != TOP_K:
        raise ValueError("registered reference mask axis mismatch")
    if arm == TOPK_ARM:
        return values.copy()
    if arm != TOP1_ARM:
        raise ValueError(arm)
    output = np.zeros_like(values)
    output[:, :, 0] = values[:, :, 0]
    return output
