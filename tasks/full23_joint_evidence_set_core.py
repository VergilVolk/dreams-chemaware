"""Joint embedding and aligned spectral-reference tensor readout for Full23."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn

from full23_tensor_set_readout_core import (
    Full23TensorSetReadout,
    TensorSetSettings,
)


SPECTRAL_NAMES = (
    "noise_v1",
    "weighted_spectral_entropy",
    "p2b_noise_v1_frozen",
    "noise_msg_fusion_stack_v1",
)


@dataclass(frozen=True)
class JointEvidenceSettings:
    embedding: TensorSetSettings
    spectral_names: tuple[str, ...] = SPECTRAL_NAMES
    spectral_token_dimension: int = 16
    spectral_summary_dimension: int = 32
    fusion_hidden: int = 64
    include_spectral_count: bool = False
    query_center_raw_residual: bool = True


class Full23JointEvidenceSetReadout(nn.Module):
    """Full23 residual that delays pooling on both representation and score axes."""

    def __init__(
        self,
        settings: JointEvidenceSettings,
        residual_bound: float,
        spectral_mean: np.ndarray,
        spectral_scale: np.ndarray,
    ):
        super().__init__()
        if tuple(settings.spectral_names) != SPECTRAL_NAMES:
            raise ValueError("spectral family registry drifted")
        mean = np.asarray(spectral_mean, dtype=np.float32)
        scale = np.asarray(spectral_scale, dtype=np.float32)
        if mean.shape != (len(SPECTRAL_NAMES),) or scale.shape != mean.shape:
            raise ValueError("spectral normalization axis mismatch")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("invalid spectral normalization")
        self.settings = settings
        self.residual_bound = float(residual_bound)
        self.embedding_readout = Full23TensorSetReadout(settings.embedding, residual_bound)
        # Its original scalar head is not used; keeping the embedding token
        # encoder in one tested implementation avoids a second representation path.
        del self.embedding_readout.residual_head
        self.register_buffer("spectral_mean", torch.as_tensor(mean))
        self.register_buffer("spectral_scale", torch.as_tensor(scale))
        self.spectral_tokens = nn.Sequential(
            nn.Linear(len(SPECTRAL_NAMES), settings.spectral_token_dimension),
            nn.GELU(), nn.LayerNorm(settings.spectral_token_dimension),
        )
        self.spectral_attention = nn.Linear(settings.spectral_token_dimension, 1, bias=False)
        spectral_width = 2 * settings.spectral_token_dimension + (
            1 if settings.include_spectral_count else 0
        )
        self.spectral_summary = nn.Sequential(
            nn.Linear(spectral_width,
                      settings.spectral_summary_dimension),
            nn.GELU(), nn.LayerNorm(settings.spectral_summary_dimension),
        )
        embedding_width = (
            len(settings.embedding.encoder_names) * settings.embedding.encoder_dimension
        )
        self.residual_head = nn.Sequential(
            nn.Linear(embedding_width + settings.spectral_summary_dimension,
                      settings.fusion_hidden),
            nn.GELU(), nn.Linear(settings.fusion_hidden, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def spectral_set_summary(
        self, spectral_references: torch.Tensor, spectral_mask: torch.Tensor,
    ) -> torch.Tensor:
        expected = (spectral_references.shape[0], self.settings.embedding.top_k,
                    len(self.settings.spectral_names))
        if spectral_references.shape != expected:
            raise ValueError("spectral reference tensor axis mismatch")
        if spectral_mask.shape != expected[:2] or not torch.all(spectral_mask[:, 0]):
            raise ValueError("spectral reference mask is malformed")
        dtype = self.spectral_tokens[0].weight.dtype
        normalized = (
            spectral_references.to(dtype=dtype) - self.spectral_mean.to(dtype=dtype)
        ) / self.spectral_scale.to(dtype=dtype)
        token = self.spectral_tokens(normalized)
        logits = self.spectral_attention(token).squeeze(-1).masked_fill(~spectral_mask, -torch.inf)
        attention = torch.softmax(logits, dim=1)
        mean = torch.sum(attention[:, :, None] * token, dim=1)
        deviation = torch.sqrt(torch.sum(
            attention[:, :, None] * (token - mean[:, None, :]).square(), dim=1,
        ).clamp_min(1e-8))
        values = [mean, deviation]
        if self.settings.include_spectral_count:
            count = spectral_mask.sum(1, keepdim=True).to(dtype=dtype) / float(
                self.settings.embedding.top_k
            )
            values.append(count)
        return self.spectral_summary(torch.cat(values, dim=-1))

    def residual(
        self,
        query_positions: torch.Tensor,
        reference_positions: torch.Tensor,
        reference_mask: torch.Tensor,
        banks: Mapping[str, torch.Tensor],
        spectral_references: torch.Tensor,
        spectral_mask: torch.Tensor,
        query_ptr: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.embedding_readout.candidate_embedding_tokens(
            query_positions, reference_positions, reference_mask, banks,
        ).flatten(1)
        spectral = self.spectral_set_summary(spectral_references, spectral_mask)
        raw = self.residual_head(torch.cat((embedding, spectral), dim=1)).squeeze(-1)
        if self.settings.query_center_raw_residual:
            if query_ptr is None:
                raise ValueError("query-centered joint residual requires query_ptr")
            ptr = np.asarray(
                query_ptr.detach().cpu().numpy() if torch.is_tensor(query_ptr) else query_ptr,
                dtype=np.int64,
            )
            if ptr.ndim != 1 or ptr[0] != 0 or ptr[-1] != len(raw) or np.any(np.diff(ptr) < 2):
                raise ValueError("query_ptr is malformed")
            centered = raw.clone()
            for left, right in zip(ptr[:-1], ptr[1:]):
                left, right = int(left), int(right)
                centered[left:right] = raw[left:right] - raw[left:right].mean()
            raw = centered
        return self.residual_bound * torch.tanh(raw / self.residual_bound)

    def forward(
        self,
        anchor: torch.Tensor,
        query_positions: torch.Tensor,
        reference_positions: torch.Tensor,
        reference_mask: torch.Tensor,
        banks: Mapping[str, torch.Tensor],
        spectral_references: torch.Tensor,
        spectral_mask: torch.Tensor,
        query_ptr: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return anchor + self.residual(
            query_positions, reference_positions, reference_mask, banks,
            spectral_references, spectral_mask, query_ptr,
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def joint_model_from_payload(payload: dict, device: str = "cpu") -> Full23JointEvidenceSetReadout:
    if payload.get("schema") != "full23_joint_evidence_set_readout_v1":
        raise ValueError("unsupported joint-evidence checkpoint schema")
    settings_body = dict(payload["settings"])
    embedding_body = dict(settings_body.pop("embedding"))
    embedding_body["encoder_names"] = tuple(embedding_body["encoder_names"])
    settings_body["spectral_names"] = tuple(settings_body["spectral_names"])
    settings = JointEvidenceSettings(
        embedding=TensorSetSettings(**embedding_body), **settings_body,
    )
    model = Full23JointEvidenceSetReadout(
        settings,
        float(payload["residual_bound"]),
        np.asarray(payload["spectral_mean"], dtype=np.float32),
        np.asarray(payload["spectral_scale"], dtype=np.float32),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


class Full23SpectralSetResidual(nn.Module):
    """Incremental spectral-reference residual over an already learned anchor.

    This intentionally cannot perturb the representation readout.  It tests
    whether within-candidate reference consistency adds information beyond an
    embedding-derived score while preserving exact identity at initialization.
    """

    def __init__(
        self,
        top_k: int,
        residual_bound: float,
        spectral_mean: np.ndarray,
        spectral_scale: np.ndarray,
        token_dimension: int = 16,
        summary_dimension: int = 32,
        hidden: int = 32,
    ):
        super().__init__()
        mean = np.asarray(spectral_mean, dtype=np.float32)
        scale = np.asarray(spectral_scale, dtype=np.float32)
        if mean.shape != (len(SPECTRAL_NAMES),) or scale.shape != mean.shape:
            raise ValueError("spectral normalization axis mismatch")
        if np.any(scale <= 0) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("invalid spectral normalization")
        self.top_k = int(top_k)
        self.residual_bound = float(residual_bound)
        self.register_buffer("spectral_mean", torch.as_tensor(mean))
        self.register_buffer("spectral_scale", torch.as_tensor(scale))
        self.tokens = nn.Sequential(
            nn.Linear(len(SPECTRAL_NAMES), token_dimension),
            nn.GELU(), nn.LayerNorm(token_dimension),
        )
        self.attention = nn.Linear(token_dimension, 1, bias=False)
        self.summary = nn.Sequential(
            nn.Linear(2 * token_dimension + 1, summary_dimension),
            nn.GELU(), nn.LayerNorm(summary_dimension),
        )
        self.head = nn.Sequential(
            nn.Linear(summary_dimension, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def residual(self, references: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expected = (references.shape[0], self.top_k, len(SPECTRAL_NAMES))
        if references.shape != expected or mask.shape != expected[:2] or not torch.all(mask[:, 0]):
            raise ValueError("spectral reference tensor is malformed")
        dtype = self.tokens[0].weight.dtype
        normalized = (
            references.to(dtype=dtype) - self.spectral_mean.to(dtype=dtype)
        ) / self.spectral_scale.to(dtype=dtype)
        token = self.tokens(normalized)
        logits = self.attention(token).squeeze(-1).masked_fill(~mask, -torch.inf)
        weight = torch.softmax(logits, dim=1)
        mean = torch.sum(weight[:, :, None] * token, dim=1)
        deviation = torch.sqrt(torch.sum(
            weight[:, :, None] * (token - mean[:, None]).square(), dim=1,
        ).clamp_min(1e-8))
        count = mask.sum(1, keepdim=True).to(dtype=dtype) / float(self.top_k)
        summary = self.summary(torch.cat((mean, deviation, count), dim=-1))
        raw = self.head(summary).squeeze(-1)
        return self.residual_bound * torch.tanh(raw / self.residual_bound)

    def forward(
        self, anchor: torch.Tensor, references: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        return anchor + self.residual(references, mask)
