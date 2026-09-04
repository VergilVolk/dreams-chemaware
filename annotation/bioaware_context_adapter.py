"""Candidate-specific biological-context embedding adapter.

The adapter consumes a candidate's universal spectrum embedding plus typed,
sample-local context edges.  It never makes reaction neighbours same-molecule
positives.  With no context (or at initialization) it exactly returns the
universal embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiologicalContextAdapter(nn.Module):
    def __init__(
        self, embedding_dim: int, relation_types: int, hidden_dim: int = 128,
        relation_dim: int = 32, delta_bound: float = 0.08,
    ):
        super().__init__()
        if min(embedding_dim, relation_types, hidden_dim, relation_dim) <= 0 or delta_bound <= 0:
            raise ValueError("invalid context-adapter dimensions")
        self.embedding_dim = int(embedding_dim)
        self.delta_bound = float(delta_bound)
        self.relation = nn.Embedding(relation_types, relation_dim)
        edge_dim = embedding_dim + relation_dim + 4
        self.edge_value = nn.Sequential(
            nn.LayerNorm(edge_dim), nn.Linear(edge_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.edge_attention = nn.Sequential(
            nn.Linear(edge_dim + embedding_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.context_gate = nn.Sequential(
            nn.LayerNorm(embedding_dim + hidden_dim + 2),
            nn.Linear(embedding_dim + hidden_dim + 2, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.candidate_value = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, hidden_dim), nn.GELU(),
        )
        self.candidate_context_fusion = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim), nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(),
        )
        self.output = nn.Linear(hidden_dim, embedding_dim)
        # Exact fallback and exact official initialization.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        candidate_embedding: torch.Tensor,
        seed_embeddings: torch.Tensor,
        relation_type: torch.Tensor,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return contextual embedding, bounded delta, and context gate.

        Shapes are candidate ``[B,D]``, seeds ``[B,K,D]``, relations ``[B,K]``,
        edge features ``[B,K,4]`` and boolean mask ``[B,K]``.  The four edge
        features are expected to encode calibrated path confidence,
        experimental-layer support, reaction completeness and conflict.
        """
        if candidate_embedding.ndim != 2 or seed_embeddings.ndim != 3:
            raise RuntimeError("candidate/seed embedding rank mismatch")
        if seed_embeddings.shape[0] != candidate_embedding.shape[0] or seed_embeddings.shape[2] != candidate_embedding.shape[1]:
            raise RuntimeError("candidate/seed embedding shape mismatch")
        if relation_type.shape != seed_embeddings.shape[:2] or edge_features.shape != seed_embeddings.shape[:2] + (4,):
            raise RuntimeError("relation/edge feature shape mismatch")
        if edge_mask.shape != seed_embeddings.shape[:2]:
            raise RuntimeError("edge mask shape mismatch")
        relation = self.relation(relation_type.long())
        edge_input = torch.cat((seed_embeddings.float(), relation, edge_features.float()), dim=-1)
        expanded_candidate = candidate_embedding[:, None, :].expand(-1, seed_embeddings.shape[1], -1)
        attention_input = torch.cat((edge_input, expanded_candidate), dim=-1)
        logits = self.edge_attention(attention_input).squeeze(-1).masked_fill(~edge_mask, -1e4)
        weights = torch.softmax(logits, dim=1) * edge_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        context = torch.sum(weights.unsqueeze(-1) * self.edge_value(edge_input), dim=1)
        has_context = edge_mask.any(dim=1)
        confidence = torch.where(
            edge_mask, edge_features[..., 0].float().clamp(0, 1), torch.zeros_like(edge_features[..., 0]),
        )
        conflict = torch.where(
            edge_mask, edge_features[..., 3].float().clamp(0, 1), torch.zeros_like(edge_features[..., 3]),
        )
        summary = torch.stack((confidence.max(dim=1).values, conflict.max(dim=1).values), dim=-1)
        gate = torch.sigmoid(self.context_gate(torch.cat((candidate_embedding.float(), context, summary), dim=-1))).squeeze(-1)
        gate = gate * has_context.float() * (1.0 - summary[:, 1])
        candidate_value = self.candidate_value(candidate_embedding.float())
        fused = self.candidate_context_fusion(torch.cat(
            (candidate_value, context, candidate_value * context), dim=-1,
        ))
        raw_delta = self.output(fused)
        norm = raw_delta.norm(dim=1, keepdim=True)
        delta = gate[:, None] * self.delta_bound * raw_delta / (1.0 + norm)
        normalized = F.normalize(candidate_embedding.float() + delta, dim=-1)
        # Preserve the original tensor bit-for-bit at exact fallback while using
        # a straight-through normalized branch.  A plain torch.where here would
        # sever every adapter gradient at zero initialization.
        zero_delta = (delta.abs().sum(dim=1) == 0)[:, None]
        straight_through = candidate_embedding.float() + (normalized - normalized.detach())
        adapted = torch.where(zero_delta, straight_through, normalized)
        return adapted, delta, gate


class BiologicalEvidenceContextAdapter(nn.Module):
    """Bounded candidate embedding update from typed sample-context summaries.

    This variant is used when seed/path messages have already been reduced to a
    candidate-specific evidence vector by an outcome-blind graph executor.  It
    still changes the candidate embedding before cosine scoring; it is not a
    scalar reranker.  Rows without biological context return the universal
    embedding exactly.
    """

    def __init__(
        self, embedding_dim: int, evidence_dim: int, hidden_dim: int = 64,
        update_rank: int = 16, delta_bound: float = 0.05,
    ):
        super().__init__()
        if min(embedding_dim, evidence_dim, hidden_dim, update_rank) <= 0:
            raise ValueError("invalid evidence-context adapter dimensions")
        if delta_bound <= 0:
            raise ValueError("delta_bound must be positive")
        self.embedding_dim = int(embedding_dim)
        self.evidence_dim = int(evidence_dim)
        self.delta_bound = float(delta_bound)
        self.candidate = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, hidden_dim), nn.GELU(),
        )
        self.evidence = nn.Sequential(
            nn.LayerNorm(evidence_dim), nn.Linear(evidence_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, update_rank), nn.Tanh(),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim + 2),
            nn.Linear(2 * hidden_dim + 2, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output = nn.Linear(update_rank, embedding_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(
        self, candidate_embedding: torch.Tensor, evidence: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate_embedding.ndim != 2 or evidence.ndim != 2:
            raise RuntimeError("candidate/evidence rank mismatch")
        if candidate_embedding.shape[0] != evidence.shape[0]:
            raise RuntimeError("candidate/evidence batch mismatch")
        if candidate_embedding.shape[1] != self.embedding_dim:
            raise RuntimeError("candidate embedding dimension mismatch")
        if evidence.shape[1] != self.evidence_dim:
            raise RuntimeError("evidence dimension mismatch")
        if context_mask.shape != (candidate_embedding.shape[0],):
            raise RuntimeError("context mask shape mismatch")
        candidate = self.candidate(candidate_embedding.float())
        evidence_hidden = self.evidence(evidence.float())
        fused = self.fusion(torch.cat(
            (candidate, evidence_hidden, candidate * evidence_hidden), dim=-1,
        ))
        support = torch.stack((
            evidence.float().abs().mean(dim=-1),
            (evidence.float().abs() > 1e-8).float().mean(dim=-1),
        ), dim=-1)
        gate = torch.sigmoid(self.gate(torch.cat(
            (candidate, evidence_hidden, support), dim=-1,
        ))).squeeze(-1)
        gate = gate * context_mask.float()
        raw_delta = self.output(fused)
        raw_norm = raw_delta.norm(dim=-1, keepdim=True)
        delta = gate[:, None] * self.delta_bound * raw_delta / (1.0 + raw_norm)
        normalized = F.normalize(candidate_embedding.float() + delta, dim=-1)
        zero_delta = (delta.abs().sum(dim=1) == 0)[:, None]
        straight_through = candidate_embedding.float() + (normalized - normalized.detach())
        adapted = torch.where(zero_delta, straight_through, normalized)
        return adapted, delta, gate


def context_training_loss(
    query_embedding: torch.Tensor,
    contextual_candidate_embeddings: torch.Tensor,
    labels: torch.Tensor,
    universal_candidate_embeddings: torch.Tensor,
    gates: torch.Tensor,
    *, temperature: float = 0.1, preserve_weight: float = 2.0,
    context_sparsity_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Candidate-listwise objective with explicit universal-space protection."""
    if contextual_candidate_embeddings.ndim != 3 or labels.ndim != 1:
        raise RuntimeError("context listwise input shape mismatch")
    if torch.any(labels < 0) or torch.any(labels >= contextual_candidate_embeddings.shape[1]):
        raise RuntimeError("positive candidate index out of range")
    scores = torch.einsum("bd,bcd->bc", query_embedding.float(), contextual_candidate_embeddings.float())
    rank = F.cross_entropy(scores / temperature, labels.long())
    preserve = torch.clamp(
        1.0 - torch.sum(contextual_candidate_embeddings * universal_candidate_embeddings, dim=-1), min=0,
    ).mean()
    sparsity = gates.mean()
    total = rank + preserve_weight * preserve + context_sparsity_weight * sparsity
    return {"total": total, "rank": rank, "preserve": preserve, "gate_sparsity": sparsity, "scores": scores}
