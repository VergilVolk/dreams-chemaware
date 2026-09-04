"""ChemAware v2 components for a candidate-independent shared DreaMS encoder.

This module deliberately does *not* reuse the historical rule-attention path.
The deployable encoder consumes one clean spectrum and returns one normalized
embedding.  Candidate structures, molecular teachers, and chemical concepts
are training-only supervision and therefore live in loss helpers rather than
in :class:`ChemAwareSharedEncoder.forward`.

The design has three fail-safe properties:

* the residual adapter is zero initialized and exactly reproduces the official
  DreaMS embedding before training;
* query and reference spectra necessarily pass through the same encoder;
* the primary objective is molecule-level listwise retrieval over complete
  candidate groups, not rule overlap, pair AUROC, or a projection-only proxy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ChemAwareEncoderOutput:
    """Inspectable output of the shared encoder."""

    embedding: torch.Tensor
    official_embedding: torch.Tensor
    delta: torch.Tensor
    support_weights: torch.Tensor
    conflict_weights: torch.Tensor
    fragment_tokens: torch.Tensor
    fragment_mask: torch.Tensor


class SignedPeakResidualAdapter(nn.Module):
    """Zero-init residual with separate support and conflict peak channels.

    The two channels reflect the empirically distinct positive-deficit and
    negative-excess error families.  They remain candidate independent: both
    gates are functions only of contextual peak tokens and observed peak
    measurements.  A bounded residual protects the strong official geometry.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 192,
        delta_bound: float = 0.12,
        gate_temperature: float = 1.0,
        gate_topk: int = 0,
        contextual_gate: bool = False,
        global_branch: bool = False,
    ) -> None:
        super().__init__()
        if (
            embedding_dim <= 0 or hidden_dim < 8 or delta_bound <= 0
            or gate_temperature <= 0 or gate_topk < 0
        ):
            raise ValueError("invalid ChemAware adapter dimensions")
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.delta_bound = float(delta_bound)
        self.gate_temperature = float(gate_temperature)
        self.gate_topk = int(gate_topk)
        self.contextual_gate = bool(contextual_gate)
        self.global_branch = bool(global_branch)

        # m/z, intensity, and precursor-minus-fragment neutral loss are the
        # only non-learned measurements supplied at inference.
        input_dim = self.embedding_dim + 3
        self.token_norm = nn.LayerNorm(self.embedding_dim)
        self.support_value = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.conflict_value = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        gate_hidden = max(hidden_dim // 2, 8)
        self.support_gate = nn.Sequential(
            nn.Linear(input_dim, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1)
        )
        self.conflict_gate = nn.Sequential(
            nn.Linear(input_dim, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1)
        )
        if self.contextual_gate:
            self.support_query = nn.Linear(self.embedding_dim, gate_hidden, bias=False)
            self.support_key = nn.Linear(self.embedding_dim, gate_hidden, bias=False)
            self.conflict_query = nn.Linear(self.embedding_dim, gate_hidden, bias=False)
            self.conflict_key = nn.Linear(self.embedding_dim, gate_hidden, bias=False)
            self.support_context_scale = nn.Parameter(torch.tensor(2.0).log())
            self.conflict_context_scale = nn.Parameter(torch.tensor(2.0).log())
        if self.global_branch:
            self.global_value = nn.Sequential(
                nn.Linear(self.embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
        output_input_dim = (3 if self.global_branch else 2) * hidden_dim
        self.output = nn.Sequential(
            nn.LayerNorm(output_input_dim),
            nn.Linear(output_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embedding_dim),
        )
        # Exact official checkpoint reproduction is a hard contract.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _masked_weights(
        logits: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
        topk: int = 0,
    ) -> torch.Tensor:
        if logits.shape != mask.shape:
            raise RuntimeError("peak gate/mask shape mismatch")
        if temperature <= 0 or topk < 0:
            raise ValueError("invalid peak-gate temperature/top-k")
        if not torch.all(mask.any(dim=1)):
            raise RuntimeError("each spectrum needs at least one valid fragment peak")
        active = mask
        scaled = logits / temperature
        if topk:
            keep = min(topk, logits.shape[1])
            indices = torch.topk(scaled.masked_fill(~mask, -1e4), keep, dim=1).indices
            selected = torch.zeros_like(mask)
            selected.scatter_(1, indices, True)
            active = mask & selected
        weights = torch.softmax(scaled.masked_fill(~active, -1e4), dim=1)
        weights = weights * active.to(weights.dtype)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        official_embedding: torch.Tensor,
        peak_tokens: torch.Tensor,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        precursor_mz: torch.Tensor,
        peak_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if peak_tokens.ndim != 3 or peak_tokens.shape[:2] != peak_mask.shape:
            raise RuntimeError("peak token/mask shape mismatch")
        if official_embedding.shape != (peak_tokens.shape[0], self.embedding_dim):
            raise RuntimeError("official embedding shape mismatch")
        if peak_mz.shape != peak_mask.shape or peak_intensity.shape != peak_mask.shape:
            raise RuntimeError("peak measurement shape mismatch")
        if precursor_mz.shape != (peak_tokens.shape[0],):
            raise RuntimeError("precursor shape mismatch")

        token = self.token_norm(peak_tokens.float())
        neutral_loss = (precursor_mz[:, None].float() - peak_mz.float()).clamp_min(0.0)
        measurement = torch.stack(
            (peak_mz.float() / 1000.0, peak_intensity.float(), neutral_loss / 1000.0),
            dim=-1,
        )
        features = torch.cat((token, measurement), dim=-1)
        support_logits = self.support_gate(features).squeeze(-1)
        conflict_logits = self.conflict_gate(features).squeeze(-1)
        if self.contextual_gate:
            official_context = official_embedding.float()
            support_query = F.normalize(self.support_query(official_context), dim=-1)
            support_key = F.normalize(self.support_key(token), dim=-1)
            conflict_query = F.normalize(self.conflict_query(official_context), dim=-1)
            conflict_key = F.normalize(self.conflict_key(token), dim=-1)
            support_logits = support_logits + self.support_context_scale.exp() * torch.sum(
                support_key * support_query[:, None, :], dim=-1
            )
            conflict_logits = conflict_logits + self.conflict_context_scale.exp() * torch.sum(
                conflict_key * conflict_query[:, None, :], dim=-1
            )
        support_weights = self._masked_weights(
            support_logits, peak_mask,
            self.gate_temperature, self.gate_topk,
        )
        conflict_weights = self._masked_weights(
            conflict_logits, peak_mask,
            self.gate_temperature, self.gate_topk,
        )
        support = torch.sum(
            support_weights.unsqueeze(-1) * self.support_value(features), dim=1
        )
        conflict = torch.sum(
            conflict_weights.unsqueeze(-1) * self.conflict_value(features), dim=1
        )
        residual_inputs = [support, conflict]
        if self.global_branch:
            residual_inputs.append(self.global_value(official_embedding.float()))
        raw_delta = self.output(torch.cat(residual_inputs, dim=-1))
        raw_norm = raw_delta.norm(dim=1, keepdim=True)
        delta = self.delta_bound * raw_delta / (1.0 + raw_norm)
        official_float = official_embedding.float()
        adapted = F.normalize(official_float + delta, dim=-1)
        # Avoid a second normalization changing an already-normalized official
        # vector by a few ULPs at exact zero initialization.
        exact_zero = torch.all(delta == 0, dim=1, keepdim=True)
        # Straight-through exact value: forward is bitwise official, backward
        # still follows the normalized residual branch so the zero-initialized
        # output layer receives the first optimization step.
        exact_official_with_adapted_gradient = official_float + (adapted - adapted.detach())
        adapted = torch.where(exact_zero, exact_official_with_adapted_gradient, adapted)
        return adapted, delta, support_weights, conflict_weights


class ChemAwareSharedEncoder(nn.Module):
    """Official DreaMS identity model plus a shared peak residual adapter.

    ``official_model`` is expected to expose ``backbone`` and ``head`` like
    :class:`tasks.train_e1_identity.IdentityEmbeddingModel`.  No candidate or
    molecular input is accepted by ``forward`` by design.
    """

    def __init__(self, official_model: nn.Module, adapter: SignedPeakResidualAdapter):
        super().__init__()
        if not hasattr(official_model, "backbone") or not hasattr(official_model, "head"):
            raise TypeError("official_model must expose backbone and head")
        self.official_model = official_model
        self.adapter = adapter

    def forward(self, spectra: torch.Tensor) -> ChemAwareEncoderOutput:
        if spectra.ndim != 3 or spectra.shape[-1] != 2 or spectra.shape[1] < 2:
            raise RuntimeError("spectra must have shape (batch, precursor+peaks, 2)")
        tokens = self.official_model.backbone(spectra, None)
        official = F.normalize(self.official_model.head(tokens[:, 0, :]), dim=-1)
        fragments = tokens[:, 1:, :]
        fragment_mz = spectra[:, 1:, 0]
        fragment_intensity = spectra[:, 1:, 1]
        fragment_mask = fragment_mz > 0
        embedding, delta, support, conflict = self.adapter(
            official,
            fragments,
            fragment_mz,
            fragment_intensity,
            spectra[:, 0, 0],
            fragment_mask,
        )
        return ChemAwareEncoderOutput(
            embedding=embedding,
            official_embedding=official,
            delta=delta,
            support_weights=support,
            conflict_weights=conflict,
            fragment_tokens=fragments,
            fragment_mask=fragment_mask,
        )


class ChemAwareEmbeddingInference(nn.Module):
    """Tensor-only deployment view; training-only outputs are discarded."""

    def __init__(self, shared_encoder: ChemAwareSharedEncoder):
        super().__init__()
        self.shared_encoder = shared_encoder

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.shared_encoder(spectra).embedding


def molecule_scores_from_spectrum_pairs(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    pair_query_index: torch.Tensor,
    molecule_ptr: torch.Tensor,
) -> torch.Tensor:
    """Aggregate spectrum-pair cosine scores to candidate molecules by max."""
    if query_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
        raise RuntimeError("embeddings must be matrices")
    if candidate_embeddings.shape[0] != pair_query_index.numel():
        raise RuntimeError("candidate/query pair mismatch")
    if molecule_ptr.ndim != 1 or molecule_ptr.numel() < 2:
        raise RuntimeError("invalid molecule pointer")
    if int(molecule_ptr[0]) != 0 or int(molecule_ptr[-1]) != len(candidate_embeddings):
        raise RuntimeError("molecule pointer does not span candidate pairs")
    pair_scores = torch.sum(
        query_embeddings[pair_query_index.long()] * candidate_embeddings, dim=1
    )
    return torch.stack([
        torch.max(pair_scores[int(left):int(right)])
        for left, right in zip(molecule_ptr[:-1], molecule_ptr[1:])
    ])


def molecule_listwise_loss_per_query(
    molecule_scores: torch.Tensor,
    query_ptr: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Return one complete-candidate cross entropy per query.

    Ties are not specially broken during training.  Formal evaluation must use
    the project's strict rank implementation where every tie counts against
    the positive molecule.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if query_ptr.ndim != 1 or query_ptr.numel() < 2:
        raise RuntimeError("invalid query pointer")
    if int(query_ptr[0]) != 0 or int(query_ptr[-1]) != len(molecule_scores):
        raise RuntimeError("query pointer does not span molecule scores")
    losses = []
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left_i, right_i = int(left), int(right)
        if right_i - left_i < 2:
            raise RuntimeError("each query needs a positive and a negative molecule")
        losses.append(-F.log_softmax(molecule_scores[left_i:right_i] / temperature, dim=0)[0])
    return torch.stack(losses)


def molecule_listwise_loss(
    molecule_scores: torch.Tensor,
    query_ptr: torch.Tensor,
    temperature: float = 0.07,
    query_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean or query-weighted complete-candidate cross entropy."""

    values = molecule_listwise_loss_per_query(
        molecule_scores, query_ptr, temperature
    )
    if query_weights is None:
        return values.mean()
    if query_weights.shape != values.shape or torch.any(query_weights < 0):
        raise RuntimeError("invalid query weights")
    weights = query_weights / query_weights.sum().clamp_min(1e-12)
    return torch.sum(values * weights)


def positive_reference_increment_loss(
    query_embeddings: torch.Tensor,
    positive_reference_embeddings: torch.Tensor,
    positive_reference_ptr: torch.Tensor,
    official_query_embeddings: torch.Tensor,
    official_positive_reference_embeddings: torch.Tensor,
    increment: float = 0.01,
    query_weights: torch.Tensor | None = None,
    aggregation: str = "mean",
) -> torch.Tensor:
    """Increase same-identity cross-spectrum similarity relative to official DreaMS.

    This training-only term exposes every positive reference spectrum available
    for a query.  Targets are anchored to the frozen official pair geometry plus
    a small bounded increment, avoiding an absolute collapse-to-one objective.
    ``aggregation='worst'`` emphasizes the least improved positive view per
    query; ``'mean'`` uses all available views.  The caller must audit reference
    multiplicity before describing the objective as multi-positive.
    """

    if increment < 0:
        raise ValueError("positive-reference increment must be non-negative")
    if aggregation not in {"mean", "worst"}:
        raise ValueError("unsupported positive-reference aggregation")
    if query_embeddings.shape != official_query_embeddings.shape:
        raise RuntimeError("new/official query embedding mismatch")
    if positive_reference_embeddings.shape != official_positive_reference_embeddings.shape:
        raise RuntimeError("new/official positive reference mismatch")
    if positive_reference_ptr.ndim != 1 or positive_reference_ptr.numel() != len(query_embeddings) + 1:
        raise RuntimeError("positive-reference pointer/query mismatch")
    if int(positive_reference_ptr[0]) != 0 or int(positive_reference_ptr[-1]) != len(positive_reference_embeddings):
        raise RuntimeError("positive-reference pointer does not span references")
    losses = []
    for query, (left, right) in enumerate(
        zip(positive_reference_ptr[:-1], positive_reference_ptr[1:])
    ):
        left_i, right_i = int(left), int(right)
        if right_i <= left_i:
            raise RuntimeError("each query requires at least one positive reference")
        new_similarity = positive_reference_embeddings[left_i:right_i] @ query_embeddings[query]
        with torch.no_grad():
            old_similarity = (
                official_positive_reference_embeddings[left_i:right_i]
                @ official_query_embeddings[query]
            )
            target = torch.clamp(old_similarity + increment, max=1.0)
        pair_loss = F.relu(target - new_similarity)
        losses.append(pair_loss.mean() if aggregation == "mean" else pair_loss.max())
    values = torch.stack(losses)
    if query_weights is None:
        return values.mean()
    if query_weights.shape != values.shape or torch.any(query_weights < 0):
        raise RuntimeError("invalid positive-reference query weights")
    weights = query_weights / query_weights.sum().clamp_min(1e-12)
    return torch.sum(values * weights)


def chemical_weighted_listwise_loss(
    molecule_scores: torch.Tensor,
    molecule_embeddings: torch.Tensor,
    query_ptr: torch.Tensor,
    temperature: float = 0.07,
    hardness_beta: float = 4.0,
    observable: torch.Tensor | None = None,
    query_weights: torch.Tensor | None = None,
    weighting: str = "relative_centered",
) -> torch.Tensor:
    """Listwise retrieval with frozen chemical-neighbor hardness weights.

    For every query, cosine similarity between the positive molecule teacher
    vector and each negative teacher vector defines relative negative
    hardness.  ``relative_centered`` normalizes negative log-weights within a
    query and therefore needs at least two observed negatives.
    ``absolute_bounded`` gives every observed negative a bounded weight in
    ``[1, 2]`` based on its absolute teacher similarity to the positive.  The
    latter remains informative for the common one-negative candidate group;
    matched marginal/permutation controls are required because it also changes
    how strongly different queries contribute.  Unobservable molecules always
    receive neutral weight 1.

    This objective acts directly on deployable spectrum-spectrum scores; no
    training-only spectrum projector can absorb the chemical gradient.
    """
    if temperature <= 0 or hardness_beta < 0:
        raise ValueError("invalid chemical listwise temperature/beta")
    if weighting not in {"relative_centered", "absolute_bounded"}:
        raise ValueError(f"unknown chemical weighting: {weighting}")
    if molecule_embeddings.ndim != 2 or len(molecule_embeddings) != len(molecule_scores):
        raise RuntimeError("chemical teacher/molecule score shape mismatch")
    if int(query_ptr[0]) != 0 or int(query_ptr[-1]) != len(molecule_scores):
        raise RuntimeError("query pointer does not span chemical molecule scores")
    if observable is None:
        observable = torch.ones(
            len(molecule_scores), dtype=torch.bool, device=molecule_scores.device
        )
    if observable.shape != (len(molecule_scores),):
        raise RuntimeError("chemical observability shape mismatch")
    losses = []
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left_i, right_i = int(left), int(right)
        if right_i - left_i < 2:
            raise RuntimeError("chemical listwise group needs a negative")
        logits = molecule_scores[left_i:right_i] / temperature
        log_weight = torch.zeros_like(logits)
        group_observable = observable[left_i:right_i].bool()
        observed_negatives = int(torch.sum(group_observable[1:]))
        minimum_negatives = 2 if weighting == "relative_centered" else 1
        if bool(group_observable[0]) and observed_negatives >= minimum_negatives:
            teacher = F.normalize(molecule_embeddings[left_i:right_i].float(), dim=-1)
            similarity = teacher[1:] @ teacher[0]
            usable = group_observable[1:]
            if weighting == "relative_centered":
                raw = hardness_beta * similarity[usable]
                # Geometric-mean normalization gives mean log-weight zero and
                # is stable even when all observed similarities are identical.
                raw = raw - raw.mean()
            else:
                # A fixed center avoids query-wise cancellation.  The sigmoid
                # and log1p make the multiplicative negative weight bounded in
                # [1, 2], preventing a molecule teacher from overwhelming the
                # clean retrieval objective.
                raw = torch.log1p(
                    torch.sigmoid(hardness_beta * (similarity[usable] - 0.5))
                )
            negative_log_weight = torch.zeros_like(similarity)
            negative_log_weight[usable] = raw
            log_weight = torch.cat((log_weight[:1], negative_log_weight))
        losses.append(-logits[0] + torch.logsumexp(logits + log_weight, dim=0))
    values = torch.stack(losses)
    if query_weights is None:
        return values.mean()
    if query_weights.shape != values.shape or torch.any(query_weights < 0):
        raise RuntimeError("invalid chemical query weights")
    normalized = query_weights / query_weights.sum().clamp_min(1e-12)
    return torch.sum(values * normalized)


def chemical_margin_listwise_loss(
    molecule_scores: torch.Tensor,
    molecule_embeddings: torch.Tensor,
    query_ptr: torch.Tensor,
    temperature: float = 0.07,
    margin_scale: float = 0.03,
    observable: torch.Tensor | None = None,
    query_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Retrieval CE with a bounded frozen-structure margin on negatives.

    A negative structurally close to the positive receives a larger additive
    training margin, forcing the shared spectrum encoder to distinguish that
    hard decoy.  The margin changes only training logits and no molecular input
    is available at inference.  Correct/permuted/marginal teacher controls are
    mandatory because two-candidate groups can otherwise reduce this signal to
    query reweighting.
    """

    if temperature <= 0 or margin_scale < 0:
        raise ValueError("invalid chemical margin temperature/scale")
    if molecule_embeddings.ndim != 2 or len(molecule_embeddings) != len(molecule_scores):
        raise RuntimeError("chemical teacher/molecule score shape mismatch")
    if int(query_ptr[0]) != 0 or int(query_ptr[-1]) != len(molecule_scores):
        raise RuntimeError("query pointer does not span chemical molecule scores")
    if observable is None:
        observable = torch.ones(
            len(molecule_scores), dtype=torch.bool, device=molecule_scores.device
        )
    if observable.shape != (len(molecule_scores),):
        raise RuntimeError("chemical observability shape mismatch")

    losses = []
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left_i, right_i = int(left), int(right)
        if right_i - left_i < 2:
            raise RuntimeError("chemical margin group needs a negative")
        group_observable = observable[left_i:right_i].bool()
        margin = torch.zeros(
            right_i - left_i, dtype=molecule_scores.dtype,
            device=molecule_scores.device,
        )
        if bool(group_observable[0]) and bool(torch.any(group_observable[1:])):
            teacher = F.normalize(molecule_embeddings[left_i:right_i].float(), dim=-1)
            similarity = torch.clamp(teacher[1:] @ teacher[0], min=0.0, max=1.0)
            usable = group_observable[1:]
            negative_margin = torch.zeros_like(similarity)
            negative_margin[usable] = margin_scale * similarity[usable]
            margin = torch.cat((margin[:1], negative_margin.to(margin.dtype)))
        logits = (molecule_scores[left_i:right_i] + margin) / temperature
        losses.append(-F.log_softmax(logits, dim=0)[0])
    values = torch.stack(losses)
    if query_weights is None:
        return values.mean()
    if query_weights.shape != values.shape or torch.any(query_weights < 0):
        raise RuntimeError("invalid chemical margin query weights")
    normalized = query_weights / query_weights.sum().clamp_min(1e-12)
    return torch.sum(values * normalized)


def targeted_chemical_margin_increment(
    molecule_scores: torch.Tensor,
    official_scores: torch.Tensor,
    molecule_embeddings: torch.Tensor,
    query_ptr: torch.Tensor,
    temperature: float = 0.07,
    margin_scale: float = 0.10,
    similarity_threshold: float = 0.40,
    observable: torch.Tensor | None = None,
    query_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extra retrieval loss only for frozen-official errors with close decoys.

    Clean CE remains responsible for every query.  This function returns only
    ``CE(with structure margin) - CE(clean)`` on official-wrong groups.  Thus
    official-correct groups receive exactly zero chemical gradient.  Teacher
    cosine below ``similarity_threshold`` also receives zero margin, preserving
    capacity for easy but chemically distant retrieval errors.
    """

    if (
        temperature <= 0 or margin_scale < 0
        or not 0 <= similarity_threshold < 1
    ):
        raise ValueError("invalid targeted chemical margin contract")
    if molecule_scores.shape != official_scores.shape:
        raise RuntimeError("new/official molecule score shape mismatch")
    if molecule_embeddings.ndim != 2 or len(molecule_embeddings) != len(molecule_scores):
        raise RuntimeError("chemical teacher/molecule score shape mismatch")
    if int(query_ptr[0]) != 0 or int(query_ptr[-1]) != len(molecule_scores):
        raise RuntimeError("query pointer does not span targeted margin scores")
    if observable is None:
        observable = torch.ones(
            len(molecule_scores), dtype=torch.bool, device=molecule_scores.device
        )
    if observable.shape != (len(molecule_scores),):
        raise RuntimeError("chemical observability shape mismatch")

    increments = []
    active = []
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left_i, right_i = int(left), int(right)
        if right_i - left_i < 2:
            raise RuntimeError("targeted chemical margin group needs a negative")
        old = official_scores[left_i:right_i]
        group_observable = observable[left_i:right_i].bool()
        selected = bool(
            ((old[0] - torch.max(old[1:])) <= 0).detach()
            and group_observable[0]
            and torch.any(group_observable[1:])
        )
        logits = molecule_scores[left_i:right_i] / temperature
        clean = -F.log_softmax(logits, dim=0)[0]
        if selected:
            teacher = F.normalize(molecule_embeddings[left_i:right_i].float(), dim=-1)
            similarity = torch.clamp(teacher[1:] @ teacher[0], min=0.0, max=1.0)
            usable = group_observable[1:]
            scaled = torch.clamp(
                (similarity - similarity_threshold) / (1.0 - similarity_threshold),
                min=0.0, max=1.0,
            )
            negative_margin = torch.zeros_like(similarity)
            negative_margin[usable] = margin_scale * scaled[usable]
            margin = torch.cat((torch.zeros_like(negative_margin[:1]), negative_margin))
            chemical = -F.log_softmax(
                (molecule_scores[left_i:right_i] + margin.to(molecule_scores.dtype))
                / temperature,
                dim=0,
            )[0]
            increments.append(chemical - clean)
        else:
            increments.append(clean * 0.0)
        active.append(selected)
    values = torch.stack(increments)
    active_tensor = torch.tensor(active, dtype=torch.bool, device=molecule_scores.device)
    if not bool(torch.any(active_tensor)):
        return values.sum() * 0.0, active_tensor
    if query_weights is None:
        weights = active_tensor.to(values.dtype)
    else:
        if query_weights.shape != values.shape or torch.any(query_weights < 0):
            raise RuntimeError("invalid targeted chemical margin query weights")
        weights = query_weights * active_tensor.to(query_weights.dtype)
    weights = weights / weights.sum().clamp_min(1e-12)
    return torch.sum(values * weights), active_tensor


def protected_margin_loss(
    new_scores: torch.Tensor,
    official_scores: torch.Tensor,
    query_ptr: torch.Tensor,
    slack: float = 0.0,
) -> torch.Tensor:
    """Protect official-correct groups by preserving their worst margin.

    The loss is zero for official-wrong groups.  For official-correct groups it
    penalizes any drop below the official positive-vs-hardest-negative margin,
    optionally allowing ``slack``.
    """
    if new_scores.shape != official_scores.shape or slack < 0:
        raise RuntimeError("invalid preservation inputs")
    penalties = []
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left_i, right_i = int(left), int(right)
        old = official_scores[left_i:right_i]
        new = new_scores[left_i:right_i]
        old_margin = old[0] - torch.max(old[1:])
        if bool((old_margin > 0).detach()):
            new_margin = new[0] - torch.max(new[1:])
            penalties.append(F.relu(old_margin.detach() - new_margin - slack))
    if not penalties:
        return new_scores.sum() * 0.0
    return torch.stack(penalties).mean()


def masked_multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    observable: torch.Tensor,
    positive_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Chemical concept supervision with explicit missing/observable states."""
    if logits.shape != targets.shape or logits.shape != observable.shape:
        raise RuntimeError("concept tensor shape mismatch")
    mask = observable.bool()
    if not torch.any(mask):
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="none", pos_weight=positive_weight
    )
    return losses[mask].mean()


def identity_equal_weights(identities: Iterable[str], device: torch.device | None = None) -> torch.Tensor:
    """Give every identity equal total weight regardless of query multiplicity."""
    values = list(map(str, identities))
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    weights = torch.tensor([1.0 / counts[value] for value in values], device=device)
    return weights / weights.sum().clamp_min(1e-12)
