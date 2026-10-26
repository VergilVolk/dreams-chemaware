"""Core contracts for the consumed-Enveda candidate evidence integrator.

The integrator operates on candidate-level scalar evidence produced by frozen
upstream experts.  It intentionally contains no spectrum encoder and never
changes a candidate universe.  A small nonlinear residual head is the
predeclared primary model; a single linear layer is the capacity-lower
additive comparator, not a capacity-matched mechanism proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


PRIMARY_ARM = "nonlinear_reliability"
CONTROL_ARM = "additive_linear_control"
ARMS = (PRIMARY_ARM, CONTROL_ARM)
CHECKPOINT_SCHEMA = "enveda_unified_candidate_integrator_v1"


def unavailable_chemaware(candidate_count: int) -> dict[str, np.ndarray]:
    """Materialize the exact optional-V2 no-op accepted by the feature builder."""
    if candidate_count < 0:
        raise ValueError("candidate count cannot be negative")
    zero = np.zeros(candidate_count, dtype=np.float32)
    return {
        "chem_valid": np.zeros(candidate_count, dtype=bool),
        "candidate_utility_correct": zero.copy(),
        "candidate_utility_zero": zero.copy(),
        "candidate_utility_reversed": zero.copy(),
        "candidate_utility_rotated": zero.copy(),
    }


def formula_fold(value: str, folds: int, seed: int) -> int:
    """Stable, outcome-blind formula fold shared by every panel occurrence."""
    text = str(value).strip()
    if not text or folds < 2:
        raise ValueError("formula must be non-empty and folds must be at least two")
    payload = f"{seed}|{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % folds


def occurrence_weights(query_keys_by_panel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Give a repeated unique query total weight one across panel occurrences."""
    counts: dict[str, int] = {}
    normalized: dict[str, np.ndarray] = {}
    for panel, keys in query_keys_by_panel.items():
        values = np.asarray(keys).astype("U")
        if values.ndim != 1 or len(set(values.tolist())) != len(values):
            raise ValueError(f"query keys must be unique within panel {panel}")
        normalized[panel] = values
        for key in values:
            counts[str(key)] = counts.get(str(key), 0) + 1
    return {
        panel: np.asarray([1.0 / counts[str(key)] for key in keys], dtype=np.float32)
        for panel, keys in normalized.items()
    }


def validate_global_query_formula(
    query_keys_by_panel: dict[str, np.ndarray],
    formulas_by_panel: dict[str, np.ndarray],
) -> None:
    """Reject a repeated query whose formula identity changes between panels."""
    observed: dict[str, str] = {}
    if set(query_keys_by_panel) != set(formulas_by_panel):
        raise ValueError("query-key and formula panels differ")
    for panel in query_keys_by_panel:
        keys = np.asarray(query_keys_by_panel[panel]).astype("U")
        formulas = np.asarray(formulas_by_panel[panel]).astype("U")
        if keys.shape != formulas.shape:
            raise ValueError(f"query/formula shape mismatch in {panel}")
        for key, formula in zip(keys, formulas):
            formula = str(formula).strip()
            if not formula:
                raise ValueError(f"empty molecular formula in {panel}")
            previous = observed.setdefault(str(key), formula)
            if previous != formula:
                raise RuntimeError(
                    f"repeated query {key!r} changed formula: {previous!r} != {formula!r}"
                )


def candidate_row_weights(
    query_ptr: np.ndarray,
    queries: np.ndarray,
    query_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected candidate rows and query-balanced per-row weights."""
    ptr = np.asarray(query_ptr, dtype=np.int64)
    selected = np.asarray(queries, dtype=np.int64)
    weights = np.asarray(query_weights, dtype=np.float64)
    if weights.shape != (len(ptr) - 1,) or np.any(weights <= 0):
        raise ValueError("query weights must be positive and align to query_ptr")
    rows: list[np.ndarray] = []
    row_weights: list[np.ndarray] = []
    for query in selected:
        left, right = int(ptr[query]), int(ptr[query + 1])
        if right <= left:
            raise ValueError("empty candidate block")
        rows.append(np.arange(left, right, dtype=np.int64))
        row_weights.append(np.full(right - left, weights[query] / (right - left)))
    if not rows:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    return np.concatenate(rows), np.concatenate(row_weights)


def fit_weighted_normalization(
    feature_blocks: list[tuple[np.ndarray, np.ndarray]],
    feature_names: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one training-only scaler while preserving ChemAware's exact zero.

    Each block is ``(candidate_features, candidate_weights)``.  Query weights
    should already be divided across candidates, so a query duplicated in two
    panels contributes total mass one rather than two.
    """
    names = tuple(map(str, feature_names))
    if not feature_blocks:
        raise ValueError("normalization needs at least one training block")
    values = np.concatenate([np.asarray(block[0], dtype=np.float64) for block in feature_blocks])
    weights = np.concatenate([np.asarray(block[1], dtype=np.float64) for block in feature_blocks])
    if values.ndim != 2 or values.shape[1] != len(names) or weights.shape != (len(values),):
        raise ValueError("normalization feature/weight shape mismatch")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("normalization inputs must be finite with positive weights")
    weights = weights / np.sum(weights)
    mean = np.sum(values * weights[:, None], axis=0)
    scale = np.sqrt(np.sum(np.square(values - mean) * weights[:, None], axis=0))
    for index, name in enumerate(names):
        if name == "chemaware_v2_available":
            mean[index], scale[index] = 0.0, 1.0
        elif name.startswith("chemaware_v2_"):
            # Zero is a semantic no-op, not a missing value to be centered.
            mean[index] = 0.0
            scale[index] = np.sqrt(np.sum(np.square(values[:, index]) * weights))
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def transform_features(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if values.ndim != 2 or mean.shape != (values.shape[1],) or scale.shape != mean.shape:
        raise ValueError("feature normalization axis mismatch")
    output = (values - mean) / scale
    if not np.all(np.isfinite(output)):
        raise ValueError("non-finite normalized evidence")
    return output.astype(np.float32, copy=False)


@dataclass(frozen=True)
class IntegratorSettings:
    scalar_features: int
    anchor_index: int
    hidden: int = 16


class CandidateEvidenceIntegrator(nn.Module):
    """Small candidate scorer with a fixed WSE anchor initialization.

    The primary residual is deliberately unbounded.  Its size is controlled by
    the training-only anchor penalty, not by a hard ceiling that could make a
    real WSE error mathematically impossible to correct.
    """

    def __init__(self, arm: str, settings: IntegratorSettings):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown integrator arm: {arm}")
        if not 0 <= settings.anchor_index < settings.scalar_features:
            raise ValueError("anchor index is outside the scalar feature axis")
        if settings.hidden < 1 or settings.scalar_features < 2:
            raise ValueError("invalid integrator dimensions")
        self.arm = arm
        self.settings = settings
        if arm == CONTROL_ARM:
            self.network = nn.Linear(settings.scalar_features, 1)
            nn.init.zeros_(self.network.weight)
            nn.init.zeros_(self.network.bias)
            with torch.no_grad():
                self.network.weight[0, settings.anchor_index] = 1.0
        else:
            self.network = nn.Sequential(
                nn.Linear(settings.scalar_features, settings.hidden),
                nn.Tanh(),
                nn.Linear(settings.hidden, 1),
            )
            # At initialization the primary is exactly the WSE ordering.
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        if scalar.ndim != 2 or scalar.shape[1] != self.settings.scalar_features:
            raise ValueError("candidate scalar evidence axis mismatch")
        if self.arm == CONTROL_ARM:
            return self.network(scalar).squeeze(-1)
        anchor = scalar[:, self.settings.anchor_index]
        return anchor + self.network(scalar).squeeze(-1)

    def residual(self, scalar: torch.Tensor) -> torch.Tensor:
        return self.forward(scalar) - scalar[:, self.settings.anchor_index]


def weighted_listwise_loss(
    logits: torch.Tensor,
    query_ptr: np.ndarray,
    labels: np.ndarray,
    query_weights: np.ndarray,
) -> torch.Tensor:
    ptr = np.asarray(query_ptr, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int8)
    weights = torch.as_tensor(query_weights, dtype=logits.dtype, device=logits.device)
    if weights.shape != (len(ptr) - 1,) or torch.any(weights <= 0):
        raise ValueError("query weights do not align with packed query blocks")
    losses: list[torch.Tensor] = []
    for left, right in zip(ptr[:-1], ptr[1:]):
        left, right = int(left), int(right)
        positive = np.flatnonzero(labels[left:right] > 0)
        if len(positive) != 1:
            raise ValueError("every candidate block must have exactly one positive")
        target = torch.tensor([int(positive[0])], dtype=torch.long, device=logits.device)
        losses.append(torch.nn.functional.cross_entropy(logits[left:right][None, :], target))
    stacked = torch.stack(losses)
    return torch.sum(stacked * weights) / torch.sum(weights)


def weighted_query_residual_penalty(
    residual: torch.Tensor,
    query_ptr: np.ndarray,
    query_weights: np.ndarray,
) -> torch.Tensor:
    """Penalize residual size without overweighting large candidate blocks.

    Mean square is computed inside each query first, then combined with the
    same 1/k cross-panel occurrence weights as the listwise objective.
    """
    ptr = np.asarray(query_ptr, dtype=np.int64)
    weights = torch.as_tensor(
        query_weights, dtype=residual.dtype, device=residual.device,
    )
    if residual.ndim != 1 or ptr[0] != 0 or ptr[-1] != len(residual):
        raise ValueError("residual/query pointer axis mismatch")
    if weights.shape != (len(ptr) - 1,) or torch.any(weights <= 0):
        raise ValueError("query weights do not align with residual blocks")
    penalties = torch.stack([
        torch.mean(torch.square(residual[int(left):int(right)]))
        for left, right in zip(ptr[:-1], ptr[1:])
    ])
    return torch.sum(penalties * weights) / torch.sum(weights)


def checkpoint_model(payload: dict, arm: str, device: torch.device | str = "cpu") -> CandidateEvidenceIntegrator:
    """Reconstruct one arm from a saved final payload."""
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported unified-integrator checkpoint")
    if arm not in payload.get("arms", {}):
        raise RuntimeError(f"checkpoint does not contain arm {arm}")
    settings = IntegratorSettings(
        scalar_features=len(payload["scalar_feature_names"]),
        anchor_index=int(payload["anchor_index"]),
        hidden=int(payload["hidden"]),
    )
    model = CandidateEvidenceIntegrator(arm, settings)
    model.load_state_dict(payload["arms"][arm])
    return model.to(device).eval()


@torch.inference_mode()
def predict_from_payload(
    payload: dict,
    feature_names: Iterable[str],
    scalar_features: np.ndarray,
    arm: str = PRIMARY_ARM,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Use the exact saved training transform for GNPS or application inference."""
    observed = tuple(map(str, feature_names))
    expected = tuple(map(str, payload.get("scalar_feature_names", ())))
    if observed != expected:
        raise RuntimeError("deployment scalar feature contract drifted")
    transformed = transform_features(
        scalar_features,
        np.asarray(payload["scalar_mean"], dtype=np.float32),
        np.asarray(payload["scalar_scale"], dtype=np.float32),
    )
    model = checkpoint_model(payload, arm, device)
    tensor = torch.as_tensor(transformed, dtype=torch.float32, device=device)
    return model(tensor).cpu().numpy().astype(np.float32)


def load_and_predict(
    checkpoint: Path,
    feature_names: Iterable[str],
    scalar_features: np.ndarray,
    arm: str = PRIMARY_ARM,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was added.
        payload = torch.load(checkpoint, map_location="cpu")
    return predict_from_payload(payload, feature_names, scalar_features, arm, device)
