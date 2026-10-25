"""Core model and data helpers for frozen-encoder MassSpecGym readouts.

The head never changes an encoder.  It compares the same candidate molecules
with one of three fixed-capacity readouts: scalar evidence, projected cosine,
or a symmetric pair head over complete embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ARM_SCALAR = "scalar_evidence"
ARM_PROJECTED = "projected_cosine"
ARM_PAIR = "embedding_pair"
ARMS = (ARM_SCALAR, ARM_PROJECTED, ARM_PAIR)


def parse_named_paths(values: list[str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"invalid NAME=PATH value {value!r}")
        output.setdefault(name, []).append(path)
    if len(output) < 2:
        raise ValueError("at least two independently frozen encoders are required")
    return output


def percentile_by_query(values: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    """Average-tie percentile within every query candidate block."""
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left, right = int(left), int(right)
        block = values[left:right]
        order = np.argsort(block, kind="stable")
        sorted_values = block[order]
        ranks = np.empty(len(block), dtype=np.float64)
        start = 0
        while start < len(block):
            stop = start + 1
            while stop < len(block) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        output[left:right] = ranks / max(len(block) - 1, 1)
    return output


def query_gap_column(values: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    output = np.empty(len(values), dtype=np.float64)
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left, right = int(left), int(right)
        ordered = np.sort(values[left:right])
        gap = 0.0 if len(ordered) < 2 else float(ordered[-1] - ordered[-2])
        output[left:right] = gap
    return output


def masked_percentile_by_query(
    values: np.ndarray, valid: np.ndarray, query_ptr: np.ndarray,
) -> np.ndarray:
    """Percentiles among available candidates; unavailable remains exact zero."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    output = np.zeros(len(values), dtype=np.float64)
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left, right = int(left), int(right)
        local_valid = valid[left:right]
        if not local_valid.any():
            continue
        local_indices = np.flatnonzero(local_valid)
        local_values = values[left:right][local_valid]
        local_ptr = np.asarray([0, len(local_values)], dtype=np.int64)
        output[left + local_indices] = percentile_by_query(local_values, local_ptr)
    return output


def masked_query_gap_column(
    values: np.ndarray, valid: np.ndarray, query_ptr: np.ndarray,
) -> np.ndarray:
    """Available-only top gap; unavailable and singleton availability are zero."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    output = np.zeros(len(values), dtype=np.float64)
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left, right = int(left), int(right)
        local_valid = valid[left:right]
        if int(local_valid.sum()) < 2:
            continue
        available = np.sort(values[left:right][local_valid])
        gap = float(available[-1] - available[-2])
        output[left:right][local_valid] = gap
    return output


def fit_scalar_normalization(
    scalar: np.ndarray, feature_names: list[str], rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on training rows while preserving the ChemAware missing-value zero.

    Ordinary evidence is z-scored.  ChemAware V2 features encode absence as
    exactly zero, so they are scaled by training RMS without centering.  The
    availability bit stays in its native 0/1 scale.
    """
    scalar = np.asarray(scalar, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.int64)
    chosen = scalar[rows]
    mean = chosen.mean(0).astype(np.float32)
    scale = chosen.std(0).astype(np.float32)
    for index, name in enumerate(feature_names):
        if name == "chemaware_v2_available":
            mean[index], scale[index] = 0.0, 1.0
        elif name.startswith("chemaware_v2_"):
            mean[index] = 0.0
            scale[index] = float(np.sqrt(np.mean(np.square(chosen[:, index]))))
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return mean, scale


def build_independent_scalar_features(
    evidence: dict[str, np.ndarray], graph,
    encoder_scores: dict[str, np.ndarray], noise_name: str,
    chemical: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """One feature block per evidence family, without P2b double counting."""
    family_edges = {
        "noise_v1": np.asarray(evidence["v1_cosine"], dtype=np.float64),
        "weighted_spectral_entropy": np.asarray(evidence["weighted_entropy"], dtype=np.float64),
        "p2b_noise_v1_frozen": np.asarray(evidence["p2b_fused"], dtype=np.float64),
    }
    family = {
        name: np.maximum.reduceat(values, graph.molecule_ptr[:-1])
        for name, values in family_edges.items()
    }
    columns: list[np.ndarray] = []
    names: list[str] = []
    for name, values in family.items():
        columns.extend((values[:, None], percentile_by_query(values, graph.query_ptr)[:, None]))
        names.extend((name, f"{name}_percentile"))
    context: list[np.ndarray] = []
    context_names: list[str] = []
    query_molecules = np.diff(graph.query_ptr).astype(np.float64)
    query_edges = np.diff(graph.molecule_ptr[graph.query_ptr]).astype(np.float64)
    for values in (query_molecules, query_edges, np.log(np.maximum(query_edges, 1))):
        context.append(np.repeat(values, np.diff(graph.query_ptr))[:, None])
    context_names.extend(("query_molecules", "query_edges", "log_query_edges"))
    for name, values in family.items():
        context.append(query_gap_column(values, graph.query_ptr)[:, None])
        context_names.append(f"{name}_top_gap")
    family_names = tuple(family)
    for left_index, left in enumerate(family_names):
        for right in family_names[left_index + 1:]:
            agree = np.empty(len(graph.molecule_label), dtype=np.float64)
            for start, stop in zip(graph.query_ptr[:-1], graph.query_ptr[1:]):
                start, stop = int(start), int(stop)
                left_top = int(np.argmax(family[left][start:stop]))
                right_top = int(np.argmax(family[right][start:stop]))
                agree[start:stop] = float(left_top == right_top)
            context.append(agree[:, None])
            context_names.append(f"{left}_{right}_top1_agree")
    columns.extend(context)
    names.extend(context_names)
    for name, scores in encoder_scores.items():
        if name == noise_name:
            continue
        columns.extend((
            scores[:, None],
            percentile_by_query(scores, graph.query_ptr)[:, None],
            query_gap_column(scores, graph.query_ptr)[:, None],
        ))
        names.extend((f"{name}_cosine", f"{name}_percentile", f"{name}_top_gap"))
    if chemical is not None:
        valid = np.asarray(chemical["chem_valid"], dtype=bool)
        correct = np.asarray(chemical["candidate_utility_correct"], dtype=np.float64)
        nulls = np.stack([
            np.asarray(chemical[name], dtype=np.float64)
            for name in (
                "candidate_utility_zero", "candidate_utility_reversed",
                "candidate_utility_rotated",
            )
        ], axis=1)
        if valid.shape != correct.shape or nulls.shape != (len(correct), 3):
            raise RuntimeError("chemical candidate evidence shape drifted")
        delta_zero = correct - nulls[:, 0]
        delta_worst = correct - np.max(nulls, axis=1)
        null_spread = np.max(nulls, axis=1) - np.min(nulls, axis=1)
        for values in (delta_zero, delta_worst, null_spread):
            values[~valid] = 0.0
        columns.extend((
            valid.astype(np.float64)[:, None], delta_zero[:, None],
            delta_worst[:, None], null_spread[:, None],
            masked_percentile_by_query(delta_worst, valid, graph.query_ptr)[:, None],
            masked_query_gap_column(delta_worst, valid, graph.query_ptr)[:, None],
        ))
        names.extend((
            "chemaware_v2_available", "chemaware_v2_correct_minus_zero",
            "chemaware_v2_correct_minus_worst_null", "chemaware_v2_null_spread",
            "chemaware_v2_delta_percentile", "chemaware_v2_delta_top_gap",
        ))
    output = np.concatenate(columns, axis=1).astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("non-finite scalar feature")
    return output, names


def padded_candidate_matrix(query_ptr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    widths = np.diff(np.asarray(query_ptr, dtype=np.int64))
    maximum = int(widths.max())
    matrix = np.full((len(widths), maximum), -1, dtype=np.int64)
    mask = np.zeros_like(matrix, dtype=bool)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        width = int(right - left)
        matrix[query, :width] = np.arange(left, right, dtype=np.int64)
        mask[query, :width] = True
    return matrix, mask


def hard_competitor_matrix(
    query_ptr: np.ndarray,
    score_families: list[np.ndarray],
    maximum_negatives: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Positive plus family-specific hardest negatives for head fitting.

    Labels are known on MassSpecGym development data and the unique positive is
    first in each query.  The held-fold evaluation still scores every candidate.
    """
    rows: list[list[int]] = []
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left, right = int(left), int(right)
        negatives: list[int] = []
        for values in score_families:
            local = np.asarray(values[left + 1:right], dtype=np.float64)
            negatives.append(left + 1 + int(np.argmax(local)))
        # Add strongest average-percentile competitors when families nominate
        # the same molecule.  This is label-aware training, never evaluation.
        block = np.stack([v[left + 1:right] for v in score_families], axis=1)
        ranks = np.stack([
            percentile_by_query(block[:, column], np.asarray([0, len(block)]))
            for column in range(block.shape[1])
        ], axis=1).mean(axis=1)
        for local in np.argsort(ranks)[::-1]:
            negatives.append(left + 1 + int(local))
        unique: list[int] = []
        for index in negatives:
            if index not in unique:
                unique.append(index)
            if len(unique) == maximum_negatives:
                break
        rows.append([left, *unique])
    width = max(map(len, rows))
    matrix = np.full((len(rows), width), -1, dtype=np.int64)
    mask = np.zeros_like(matrix, dtype=bool)
    for query, row in enumerate(rows):
        matrix[query, :len(row)] = row
        mask[query, :len(row)] = True
    return matrix, mask


def strict_ranks(scores: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(query_ptr) - 1, dtype=np.int32)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        block = np.asarray(scores[int(left):int(right)], dtype=np.float64)
        ranks[query] = 1 + int(np.sum(block[1:] >= block[0]))
    return ranks


def rank_metrics(ranks: np.ndarray, near: np.ndarray) -> dict[str, float | int]:
    ranks = np.asarray(ranks, dtype=np.int32)
    near = np.asarray(near, dtype=bool)
    result: dict[str, float | int] = {
        "queries": int(len(ranks)),
        "recall1": float(np.mean(ranks <= 1)),
        "recall2": float(np.mean(ranks <= 2)),
        "recall3": float(np.mean(ranks <= 3)),
        "recall5": float(np.mean(ranks <= 5)),
        "recall10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "near_queries": int(near.sum()),
    }
    if near.any():
        result["near_recall1"] = float(np.mean(ranks[near] == 1))
        result["near_mrr"] = float(np.mean(1.0 / ranks[near]))
    return result


def formula_cluster_ci(
    values: np.ndarray, formulas: np.ndarray, repeats: int, seed: int,
    hypotheses: int = 24,
) -> dict[str, float | int]:
    unique, inverse = np.unique(np.asarray(formulas).astype(str), return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64))
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        take = rng.integers(0, len(unique), len(unique))
        boot[index] = sums[take].sum() / counts[take].sum()
    tail = 0.05 / (2 * hypotheses)
    return {
        "delta": float(np.mean(values)), "delta_pp": float(100 * np.mean(values)),
        "ci_low_pp": float(100 * np.quantile(boot, tail)),
        "ci_high_pp": float(100 * np.quantile(boot, 1 - tail)),
        "formula_clusters": int(len(unique)), "familywise_hypotheses": hypotheses,
    }


def comparison(
    candidate: np.ndarray, baseline: np.ndarray, query_ptr: np.ndarray,
    formulas: np.ndarray, repeats: int, seed: int,
) -> dict:
    candidate_rank = strict_ranks(candidate, query_ptr)
    baseline_rank = strict_ranks(baseline, query_ptr)
    candidate_ok, baseline_ok = candidate_rank == 1, baseline_rank == 1
    delta = candidate_ok.astype(float) - baseline_ok.astype(float)
    corrected = int(np.sum(candidate_ok & ~baseline_ok))
    introduced = int(np.sum(~candidate_ok & baseline_ok))
    return {
        "corrected": corrected, "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "paired_formula_ci": formula_cluster_ci(delta, formulas, repeats, seed),
    }


@dataclass(frozen=True)
class ReadoutSettings:
    scalar_features: int
    encoder_names: tuple[str, ...]
    embedding_dimension: int
    projection_dimension: int = 16
    scalar_width: int = 64
    # Registered production input (four encoders plus six V2 fields) yields a
    # scalar control within a few percent of the projected-cosine parameter count.
    scalar_control_width: int = 255


class FrozenEmbeddingReadout(nn.Module):
    """Candidate scorer over fixed scalar evidence and frozen embeddings."""

    def __init__(self, arm: str, settings: ReadoutSettings):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.settings = settings
        if arm == ARM_SCALAR:
            width = settings.scalar_control_width
            self.scalar = nn.Sequential(
                nn.Linear(settings.scalar_features, width), nn.GELU(),
                nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1),
            )
            self.projections = nn.ModuleDict()
            self.pair_heads = nn.ModuleDict()
        else:
            width = settings.scalar_width
            self.scalar = nn.Sequential(
                nn.Linear(settings.scalar_features, width), nn.GELU(),
            )
            self.projections = nn.ModuleDict({
                name: nn.Linear(
                    settings.embedding_dimension,
                    settings.projection_dimension,
                    bias=False,
                )
                for name in settings.encoder_names
            })
            self.pair_heads = nn.ModuleDict()
            if arm == ARM_PAIR:
                self.pair_heads = nn.ModuleDict({
                    name: nn.Sequential(
                        nn.Linear(2 * settings.projection_dimension,
                                  settings.projection_dimension),
                        nn.GELU(), nn.Linear(settings.projection_dimension, 1),
                    )
                    for name in settings.encoder_names
                })
            self.final = nn.Sequential(
                nn.Linear(width + len(settings.encoder_names), width),
                nn.GELU(), nn.Linear(width, 1),
            )

    def forward(
        self,
        scalar: torch.Tensor,
        query_rows: torch.Tensor | None = None,
        reference_rows: torch.Tensor | None = None,
        banks: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.arm == ARM_SCALAR:
            return self.scalar(scalar).squeeze(-1)
        if query_rows is None or reference_rows is None or banks is None:
            raise ValueError("embedding arms require query/reference rows and banks")
        scalar_hidden = self.scalar(scalar)
        evidence: list[torch.Tensor] = []
        for encoder_index, name in enumerate(self.settings.encoder_names):
            bank = banks[name]
            dtype = self.projections[name].weight.dtype
            query = bank[query_rows].to(dtype=dtype)
            reference = bank[reference_rows[:, encoder_index]].to(dtype=dtype)
            q = F.normalize(self.projections[name](query), dim=-1)
            r = F.normalize(self.projections[name](reference), dim=-1)
            if self.arm == ARM_PROJECTED:
                value = torch.sum(q * r, dim=-1)
            else:
                pair = torch.cat((torch.abs(q - r), q * r), dim=-1)
                value = self.pair_heads[name](pair).squeeze(-1)
            evidence.append(value)
        return self.final(torch.cat((scalar_hidden, torch.stack(evidence, 1)), 1)).squeeze(-1)

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
