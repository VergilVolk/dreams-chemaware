#!/usr/bin/env python
"""Fixed feature contracts for the unified Additive contribution matrix.

The matrix deliberately changes only which *frozen* evidence families are
visible to the same learned linear candidate scorer.  Candidate-varying
reliability summaries are rebuilt from the retained experts; they are never
copied from the full model after an expert has been removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from enveda_unified_candidate_integrator_core import (
    CONTROL_ARM,
    CandidateEvidenceIntegrator,
    IntegratorSettings,
    transform_features,
)
from massspecgym_embedding_readout_core import percentile_by_query


CHECKPOINT_SCHEMA = "enveda_unified_additive_ablation_matrix_v1"
FULL = "full_23_additive"
OFFICIAL_WSE = "official_dreams_wse_additive"
NO_CHEMAWARE_REPRESENTATIONS = "without_chemaware_representation_group"
NO_CHEMAWARE_V2 = "without_chemaware_v2"
NO_LOCAL_RERANKERS = "without_p2b_msg"
VARIANT_ORDER = (
    FULL,
    OFFICIAL_WSE,
    NO_CHEMAWARE_REPRESENTATIONS,
    NO_CHEMAWARE_V2,
    NO_LOCAL_RERANKERS,
)
ABLATION_ORDER = VARIANT_ORDER[1:]

WSE = "weighted_spectral_entropy"
P2B = "p2b_noise_v1_frozen"
MSG = "noise_msg_fusion_stack_v1"
CHEM_PREFIX = "chemaware_v2_"


@dataclass(frozen=True)
class VariantSpec:
    experts: tuple[str, ...]
    include_chemistry: bool
    question: str


VARIANTS: dict[str, VariantSpec] = {
    FULL: VariantSpec(
        experts=(
            "noise_v1", WSE, P2B, MSG,
            "chemaware_stage1", "chemaware_phasea", "noise_chemaware_fullmsg",
        ),
        include_chemistry=True,
        question="fixed 23-feature Additive reference",
    ),
    OFFICIAL_WSE: VariantSpec(
        experts=("official_dreams", WSE),
        include_chemistry=False,
        question="can two off-the-shelf baselines explain the full learned combination",
    ),
    NO_CHEMAWARE_REPRESENTATIONS: VariantSpec(
        experts=("noise_v1", WSE, P2B, MSG),
        include_chemistry=True,
        question="increment from Stage-1, Phase-A and FullMSG representations as one group",
    ),
    NO_CHEMAWARE_V2: VariantSpec(
        experts=(
            "noise_v1", WSE, P2B, MSG,
            "chemaware_stage1", "chemaware_phasea", "noise_chemaware_fullmsg",
        ),
        include_chemistry=False,
        question="increment from continuous ChemAware V2 evidence",
    ),
    NO_LOCAL_RERANKERS: VariantSpec(
        experts=(
            "noise_v1", WSE,
            "chemaware_stage1", "chemaware_phasea", "noise_chemaware_fullmsg",
        ),
        include_chemistry=True,
        question="increment from the frozen P2b and MSG local-reranker group",
    ),
}


# Raw score names must exactly reproduce the retained columns in run 2413878.
FEATURE_NAMES: dict[str, tuple[str, str]] = {
    "official_dreams": ("official_dreams", "official_dreams_percentile"),
    "noise_v1": ("noise_v1", "noise_v1_percentile"),
    WSE: (WSE, f"{WSE}_percentile"),
    P2B: (P2B, f"{P2B}_percentile"),
    MSG: (MSG, f"{MSG}_percentile"),
    "chemaware_stage1": ("chemaware_stage1_cosine", "chemaware_stage1_percentile"),
    "chemaware_phasea": ("chemaware_phasea_cosine", "chemaware_phasea_percentile"),
    "noise_chemaware_fullmsg": (
        "noise_chemaware_fullmsg_cosine", "noise_chemaware_fullmsg_percentile",
    ),
}

RELIABILITY_NAMES = (
    "reliability_top1_vote_fraction",
    "reliability_mean_percentile",
    "reliability_percentile_std",
)


def _candidate_reliability(
    scores: Mapping[str, np.ndarray], expert_names: tuple[str, ...],
    query_ptr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute the three candidate-varying summaries for retained experts."""
    ptr = np.asarray(query_ptr, dtype=np.int64)
    percentiles = {
        name: percentile_by_query(np.asarray(scores[name], dtype=np.float64), ptr)
        for name in expert_names
    }
    stacked = np.stack([percentiles[name] for name in expert_names], axis=1)
    votes = np.zeros(stacked.shape[0], dtype=np.float64)
    for left, right in zip(ptr[:-1], ptr[1:]):
        left, right = int(left), int(right)
        for name in expert_names:
            local = np.asarray(scores[name][left:right], dtype=np.float64)
            winners = np.flatnonzero(local == np.max(local))
            # Preserve the original tie policy: a tied expert casts no vote.
            if len(winners) == 1:
                votes[left + int(winners[0])] += 1.0
    votes /= len(expert_names)
    return votes, np.mean(stacked, axis=1), np.std(stacked, axis=1)


def build_variant_features(
    scores: Mapping[str, np.ndarray], query_ptr: np.ndarray,
    retained_full_scalar: np.ndarray, retained_full_names: tuple[str, ...],
    variant: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build one strict Additive arm and recompute all dependent summaries."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown additive ablation: {variant}")
    spec = VARIANTS[variant]
    ptr = np.asarray(query_ptr, dtype=np.int64)
    if ptr.ndim != 1 or len(ptr) < 2 or ptr[0] != 0 or np.any(np.diff(ptr) < 2):
        raise ValueError("variant builder requires non-empty multi-candidate queries")
    count = int(ptr[-1])
    columns: list[np.ndarray] = []
    names: list[str] = []
    normalized_scores: dict[str, np.ndarray] = {}
    for expert in spec.experts:
        if expert not in scores or expert not in FEATURE_NAMES:
            raise RuntimeError(f"variant {variant} misses frozen expert {expert}")
        values = np.asarray(scores[expert], dtype=np.float64)
        if values.shape != (count,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"variant {variant} has malformed expert {expert}")
        normalized_scores[expert] = values
        percentile = percentile_by_query(values, ptr)
        columns.extend((values[:, None], percentile[:, None]))
        names.extend(FEATURE_NAMES[expert])

    reliability = _candidate_reliability(normalized_scores, spec.experts, ptr)
    columns.extend(value[:, None] for value in reliability)
    names.extend(RELIABILITY_NAMES)

    full = np.asarray(retained_full_scalar, dtype=np.float32)
    full_names = tuple(map(str, retained_full_names))
    if full.shape != (count, len(full_names)):
        raise RuntimeError("retained full feature block is malformed")
    if spec.include_chemistry:
        chemistry = [index for index, name in enumerate(full_names) if name.startswith(CHEM_PREFIX)]
        chemistry_names = [full_names[index] for index in chemistry]
        if len(chemistry) != 6 or chemistry_names[0] != "chemaware_v2_available":
            raise RuntimeError("retained full block lacks the six fixed V2 columns")
        columns.append(full[:, chemistry])
        names.extend(chemistry_names)

    output = np.concatenate(columns, axis=1).astype(np.float32)
    if not np.all(np.isfinite(output)) or output.shape[1] != len(names):
        raise RuntimeError(f"variant {variant} produced malformed features")
    return output, tuple(names)


def assert_full_reconstruction(
    observed: np.ndarray, observed_names: tuple[str, ...],
    expected: np.ndarray, expected_names: tuple[str, ...],
) -> None:
    """Fail if the matrix's full arm is not the exact 23-feature reference."""
    if tuple(observed_names) != tuple(expected_names):
        raise RuntimeError("full Additive feature names drifted from the 23-feature reference")
    delta = float(np.max(np.abs(np.asarray(observed) - np.asarray(expected))))
    if not np.isfinite(delta) or delta > 1e-7:
        raise RuntimeError(f"full Additive feature values drifted: max_abs={delta}")


@torch.inference_mode()
def predict_variant(
    entry: dict, feature_names: tuple[str, ...], scalar: np.ndarray, device: str,
) -> np.ndarray:
    expected = tuple(map(str, entry.get("feature_names", ())))
    if tuple(feature_names) != expected:
        raise RuntimeError("additive ablation deployment feature names drifted")
    mean = np.asarray(entry["scalar_mean"], dtype=np.float32)
    scale = np.asarray(entry["scalar_scale"], dtype=np.float32)
    anchor = expected.index(WSE)
    if int(entry.get("anchor_index", -1)) != anchor:
        raise RuntimeError("additive ablation WSE anchor drifted")
    model = CandidateEvidenceIntegrator(
        CONTROL_ARM, IntegratorSettings(len(expected), anchor, int(entry["hidden"])),
    ).to(device)
    model.load_state_dict(entry["state_dict"], strict=True)
    model.eval()
    values = transform_features(scalar, mean, scale)
    return model(torch.as_tensor(values, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float32)
