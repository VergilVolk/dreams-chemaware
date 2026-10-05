"""Pure helpers for candidate-specific ChemAware predicate support.

The helpers deliberately contain no retrieval labels.  They estimate a
formula/domain-centred spectrum-observation to structure-predicate map and
apply it only through candidate-set-centred structure differences.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np


def stable_nontrivial_formula_permutation(
    identities: np.ndarray, formulas: np.ndarray, seed: int,
) -> np.ndarray:
    """Return a deterministic within-formula permutation with no fixed points.

    Singleton formula groups remain fixed because no equal-formula control is
    possible.  Every group with at least two identities is cyclically shifted
    by a content-derived non-zero offset.
    """
    identities = np.asarray(identities, dtype=str)
    formulas = np.asarray(formulas, dtype=str)
    if identities.ndim != 1 or formulas.shape != identities.shape:
        raise ValueError("identity/formula arrays are misaligned")
    if len(np.unique(identities)) != len(identities):
        raise ValueError("identities must be unique")
    output = np.arange(len(identities), dtype=np.int64)
    for formula in np.unique(formulas):
        group = np.flatnonzero(formulas == formula)
        if len(group) < 2:
            continue
        ordered = group[np.argsort(identities[group], kind="stable")]
        digest = hashlib.sha256(f"{seed}|{formula}".encode()).digest()
        shift = 1 + int.from_bytes(digest[:8], "little") % (len(ordered) - 1)
        output[ordered] = np.roll(ordered, shift)
    return output


def formula_domain_centered_design(
    observation: np.ndarray,
    predicate: np.ndarray,
    formulas: np.ndarray,
    domains: np.ndarray,
    positions: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Build a formula-balanced within-(formula, domain) contrast design."""
    observation = np.asarray(observation, dtype=np.float64)
    predicate = np.asarray(predicate, dtype=np.float64)
    formulas = np.asarray(formulas, dtype=str)
    domains = np.asarray(domains, dtype=str)
    positions = np.asarray(tuple(positions), dtype=np.int64)
    if observation.ndim != 2 or predicate.ndim != 2:
        raise ValueError("observation and predicate must be matrices")
    if not (
        len(observation) == len(predicate) == len(formulas) == len(domains)
    ):
        raise ValueError("unit arrays are misaligned")
    if np.any((positions < 0) | (positions >= len(observation))):
        raise ValueError("training position lies outside the unit matrix")

    keys: dict[tuple[str, str], list[int]] = {}
    for position in positions:
        key = (str(formulas[position]), str(domains[position]))
        keys.setdefault(key, []).append(int(position))
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    retained_formula: set[str] = set()
    retained_groups = 0
    for (formula, _domain), group_list in sorted(keys.items()):
        group = np.asarray(group_list, dtype=np.int64)
        if len(group) < 2:
            continue
        y = predicate[group]
        if not np.any(np.ptp(y, axis=0) > 0):
            continue
        x = observation[group]
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean(axis=0, keepdims=True)
        # Each formula-domain block has total squared weight one.  Large,
        # heavily measured chemical families cannot dominate the map.
        weight = 1.0 / np.sqrt(len(group))
        x_blocks.append(x * weight)
        y_blocks.append(y * weight)
        retained_formula.add(formula)
        retained_groups += 1
    if not x_blocks:
        raise RuntimeError("no within-formula predicate contrast is estimable")
    x_design = np.vstack(x_blocks)
    y_design = np.vstack(y_blocks)
    report = {
        "input_units": int(len(positions)),
        "retained_units": int(len(x_design)),
        "retained_formula_domain_groups": int(retained_groups),
        "retained_formulas": int(len(retained_formula)),
        "varying_observation_channels": int(np.sum(np.std(x_design, axis=0) > 1e-12)),
        "varying_predicates": int(np.sum(np.std(y_design, axis=0) > 1e-12)),
    }
    return x_design, y_design, report


def ridge_support_family(
    x_design: np.ndarray,
    y_design: np.ndarray,
    ridges: Iterable[float],
    ranks: Iterable[int],
    support_only: bool,
) -> tuple[dict[tuple[float, int], np.ndarray], np.ndarray, dict[str, object]]:
    """Fit ridge maps and optionally retain a clipped low-rank reconstruction."""
    x = np.asarray(x_design, dtype=np.float64)
    y = np.asarray(y_design, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("ridge designs are misaligned")
    scale = np.sqrt(np.mean(x * x, axis=0))
    active = scale > 1e-12
    safe_scale = np.where(active, scale, 1.0)
    xs = x / safe_scale
    gram = xs.T @ xs
    cross = xs.T @ y
    output: dict[tuple[float, int], np.ndarray] = {}
    spectra: dict[str, list[float]] = {}
    for ridge_value in ridges:
        ridge_value = float(ridge_value)
        if ridge_value <= 0:
            raise ValueError("ridge values must be positive")
        weight = np.linalg.solve(
            gram + ridge_value * np.eye(gram.shape[0], dtype=np.float64),
            cross,
        )
        if support_only:
            weight = np.maximum(weight, 0.0)
        u, singular, vt = np.linalg.svd(weight, full_matrices=False)
        spectra[str(ridge_value)] = singular[: min(12, len(singular))].tolist()
        for rank_value in ranks:
            rank_value = int(rank_value)
            if rank_value < 0:
                raise ValueError("rank cannot be negative")
            if rank_value == 0 or rank_value >= len(singular):
                reconstructed = weight.copy()
            else:
                reconstructed = (u[:, :rank_value] * singular[:rank_value]) @ vt[:rank_value]
                if support_only:
                    reconstructed = np.maximum(reconstructed, 0.0)
            reconstructed[~active] = 0.0
            output[(ridge_value, rank_value)] = reconstructed.astype(np.float32)
    report: dict[str, object] = {
        "support_only": bool(support_only),
        "active_observation_channels": int(np.sum(active)),
        "observation_scale_quantiles": np.quantile(
            scale[active], [0.0, 0.25, 0.5, 0.75, 1.0]
        ).tolist(),
        "leading_singular_values": spectra,
    }
    return output, safe_scale.astype(np.float32), report


def candidate_compatibility(
    query_observation: np.ndarray,
    candidate_predicate: np.ndarray,
    weight: np.ndarray,
    observation_scale: np.ndarray,
    support_only: bool,
) -> np.ndarray:
    """Compute candidate-centred cosine compatibility for one query.

    Any predicate shared by all same-formula candidates cancels exactly.  A
    query without usable observation evidence produces an exact no-op vector.
    """
    observation = np.asarray(query_observation, dtype=np.float64)
    candidate = np.asarray(candidate_predicate, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    scale = np.asarray(observation_scale, dtype=np.float64)
    if candidate.ndim != 2 or observation.ndim != 1:
        raise ValueError("candidate predicate must be a matrix and observation a vector")
    if weight.shape != (len(observation), candidate.shape[1]) or scale.shape != observation.shape:
        raise ValueError("compatibility dimensions do not align")
    x = observation / np.maximum(scale, 1e-12)
    if support_only:
        x = np.maximum(x, 0.0)
    decoded = x @ weight
    decoded_norm = float(np.linalg.norm(decoded))
    if decoded_norm <= 1e-12:
        return np.zeros(len(candidate), dtype=np.float32)
    centred = candidate - candidate.mean(axis=0, keepdims=True)
    denominator = np.linalg.norm(centred, axis=1) * decoded_norm
    numerator = centred @ decoded
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(len(candidate), dtype=np.float64),
        where=denominator > 1e-12,
    ).astype(np.float32)


def choose_abstention_threshold(
    confidence: np.ndarray,
    baseline_rank: np.ndarray,
    proposal_rank: np.ndarray,
    formulas: np.ndarray,
    minimum_formulas: int,
) -> dict[str, object]:
    """Freeze a one-dimensional, risk-aware no-op gate without a classifier."""
    confidence = np.asarray(confidence, dtype=np.float64)
    baseline_rank = np.asarray(baseline_rank)
    proposal_rank = np.asarray(proposal_rank)
    formulas = np.asarray(formulas, dtype=str)
    if not (confidence.shape == baseline_rank.shape == proposal_rank.shape == formulas.shape):
        raise ValueError("threshold arrays are misaligned")
    finite = np.isfinite(confidence)
    candidates = np.unique(
        np.concatenate((
            np.asarray([np.inf, -np.inf]),
            confidence[finite],
        ))
    )
    rows: list[dict[str, object]] = []
    for threshold in candidates:
        selected = finite & (confidence >= threshold)
        selected_formulas = len(np.unique(formulas[selected]))
        if selected_formulas < minimum_formulas and np.isfinite(threshold):
            continue
        rank = np.where(selected, proposal_rank, baseline_rank)
        corrected = int(np.sum((baseline_rank > 1) & (rank == 1)))
        introduced = int(np.sum((baseline_rank == 1) & (rank > 1)))
        rows.append({
            "threshold": float(threshold),
            "selected_queries": int(np.sum(selected)),
            "selected_formulas": int(selected_formulas),
            "corrected_at_1": corrected,
            "introduced_at_1": introduced,
            "risk_utility_at_1": corrected - 2 * introduced,
            "delta_recall1": float(np.mean(rank == 1) - np.mean(baseline_rank == 1)),
            "delta_mrr": float(np.mean(1.0 / rank - 1.0 / baseline_rank)),
        })
    if not rows:
        raise RuntimeError("no admissible abstention threshold")
    return max(
        rows,
        key=lambda row: (
            int(row["risk_utility_at_1"]),
            float(row["delta_recall1"]),
            float(row["delta_mrr"]),
            -int(row["selected_queries"]),
            float(row["threshold"]),
        ),
    )
