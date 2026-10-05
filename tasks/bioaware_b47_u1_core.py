#!/usr/bin/env python
"""Pure NumPy core for BioAware B47-U1 spectrum-only unary calibration.

U1 changes only how repeated reference spectra for one candidate identity are
aggregated.  It never consumes reaction edges, sample phenotype, B47 truth, or
downstream BioAware outputs.  The functions here are deliberately small and
dependency-free so the scientific contract can be unit tested locally.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


REQUIRED_GRAPH_ARRAYS = {
    "feature_names", "features", "pair_candidate_row", "query_ptr",
    "molecule_ptr", "molecule_label", "molecule_ik14", "molecule_formula",
    "molecule_mces_grade", "query_row", "query_ik14", "query_formula",
    "query_has_near",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CandidateGraph:
    """Dependency-free reader for the immutable corrected graph schema."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as body:
            missing = REQUIRED_GRAPH_ARRAYS - set(body.files)
            if missing:
                raise RuntimeError(f"candidate graph missing arrays: {sorted(missing)}")
            for name in body.files:
                setattr(self, name, body[name])
        self.feature_names = list(map(str, self.feature_names))
        self.features = np.asarray(self.features, dtype=np.float32)
        self.pair_candidate_row = np.asarray(self.pair_candidate_row, dtype=np.int64)
        self.query_ptr = np.asarray(self.query_ptr, dtype=np.int64)
        self.molecule_ptr = np.asarray(self.molecule_ptr, dtype=np.int64)
        self.molecule_label = np.asarray(self.molecule_label, dtype=np.int8)
        self.molecule_mces_grade = np.asarray(self.molecule_mces_grade, dtype=np.int8)
        self.molecule_ik14 = np.asarray(self.molecule_ik14, dtype=str)
        self.molecule_formula = np.asarray(self.molecule_formula, dtype=str)
        self.query_row = np.asarray(self.query_row, dtype=np.int64)
        self.query_ik14 = np.asarray(self.query_ik14, dtype=str)
        self.query_formula = np.asarray(self.query_formula, dtype=str)
        self.query_has_near = np.asarray(self.query_has_near, dtype=bool)
        self.n_queries = len(self.query_ptr) - 1
        self._validate()

    @property
    def dreams_column(self) -> int:
        try:
            return self.feature_names.index("dreams_similarity")
        except ValueError as error:
            raise RuntimeError("candidate graph has no dreams_similarity column") from error

    def _validate(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] != len(self.feature_names):
            raise RuntimeError("feature matrix/name mismatch")
        if self.query_ptr[0] != 0 or self.query_ptr[-1] != len(self.molecule_label):
            raise RuntimeError("query_ptr does not span candidate molecules")
        if self.molecule_ptr[0] != 0 or self.molecule_ptr[-1] != len(self.features):
            raise RuntimeError("molecule_ptr does not span spectrum pairs")
        if len(self.pair_candidate_row) != len(self.features):
            raise RuntimeError("candidate rows do not align to spectrum pairs")
        if len(self.query_row) != self.n_queries:
            raise RuntimeError("query metadata is not aligned")
        if np.any(np.diff(self.query_ptr) < 2) or np.any(np.diff(self.molecule_ptr) < 1):
            raise RuntimeError("each query needs >=2 molecules; each molecule >=1 spectrum")
        for left, right in zip(self.query_ptr[:-1], self.query_ptr[1:]):
            labels = self.molecule_label[int(left):int(right)]
            if labels[0] != 1 or labels.sum() != 1:
                raise RuntimeError("positive molecule must be unique and first")


@dataclass(frozen=True)
class UnaryFeatures:
    maximum: np.ndarray
    mean: np.ndarray
    top2_mean: np.ndarray
    standard_deviation: np.ndarray
    log_reference_count: np.ndarray
    reference_count: np.ndarray


@dataclass(frozen=True)
class UnaryRecipe:
    name: str
    support: str
    shrinkage: float
    log_count_weight: float


def stable_formula_folds(formula: np.ndarray, folds: int, seed: int) -> np.ndarray:
    if folds < 3:
        raise ValueError("at least three formula folds are required")
    return np.asarray([
        int.from_bytes(
            hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()[:8],
            "little",
        ) % folds
        for value in np.asarray(formula, dtype=str)
    ], dtype=np.int8)


def unary_features(pair_score: np.ndarray, molecule_ptr: np.ndarray) -> UnaryFeatures:
    score = np.asarray(pair_score, dtype=np.float64)
    ptr = np.asarray(molecule_ptr, dtype=np.int64)
    if (
        score.ndim != 1 or ptr.ndim != 1 or len(ptr) < 2
        or ptr[0] != 0 or ptr[-1] != len(score)
        or np.any(np.diff(ptr) < 1) or not np.all(np.isfinite(score))
    ):
        raise RuntimeError("malformed pair-score/molecule pointer arrays")
    count = np.diff(ptr).astype(np.int32)
    maximum = np.maximum.reduceat(score, ptr[:-1])
    total = np.add.reduceat(score, ptr[:-1])
    mean = total / count
    second_maybe = np.maximum.reduceat(
        np.where(score < np.repeat(maximum, count), score, -np.inf), ptr[:-1],
    )
    maximum_multiplicity = np.add.reduceat(
        np.isclose(
            score, np.repeat(maximum, count), rtol=0.0, atol=1e-12,
        ).astype(np.int32),
        ptr[:-1],
    )
    second = np.where(
        count == 1,
        maximum,
        np.where(maximum_multiplicity >= 2, maximum, second_maybe),
    )
    if not np.all(np.isfinite(second)):
        raise RuntimeError("failed to recover a finite second-best spectrum score")
    top2_mean = 0.5 * (maximum + second)
    variance = np.maximum(
        np.add.reduceat(score * score, ptr[:-1]) / count - mean * mean,
        0.0,
    )
    return UnaryFeatures(
        maximum=maximum.astype(np.float32),
        mean=mean.astype(np.float32),
        top2_mean=top2_mean.astype(np.float32),
        standard_deviation=np.sqrt(variance).astype(np.float32),
        log_reference_count=np.log1p(count).astype(np.float32),
        reference_count=count,
    )


def fixed_recipes() -> tuple[UnaryRecipe, ...]:
    """Return the small preregistered family; baseline is always included."""
    recipes = [UnaryRecipe("max", "max", 0.0, 0.0)]
    for support in ("mean", "top2_mean"):
        for shrinkage in (0.01, 0.02, 0.04, 0.06):
            for count_weight in (0.0, 0.001, 0.002, 0.003):
                recipes.append(UnaryRecipe(
                    name=(
                        f"max_{support}_a{shrinkage:.2f}_"
                        f"logn{count_weight:.3f}"
                    ),
                    support=support,
                    shrinkage=shrinkage,
                    log_count_weight=count_weight,
                ))
    return tuple(recipes)


def score_recipe(features: UnaryFeatures, recipe: UnaryRecipe) -> np.ndarray:
    if recipe.support == "max":
        support = features.maximum
    elif recipe.support == "mean":
        support = features.mean
    elif recipe.support == "top2_mean":
        support = features.top2_mean
    else:
        raise ValueError(f"unknown unary support: {recipe.support}")
    value = (
        (1.0 - recipe.shrinkage) * features.maximum
        + recipe.shrinkage * support
        + recipe.log_count_weight * features.log_reference_count
    )
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"non-finite score for recipe {recipe.name}")
    return np.asarray(value, dtype=np.float32)


def strict_ranks(score: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    value = np.asarray(score, dtype=np.float64)
    ptr = np.asarray(query_ptr, dtype=np.int64)
    if (
        value.ndim != 1 or ptr.ndim != 1 or ptr[0] != 0
        or ptr[-1] != len(value) or np.any(np.diff(ptr) < 2)
        or not np.all(np.isfinite(value))
    ):
        raise RuntimeError("malformed molecule-score/query pointer arrays")
    positive = value[ptr[:-1]]
    positive_per_molecule = np.repeat(positive, np.diff(ptr))
    # The positive is the first molecule and is included once in >=; subtract it.
    return 1 + np.add.reduceat(
        (value >= positive_per_molecule).astype(np.int16), ptr[:-1],
    ) - 1


def same_top_unique_veto(
    baseline_score: np.ndarray,
    calibrated_score: np.ndarray,
    query_ptr: np.ndarray,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply calibration only to a baseline-unique, winner-preserving query.

    Requiring the baseline winner to be unique is essential.  Candidate graphs
    used for supervised evaluation place the positive molecule first, so using
    ``argmax`` to choose a winner from a baseline tie would leak candidate
    order.  A safe veto may preserve an already unique identity; it may never
    manufacture a deployable tie-break from evaluation ordering.
    """
    baseline = np.asarray(baseline_score, dtype=np.float32)
    calibrated = np.asarray(calibrated_score, dtype=np.float32)
    ptr = np.asarray(query_ptr, dtype=np.int64)
    if baseline.shape != calibrated.shape or ptr[-1] != len(baseline):
        raise RuntimeError("safe-veto score arrays do not align")
    output = baseline.copy()
    gate = np.zeros(len(ptr) - 1, dtype=bool)
    for query, (left_value, right_value) in enumerate(zip(ptr[:-1], ptr[1:])):
        left, right = int(left_value), int(right_value)
        old = baseline[left:right]
        new = calibrated[left:right]
        old_maximum = float(np.max(old))
        old_top_count = int(np.sum(np.isclose(
            old.astype(np.float64), old_maximum, rtol=0.0, atol=tolerance,
        )))
        if old_top_count != 1:
            continue
        old_top = int(np.argmax(old))
        new_top = int(np.argmax(new))
        if old_top != new_top:
            continue
        second = float(np.partition(new.astype(np.float64), -2)[-2])
        if float(new[new_top]) - second <= tolerance:
            continue
        output[left:right] = new
        gate[query] = True
    return output, gate


def query_metrics(ranks: np.ndarray, near: np.ndarray) -> dict[str, float | int]:
    rank = np.asarray(ranks, dtype=np.int32)
    near_mask = np.asarray(near, dtype=bool)
    if rank.ndim != 1 or near_mask.shape != rank.shape or np.any(rank < 1):
        raise RuntimeError("rank/near arrays are malformed")
    result: dict[str, float | int] = {
        "queries": int(len(rank)),
        "mrr": float(np.mean(1.0 / rank)),
        "near_queries": int(near_mask.sum()),
        "near_mrr": float(np.mean(1.0 / rank[near_mask])),
    }
    for cutoff in (1, 2, 5, 10, 20):
        result[f"recall@{cutoff}"] = float(np.mean(rank <= cutoff))
        result[f"near_recall@{cutoff}"] = float(np.mean(rank[near_mask] <= cutoff))
    return result


def paired_summary(
    baseline_rank: np.ndarray,
    candidate_rank: np.ndarray,
    formulas: np.ndarray,
    near: np.ndarray,
    selection: np.ndarray | None = None,
) -> dict[str, float | int | bool]:
    old = np.asarray(baseline_rank, dtype=np.int32)
    new = np.asarray(candidate_rank, dtype=np.int32)
    formula = np.asarray(formulas, dtype=str)
    near_mask = np.asarray(near, dtype=bool)
    use = np.ones(len(old), dtype=bool) if selection is None else np.asarray(selection, dtype=bool)
    if not (old.shape == new.shape == formula.shape == near_mask.shape == use.shape):
        raise RuntimeError("paired summary arrays do not align")
    if not np.any(use):
        raise RuntimeError("paired summary selection is empty")
    old_top1, new_top1 = old == 1, new == 1
    corrected = (~old_top1 & new_top1) & use
    introduced = (old_top1 & ~new_top1) & use
    delta = new_top1.astype(np.float64) - old_top1.astype(np.float64)
    utility = corrected.astype(np.float64) - 2.0 * introduced.astype(np.float64)
    unique, inverse = np.unique(formula[use], return_inverse=True)
    formula_count = np.bincount(inverse)
    formula_delta = np.bincount(inverse, weights=delta[use]) / formula_count
    formula_utility = np.bincount(inverse, weights=utility[use]) / formula_count
    near_use = use & near_mask
    return {
        "queries": int(use.sum()),
        "formulas": int(len(unique)),
        "delta_recall1": float(np.mean(delta[use])),
        "delta_mrr": float(np.mean((1.0 / new - 1.0 / old)[use])),
        "near_delta_recall1": float(np.mean(delta[near_use])) if np.any(near_use) else 0.0,
        "corrected": int(corrected.sum()),
        "introduced": int(introduced.sum()),
        "risk_net_lambda2": int(corrected.sum() - 2 * introduced.sum()),
        "formula_equal_delta_recall1": float(np.mean(formula_delta)),
        "formula_equal_risk_utility": float(np.mean(formula_utility)),
        "corrected_gt_2x_introduced": bool(corrected.sum() > 2 * introduced.sum()),
    }


def choose_recipe(
    summaries: dict[str, dict[str, float | int | bool]],
) -> str:
    """Choose conservatively; fall back to max when no recipe passes safety."""
    eligible = []
    for name, row in summaries.items():
        if name == "max":
            continue
        if (
            bool(row["corrected_gt_2x_introduced"])
            and float(row["delta_recall1"]) > 0.0
            and float(row["delta_mrr"]) >= 0.0
            and float(row["near_delta_recall1"]) >= 0.0
            and float(row["formula_equal_delta_recall1"]) > 0.0
            and float(row["formula_equal_risk_utility"]) > 0.0
        ):
            eligible.append((
                float(row["formula_equal_delta_recall1"]),
                float(row["delta_recall1"]),
                float(row["formula_equal_risk_utility"]),
                -float(row["introduced"]),
                name,
            ))
    return max(eligible)[-1] if eligible else "max"


def formula_cluster_bootstrap(
    delta: np.ndarray,
    formula: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    value = np.asarray(delta, dtype=np.float64)
    group = np.asarray(formula, dtype=str)
    if value.shape != group.shape or repeats < 100:
        raise RuntimeError("invalid formula bootstrap input")
    unique, inverse = np.unique(group, return_inverse=True)
    group_sum = np.bincount(inverse, weights=value)
    group_count = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draw = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = rng.integers(0, len(unique), size=len(unique))
        draw[index] = group_sum[sampled].sum() / group_count[sampled].sum()
    return {
        "mean": float(np.mean(value)),
        "ci_low": float(np.quantile(draw, 0.025)),
        "ci_high": float(np.quantile(draw, 0.975)),
        "clusters": int(len(unique)),
        "resamples": int(repeats),
    }
