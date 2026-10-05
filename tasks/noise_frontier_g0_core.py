"""Core utilities for the frozen DreaMS frontier G0 audit.

G0 is deliberately a readout/headroom experiment.  It never updates DreaMS
and it never claims that a candidate-aware score is already a deployable
single-spectrum embedding.  The important invariant is that channels are
combined on the same query--reference-spectrum pair *before* the best
reference spectrum is selected for a candidate molecule.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


REQUIRED_GRAPH_ARRAYS = {
    "feature_names", "features", "pair_candidate_row", "query_ptr",
    "molecule_ptr", "molecule_label", "molecule_ik14", "molecule_formula",
    "molecule_mces_grade", "query_row", "query_ik14", "query_formula",
    "query_has_near",
}


class FrozenCandidateGraph:
    """Torch-free, read-only loader for the corrected retrieval graph."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=True) as body:
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
            raise RuntimeError("each query needs >=2 molecules and each molecule >=1 spectrum")
        for left, right in zip(self.query_ptr[:-1], self.query_ptr[1:], strict=True):
            labels = self.molecule_label[left:right]
            if labels[0] != 1 or labels.sum() != 1:
                raise RuntimeError("positive molecule must be unique and first")
            if len(set(self.molecule_ik14[left:right])) != right - left:
                raise RuntimeError("candidate identities are not unique inside a query")

    @property
    def dreams_column(self) -> int:
        try:
            return self.feature_names.index("dreams_similarity")
        except ValueError as error:
            raise RuntimeError("candidate graph has no dreams_similarity column") from error


def load_embedding_cache(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    with np.load(path) as body:
        rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    if rows.ndim != 1 or embeddings.ndim != 2 or len(rows) != len(embeddings):
        raise RuntimeError("official embedding cache is malformed")
    if len(np.unique(rows)) != len(rows) or not np.all(np.isfinite(embeddings)):
        raise RuntimeError("official embedding cache has duplicate rows or non-finite values")
    embeddings /= np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    return rows, embeddings, {int(row): index for index, row in enumerate(rows)}


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_group_fold(value: str, folds: int, seed: int) -> int:
    if folds < 2:
        raise ValueError("folds must be >=2")
    payload = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % folds


def formula_folds(formulas: np.ndarray, folds: int, seed: int) -> np.ndarray:
    return np.asarray([stable_group_fold(str(value), folds, seed) for value in formulas], dtype=np.int8)


def grouped_max(values: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ptr = np.asarray(ptr, dtype=np.int64)
    if ptr.ndim != 1 or ptr[0] != 0 or ptr[-1] != len(values) or np.any(np.diff(ptr) <= 0):
        raise ValueError("invalid grouped-max pointer")
    return np.maximum.reduceat(values, ptr[:-1])


def pair_query_index(query_ptr: np.ndarray, molecule_ptr: np.ndarray) -> np.ndarray:
    query_ptr = np.asarray(query_ptr, dtype=np.int64)
    molecule_ptr = np.asarray(molecule_ptr, dtype=np.int64)
    molecule_query = np.repeat(np.arange(len(query_ptr) - 1, dtype=np.int64), np.diff(query_ptr))
    pair_molecule = np.repeat(np.arange(len(molecule_ptr) - 1, dtype=np.int64), np.diff(molecule_ptr))
    return molecule_query[pair_molecule]


def within_query_zscore(values: np.ndarray, query_pair_ptr: np.ndarray) -> np.ndarray:
    """Standardize one pair-level channel independently inside every query."""

    values = np.asarray(values, dtype=np.float64)
    ptr = np.asarray(query_pair_ptr, dtype=np.int64)
    if ptr[0] != 0 or ptr[-1] != len(values) or np.any(np.diff(ptr) <= 0):
        raise ValueError("invalid query-pair pointer")
    output = np.empty_like(values, dtype=np.float64)
    for left, right in zip(ptr[:-1], ptr[1:], strict=True):
        block = values[left:right]
        scale = max(float(np.std(block)), 1e-6)
        output[left:right] = (block - float(np.mean(block))) / scale
    return output


def ranks_from_pair_scores(
    pair_scores: np.ndarray,
    query_ptr: np.ndarray,
    molecule_ptr: np.ndarray,
    molecule_label: np.ndarray,
) -> np.ndarray:
    """Strict positive ranks; every negative tie counts against the positive."""

    molecule = grouped_max(pair_scores, molecule_ptr)
    query_ptr = np.asarray(query_ptr, dtype=np.int64)
    labels = np.asarray(molecule_label, dtype=bool)
    ranks = np.empty(len(query_ptr) - 1, dtype=np.int16)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:], strict=True)):
        local_label = labels[left:right]
        if local_label.sum() != 1 or not local_label[0]:
            raise RuntimeError("positive candidate must be unique and first")
        block = molecule[left:right]
        ranks[query] = 1 + int(np.sum(block[1:] >= block[0]))
    return ranks


def retrieval_summary(baseline: np.ndarray, candidate: np.ndarray, mask: np.ndarray | None = None) -> dict:
    baseline = np.asarray(baseline, dtype=np.int32)
    candidate = np.asarray(candidate, dtype=np.int32)
    if baseline.shape != candidate.shape:
        raise ValueError("rank vectors differ")
    if mask is None:
        mask = np.ones(len(baseline), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != baseline.shape or not np.any(mask):
        raise ValueError("empty or malformed evaluation mask")
    old = baseline[mask]
    new = candidate[mask]
    old_hit = old == 1
    new_hit = new == 1
    return {
        "n_queries": int(np.sum(mask)),
        "baseline_recall1": float(np.mean(old_hit)),
        "recall1": float(np.mean(new_hit)),
        "delta_recall1": float(np.mean(new_hit) - np.mean(old_hit)),
        "baseline_mrr": float(np.mean(1.0 / old)),
        "mrr": float(np.mean(1.0 / new)),
        "delta_mrr": float(np.mean(1.0 / new) - np.mean(1.0 / old)),
        "corrected": int(np.sum((~old_hit) & new_hit)),
        "introduced": int(np.sum(old_hit & (~new_hit))),
        "risk_utility_lambda2": int(np.sum((~old_hit) & new_hit) - 2 * np.sum(old_hit & (~new_hit))),
    }


def clustered_bootstrap(
    delta: np.ndarray,
    cluster: np.ndarray,
    draws: int,
    seed: int,
) -> dict:
    delta = np.asarray(delta, dtype=np.float64)
    cluster = np.asarray(cluster, dtype=str)
    if delta.shape != cluster.shape or draws < 100:
        raise ValueError("invalid clustered bootstrap inputs")
    names, inverse = np.unique(cluster, return_inverse=True)
    means = np.asarray([np.mean(delta[inverse == index]) for index in range(len(names))])
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        samples[draw] = float(np.mean(means[rng.integers(0, len(means), len(means))]))
    return {
        "mean_delta": float(np.mean(means)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "clusters": int(len(names)),
        "resamples": int(draws),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    output = np.empty(len(values), dtype=np.float32)
    left = 0
    while left < len(order):
        right = left + 1
        while right < len(order) and values[order[right]] == values[order[left]]:
            right += 1
        output[order[left:right]] = 0.5 * (left + right - 1)
        left = right
    if len(values) > 1:
        output /= float(len(values) - 1)
    return output


def landmark_rank_profiles(responses: np.ndarray) -> np.ndarray:
    """Convert node x landmark x channel responses to comparable rank profiles."""

    responses = np.asarray(responses, dtype=np.float32)
    if responses.ndim != 3 or responses.shape[1] < 4:
        raise ValueError("landmark response tensor is malformed")
    output = np.empty_like(responses, dtype=np.float32)
    for node in range(len(responses)):
        for channel in range(responses.shape[2]):
            output[node, :, channel] = average_ranks(responses[node, :, channel])
    return output


def landmark_concordance(
    query_profile: np.ndarray,
    reference_profile: np.ndarray,
    valid_landmark: np.ndarray,
) -> float:
    """A1-faithful mean rank concordance across observable channels."""

    valid = np.asarray(valid_landmark, dtype=bool)
    if np.sum(valid) < 4:
        return 0.0
    difference = np.abs(query_profile[valid] - reference_profile[valid])
    return float(np.clip(1.0 - np.mean(difference), 0.0, 1.0))


def landmark_response_concordance(
    query_response: np.ndarray,
    reference_response: np.ndarray,
    valid_landmark: np.ndarray,
) -> float:
    """Exact A1 concordance, reranking after excluded identities are removed."""

    valid = np.asarray(valid_landmark, dtype=bool)
    if np.sum(valid) < 4:
        return 0.0
    query_response = np.asarray(query_response, dtype=np.float32)
    reference_response = np.asarray(reference_response, dtype=np.float32)
    if query_response.shape != reference_response.shape or query_response.ndim != 2:
        raise ValueError("landmark response matrices differ")
    channel = []
    for index in range(query_response.shape[1]):
        left = average_ranks(query_response[valid, index])
        right = average_ranks(reference_response[valid, index])
        channel.append(float(np.clip(1.0 - np.mean(np.abs(left - right)), 0.0, 1.0)))
    return float(np.mean(channel))


def selection_key(summary: dict, near: dict, weight_sum: float) -> tuple:
    """Risk-first selection; an unsafe arm always loses to the zero update."""

    safe = (
        summary["delta_recall1"] >= -1e-12
        and summary["delta_mrr"] >= -1e-12
        and near["delta_recall1"] >= -1e-12
        and summary["corrected"] >= summary["introduced"]
    )
    return (
        int(safe),
        summary["risk_utility_lambda2"],
        summary["delta_recall1"],
        near["delta_recall1"],
        summary["delta_mrr"],
        -float(weight_sum),
    )
