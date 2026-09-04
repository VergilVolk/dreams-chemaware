"""Shared, fail-closed utilities for the final noise-finetuning programme.

This module intentionally contains no P2b/reranker code.  It operates on the
strict-10ppm candidate graph and produces candidate-independent query
embeddings.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REQUIRED_GRAPH_ARRAYS = {
    "feature_names", "features", "pair_candidate_row", "query_ptr",
    "molecule_ptr", "molecule_label", "molecule_ik14", "molecule_formula",
    "molecule_mces_grade", "query_row", "query_ik14", "query_formula",
    "query_has_near",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fold(value: str, folds: int, seed: int) -> int:
    payload = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % folds


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CandidateGraph:
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
            raise RuntimeError("each query needs >=2 molecules; each molecule >=1 spectrum")
        for left, right in zip(self.query_ptr[:-1], self.query_ptr[1:]):
            labels = self.molecule_label[left:right]
            if labels[0] != 1 or labels.sum() != 1:
                raise RuntimeError("positive molecule must be unique and first")
            if len(set(self.molecule_ik14[left:right])) != right - left:
                raise RuntimeError("candidate molecule identities are not unique within a query")

    @property
    def dreams_column(self) -> int:
        try:
            return self.feature_names.index("dreams_similarity")
        except ValueError as error:
            raise RuntimeError("candidate graph has no dreams_similarity column") from error

    def query_block(self, query: int) -> tuple[slice, np.ndarray, np.ndarray, int]:
        molecule_left, molecule_right = map(int, self.query_ptr[query:query + 2])
        pair_left = int(self.molecule_ptr[molecule_left])
        pair_right = int(self.molecule_ptr[molecule_right])
        local_ptr = self.molecule_ptr[molecule_left:molecule_right + 1] - pair_left
        return (
            slice(pair_left, pair_right),
            self.pair_candidate_row[pair_left:pair_right],
            local_ptr.astype(np.int64, copy=False),
            molecule_left,
        )

    def official_molecule_scores(self, query: int) -> np.ndarray:
        pair_slice, _, ptr, _ = self.query_block(query)
        pair_scores = self.features[pair_slice, self.dreams_column]
        return np.maximum.reduceat(pair_scores, ptr[:-1])


def strict_rank(molecule_scores: np.ndarray) -> int:
    scores = np.asarray(molecule_scores, dtype=np.float64)
    if len(scores) < 2 or not np.all(np.isfinite(scores)):
        raise RuntimeError("invalid molecule score vector")
    return 1 + int(np.sum(scores[1:] >= scores[0]))


def strict_metrics(ranks: np.ndarray, near: np.ndarray) -> dict[str, float | int]:
    ranks = np.asarray(ranks, dtype=np.int32)
    near = np.asarray(near, dtype=bool)
    output: dict[str, float | int] = {
        "n_queries": int(len(ranks)),
        "recall1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1.0 / ranks)),
        "errors": int(np.sum(ranks != 1)),
        "n_near": int(np.sum(near)),
    }
    if np.any(near):
        output["near_recall1"] = float(np.mean(ranks[near] == 1))
        output["near_errors"] = int(np.sum(ranks[near] != 1))
    return output


def load_embedding_cache(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    with np.load(path) as body:
        rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = np.asarray(body["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(rows) != len(embeddings):
        raise RuntimeError("official embedding cache is malformed")
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("official embedding cache has duplicate spectrum rows")
    embeddings /= np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    return rows, embeddings, {int(row): index for index, row in enumerate(rows)}


class ZeroInitPeakAdapter(nn.Module):
    """Candidate-independent, norm-bounded residual over contextual peak tokens."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 128, delta_bound: float = 0.15):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.delta_bound = float(delta_bound)
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.value = nn.Sequential(
            nn.Linear(embedding_dim + 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim + 2, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output = nn.Linear(hidden_dim, embedding_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        official_embedding: torch.Tensor,
        peak_tokens: torch.Tensor,
        peak_mz: torch.Tensor,
        peak_intensity: torch.Tensor,
        peak_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if peak_tokens.ndim != 3 or peak_tokens.shape[:2] != peak_mask.shape:
            raise RuntimeError("peak token/mask shape mismatch")
        token = self.token_norm(peak_tokens.float())
        auxiliary = torch.stack((peak_mz.float() / 1000.0, peak_intensity.float()), dim=-1)
        features = torch.cat((token, auxiliary), dim=-1)
        logits = self.gate(features).squeeze(-1)
        logits = logits.masked_fill(~peak_mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        weights = weights * peak_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        context = torch.sum(weights.unsqueeze(-1) * self.value(features), dim=1)
        raw_delta = self.output(context)
        norm = raw_delta.norm(dim=1, keepdim=True)
        delta = self.delta_bound * raw_delta / (1.0 + norm)
        adapted = F.normalize(official_embedding.float() + delta, dim=-1)
        return adapted, delta, weights


def molecule_scores_from_pairs(
    query_embedding: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    local_ptr: np.ndarray,
) -> torch.Tensor:
    pair_scores = candidate_embeddings @ query_embedding
    scores = []
    for left, right in zip(local_ptr[:-1], local_ptr[1:]):
        scores.append(torch.max(pair_scores[int(left):int(right)]))
    return torch.stack(scores)


def json_dump(path: Path, body: dict) -> None:
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

