"""NumPy-only sampling helpers shared by ChemAware audits.

This module intentionally has no Torch dependency so action-ledger and triplet
construction can run locally without importing any training stack.
"""
from __future__ import annotations

import hashlib

import numpy as np


def stable_formula_folds(formulas: np.ndarray, folds: int, seed: int) -> np.ndarray:
    if folds < 3:
        raise ValueError("at least three folds are required")
    return np.asarray([
        int.from_bytes(
            hashlib.sha256(f"{seed}|{str(value)}".encode()).digest()[:8], "little",
        ) % folds
        for value in formulas
    ], dtype=np.int16)


def identity_balanced_queries(
    query: np.ndarray, identities: np.ndarray, rng: np.random.Generator, limit: int = 0,
) -> np.ndarray:
    by_identity: dict[str, list[int]] = {}
    for value in np.asarray(query, dtype=np.int64):
        by_identity.setdefault(str(identities[value]), []).append(int(value))
    keys = np.asarray(sorted(by_identity), dtype=object)
    rng.shuffle(keys)
    if limit:
        keys = keys[:limit]
    selected = [
        by_identity[str(key)][int(rng.integers(len(by_identity[str(key)])))]
        for key in keys
    ]
    return np.asarray(selected, dtype=np.int64)


def formula_bootstrap(
    delta: np.ndarray, formula: np.ndarray, seed: int, draws: int,
) -> dict[str, object]:
    unique = np.unique(formula)
    macro = np.asarray([np.mean(delta[formula == value]) for value in unique])
    rng = np.random.default_rng(seed)
    estimate = np.asarray([
        np.mean(macro[rng.integers(0, len(macro), len(macro))])
        for _ in range(draws)
    ])
    return {
        "formula_macro_delta": float(np.mean(macro)),
        "formula_cluster_bootstrap_95ci": [
            float(np.quantile(estimate, 0.025)),
            float(np.quantile(estimate, 0.975)),
        ],
        "formula_clusters": int(len(unique)),
        "draws": int(draws),
    }


def official_outcomes(
    body: dict[str, np.ndarray], queries: np.ndarray, official: np.ndarray,
    row_position: dict[int, int], molecule_allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    error = np.zeros(len(body["query_row"]), dtype=bool)
    margin = np.full(len(body["query_row"]), np.nan, dtype=np.float32)
    for query in np.asarray(queries, dtype=np.int64):
        qpos = row_position[int(body["query_row"][query])]
        left, right = map(int, body["query_ptr"][query:query + 2])
        scores = []
        for molecule in range(left, right):
            if molecule_allowed is not None and not molecule_allowed[molecule]:
                continue
            rleft, rright = map(int, body["molecule_ptr"][molecule:molecule + 2])
            positions = [
                row_position[int(row)]
                for row in body["pair_candidate_row"][rleft:rright]
                if int(row) in row_position
            ]
            if positions:
                scores.append(float(np.max(official[positions] @ official[qpos])))
        if len(scores) < 2:
            raise RuntimeError("training outcome query has fewer than two cached candidates")
        rank = 1 + int(np.sum(np.asarray(scores[1:]) >= scores[0]))
        error[int(query)] = rank != 1
        margin[int(query)] = scores[0] - max(scores[1:])
    return error, margin
