"""Rule-coherence score (Layer-1 confidence).

Single-spectrum scalar computed on DreaMS's frozen representation, at the data-flow
point *before* the final cosine retrieval. The frozen embedding's *decoded* rule
probabilities (a frozen linear probe over a subset of the 335 main rules) are
compared to the spectrum's *observed* rule-hit vector (exact-mass pattern
matching, 0.02 Da, ``annotation.rule_evidence.spectrum_rule_vector``).

The score is the per-spectrum AUROC of "does the embedding rank observed-present
rules above observed-absent rules". Range [0, 1]; 0.5 = random; a chemically
coherent spectrum should exceed 0.5. This is our own definition (permitted by the
two-layer design): reproducible (frozen backbone + frozen probe) and comparable
(the same scalar for known compounds and dark matter alike -- no library cosine).

Honest limit: the score is a *chemical self-consistency* measure, not a structure
annotation. It does not identify a fragment structure or mechanism, and a high
score does not by itself imply a correct library annotation.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-sum AUROC with average-rank tie handling (robust to discrete scores)."""
    y = np.asarray(y, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5  # uninformative: no present or no absent rule
    order = np.argsort(score, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for group in np.flatnonzero(counts > 1):
            mask = inverse == group
            ranks[mask] = ranks[mask].mean()
    return float((ranks[y].sum() - 0.5 * n_pos * (n_pos + 1)) / (n_pos * n_neg))


def rule_coherence_scores(
    embeddings: np.ndarray,       # [N, D] L2-normalized frozen embeddings
    rule_hits: np.ndarray,        # [N, n_rules] binary observed rule hits (full 335)
    rule_indices: np.ndarray,     # [K] indices of the K rules the probe covers
    weight: np.ndarray,           # [K, D] probe weight
    bias: np.ndarray,             # [K] probe bias
    mean: np.ndarray,             # [D] probe training mean
    std: np.ndarray,              # [D] probe training std
) -> np.ndarray:
    """Per-spectrum rule-coherence AUROC score (range [0, 1], 0.5 = random).

    Decodes rule probabilities ``p = sigmoid((emb - mean) / std @ W.T + b)`` from
    the frozen embedding, then AUROCs ``p`` against the observed rule vector. A
    spectrum with no observed rule (or no absent rule) yields 0.5.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    rule_hits = np.asarray(rule_hits, dtype=np.uint8)[:, rule_indices]
    x = (embeddings - mean.astype(np.float32)) / std.astype(np.float32)
    probs = expit(x @ weight.astype(np.float32).T + bias.astype(np.float32))
    scores = np.empty(len(embeddings), dtype=np.float32)
    for i in range(len(embeddings)):
        scores[i] = roc_auc(rule_hits[i], probs[i])
    return scores
