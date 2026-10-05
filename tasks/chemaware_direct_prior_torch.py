"""Torch objective for minimal-drift direct chemical residual transfer.

For each active query, molecule-level shared-embedding scores are compared with
the frozen official DreaMS scores.  Their score change is candidate-centred and
matched to an independently frozen chemical residual.  Candidate centring is
the minimum-L2 representative of the pairwise score differences, so the loss
does not spend capacity on ranking-irrelevant common offsets.

Inactive queries, and every arm with ``alpha == 0``, have a literal zero
chemical loss.  In particular, an alpha-zero control is not a hidden
distillation-to-official objective.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def select_official_boundary_batch(
    body: dict[str, np.ndarray],
    queries: np.ndarray,
    row_position: dict[int, int],
    official_embedding: np.ndarray,
    references_per_molecule: int,
    rng: np.random.Generator,
    molecule_allowed: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build a batch that always contains the frozen DreaMS top reference.

    Random-only reference subsampling changes the active candidate boundary.
    This sampler first includes the official highest-scoring reference for
    every molecule, then adds random distinct references up to the requested
    count.  The first candidate molecule must remain the labelled truth.
    """
    selected_query = np.asarray(queries, dtype=np.int64)
    if selected_query.ndim != 1 or not len(selected_query):
        raise ValueError("queries must be a non-empty vector")
    if references_per_molecule < 1:
        raise ValueError("references_per_molecule must be positive")
    query_cache = np.asarray(
        [row_position[int(body["query_row"][q])] for q in selected_query],
        dtype=np.int64,
    )
    reference_cache: list[int] = []
    molecule_index: list[int] = []
    candidate_ptr = [0]
    reference_ptr = [0]
    for query_position, query in enumerate(selected_query):
        left, right = map(int, body["query_ptr"][query : query + 2])
        kept = []
        for molecule in range(left, right):
            if molecule_allowed is not None and not molecule_allowed[molecule]:
                continue
            rleft, rright = map(int, body["molecule_ptr"][molecule : molecule + 2])
            available_row = np.asarray(
                [
                    int(row)
                    for row in body["pair_candidate_row"][rleft:rright]
                    if int(row) in row_position
                ],
                dtype=np.int64,
            )
            if not len(available_row):
                raise RuntimeError("candidate molecule has no cached reference spectrum")
            available_position = np.asarray(
                [row_position[int(row)] for row in available_row], dtype=np.int64
            )
            qvec = np.asarray(official_embedding[query_cache[query_position]])
            score = np.asarray(official_embedding[available_position]) @ qvec
            top = int(np.argmax(score))
            chosen = [top]
            remaining = np.delete(np.arange(len(available_row), dtype=np.int64), top)
            take_random = min(references_per_molecule - 1, len(remaining))
            if take_random:
                chosen.extend(
                    map(int, rng.choice(remaining, size=take_random, replace=False))
                )
            reference_cache.extend(map(int, available_position[chosen]))
            reference_ptr.append(len(reference_cache))
            molecule_index.append(molecule)
            kept.append(molecule)
        if len(kept) < 2:
            raise RuntimeError("query has fewer than two eligible candidate molecules")
        if int(body["molecule_label"][kept[0]]) != 1:
            raise RuntimeError("candidate filtering removed or reordered the true molecule")
        candidate_ptr.append(len(molecule_index))
    unique_reference, reference_edge = np.unique(
        np.asarray(reference_cache, dtype=np.int64), return_inverse=True
    )
    return {
        "query_cache": query_cache,
        "reference_cache": unique_reference,
        "reference_edge": reference_edge.astype(np.int64),
        "molecule_index": np.asarray(molecule_index, dtype=np.int64),
        "candidate_ptr": np.asarray(candidate_ptr, dtype=np.int64),
        "reference_ptr": np.asarray(reference_ptr, dtype=np.int64),
    }


def molecule_max_scores(
    query_embedding: torch.Tensor,
    reference_embedding: torch.Tensor,
    candidate_ptr: np.ndarray,
    reference_ptr: np.ndarray,
    reference_edge: np.ndarray,
) -> torch.Tensor:
    """Return flat molecule scores, preserving differentiable max pooling."""
    if query_embedding.ndim != 2 or reference_embedding.ndim != 2:
        raise ValueError("query and reference embeddings must be matrices")
    ptr = np.asarray(candidate_ptr, dtype=np.int64)
    rptr = np.asarray(reference_ptr, dtype=np.int64)
    edge = torch.as_tensor(reference_edge, device=reference_embedding.device)
    if ptr.ndim != 1 or len(ptr) != len(query_embedding) + 1 or ptr[0] != 0:
        raise ValueError("candidate_ptr does not align with the query batch")
    if rptr.ndim != 1 or len(rptr) != int(ptr[-1]) + 1 or rptr[0] != 0:
        raise ValueError("reference_ptr does not align with candidate_ptr")
    if rptr[-1] != len(edge) or np.any(np.diff(ptr) < 2) or np.any(np.diff(rptr) < 1):
        raise ValueError("every query needs two molecules and every molecule a reference")
    output = []
    for query, (left, right) in enumerate(zip(ptr[:-1], ptr[1:])):
        for molecule in range(int(left), int(right)):
            rleft, rright = map(int, rptr[molecule : molecule + 2])
            output.append(
                (reference_embedding[edge[rleft:rright]] @ query_embedding[query]).max()
            )
    return torch.stack(output)


def candidate_centered_residual_loss(
    student_score: torch.Tensor,
    official_score: torch.Tensor,
    centered_prior: torch.Tensor,
    query_ptr: np.ndarray,
    *,
    alpha: float,
    huber_delta: float,
    query_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Query-equal Huber transfer of a frozen candidate-centred residual."""
    if student_score.ndim != 1 or official_score.shape != student_score.shape:
        raise ValueError("student and official scores must be aligned vectors")
    if centered_prior.shape != student_score.shape:
        raise ValueError("chemical residual does not align with molecule scores")
    ptr = np.asarray(query_ptr, dtype=np.int64)
    if (
        ptr.ndim != 1
        or len(ptr) < 2
        or ptr[0] != 0
        or ptr[-1] != len(student_score)
        or np.any(np.diff(ptr) < 2)
    ):
        raise ValueError("query_ptr is not a valid candidate partition")
    if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("alpha must be nonnegative and Huber delta positive")
    if not torch.all(torch.isfinite(official_score)) or not torch.all(torch.isfinite(centered_prior)):
        raise ValueError("official score and chemical residual must be finite")

    # This early return is part of the causal contract.  It must not pull the
    # concurrently updated encoder back toward the official initialization.
    if alpha == 0:
        return student_score.sum() * 0.0

    if query_weight is None:
        weight = torch.ones(
            len(ptr) - 1, device=student_score.device, dtype=student_score.dtype
        )
    else:
        weight = query_weight.to(device=student_score.device, dtype=student_score.dtype)
    if (
        weight.shape != (len(ptr) - 1,)
        or not torch.all(torch.isfinite(weight))
        or torch.any(weight <= 0)
    ):
        raise ValueError("query weights must be one finite positive value per query")

    losses = []
    for left, right in zip(ptr[:-1], ptr[1:]):
        left, right = int(left), int(right)
        prior = centered_prior[left:right]
        if float(torch.abs(prior.sum()).detach().cpu()) > 1e-5:
            raise ValueError("chemical prior must be candidate-centred per query")
        observed = student_score[left:right] - official_score[left:right].detach()
        observed = observed - observed.mean()
        error = observed - float(alpha) * prior.detach()
        losses.append(F.smooth_l1_loss(
            error,
            torch.zeros_like(error),
            beta=huber_delta,
            reduction="mean",
        ))
    stacked = torch.stack(losses)
    return torch.sum(stacked * weight) / torch.sum(weight)
