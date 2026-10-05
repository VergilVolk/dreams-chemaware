"""GLM listwise objective core (pure torch; pre-registered 2026-09-30).

The candidate-set listwise objective of
docs/GLM_CHEMAWARE_LISTWISE_TWO_ARM_PREREGISTRATION_20260930.md, isolated from
the Lightning trainer so the loss mathematics is testable on any host:

    S_tau(q, c) = tau * ( logsumexp_{r in R(c)} cos(z_q, z_r) / tau
                          - log |R(c)| )
    L(q) = softmax_cross_entropy( (S_tau(q, .) + margin(.)) / T, target = 0 )

``S_tau`` is the reference-count-corrected smooth maximum over each candidate
molecule's reference spectra: as tau -> 0 it approaches the frozen evaluation's
``max_r cos(z_q, z_r)`` decision, while duplicated references of one molecule
do not inflate the score (log-mean-exp, not log-sum-exp).  Molecule 0 of each
group is the positive target; margins live on candidate molecules only.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset

SHARED_ARRAY_ORDER = (
    "query_row", "group_ptr", "molecule_ref_ptr", "ref_row",
    "molecule_label", "val_query_mask",
)


def shared_arrays_sha256(pool: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in SHARED_ARRAY_ORDER:
        digest.update(np.ascontiguousarray(pool[name]).tobytes())
    return digest.hexdigest()


def listwise_group_loss(
    query_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    molecule_of_reference: torch.Tensor,
    slot_of_reference: torch.Tensor,
    reference_counts: torch.Tensor,
    margins: torch.Tensor,
    group_ptr: torch.Tensor,
    tau: float,
    temperature: float,
) -> torch.Tensor:
    """Smooth-maximum candidate-group cross-entropy for one batch.

    ``query_embeddings``: (B, h); ``reference_embeddings``: (R, h);
    ``molecule_of_reference``/``slot_of_reference``: (R,) long tensors placing
    each reference into a (molecule, slot) grid; ``reference_counts``: (M,);
    ``margins``: (M,); ``group_ptr``: (B+1,) long tensor.  Molecule 0 of each
    group is the positive target; references must not cross query boundaries.
    """
    if tau <= 0 or temperature <= 0:
        raise ValueError("tau and temperature must be positive")
    if len(group_ptr) != len(query_embeddings) + 1:
        raise RuntimeError("group pointer does not span the batch queries")
    molecules_per_group = torch.diff(group_ptr)
    if int(molecules_per_group.min().item()) < 2:
        raise RuntimeError("a training group carries fewer than two candidates")
    molecule_count = int(reference_counts.shape[0])
    if int(molecules_per_group.sum().item()) != molecule_count:
        raise RuntimeError("molecule count disagrees with the group pointer")
    if len(molecule_of_reference) != len(reference_embeddings):
        raise RuntimeError("reference layout arrays disagree")
    if len(slot_of_reference) != len(reference_embeddings):
        raise RuntimeError("reference slot layout disagrees")
    if int(reference_counts.sum().item()) != len(reference_embeddings):
        raise RuntimeError("reference counts do not span the references")
    if margins.shape[0] != molecule_count:
        raise RuntimeError("margin vector does not span the molecules")

    molecule_batch = torch.repeat_interleave(
        torch.arange(len(query_embeddings), device=query_embeddings.device),
        molecules_per_group,
    )
    if not torch.all(molecule_of_reference[1:] >= molecule_of_reference[:-1]):
        raise RuntimeError("references must be grouped by molecule contiguously")
    if not torch.all(
        slot_of_reference < reference_counts[molecule_of_reference],
    ):
        raise RuntimeError("a reference slot exceeds its molecule's count")
    reference_batch = molecule_batch[molecule_of_reference]
    max_slots = int(reference_counts.max().item())
    cosines = (
        query_embeddings[reference_batch] * reference_embeddings
    ).sum(dim=-1)
    padded = cosines.new_full((molecule_count, max_slots), float("-inf"))
    padded[molecule_of_reference, slot_of_reference] = cosines / tau
    maximum = padded.max(dim=1).values
    logsumexp = maximum + torch.log(
        torch.exp(padded - maximum.unsqueeze(1)).sum(dim=1).clamp_min(1e-30)
    )
    candidate_scores = tau * (logsumexp - torch.log(reference_counts.float()))
    # Margins are expressed in cosine-similarity units, matching the native
    # DreaMS triplet margin.  Adding them after division by temperature would
    # silently shrink a 0.05 chemical boundary to 0.0025 similarity units.
    logits = (candidate_scores + margins) / temperature
    losses = []
    for index in range(len(query_embeddings)):
        left = int(group_ptr[index])
        right = int(group_ptr[index + 1])
        losses.append(torch.nn.functional.cross_entropy(
            logits[left:right].unsqueeze(0),
            torch.zeros(1, dtype=torch.long, device=logits.device),
        ))
    return torch.stack(losses).mean()


def candidate_scores_from_cosines(
    cosines_by_molecule: list[list[float]], tau: float,
) -> torch.Tensor:
    """Reference-count-corrected smooth-maximum scores (test/inspection aid)."""
    molecule_count = len(cosines_by_molecule)
    max_slots = max(len(rows) for rows in cosines_by_molecule)
    padded = torch.full((molecule_count, max_slots), float("-inf"))
    counts = torch.zeros(molecule_count, dtype=torch.long)
    for molecule, rows in enumerate(cosines_by_molecule):
        for slot, cosine in enumerate(rows):
            padded[molecule, slot] = float(cosine) / tau
        counts[molecule] = len(rows)
    maximum = padded.max(dim=1).values
    logsumexp = maximum + torch.log(
        torch.exp(padded - maximum.unsqueeze(1)).sum(dim=1).clamp_min(1e-30)
    )
    return tau * (logsumexp - torch.log(counts.float()))


class GroupTensorDataset(Dataset):
    """Deterministic per-group tensors; collation flattens the batch."""

    def __init__(
        self,
        query_row: np.ndarray,
        group_ptr: np.ndarray,
        molecule_ref_ptr: np.ndarray,
        ref_row: np.ndarray,
        margins: np.ndarray,
        spectra: dict[int, torch.Tensor],
    ) -> None:
        self.query_row = np.asarray(query_row, dtype=np.int64)
        self.group_ptr = np.asarray(group_ptr, dtype=np.int64)
        self.molecule_ref_ptr = np.asarray(molecule_ref_ptr, dtype=np.int64)
        self.ref_row = np.asarray(ref_row, dtype=np.int64)
        self.margins = np.asarray(margins, dtype=np.float32)
        self.spectra = spectra

    def __len__(self) -> int:
        return len(self.query_row)

    def __getitem__(self, index: int) -> dict[str, object]:
        left, right = map(int, self.group_ptr[index:index + 2])
        molecule_ids = list(range(left, right))
        molecule_of_ref: list[int] = []
        slot_of_ref: list[int] = []
        refs: list[torch.Tensor] = []
        for molecule in molecule_ids:
            r_left = int(self.molecule_ref_ptr[molecule])
            r_right = int(self.molecule_ref_ptr[molecule + 1])
            for slot, ref_index in enumerate(range(r_left, r_right)):
                molecule_of_ref.append(molecule - left)
                slot_of_ref.append(slot)
                refs.append(self.spectra[int(self.ref_row[ref_index])])
        return {
            "query": self.spectra[int(self.query_row[index])],
            "refs": refs,
            "local_molecule_of_ref": np.asarray(molecule_of_ref, dtype=np.int64),
            "slot_of_ref": np.asarray(slot_of_ref, dtype=np.int64),
            "margins": self.margins[left:right],
        }


def collate_groups(samples: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    query = torch.stack([torch.as_tensor(sample["query"]) for sample in samples])
    refs = torch.stack([
        torch.as_tensor(reference)
        for sample in samples for reference in sample["refs"]
    ])
    molecules_per_sample = [
        len(np.asarray(sample["margins"], dtype=np.float32))
        for sample in samples
    ]
    molecule_offsets = np.cumsum([0] + molecules_per_sample)
    molecule_of_reference = torch.as_tensor(np.concatenate([
        np.asarray(sample["local_molecule_of_ref"], dtype=np.int64)
        + int(molecule_offsets[index])
        for index, sample in enumerate(samples)
    ]))
    slot_of_reference = torch.as_tensor(np.concatenate([
        np.asarray(sample["slot_of_ref"], dtype=np.int64)
        for sample in samples
    ]))
    margins = torch.as_tensor(np.concatenate([
        np.asarray(sample["margins"], dtype=np.float32)
        for sample in samples
    ]))
    reference_counts = torch.bincount(
        molecule_of_reference, minlength=int(molecule_offsets[-1]),
    ).long()
    return {
        "spec": query,
        "ref_specs": refs,
        "molecule_of_reference": molecule_of_reference,
        "slot_of_reference": slot_of_reference,
        "reference_counts": reference_counts,
        "margins": margins,
        "group_ptr": torch.as_tensor(molecule_offsets, dtype=torch.long),
    }
