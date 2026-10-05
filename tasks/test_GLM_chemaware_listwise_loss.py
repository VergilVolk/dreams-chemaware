"""Loss-math contracts for GLM_chemaware_listwise_loss (pure torch, CPU).

Verified properties:
- tau -> 0 recovers the frozen evaluation's molecule-max decision;
- duplicated references of one molecule do not inflate the candidate score
  (reference-count correction: identical references give the identical score);
- a chemical margin on a false candidate strictly increases the loss;
- zero margins (arm 1) equal the no-margin objective;
- gradients flow into query AND reference embeddings;
- dataset + collation reproduce the flat (molecule, slot) layout loss expects,
  and inconsistent layouts fail closed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from GLM_chemaware_listwise_loss import (  # noqa: E402
    GroupTensorDataset,
    candidate_scores_from_cosines,
    collate_groups,
    listwise_group_loss,
)


def _reference_with_cosine(cosine: float, dimension: int) -> torch.Tensor:
    vector = torch.zeros(dimension, dtype=torch.float64)
    vector[0] = float(cosine)
    vector[1] = float(np.sqrt(max(0.0, 1.0 - cosine ** 2)))
    return vector


def explicit_batch(
    groups: list[list[list[float]]],
    margins: list[float] | None = None,
    dimension: int = 8,
) -> dict[str, torch.Tensor]:
    """groups: per query, per molecule, the list of reference cosines."""
    references: list[torch.Tensor] = []
    molecule_of_reference: list[int] = []
    slot_of_reference: list[int] = []
    counts: list[int] = []
    group_ptr = [0]
    molecule = 0
    for per_group in groups:
        for cosines in per_group:
            for slot, cosine in enumerate(cosines):
                references.append(_reference_with_cosine(cosine, dimension))
                molecule_of_reference.append(molecule)
                slot_of_reference.append(slot)
            counts.append(len(cosines))
            molecule += 1
        group_ptr.append(molecule)
    margin_vector = (
        torch.zeros(molecule, dtype=torch.float64)
        if margins is None
        else torch.as_tensor(margins, dtype=torch.float64)
    )
    query_basis = torch.zeros(dimension, dtype=torch.float64)
    query_basis[0] = 1.0
    return {
        "query_embeddings": torch.stack(
            [query_basis.clone() for _ in groups],
        ),
        "reference_embeddings": torch.stack(references),
        "molecule_of_reference": torch.as_tensor(molecule_of_reference),
        "slot_of_reference": torch.as_tensor(slot_of_reference),
        "reference_counts": torch.as_tensor(counts),
        "margins": margin_vector,
        "group_ptr": torch.as_tensor(group_ptr),
    }


def loss_of(batch: dict[str, torch.Tensor], tau: float, temperature: float):
    return listwise_group_loss(
        batch["query_embeddings"], batch["reference_embeddings"],
        batch["molecule_of_reference"], batch["slot_of_reference"],
        batch["reference_counts"], batch["margins"], batch["group_ptr"],
        tau, temperature,
    )


def test_smooth_max_approaches_max_as_tau_shrinks() -> None:
    scores_tau = {
        tau: float(candidate_scores_from_cosines([[0.90, 0.70], [0.50]], tau=tau)[0])
        for tau in (0.5, 0.1, 0.01, 0.001)
    }
    # Smooth-max is bounded by max >= S_tau >= mean of the reference cosines.
    assert 0.80 - 1e-6 <= scores_tau[0.5] <= 0.90 + 1e-6
    assert abs(scores_tau[0.01] - 0.90) < 0.01
    assert abs(scores_tau[0.001] - 0.90) < 1e-3
    assert scores_tau[0.001] > scores_tau[0.01] > scores_tau[0.1] > scores_tau[0.5]


def test_reference_count_correction_neutralises_duplicates() -> None:
    single = candidate_scores_from_cosines([[0.60], [0.50]], tau=0.1)
    duplicated = candidate_scores_from_cosines(
        [[0.60], [0.50, 0.50, 0.50]], tau=0.1,
    )
    assert float(single[0]) == pytest.approx(float(duplicated[0]), abs=1e-7)
    assert float(single[1]) == pytest.approx(float(duplicated[1]), abs=1e-7)

    # End-to-end: the loss is unchanged when a false candidate's single
    # reference is duplicated three times (no count inflation).
    base = explicit_batch([[[0.60], [0.50]]])
    duplicate = explicit_batch([[[0.60], [0.50, 0.50, 0.50]]])
    assert float(loss_of(base, 0.1, 0.05)) == pytest.approx(
        float(loss_of(duplicate, 0.1, 0.05)), abs=1e-6,
    )


def test_margin_on_false_candidate_increases_loss() -> None:
    groups = [[[0.60], [0.55]]]
    plain = explicit_batch(groups)
    margined = explicit_batch(groups, margins=[0.0, 0.05])
    assert float(loss_of(margined, 0.1, 0.05)) > float(
        loss_of(plain, 0.1, 0.05)
    )


def test_margin_is_measured_in_similarity_units_before_temperature() -> None:
    """A 0.05 margin must equal raising the false similarity by 0.05."""
    margined = explicit_batch([[[0.60], [0.50]]], margins=[0.0, 0.05])
    shifted = explicit_batch([[[0.60], [0.55]]])
    assert float(loss_of(margined, 0.1, 0.05)) == pytest.approx(
        float(loss_of(shifted, 0.1, 0.05)), abs=1e-7,
    )


def test_zero_margin_equals_no_margin() -> None:
    groups = [[[0.60, 0.50], [0.55], [0.30]]]
    plain = explicit_batch(groups)
    zeroed = explicit_batch(groups, margins=[0.0, 0.0, 0.0])
    assert float(loss_of(plain, 0.1, 0.05)) == pytest.approx(
        float(loss_of(zeroed, 0.1, 0.05)), abs=0.0,
    )


def test_gradients_reach_query_and_references() -> None:
    batch = explicit_batch([[[0.60], [0.55]], [[0.70], [0.68], [0.20]]])
    query = batch["query_embeddings"].clone().requires_grad_(True)
    reference = batch["reference_embeddings"].clone().requires_grad_(True)
    loss = listwise_group_loss(
        query, reference,
        batch["molecule_of_reference"], batch["slot_of_reference"],
        batch["reference_counts"], batch["margins"], batch["group_ptr"],
        0.1, 0.05,
    )
    loss.backward()
    assert query.grad is not None and float(query.grad.abs().sum()) > 0
    assert reference.grad is not None and float(
        reference.grad.abs().sum()
    ) > 0


def test_better_positive_lowers_loss() -> None:
    weak = explicit_batch([[[0.58], [0.55]]])
    strong = explicit_batch([[[0.80], [0.55]]])
    assert float(loss_of(strong, 0.1, 0.05)) < float(loss_of(weak, 0.1, 0.05))


def test_layout_contracts_fail_closed() -> None:
    batch = explicit_batch([[[0.60], [0.55]]])
    with pytest.raises(RuntimeError):
        broken = dict(batch)
        broken["reference_counts"] = batch["reference_counts"] + 1
        loss_of(broken, 0.1, 0.05)
    with pytest.raises(RuntimeError):
        broken = dict(batch)
        broken["group_ptr"] = torch.as_tensor([0, 1])
        loss_of(broken, 0.1, 0.05)
    with pytest.raises(ValueError):
        loss_of(batch, 0.0, 0.05)


def test_dataset_and_collation_reproduce_layout() -> None:
    query_row = np.asarray([101, 202], dtype=np.int64)
    group_ptr = np.asarray([0, 2, 5], dtype=np.int64)
    molecule_ref_ptr = np.asarray([0, 2, 3, 5, 6, 7], dtype=np.int64)
    ref_row = np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    margins = np.asarray([0.0, 0.0, 0.05, 0.0, 0.0], dtype=np.float32)
    spectra = {
        row: torch.full((4, 2), float(row), dtype=torch.float32)
        for row in range(1, 8)
    }
    spectra[101] = torch.full((4, 2), 101.0, dtype=torch.float32)
    spectra[202] = torch.full((4, 2), 202.0, dtype=torch.float32)
    dataset = GroupTensorDataset(
        query_row, group_ptr, molecule_ref_ptr, ref_row, margins, spectra,
    )
    assert len(dataset) == 2
    batch = collate_groups([dataset[0], dataset[1]])
    assert batch["spec"].shape == (2, 4, 2)
    assert batch["ref_specs"].shape == (7, 4, 2)
    assert batch["group_ptr"].tolist() == [0, 2, 5]
    assert batch["reference_counts"].tolist() == [2, 1, 2, 1, 1]
    assert batch["molecule_of_reference"].tolist() == [0, 0, 1, 2, 2, 3, 4]
    assert batch["slot_of_reference"].tolist() == [0, 1, 0, 0, 1, 0, 0]
    assert batch["margins"].tolist() == pytest.approx(
        [0.0, 0.0, 0.05, 0.0, 0.0],
    )
    # Reference identity check: rows 1..7 in flat order.
    assert torch.allclose(
        batch["ref_specs"][:, 0, 0],
        torch.as_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
    )

    # The collated batch plugs straight into the loss with fake embeddings.
    embeddings = torch.eye(9)[:7].clone()
    query_embeddings = torch.zeros(2, 9, dtype=torch.float32)
    query_embeddings[0, 0] = 1.0
    query_embeddings[1, 1] = 1.0
    # cosines: query0 vs refs 0..2, query1 vs refs 3..6 (orthogonal -> 0).
    loss = listwise_group_loss(
        query_embeddings, embeddings,
        batch["molecule_of_reference"], batch["slot_of_reference"],
        batch["reference_counts"], batch["margins"], batch["group_ptr"],
        0.1, 0.05,
    )
    assert torch.isfinite(loss)
