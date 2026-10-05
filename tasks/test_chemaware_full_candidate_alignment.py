"""Regression tests for full-manifest candidate alignment."""
from __future__ import annotations

import numpy as np
import torch

from train_chemaware_full_candidate_alignment import (
    ResidualSpectrumProjector, identity_balanced_queries,
    error_curriculum_queries, evaluate_crossmodal, formula_identity_epoch_weights,
    learning_rate_scale,
    listwise_losses, sample_training_batch,
)
from dreams.models.chem_aware.global_embedding_adapter import SharedEmbeddingPostAdapter


def main() -> None:
    rng = np.random.default_rng(7)
    identities = np.asarray(["A", "A", "B", "C"])
    selected = identity_balanced_queries(np.arange(4), identities, rng)
    assert len(selected) == 3 and len(set(identities[selected])) == 3
    curriculum = error_curriculum_queries(
        np.arange(4), identities, np.asarray([True, False, False, False]),
        np.asarray([-0.1, 0.2, 0.01, 0.8], dtype=np.float32),
        np.random.default_rng(9), 2, 0.5,
    )
    assert len(curriculum) == 2 and 0 in curriculum
    boundary = error_curriculum_queries(
        np.arange(4), identities, np.asarray([True, False, False, False]),
        np.asarray([-0.1, 0.2, 0.01, 0.8], dtype=np.float32),
        np.random.default_rng(9), 3, 1 / 3, "boundary_mixture",
    )
    assert len(boundary) == 3 and 0 in boundary and 2 in boundary
    formula_weight = formula_identity_epoch_weights(
        np.arange(4), np.asarray(["F1", "F1", "F1", "F2"]), "formula_identity",
    )
    assert np.isclose(formula_weight[:3].sum(), formula_weight[3:].sum())
    assert np.isclose(formula_weight.mean(), 1.0)

    body = {
        "query_row": np.asarray([10]),
        "query_ptr": np.asarray([0, 2]),
        "molecule_ptr": np.asarray([0, 2, 3]),
        "molecule_label": np.asarray([1, 0]),
        "pair_candidate_row": np.asarray([11, 12, 13]),
    }
    sampled = sample_training_batch(
        body, np.asarray([0]), {10: 0, 11: 1, 12: 2, 13: 3},
        np.asarray([5, 6]), 2, np.random.default_rng(1),
    )
    assert sampled["candidate_ptr"].tolist() == [0, 2]
    assert sampled["reference_ptr"].tolist() == [0, 2, 3]
    assert len(sampled["reference_cache"]) == 3
    assert sampled["reference_edge"].tolist() == [0, 1, 2]
    assert sampled["molecule_teacher"].tolist() == [5, 6]
    try:
        sample_training_batch(
            body, np.asarray([0]), {10: 0, 11: 1, 12: 2, 13: 3},
            np.asarray([5, 6]), 2, np.random.default_rng(1),
            np.asarray([True, False]),
        )
    except RuntimeError as error:
        assert "no legal negative" in str(error)
    else:
        raise AssertionError("single-candidate training query was not rejected")

    model = ResidualSpectrumProjector(4, 8, 0.0)
    x = torch.nn.functional.normalize(torch.randn(4, 4), dim=1)
    assert torch.allclose(model(x), x, atol=1e-6)
    wrapped = SharedEmbeddingPostAdapter(torch.nn.Identity(), model)
    assert torch.allclose(wrapped(x), x, atol=1e-6)
    query = x[:1]
    reference = torch.vstack((query, -query, -query))
    molecule = torch.vstack((query, -query))
    (spectrum_loss, molecule_loss, inbatch_spectrum,
     inbatch_molecule, margin_floor) = listwise_losses(
        query, reference, molecule,
        np.asarray([0, 2]), np.asarray([0, 1, 3]), np.asarray([0, 1, 2]), 0.1,
        query, reference,
    )
    assert spectrum_loss.item() < 1e-3 and molecule_loss.item() < 1e-3
    assert inbatch_spectrum.item() == 0.0 and inbatch_molecule.item() == 0.0
    assert margin_floor.item() == 0.0
    assert learning_rate_scale(1, 10, 2) == 0.5
    assert learning_rate_scale(2, 10, 2) == 1.0
    assert learning_rate_scale(10, 10, 2) == 0.0
    crossmodal = evaluate_crossmodal(
        body | {"query_ik14": np.asarray(["A"])}, np.asarray([0]),
        x.numpy(), molecule.numpy(), {10: 0}, np.asarray([0, 1]),
    )
    assert crossmodal["recall1"] == 1.0
    print("PASS: full candidate alignment")


if __name__ == "__main__":
    main()
