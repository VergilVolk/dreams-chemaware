#!/usr/bin/env python3
"""Contracts for the joint representation/spectral reference-set readout."""
from __future__ import annotations

import numpy as np
import torch

from full23_joint_evidence_set_core import (
    Full23JointEvidenceSetReadout,
    JointEvidenceSettings,
    joint_model_from_payload,
)
from full23_tensor_set_readout_core import TensorSetSettings


def main() -> None:
    torch.manual_seed(17)
    embedding = TensorSetSettings(
        encoder_names=("chemaware_stage1", "chemaware_phasea"), embedding_dimension=8,
        pair_dimension=4, encoder_dimension=4, attention_heads=2,
        fusion_hidden=8, top_k=3, include_reference_count=False,
    )
    model = Full23JointEvidenceSetReadout(
        JointEvidenceSettings(
            embedding=embedding, spectral_token_dimension=4,
            spectral_summary_dimension=6, fusion_hidden=8,
        ),
        residual_bound=0.5,
        spectral_mean=np.zeros(4, dtype=np.float32),
        spectral_scale=np.ones(4, dtype=np.float32),
    )
    banks = {name: torch.randn(9, 8) for name in embedding.encoder_names}
    query = torch.tensor([[0, 0], [1, 1], [2, 2], [0, 0]])
    refs = torch.tensor([
        [[3, 4, 5], [3, 4, 5]],
        [[4, 5, 6], [4, 5, 6]],
        [[6, 7, 8], [6, 7, 8]],
        [[3, 5, 7], [3, 5, 7]],
    ])
    mask = torch.tensor([
        [[True, True, False], [True, True, False]],
        [[True, True, True], [True, True, True]],
        [[True, False, False], [True, False, False]],
        [[True, True, True], [True, True, True]],
    ])
    spectral = torch.randn(4, 3, 4)
    spectral_mask = torch.tensor([
        [True, True, False], [True, True, True], [True, False, False],
        [True, True, True],
    ])
    anchor = torch.tensor([0.2, -0.1, 0.4, 0.0])
    query_ptr = np.asarray([0, 2, 4], dtype=np.int64)
    output = model(anchor, query, refs, mask, banks, spectral, spectral_mask, query_ptr)
    assert torch.equal(output, anchor), "zero initialization must preserve Full23 exactly"

    # Masked reference values are semantically absent and cannot change output.
    changed = spectral.clone()
    changed[~spectral_mask] = 1e6
    assert torch.equal(
        model(anchor, query, refs, mask, banks, changed, spectral_mask, query_ptr), output,
    )

    # Reference multiplicity is deliberately not a shortcut in the successor.
    assert model.settings.include_spectral_count is False
    assert model.settings.embedding.include_reference_count is False

    target = torch.tensor([0.8, -0.7, 0.1, -0.2])
    for _ in range(2):
        loss = (model(anchor, query, refs, mask, banks, spectral, spectral_mask, query_ptr) - target).square().mean()
        loss.backward()
        if _ == 0:
            assert model.spectral_tokens[0].weight.grad is not None
            assert model.embedding_readout.pair_encoders["chemaware_stage1"][0].weight.grad is not None
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.data.add_(parameter.grad, alpha=-0.01)
                parameter.grad = None
    assert not torch.equal(
        model(anchor, query, refs, mask, banks, spectral, spectral_mask, query_ptr), anchor,
    )

    payload = {
        "schema": "full23_joint_evidence_set_readout_v1",
        "settings": {
            "embedding": {
                **embedding.__dict__,
                "encoder_names": list(embedding.encoder_names),
            },
            "spectral_names": [
                "noise_v1", "weighted_spectral_entropy",
                "p2b_noise_v1_frozen", "noise_msg_fusion_stack_v1",
            ],
            "spectral_token_dimension": 4,
            "spectral_summary_dimension": 6,
            "fusion_hidden": 8,
            "include_spectral_count": False,
            "query_center_raw_residual": True,
        },
        "residual_bound": 0.5,
        "spectral_mean": np.zeros(4, dtype=np.float32),
        "spectral_scale": np.ones(4, dtype=np.float32),
        "state_dict": model.state_dict(),
    }
    restored = joint_model_from_payload(payload)
    assert torch.allclose(
        restored(anchor, query, refs, mask, banks, spectral, spectral_mask, query_ptr),
        model(anchor, query, refs, mask, banks, spectral, spectral_mask, query_ptr),
    )
    print("PASS: Full23 joint embedding/spectral reference-set contracts")


if __name__ == "__main__":
    main()
