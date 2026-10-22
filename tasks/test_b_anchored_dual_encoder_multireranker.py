#!/usr/bin/env python
"""Fast CPU contracts for the B-anchored integration primitives."""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from train_b_anchored_dual_encoder_multireranker import (
    AnchoredResidual,
    ARM_EXPERTS,
    EXPERTS,
    build_features,
    feature_columns,
    fit_arm,
    fit_scale,
)


def synthetic_bundle() -> dict:
    names = np.asarray((
        "weighted_spectral_entropy", "noise_v1", "p2b_noise_v1_frozen",
        "chemaware_stage1_encoder", "chemaware_phase_a_encoder",
        "noise_msg_fusion_stack_v1", "noise_rrf_1_7",
    ))
    scores = np.asarray([
        [0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1], [0.4, 0.1, 0.2, 0.3],
        [0.3, 0.5, 0.2, 0.0], [0.1, 0.6, 0.5, 0.2],
        [0.5, 0.2, 0.3, 0.2], [0.4, 0.2, 0.1, 0.5],
    ], dtype=np.float32)
    availability = np.ones_like(scores)
    availability[4, 2:] = 0
    return {"module_names": names, "scores": scores, "availability": availability,
            "query_ptr": np.asarray([0, 2, 4], dtype=np.int64)}


def main() -> None:
    features, names = build_features(synthetic_bundle())
    assert features.shape == (4, 4 * len(EXPERTS))
    assert len(names) == features.shape[1]
    # Missing Phase-A evidence is exactly zero, including after scaling.
    phase = feature_columns((1,))
    assert np.all(features[2:, phase] == 0)
    scale = fit_scale(features, np.arange(4))
    assert np.all(features[2:, phase] / scale[phase] == 0)
    assert np.all(scale > 0)

    torch.manual_seed(1)
    for linear in (False, True):
        model = AnchoredResidual(features.shape[1], hidden=4, bound=0.5, linear=linear)
        x = torch.as_tensor(features)
        base = torch.arange(4, dtype=torch.float32)
        assert torch.equal(model(base, torch.zeros_like(x)), base)
        assert torch.all(torch.abs(model.residual(x)) <= 0.5 + 1e-7)

    # Exercise every fixed arm through the real listwise optimizer.
    repeated = np.tile(features, (3, 1))
    labels = np.tile(np.asarray([1, 0, 1, 0], dtype=np.int8), 3)
    base = np.tile(np.asarray([0.2, 0.1, 0.1, 0.2], dtype=np.float32), 3)
    panel = {
        "features": repeated,
        "query_ptr": np.arange(0, 13, 2, dtype=np.int64),
        "labels": labels,
        "parent": {"B": base},
    }
    recipe = {"hidden": 4, "learning_rate": 0.01, "epochs": 2, "trust_weight": 0.05}
    queries = np.arange(6, dtype=np.int64)
    for arm, experts in ARM_EXPERTS.items():
        fitted = fit_arm(panel, queries, experts, recipe, 7, "cpu",
                         linear=(arm == "complete_spectral_additive"))
        assert np.isfinite(fitted[-1])
    sbatch = (Path(__file__).resolve().parent / "run_b_anchored_dual_encoder_multireranker_gnps.sbatch").read_text()
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --cpus-per-task=10" in sbatch
    assert "enveda" not in sbatch.lower()
    assert "massspecgym" not in sbatch.lower()
    for module in ("chemaware_stage1_encoder", "chemaware_phase_a_encoder",
                   "noise_msg_fusion_stack_v1", "noise_rrf_1_7"):
        assert f'--module {module}' in sbatch
    print("PASS: B-anchored dual-encoder/multi-reranker contracts")


if __name__ == "__main__":
    main()
