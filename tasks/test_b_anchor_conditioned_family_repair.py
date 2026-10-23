#!/usr/bin/env python
"""Fast CPU contracts for the one-shot anchor-conditioned repair."""
from pathlib import Path

import numpy as np

from train_b_anchor_conditioned_family_repair import (
    anchored_features,
    fit_standardizer,
    msg_logit,
    static_family_features,
    strongest_false_positions,
)


def main() -> None:
    names = np.asarray((
        "chemaware_stage1_encoder", "chemaware_phase_a_encoder",
        "noise_msg_fusion_stack_v1",
    ))
    bundle = {
        "module_names": names,
        "scores": np.asarray([
            [0.9, 0.2, 0.1, 0.8],
            [0.8, 0.3, 0.2, 0.7],
            [0.99, 0.01, 0.10, 0.90],
        ], dtype=np.float32),
        "availability": np.ones((3, 4), dtype=np.int8),
        "query_ptr": np.asarray([0, 2, 4], dtype=np.int64),
    }
    static, names = static_family_features(bundle)
    assert static.shape == (4, 9) and len(names) == 9
    assert np.allclose(msg_logit(np.asarray([0.5], dtype=np.float32)), 0.0)

    first_b = np.asarray([0.8, 0.1, 0.2, 0.7], dtype=np.float32)
    second_b = np.asarray([0.1, 0.8, 0.7, 0.2], dtype=np.float32)
    first, feature_names = anchored_features(static, first_b, bundle["query_ptr"])
    second, _ = anchored_features(static, second_b, bundle["query_ptr"])
    assert len(feature_names) == 12
    assert not np.array_equal(first[:, :3], second[:, :3])
    assert np.array_equal(first[:, 3:], second[:, 3:])
    mean, scale = fit_standardizer(first, np.arange(4))
    assert np.all(np.isfinite(mean)) and np.all(scale > 0)

    labels = np.asarray([1, 0, 0, 1], dtype=np.int8)
    positive, negative = strongest_false_positions(first_b, labels, bundle["query_ptr"])
    assert positive.tolist() == [0, 3]
    assert negative.tolist() == [1, 2]

    sbatch = (Path(__file__).resolve().parent / "run_b_anchor_conditioned_family_repair.sbatch").read_text()
    assert "#SBATCH --gpus=1" in sbatch and "#SBATCH --cpus-per-task=10" in sbatch
    assert "--mem" not in sbatch
    assert "enveda" not in sbatch.lower() and "massspecgym" not in sbatch.lower()
    assert "--prior-nested-predictions" in sbatch
    print("PASS: B anchor-conditioned family repair contracts")


if __name__ == "__main__":
    main()
