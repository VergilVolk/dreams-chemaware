"""Small deterministic tests for the E2 corrective action executor."""

from __future__ import annotations

import numpy as np
import torch

from run_noise_final_e2_corrective_scan import (
    action_targets,
    closest_cluster_probabilities,
    matched_control_sequences,
)


def spectrum() -> torch.Tensor:
    value = torch.zeros((8, 2), dtype=torch.float32)
    value[0] = torch.tensor([500.0, 1.0])
    value[1:] = torch.tensor([
        [50.0, 0.10], [75.0, 0.20], [100.0, 0.30], [125.0, 0.40],
        [150.0, 0.50], [175.0, 0.60], [200.0, 0.70],
    ])
    return value


def main() -> None:
    spec = spectrum()
    probabilities = closest_cluster_probabilities(
        np.asarray([49.99, 75.03, 150.0]),
        np.asarray([50.0, 75.0, 150.01]),
        np.asarray([0.1, 0.2, 0.9]),
        0.02,
    )
    assert np.allclose(probabilities[[0, 2]], [0.1, 0.9])
    assert np.isnan(probabilities[1])

    tokens = np.arange(1, 8, dtype=np.int64)
    roles = np.asarray([-1, 1, 1, 0, 2, 1, 1, 1], dtype=np.int8)
    gains = np.asarray([-np.inf, 0.1, 0.5, 0.2, -0.1, 0.4, 0.3, 0.0], dtype=np.float32)
    ranks = np.asarray([-1, 4, 1, 3, -1, 2, 5, -1], dtype=np.int16)
    missing = np.asarray([np.nan, 0.1, 0.7, 0.5, 0.3, 0.8, 0.6, 0.2], dtype=np.float32)

    target, same_role = action_targets(
        "candidate_gradient", 3, 0.5, spec, tokens, roles, gains, ranks, missing, 7,
    )
    assert target.tolist() == [2, 5, 6]
    assert same_role
    target, same_role = action_targets(
        "conditional_missingness_x_confounder", 0, 0.3, spec, tokens,
        roles, gains, ranks, missing, 7,
    )
    assert len(target) == 3 and set(target).issubset({1, 2, 5, 6, 7})
    assert same_role

    controls, levels = matched_control_sequences(spec, np.asarray([1, 2]), roles, 3, 19, True)
    assert len(controls) == 3
    assert levels == ["role_intensity_mz"] * 3
    for control in controls:
        assert len(control) == 2
        assert not (set(control) & {1, 2})
        assert all(roles[token] == 1 for token in control)
    repeated, repeated_levels = matched_control_sequences(spec, np.asarray([1, 2]), roles, 3, 19, True)
    assert repeated_levels == levels
    assert all(np.array_equal(left, right) for left, right in zip(controls, repeated))

    fallback_roles = roles.copy()
    fallback_roles[5:] = 2
    fallback, fallback_levels = matched_control_sequences(
        spec, np.asarray([1, 2]), fallback_roles, 1, 19, True,
    )
    assert len(fallback) == 1
    assert fallback_levels == ["intensity_mz_role_fallback"]
    print("[test_noise_final_e2_corrective_scan] PASS")


if __name__ == "__main__":
    main()
