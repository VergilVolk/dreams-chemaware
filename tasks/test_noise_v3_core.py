from __future__ import annotations

import numpy as np
import torch

from noise_v3_core import (
    CONFOUNDER_ONLY, IDENTITY_ONLY, SHARED, UNMATCHED,
    attenuate_and_renormalize, attenuate_sequence, candidate_peak_roles,
    matched_control_tokens, matched_control_tokens_strict,
    matched_control_tokens_strict_excluding, rank_gradient_targets,
    rank_role_targets, select_gradient_target, select_role_target,
)


def spec(rows):
    return torch.tensor([[400.0, 1.1], *rows, *([[0.0, 0.0]] * (5 - len(rows)))])


def test_roles_and_precursor_protection():
    query = spec([[50.0, 1.0], [75.0, 0.8], [100.0, 0.6], [150.0, 0.4]])
    positive = spec([[50.01, 1.0], [100.01, 0.5]])
    negative = spec([[75.01, 1.0], [100.01, 0.5]])
    roles = candidate_peak_roles(query, positive, negative, 0.02)
    assert roles.tolist()[:5] == [-1, IDENTITY_ONLY, CONFOUNDER_ONLY, SHARED, UNMATCHED]


def test_attenuation_renormalizes_without_touching_precursor():
    clean = spec([[50.0, 1.0], [75.0, 0.5]])
    changed = attenuate_and_renormalize(clean, 1, 0.5)
    assert changed[0].tolist() == clean[0].tolist()
    assert np.isclose(float(changed[1, 1]), 1.0)
    assert np.isclose(float(changed[2, 1]), 1.0)
    deleted = attenuate_and_renormalize(clean, 1, 1.0)
    assert deleted[1].tolist() == [0.0, 0.0]
    assert np.isclose(float(deleted[2, 1]), 1.0)


def test_gradient_target_and_controls():
    clean = spec([[50.0, 1.0], [75.0, 0.8], [100.0, 0.6], [150.0, 0.4]])
    roles = np.asarray([-1, IDENTITY_ONLY, CONFOUNDER_ONLY, SHARED, UNMATCHED, -1])
    gradient = np.asarray([0.0, -10.0, -1.0, -3.0, 1.0, 0.0])
    # Identity-only has the largest hypothetical gain but is protected.
    assert select_gradient_target(clean, gradient, roles, 0.5) == 3
    controls = matched_control_tokens(clean, 3, roles, 2, 7, same_role=False)
    assert 3 not in controls
    assert 0 not in controls


def test_role_targets_are_symmetric_and_never_select_precursor():
    clean = spec([[50.0, 0.4], [75.0, 0.9], [100.0, 0.8], [150.0, 0.2]])
    roles = np.asarray([-1, IDENTITY_ONLY, CONFOUNDER_ONLY, IDENTITY_ONLY, UNMATCHED, -1])
    assert select_role_target(clean, roles, IDENTITY_ONLY) == 3
    assert select_role_target(clean, roles, CONFOUNDER_ONLY) == 2
    assert select_role_target(clean, roles, SHARED) is None


def test_strict_controls_never_fall_back_to_another_role():
    clean = spec([[50.0, 1.0], [75.0, 0.8], [100.0, 0.6], [150.0, 0.4]])
    roles = np.asarray([-1, IDENTITY_ONLY, CONFOUNDER_ONLY, SHARED, SHARED, -1])
    # Only one confounder-only token exists, hence no strict controls.
    assert matched_control_tokens_strict(clean, 2, roles, 1, 7).size == 0
    controls = matched_control_tokens_strict(clean, 3, roles, 1, 7)
    assert controls.tolist() == [4]
    assert roles[controls[0]] == roles[3]


def test_one_selection_and_controls_are_reused_across_doses():
    clean = spec([[50.0, 1.0], [75.0, 0.8], [100.0, 0.6], [150.0, 0.4]])
    roles = np.asarray([-1, SHARED, SHARED, SHARED, UNMATCHED, -1])
    target = select_role_target(clean, roles, SHARED)
    controls = matched_control_tokens_strict(clean, target, roles, 2, 17)
    assert target == 1
    assert controls.tolist() == matched_control_tokens_strict(
        clean, target, roles, 2, 17,
    ).tolist()
    variants = [
        (dose, target, tuple(controls)) for dose in (0.25, 0.50, 0.75, 1.0)
    ]
    assert len({item[1:] for item in variants}) == 1


def test_topk_gradient_targets_are_unique_ordered_and_identity_protected():
    clean = spec([[50.0, 1.0], [75.0, 0.8], [100.0, 0.6], [150.0, 0.4]])
    roles = np.asarray([-1, IDENTITY_ONLY, CONFOUNDER_ONLY, SHARED, UNMATCHED, -1])
    gradient = np.asarray([0.0, -10.0, -2.0, -3.0, -1.0, 0.0])
    # Gains at 50%: token 3=.9, token 2=.8, token 4=.2; token 1 protected.
    targets = rank_gradient_targets(clean, gradient, roles, 0.5, 5, True)
    assert targets.tolist() == [3, 2, 4]
    assert len(np.unique(targets)) == len(targets)


def test_topk_role_targets_use_intensity_then_token_tie_break():
    clean = spec([[50.0, 0.8], [75.0, 0.8], [100.0, 0.6], [150.0, 0.4]])
    roles = np.asarray([-1, CONFOUNDER_ONLY, CONFOUNDER_ONLY, CONFOUNDER_ONLY, SHARED, -1])
    assert rank_role_targets(clean, roles, CONFOUNDER_ONLY, 3).tolist() == [1, 2, 3]


def test_strict_excluding_controls_never_reuse_path_tokens():
    clean = spec([[50.0, 1.0], [75.0, 0.8], [100.0, 0.6], [150.0, 0.4]])
    roles = np.asarray([-1, CONFOUNDER_ONLY, CONFOUNDER_ONLY, CONFOUNDER_ONLY, CONFOUNDER_ONLY, -1])
    controls = matched_control_tokens_strict_excluding(
        clean, 1, roles, repeats=2, seed=7, excluded={2},
    )
    assert len(controls) == 2
    assert 1 not in controls and 2 not in controls
    assert len(set(map(int, controls))) == 2


def test_attenuate_sequence_is_unique_and_preserves_precursor():
    clean = spec([[50.0, 1.0], [75.0, 0.8], [100.0, 0.6], [150.0, 0.4]])
    output = attenuate_sequence(clean, [1, 3], 0.5)
    assert torch.equal(output[0], clean[0])
    assert output[1, 1] < clean[1, 1]
    assert output[3, 1] < clean[3, 1]
    try:
        attenuate_sequence(clean, [1, 1], 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate sequential token was accepted")


def main():
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}", flush=True)
    print(f"Noise-v3 core tests passed: {len(tests)}", flush=True)


if __name__ == "__main__":
    main()
