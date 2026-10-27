import numpy as np

from run_contrast_atlas_open_local_edge_enrichment import (
    StudyProfiles,
    bh_adjust,
    build_profile_reassignments,
    count_and_strength,
)


def profile(entities, status, weight, positive=None, negative=None):
    weight = np.asarray(weight, dtype=np.float32)
    shape = weight.shape
    return StudyProfiles(
        entities=np.asarray(entities, dtype=str),
        contrasts=[f"c{i}" for i in range(shape[1])],
        status=np.asarray(status, dtype=str),
        positive=np.zeros(shape, dtype=bool) if positive is None else np.asarray(positive),
        negative=np.zeros(shape, dtype=bool) if negative is None else np.asarray(negative),
        weight=weight,
        beta=weight.copy(),
        se=np.ones(shape, dtype=np.float32),
        q=np.ones(shape, dtype=np.float32),
    )


def test_continuous_strength_does_not_require_normal_approximation_fdr():
    left = profile(["a", "b"], ["unresolved_entity"] * 2, [[0.8], [-0.7]])
    right = profile(["x", "y"], ["unresolved_entity"] * 2, [[0.6], [-0.5]])
    same, opposite, same_strength, opposite_strength = count_and_strength(
        left,
        right,
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        np.asarray([1.0, 0.5]),
    )
    assert same.item() == 0
    assert opposite.item() == 0
    assert np.isclose(same_strength.item(), 0.8 * 0.6 + 0.5 * 0.7 * 0.5)
    assert opposite_strength.item() == 0


def test_profile_reassignment_preserves_identity_strata():
    profiles = {
        "study": profile(
            ["a", "b", "c", "d"],
            ["library_supported", "library_supported", "unresolved_entity", "unresolved_entity"],
            [[0.1], [0.2], [0.3], [0.4]],
        )
    }
    donors = build_profile_reassignments(profiles, permutations=32, seed=7)["study"]
    assert donors.shape == (32, 4)
    assert all(set(row[:2]) == {0, 1} for row in donors)
    assert all(set(row[2:]) == {2, 3} for row in donors)


def test_bh_adjust_restores_original_order():
    observed = bh_adjust(np.asarray([0.04, 0.001, 0.02, 0.5]))
    assert np.allclose(observed, [0.05333333333333334, 0.004, 0.04, 0.5])
