from __future__ import annotations

import numpy as np

from annotation.bioaware_global_peak import (
    UnionFind,
    mass_relation_pairs,
    normalize_intensity_matrix,
    pair_evidence,
    panel_relations,
)


def test_mass_relation_pair_requires_mass_and_rt() -> None:
    mz = np.asarray([100.0, 121.981942, 121.981942, 137.955882])
    rt = np.asarray([10.0, 12.5, 14.0, 12.0])
    pairs = mass_relation_pairs(
        mz, rt, panel_relations("pos_rp"), ppm=10.0,
        absolute_floor_da=0.002, rt_tolerance_sec=3.0,
    )
    observed = {(a, b, relation.name) for a, b, relation, _, _ in pairs}
    assert (0, 1, "adduct_Na_vs_H") in observed
    assert (0, 2, "adduct_Na_vs_H") not in observed
    assert (0, 3, "adduct_K_vs_H") in observed


def test_pair_evidence_recovers_correlated_abundance() -> None:
    normalized = normalize_intensity_matrix(np.asarray([
        [1, 2, 3, 4, 5, 6],
        [2, 4, 6, 8, 10, 12],
        [20, 10, 30, 15, 25, 12],
    ], dtype=float))
    assert normalized.shape == (3, 6)
    evidence = pair_evidence(
        np.asarray([1, 2, 3, 4, 5, 6], dtype=float),
        np.asarray([2, 4, 6, 8, 10, 12], dtype=float),
    )
    assert evidence["n_joint_detected"] == 6
    assert evidence["abundance_pearson"] > 0.99
    assert evidence["pearson_half_even"] > 0.99
    assert evidence["pearson_half_odd"] > 0.99


def test_union_find_labels_components() -> None:
    union = UnionFind(5)
    union.union(0, 2)
    union.union(2, 4)
    labels = union.labels()
    assert labels[0] == labels[2] == labels[4]
    assert labels[1] != labels[0]
    assert labels[3] != labels[0]
