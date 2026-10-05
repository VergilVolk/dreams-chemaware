"""CPU contracts for candidate-level ChemAware evidence aggregation."""
from __future__ import annotations

import numpy as np

from audit_chemaware_candidate_evidence_policy import (
    DESCRIPTOR_NAMES,
    FEATURE_NAMES,
    build_candidate_table,
    max_only_indices,
    reference_descriptor,
)
from audit_chemaware_mass_kernel_embedding import content_permutation_offset


def main() -> None:
    descriptor = reference_descriptor(
        np.asarray([0.2, 0.8]), np.asarray([0.1, 0.9]), np.asarray([0.7, 0.3]),
    )
    assert descriptor.shape == (len(DESCRIPTOR_NAMES),)
    assert np.isfinite(descriptor).all()
    assert float(descriptor[DESCRIPTOR_NAMES.index("argmax_official_mass")]) == 1.0
    assert float(descriptor[DESCRIPTOR_NAMES.index("argmax_official_rule")]) == 0.0

    actions = [(0.0, 0.1), (0.1, 0.0), (0.1, 0.2)]
    scored = {
        "query": np.asarray([7]),
        "reference_ptr": np.asarray([np.asarray([0, 2, 3, 5])], dtype=object),
        "labels": np.asarray([np.asarray([False, True, False])], dtype=object),
        "global": np.asarray([np.asarray([0.90, 0.80, 0.70, 0.60, 0.50])], dtype=object),
        "mass": np.asarray([np.asarray([0.10, 0.10, 0.95, 0.10, 0.10])], dtype=object),
        "rule_response": np.asarray([np.asarray([0.10, 0.10, 0.95, 0.10, 0.10])], dtype=object),
        "old_rank": np.asarray([2], dtype=np.int16),
    }
    table = build_candidate_table(scored, actions, global_action=2, rule_key="rule_response")
    assert table["feature"].shape == (1, 2, len(FEATURE_NAMES))
    assert int(table["valid"].sum()) == 2
    assert list(map(int, table["proposed_candidate"][0])) == [1, 2]
    assert int(table["rank"][0, 0]) == 1
    assert bool(table["benefit"][0, 0])
    assert not bool(table["benefit"][0, 1])
    assert not bool(table["harmful"].any())

    maximum = max_only_indices()
    selected_names = [FEATURE_NAMES[index] for index in maximum]
    assert "candidate_official_max" in selected_names
    assert "candidate_official_mean" not in selected_names
    assert len(maximum) < len(FEATURE_NAMES)

    mz = np.asarray([50.0, 75.25, 101.1])
    intensity = np.asarray([0.6, 0.3, 0.1])
    offset = content_permutation_offset(mz, intensity, 150.0, 316)
    assert offset == content_permutation_offset(mz, intensity, 150.0, 316)
    salted = {
        content_permutation_offset(mz, intensity, 150.0, 316, salt)
        for salt in (b"", b"chemaware-null-b", b"chemaware-null-c")
    }
    assert len(salted) >= 2
    changed_offset = content_permutation_offset(mz + np.asarray([0.0, 0.0, 1e-3]), intensity, 150.0, 316)
    assert offset != changed_offset
    assert 1 <= offset < 316 and 1 <= changed_offset < 316
    print("PASS: 8 candidate-evidence ChemAware contracts")


if __name__ == "__main__":
    main()
