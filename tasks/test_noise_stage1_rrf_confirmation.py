#!/usr/bin/env python
"""CPU unit tests for the frozen Stage-1 RRF confirmation recipe."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from noise_corrected_fullgraph_evaluation import GraphScores
from noise_stage1_rrf_core import (
    RRF_K,
    build_rrf_graph_scores,
    fixed_rrf_score,
    tie_aware_midranks,
)


def test_tie_aware_midrank_is_order_independent() -> None:
    got = tie_aware_midranks(np.asarray([1.0, 0.0, 1.0, 0.5]))
    assert np.array_equal(got, np.asarray([1.5, 4.0, 1.5, 3.0]))
    perm = np.asarray([2, 1, 0, 3])
    permuted = tie_aware_midranks(np.asarray([1.0, 0.0, 1.0, 0.5])[perm])
    restored = np.empty_like(permuted)
    restored[perm] = permuted
    assert np.array_equal(restored, got)


def test_fixed_recipe_requires_three_votes_and_k60() -> None:
    values = np.asarray([0.3, 0.2, 0.1])
    assert np.all(np.isfinite(fixed_rrf_score(values, values, values)))
    try:
        fixed_rrf_score(values, values)
        raise AssertionError("two-vote recipe was accepted")
    except RuntimeError:
        pass
    try:
        fixed_rrf_score(values, values, values, k=RRF_K + 1)
        raise AssertionError("tuned RRF k was accepted")
    except RuntimeError:
        pass


def test_full_candidate_block_uses_label_blind_two_of_three_vote() -> None:
    graph = SimpleNamespace(
        n_queries=1,
        query_row=np.asarray([0]),
        query_ptr=np.asarray([0, 2]),
        molecule_ptr=np.asarray([0, 1, 2]),
        pair_candidate_row=np.asarray([1, 2]),
        molecule_label=np.asarray([1, 0]),
    )
    spectra = {
        0: (np.asarray([100.0, 200.0]), np.asarray([1.0, 1.0])),
        1: (np.asarray([100.0, 200.0]), np.asarray([1.0, 1.0])),
        2: (np.asarray([100.0, 300.0]), np.asarray([1.0, 1.0])),
    }
    dreams = GraphScores(
        pair=np.asarray([0.8, 0.9], dtype=np.float32),
        molecule=np.asarray([0.8, 0.9], dtype=np.float32),
    )
    fused, audit = build_rrf_graph_scores(
        graph, dreams, np.asarray([0]), lambda row: spectra[row], progress_every=0,
    )
    assert fused.molecule[0] > fused.molecule[1]
    assert np.array_equal(fused.pair, fused.molecule)
    assert audit.spectrum_comparisons == 2
    assert audit.excluded_self_edges == 0


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_stage1_rrf_confirmation] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
