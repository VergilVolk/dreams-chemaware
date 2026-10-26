#!/usr/bin/env python
"""Contract checks for the strict unified Additive refit matrix."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from enveda_unified_additive_ablation_core import (
    FULL,
    NO_CHEMAWARE_V2,
    NO_LOCAL_RERANKERS,
    NO_CHEMAWARE_REPRESENTATIONS,
    OFFICIAL_WSE,
    RELIABILITY_NAMES,
    build_variant_features,
)
from enveda_unified_candidate_integrator_core import query_constant_context_names
from massspecgym_embedding_readout_core import build_independent_scalar_features


def main() -> None:
    query_ptr = np.asarray([0, 3, 6], dtype=np.int64)
    graph = SimpleNamespace(
        molecule_ptr=np.arange(7, dtype=np.int64),
        query_ptr=query_ptr,
        molecule_label=np.asarray([1, 0, 0, 0, 1, 0], dtype=np.int8),
    )
    noise = np.asarray([0.9, 0.4, 0.1, 0.2, 0.8, 0.4], dtype=np.float32)
    wse = np.asarray([0.8, 0.7, 0.2, 0.7, 0.4, 0.1], dtype=np.float32)
    p2b = np.asarray([0.6, 0.9, 0.1, 0.1, 0.7, 0.5], dtype=np.float32)
    msg = np.asarray([0.3, 0.2, 0.95, 0.8, 0.4, 0.2], dtype=np.float32)
    stage1 = np.asarray([0.7, 0.1, 0.8, 0.2, 0.4, 0.9], dtype=np.float32)
    phasea = np.asarray([0.4, 0.2, 0.7, 0.3, 0.9, 0.1], dtype=np.float32)
    fullmsg = np.asarray([0.1, 0.8, 0.2, 0.9, 0.3, 0.2], dtype=np.float32)
    valid = np.asarray([True, True, False, True, False, True])
    chemical = {
        "chem_valid": valid,
        "candidate_utility_correct": np.asarray([.7, .4, 0, .8, 0, .2]),
        "candidate_utility_zero": np.asarray([.2, .3, 0, .1, 0, .1]),
        "candidate_utility_reversed": np.asarray([.1, .2, 0, .4, 0, .3]),
        "candidate_utility_rotated": np.asarray([.3, .1, 0, .2, 0, .1]),
    }
    scalar54, names54 = build_independent_scalar_features(
        {"v1_cosine": noise, "weighted_entropy": wse, "p2b_fused": p2b},
        graph,
        {
            "noise_v1": noise, "chemaware_stage1": stage1,
            "chemaware_phasea": phasea, "noise_chemaware_fullmsg": fullmsg,
        },
        "noise_v1", chemical,
        auxiliary_scores={"noise_msg_fusion_stack_v1": msg},
        include_reliability_context=True,
    )
    excluded = set(query_constant_context_names(names54))
    keep = [index for index, name in enumerate(names54) if name not in excluded]
    scalar23 = scalar54[:, keep]
    names23 = tuple(names54[index] for index in keep)
    assert scalar23.shape == (6, 23)

    scores = {
        "official_dreams": np.asarray([.5, .4, .3, .2, .3, .9]),
        "noise_v1": noise,
        "weighted_spectral_entropy": wse,
        "p2b_noise_v1_frozen": p2b,
        "noise_msg_fusion_stack_v1": msg,
        "chemaware_stage1": stage1,
        "chemaware_phasea": phasea,
        "noise_chemaware_fullmsg": fullmsg,
    }
    blocks = {
        variant: build_variant_features(scores, query_ptr, scalar23, names23, variant)
        for variant in (
            FULL, OFFICIAL_WSE, NO_CHEMAWARE_REPRESENTATIONS,
            NO_CHEMAWARE_V2, NO_LOCAL_RERANKERS,
        )
    }
    full, full_names = blocks[FULL]
    assert full_names == names23
    assert np.max(np.abs(full - scalar23)) <= 1e-7
    assert {name: block[0].shape[1] for name, block in blocks.items()} == {
        FULL: 23,
        OFFICIAL_WSE: 7,
        NO_CHEMAWARE_REPRESENTATIONS: 17,
        NO_CHEMAWARE_V2: 17,
        NO_LOCAL_RERANKERS: 19,
    }
    assert not any(name.startswith("chemaware_v2_") for name in blocks[NO_CHEMAWARE_V2][1])
    assert "p2b_noise_v1_frozen" not in blocks[NO_LOCAL_RERANKERS][1]
    assert "noise_msg_fusion_stack_v1" not in blocks[NO_LOCAL_RERANKERS][1]
    assert "chemaware_stage1_cosine" not in blocks[NO_CHEMAWARE_REPRESENTATIONS][1]
    assert blocks[OFFICIAL_WSE][1][:4] == (
        "official_dreams", "official_dreams_percentile",
        "weighted_spectral_entropy", "weighted_spectral_entropy_percentile",
    )

    # A strict ablation must rebuild these summaries from the retained experts.
    full_rel = full[:, [full_names.index(name) for name in RELIABILITY_NAMES]]
    no_rep, no_rep_names = blocks[NO_CHEMAWARE_REPRESENTATIONS]
    no_rep_rel = no_rep[:, [no_rep_names.index(name) for name in RELIABILITY_NAMES]]
    assert np.max(np.abs(full_rel - no_rep_rel)) > 0

    print("PASS: unified Additive strict-refit ablation contracts")


if __name__ == "__main__":
    main()
