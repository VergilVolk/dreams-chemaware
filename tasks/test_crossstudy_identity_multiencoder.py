#!/usr/bin/env python3
"""Small numerical contracts for complete identity-candidate scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile

from massspecgym_embedding_readout_core import (
    ARM_PROJECTED, ARM_SCALAR, REQUIRED_ENCODERS, FrozenEmbeddingReadout,
    ReadoutSettings,
)
from score_crossstudy_identity_multiencoder import (
    apply_unified_model, frozen_p2b_block, matched_peaks, minmax,
)


def main() -> None:
    assert np.allclose(minmax(np.asarray([2.0, 4.0, 3.0])), [0.0, 1.0, 0.5])
    assert np.allclose(minmax(np.asarray([7.0, 7.0])), [0.0, 0.0])
    left = np.asarray([[100.000, 1.0], [150.000, 2.0], [200.000, 3.0]])
    right = np.asarray([[99.991, 1.0], [150.019, 2.0], [175.000, 3.0]])
    assert matched_peaks(left, right, 0.02) == 2
    p2b = frozen_p2b_block(
        np.asarray([0.7, 0.6]), np.asarray([0.2, 0.9]),
        np.asarray([0.3, 0.1]), np.asarray([0.1, 0.8]),
    )
    assert p2b.shape == (2,) and np.isfinite(p2b).all()

    # Deployment uses the same candidate features and preserves singleton
    # queries without labels or response outcomes.
    pairs = pd.DataFrame({
        "query_id": ["q1", "q1", "q2"], "candidate_id": ["c1", "c2", "c3"],
        "WSE_score": [.8, .2, .5], "P2b_score": [.7, .3, .5],
        "noise_v1_score": [.9, .1, .6],
        "chemaware_stage1_score": [.8, .2, .4],
        "chemaware_phasea_score": [.75, .25, .45],
        "noise_chemaware_fullmsg_score": [.85, .15, .55],
    })
    settings = ReadoutSettings(
        scalar_features=30, encoder_names=REQUIRED_ENCODERS,
        embedding_dimension=4, projection_dimension=2,
    )
    model = FrozenEmbeddingReadout(ARM_PROJECTED, settings)
    names = [
        "noise_v1", "noise_v1_percentile", "weighted_spectral_entropy",
        "weighted_spectral_entropy_percentile", "p2b_noise_v1_frozen",
        "p2b_noise_v1_frozen_percentile", "query_molecules", "query_edges",
        "log_query_edges", "noise_v1_top_gap", "weighted_spectral_entropy_top_gap",
        "p2b_noise_v1_frozen_top_gap", "noise_v1_weighted_spectral_entropy_top1_agree",
        "noise_v1_p2b_noise_v1_frozen_top1_agree",
        "weighted_spectral_entropy_p2b_noise_v1_frozen_top1_agree",
        "chemaware_stage1_cosine", "chemaware_stage1_percentile", "chemaware_stage1_top_gap",
        "chemaware_phasea_cosine", "chemaware_phasea_percentile", "chemaware_phasea_top_gap",
        "noise_chemaware_fullmsg_cosine", "noise_chemaware_fullmsg_percentile",
        "noise_chemaware_fullmsg_top_gap",
        "chemaware_v2_available", "chemaware_v2_correct_minus_zero",
        "chemaware_v2_correct_minus_worst_null", "chemaware_v2_null_spread",
        "chemaware_v2_delta_percentile", "chemaware_v2_delta_top_gap",
    ]
    rng = np.random.default_rng(4)
    vectors = {entity: value for entity, value in zip(
        ["q1", "q2", "c1", "c2", "c3"],
        rng.normal(size=(5, 4)).astype(np.float32), strict=True,
    )}
    caches = {name: vectors for name in settings.encoder_names}
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "model.pt"
        payload = {
            "arm": ARM_PROJECTED, "state_dict": model.state_dict(),
            "selection_class": "full_embedding_integration",
            "non_promotable_control": ARM_SCALAR,
            "representation_increment_verdict": "TEST_ONLY",
            "settings": settings.__dict__, "scalar_mean": np.zeros(30, np.float32),
            "scalar_scale": np.ones(30, np.float32), "scalar_feature_names": names,
            "encoder_names": settings.encoder_names,
        }
        torch.save(payload, path)
        scores, provenance = apply_unified_model(pairs, caches, path, torch.device("cpu"))
        payload["arm"] = ARM_SCALAR
        torch.save(payload, path)
        try:
            apply_unified_model(pairs, caches, path, torch.device("cpu"))
        except RuntimeError as error:
            assert "full-embedding integration arm" in str(error)
        else:
            raise AssertionError("scalar control was accepted as unified deployment")
    assert scores.shape == (3,) and np.isfinite(scores).all()
    assert provenance["chemaware_v2"].startswith("unavailable")
    print("PASS: cross-study identity multi-encoder contracts", flush=True)


if __name__ == "__main__":
    main()
