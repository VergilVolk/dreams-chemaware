#!/usr/bin/env python
"""Small deterministic unit tests for BioAware v2 feature-graph primitives."""
from __future__ import annotations

import numpy as np
import pandas as pd

from build_mtbls13729_bioaware_v2_feature_graph import (
    aggregate_feature_embeddings,
    map_ms2_to_features,
    paired_spearman,
)
from evaluate_bioaware_v2_two_layer import attach_two_layer_support


def main() -> None:
    manifest = pd.DataFrame(
        {
            "file_name": ["S1", "S1", "S2"],
            "precursor_mz": [100.0002, 200.0001, 100.0001],
            "RT": [1.0, 2.0, 1.0],
        }
    )
    targets = pd.DataFrame(
        {
            "feature_id": [10, 20],
            "mz": [100.0, 200.0],
            "rt_sec": [60.0, 120.0],
        }
    )
    mapping = map_ms2_to_features(manifest, targets, ppm=10.0, rt_sec=5.0)
    assert mapping["feature_id"].tolist() == [10, 20, 10]
    embeddings = np.asarray([[1, 0], [0, 1], [0, 2]], dtype=np.float32)
    feature_embedding, counts = aggregate_feature_embeddings(embeddings, mapping)
    assert counts["feature_id"].tolist() == [10, 20]
    # Equal sample weighting for feature 10: S1=[1,0], S2=[0,1].
    assert np.allclose(feature_embedding[0], [2**-0.5, 2**-0.5], atol=1e-6)
    rho, joint, jaccard = paired_spearman(
        np.asarray([1.0, 2.0, np.nan, 4.0]),
        np.asarray([2.0, 4.0, 3.0, 8.0]),
        minimum_joint=3,
    )
    assert joint == 3 and np.isclose(rho, 1.0) and np.isclose(jaccard, 0.75)
    candidates = pd.DataFrame(
        {
            "query_id": ["neg_rp:1", "neg_rp:1"],
            "candidate_id": ["TRUTH", "WRONG"],
            "spectral_score": [0.90, 0.89],
            "truth_candidate_id": ["TRUTH", "TRUTH"],
        }
    )
    paths = pd.DataFrame(
        {
            "candidate_id": ["WRONG", "WRONG"],
            "seed_compound_id": ["SEED", "TRUTH"],
            "seed_query_id": ["neg_rp:2", "neg_rp:3"],
            "reaction_id": ["R1", "R2"],
            "contribution": [0.8, 0.9],
        }
    )
    edges = pd.DataFrame(
        {
            "query_id": ["neg_rp:1", "neg_rp:1"],
            "seed_query_id": ["neg_rp:2", "neg_rp:3"],
            "experimental_similarity": [0.5, 0.9],
            "dual_data_support": [True, True],
        }
    )
    supported, explanations = attach_two_layer_support(
        candidates, paths, edges, truth_col="truth_candidate_id"
    )
    wrong = supported[supported["candidate_id"] == "WRONG"].iloc[0]
    # Truth-identity seed is excluded; remaining two-layer support is 0.8*0.5.
    assert np.isclose(wrong.network_support, 0.4)
    assert int(wrong.network_path_count) == 1 and len(explanations) == 1
    print("[test_mtbls13729_bioaware_v2_feature_graph] PASS", flush=True)


if __name__ == "__main__":
    main()
