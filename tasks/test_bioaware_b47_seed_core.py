#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from bioaware_b47_seed_core import (
    PRIMARY_POLICY, STRICT_POLICY, attach_absolute_gates, feature_consensus,
    query_summaries, select_sample_seeds,
)


def test_query_tie_is_not_seed() -> None:
    frame = pd.DataFrame([
        {"query_id": "q1", "candidate_id": "A", "spectral_score": 0.9},
        {"query_id": "q1", "candidate_id": "B", "spectral_score": 0.9},
    ])
    summary = query_summaries(frame)
    assert not bool(summary.loc[0, "unique_top1"])
    assert summary.loc[0, "top_candidate_id"] == ""


def test_consensus_and_sample_identity_collapse() -> None:
    rows = []
    for query_id, sample, feature, candidate, score, second in (
        ("q1", "s1", "f1", "A", 0.92, 0.80),
        ("q2", "s2", "f1", "A", 0.91, 0.79),
        ("q3", "s1", "f2", "A", 0.88, 0.70),
        ("q4", "s2", "f2", "B", 0.88, 0.70),
    ):
        rows.extend([
            {"query_id": query_id, "candidate_id": candidate, "spectral_score": score},
            {"query_id": query_id, "candidate_id": "Z" + query_id, "spectral_score": second},
        ])
    summary = query_summaries(pd.DataFrame(rows))
    metadata = pd.DataFrame([
        {"query_id": "q1", "study": "X", "sample": "s1", "feature_id": "f1"},
        {"query_id": "q2", "study": "X", "sample": "s2", "feature_id": "f1"},
        {"query_id": "q3", "study": "X", "sample": "s1", "feature_id": "f2"},
        {"query_id": "q4", "study": "X", "sample": "s2", "feature_id": "f2"},
    ])
    summary = metadata.merge(summary, on="query_id", validate="one_to_one")
    policies = (PRIMARY_POLICY, STRICT_POLICY)
    summary = attach_absolute_gates(summary, policies)
    consensus = feature_consensus(summary, policies)
    f1 = consensus[consensus.feature_id.eq("f1")].iloc[0]
    f2 = consensus[consensus.feature_id.eq("f2")].iloc[0]
    assert bool(f1.primary_feature_gate)
    assert not bool(f1.strict_feature_gate)  # only two samples, strict needs three
    assert not bool(f2.primary_feature_gate)  # tied modal identity
    seeds, audit = select_sample_seeds(
        summary, consensus, PRIMARY_POLICY, {"A": 2, "B": 3}, set(), 250
    )
    assert len(seeds) == 2 and set(seeds.seed_compound_id) == {"A"}
    assert audit["rows_after_sample_candidate_collapse"] == 2


def test_gpu_bridge_copies_read_only_memmaps() -> None:
    source = Path(__file__).with_name("build_bioaware_b47_truthblind_seeds.py").read_text(
        encoding="utf-8"
    )
    assert "np.array(query_embedding, dtype=np.float32, copy=True, order=\"C\")" in source
    assert "np.array(reference_embedding, dtype=np.float32, copy=True, order=\"C\")" in source
    assert "torch.from_numpy(np.asarray(query_embedding))" not in source
    assert "torch.from_numpy(np.asarray(reference_embedding))" not in source


if __name__ == "__main__":
    test_query_tie_is_not_seed()
    test_consensus_and_sample_identity_collapse()
    test_gpu_bridge_copies_read_only_memmaps()
    print("[test_bioaware_b47_seed_core] PASS")
