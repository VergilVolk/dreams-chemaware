"""Contract tests for the corrected full-graph evaluator."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from noise_corrected_fullgraph_evaluation import (
    HELD_METRIC_EVIDENCE_SCHEMA, full_metrics, held_metric_evidence,
    official_scores, paired_outcome_table, query_table,
    score_embedding_query_subset, score_embeddings,
)


def graph():
    # q0: positive max=.9, negative=.8 -> rank 1
    # q1: positive=.3, negative=.4 -> rank 2 and near subset
    return SimpleNamespace(
        n_queries=2,
        query_ptr=np.asarray([0, 2, 4]),
        molecule_ptr=np.asarray([0, 2, 3, 4, 5]),
        molecule_label=np.asarray([1, 0, 1, 0], dtype=np.int8),
        pair_candidate_row=np.asarray([2, 3, 4, 5, 6]),
        query_row=np.asarray([0, 1]),
        query_ik14=np.asarray(["A", "B"]),
        query_formula=np.asarray(["F0", "F1"]),
        query_has_near=np.asarray([False, True]),
        features=np.asarray([[.9], [.7], [.8], [.3], [.4]], dtype=np.float32),
        dreams_column=0,
    )


def test_official_metric_contract_and_mh_pairwise_name() -> None:
    g = graph()
    scores = official_scores(g)
    report, table = full_metrics(
        g, scores, query_adduct=np.asarray(["[M+H]+", "[M+Na]+"]),
    )
    assert report["retrieval"]["recall@1"] == 0.5
    assert report["retrieval"]["recall@2"] == 1.0
    assert report["retrieval"]["mrr"] == 0.75
    assert report["near_subset"]["recall@1"] == 0.0
    assert report["massspecgym_10ppm_pooled_pairwise"]["spectrum_pairs"] == 5
    assert report["massspecgym_mh_10ppm_pooled_pairwise"]["spectrum_pairs"] == 3
    assert report["massspecgym_mh_10ppm_pooled_pairwise"]["exact_nist20_paper_replication"] is False
    assert table["rank"].tolist() == [1, 2]
    assert np.allclose(table["top1_top2_gap"], [0.1, 0.1], atol=1e-6)
    assert np.allclose(table["signed_top1_top2_gap"], [0.1, -0.1], atol=1e-6)
    assert np.isclose(report["retrieval"]["mean_signed_top1_top2_gap"], 0.0, atol=1e-6)


def test_embedding_scorer_reproduces_known_edges() -> None:
    g = graph()
    rows = np.arange(7, dtype=np.int64)
    embeddings = np.asarray([
        [1.0, 0.0], [0.0, 1.0],
        [.9, np.sqrt(1 - .9 ** 2)], [.7, np.sqrt(1 - .7 ** 2)],
        [.8, .6], [np.sqrt(1 - .3 ** 2), .3],
        [np.sqrt(1 - .4 ** 2), .4],
    ], dtype=np.float32)
    scores = score_embeddings(g, rows, embeddings, chunk_edges=2)
    assert np.allclose(scores.pair, [.9, .7, .8, .3, .4], atol=1e-6)
    assert np.allclose(scores.molecule, [.9, .8, .3, .4], atol=1e-6)

    subset = score_embedding_query_subset(
        g, rows[[1, 5, 6]], embeddings[[1, 5, 6]], np.asarray([1]),
    )
    assert np.allclose(subset.pair, [0, 0, 0, .3, .4], atol=1e-6)
    assert np.allclose(subset.molecule, [0, 0, .3, .4], atol=1e-6)
    subset_report, subset_table = full_metrics(
        g, subset, queries=np.asarray([1]),
    )
    assert subset_report["retrieval"]["queries"] == 1
    assert subset_report["retrieval"]["recall@1"] == 0.0
    assert subset_table["query_index"].tolist() == [1]


def test_paired_outcomes_count_corrections_and_introductions() -> None:
    g = graph()
    official = query_table(g, official_scores(g))
    changed = official_scores(g)
    changed.pair[:] = np.asarray([.6, .6, .8, .7, .4], dtype=np.float32)
    changed.molecule[:] = np.asarray([.6, .8, .7, .4], dtype=np.float32)
    candidate = query_table(g, changed)
    paired = paired_outcome_table(official, candidate)
    assert paired.corrected.tolist() == [False, True]
    assert paired.introduced.tolist() == [True, False]
    assert "risk_net" not in paired.columns
    assert int(paired.risk_net_lambda2.sum()) == -1


def test_held_metric_evidence_is_exact_replay_ledger() -> None:
    g = graph()
    scores = official_scores(g)
    evidence = held_metric_evidence(
        g,
        {"official": scores, "initial_E8": scores, "candidate": scores},
        np.asarray(["[M+H]+", "[M+Na]+"]),
        np.asarray([0, 1], dtype=np.int64),
    )
    assert str(evidence["schema_version"].item()) == HELD_METRIC_EVIDENCE_SCHEMA
    assert evidence["query_index"].dtype == np.int64
    assert evidence["query_index"].tolist() == [0, 1]
    assert evidence["molecule_label"].tolist() == [1, 0, 1, 0]
    assert evidence["pair_label"].tolist() == [1, 1, 0, 1, 0]
    assert evidence["pair_is_mh"].dtype == np.bool_
    assert evidence["pair_is_mh"].tolist() == [True, True, True, False, False]
    assert np.allclose(
        evidence["official_molecule_score"], [.9, .8, .3, .4], atol=1e-6,
    )
    assert np.allclose(
        evidence["candidate_pair_score"], [.9, .7, .8, .3, .4], atol=1e-6,
    )
    assert set(evidence) == {
        "schema_version", "query_index", "molecule_label", "pair_label",
        "pair_is_mh", "official_molecule_score", "official_pair_score",
        "initial_E8_molecule_score", "initial_E8_pair_score",
        "candidate_molecule_score", "candidate_pair_score",
    }


def main() -> None:
    test_official_metric_contract_and_mh_pairwise_name()
    test_embedding_scorer_reproduces_known_edges()
    test_paired_outcomes_count_corrections_and_introductions()
    test_held_metric_evidence_is_exact_replay_ledger()
    print("[test_noise_corrected_fullgraph_evaluation] PASS tests=4")


if __name__ == "__main__":
    main()
