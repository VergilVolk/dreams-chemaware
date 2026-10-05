from __future__ import annotations

import numpy as np

from train_chemaware_crossview_event_peft import (
    auc_metrics,
    candidate_reference_views,
    matched_random_events,
    numerical_replay_audit,
    retrieval_diagnostics,
)


def main() -> None:
    base = np.asarray([
        [1.0, 0.0], [0.8, 0.6], [0.7, np.sqrt(0.51)],
        [0.9, np.sqrt(0.19)], [0.4, np.sqrt(0.84)],
        [0.3, np.sqrt(0.91)],
    ], dtype=np.float32)
    graph = {
        "rows": np.asarray([100, 10, 11, 20, 21, 22]),
        "base": base,
        "query": np.asarray([0]),
        "candidate": [np.asarray([1, 2, 3, 4, 5])],
        "candidate_identity": [np.asarray(["T", "T", "N", "N", "N"])],
    }
    body = {
        "query_ptr": np.asarray([0, 2]),
        "molecule_ptr": np.asarray([0, 2, 5]),
        "pair_candidate_row": np.asarray([10, 11, 20, 21, 22]),
        "molecule_label": np.asarray([True, False]),
    }
    reference, valid, positive, baseline, score = candidate_reference_views(
        np.asarray([0]), body, graph, views=2,
    )
    assert reference.shape == (1, 2, 2) and np.all(valid)
    assert reference[0, 0].tolist() == [1, 2]
    assert reference[0, 1].tolist() == [3, 4]
    assert positive.tolist() == [0] and baseline.tolist() == [1]
    assert np.allclose(score[0, :2], [0.8, 0.9], atol=1e-6)

    diagnostics = retrieval_diagnostics(base, graph, np.asarray(["T"]))
    assert diagnostics["rank"].tolist() == [2]
    assert diagnostics["query_auc"].tolist() == [0.0]
    metrics = auc_metrics(diagnostics, np.asarray(["F"]))
    assert metrics["micro_auc"] == 0.0 and metrics["macro_formula_auc"] == 0.0

    n = 20
    baseline_rank = np.asarray([2, 3, 1] + [2] * 9 + [1] * 8)
    baseline_candidate = np.where(baseline_rank > 1, 1, 0)
    positive_candidate = np.zeros(n, dtype=np.int64)
    candidate_score = np.empty((n, 2), dtype=np.float64)
    candidate_score[:, 0] = np.where(baseline_rank > 1, 0.40, 0.60)
    candidate_score[:, 1] = 0.50
    role = np.zeros(n, dtype=np.int8); role[:3] = [1, 1, 2]
    correct_event = {
        "role": role, "active": role > 0,
        "positive": np.where(role > 0, 0, -1),
        "negative": np.where(role > 0, 1, -1),
    }
    matched, audit = matched_random_events(
        correct_event, baseline_rank, baseline_candidate, positive_candidate,
        candidate_score, np.asarray([f"I{i}" for i in range(n)]), seed=7,
    )
    assert int(np.sum(matched["role"] == 1)) == 2
    assert int(np.sum(matched["role"] == 2)) == 1
    assert not np.any(matched["active"][:3])
    assert audit["correct_teacher_identities_excluded"] is True

    panel = np.arange(6739, dtype=np.int64)
    formula = np.asarray([f"F{i // 4}" for i in panel])
    expected = np.ones(len(panel), dtype=np.int64)
    observed = expected.copy(); observed[[7, 100, 6000]] = 2
    margin = np.zeros(len(panel), dtype=np.float64)
    stable, mismatch, maximum, replay_audit = numerical_replay_audit(
        panel, formula, expected, observed, margin,
    )
    assert maximum == 7 and mismatch.tolist() == [7, 100, 6000]
    assert not np.any(stable[mismatch]) and len(replay_audit) == 3
    too_many = expected.copy(); too_many[:8] = 2
    try:
        numerical_replay_audit(panel, formula, expected, too_many, margin)
    except RuntimeError as error:
        assert "observed=8 maximum=7" in str(error)
    else:
        raise AssertionError("excessive replay drift must fail closed")
    print("PASS: ChemAware multiview, matched-random, and AUC math contracts")


if __name__ == "__main__":
    main()
