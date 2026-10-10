#!/usr/bin/env python
"""Contracts for applying frozen official-geometry V2 actions to another scorer."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import export_chemaware_v2_grand_fusion_scores as exporter


ROOT = Path(__file__).resolve().parents[1]


def test_action_transfer_preserves_selector_and_tracks_effect() -> None:
    body = {
        "query_row": np.asarray([10, 11, 12], dtype=np.int64),
        "query_ptr": np.asarray([0, 2, 4, 6], dtype=np.int64),
        "molecule_ptr": np.arange(7, dtype=np.int64),
        "pair_candidate_row": np.asarray([20, 21, 22, 23, 24, 25], dtype=np.int64),
    }
    official_global = np.asarray(
        [[0.90, 0.80], [0.80, 0.90], [0.70, 0.60]], dtype=np.float32,
    )
    deployment = np.asarray(
        [0.90, 0.10, 0.10, 0.90, 0.80, 0.20], dtype=np.float32,
    )

    def fake_score(queries, *_args, **_kwargs):
        global_values = np.empty(len(queries), dtype=object)
        pointers = np.empty(len(queries), dtype=object)
        for out, query in enumerate(map(int, queries)):
            global_values[out] = official_global[query]
            pointers[out] = np.asarray([0, 1, 2], dtype=np.int32)
        return {"global": global_values, "reference_ptr": pointers}

    def fake_predict(_policy, scored, _controls):
        query = np.asarray(scored["query"] if "query" in scored else [0, 1, 2])
        selected = np.asarray([1, 1, 0], dtype=np.int16)[query]
        abstained = np.asarray([False, False, True])[query]
        return {"selected_candidate": selected, "abstained": abstained}

    old_score = exporter.score_queries_truthblind
    old_predict = exporter.predict_truthblind_policy
    exporter.score_queries_truthblind = fake_score
    exporter.predict_truthblind_policy = fake_predict
    try:
        scores, counts, ledger = exporter._promote_frozen_actions(
            body=body,
            policy={"control_rule_keys": []},
            cache=object(),
            official=np.empty((0, 0), dtype=np.float32),
            row_position={},
            variants=(),
            batch_queries=3,
            deployment_base=deployment,
        )
    finally:
        exporter.score_queries_truthblind = old_score
        exporter.predict_truthblind_policy = old_predict

    assert counts == {
        "queries": 3,
        "selected": 2,
        "abstained": 1,
        "changed_deployment_top1": 1,
        "already_deployment_top1": 1,
    }
    assert ledger["official_top_candidate"].tolist() == [0, 1, 0]
    assert ledger["deployment_top_candidate"].tolist() == [0, 1, 0]
    assert ledger["selected_candidate"].tolist() == [1, 1, -1]
    assert ledger["changed_deployment_top1"].tolist() == [True, False, False]
    assert ledger["output_top_candidate"].tolist() == [1, 1, 0]
    assert scores[1] > scores[0]
    assert np.array_equal(scores[2:4], deployment[2:4])
    assert np.array_equal(scores[4:], deployment[4:])


def test_sbatch_is_gnps_only_and_fail_closed() -> None:
    text = (ROOT / "tasks/run_gnps_noise_chemaware_v2_transfer.sbatch").read_text(
        encoding="utf-8",
    )
    lowered = text.lower()
    assert "massspecgym" not in lowered
    directives = [line.strip() for line in text.splitlines() if line.startswith("#SBATCH")]
    assert directives.count("#SBATCH --partition=gpu") == 1
    assert directives.count("#SBATCH --gpus=1") == 1
    assert not any("--cpus-per-task" in line for line in directives)
    assert not any("--mem" in line for line in directives)
    assert 'export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"' in text
    assert "audit_chemaware_orthogonal_rule_residual_policy.py" in text
    assert "recovered_frozen_v2_policy" in text
    assert "--contrast-representation symmetric_summary" in text
    assert "--selection-control-mode deployment_safe" in text
    assert "--deployment-base-score-cache \"$OUT/noise_v1_cache\"" in text
    assert '"$POLICY_DIR/truthblind_policy.joblib"' in text
    assert '"$POLICY_DIR/report.json"' in text
    assert "evaluate_gnps_gold_silver_10ppm_pair_scores.py" in text
    assert "--bootstrap-resamples 10000" in text


def main() -> None:
    test_action_transfer_preserves_selector_and_tracks_effect()
    test_sbatch_is_gnps_only_and_fail_closed()
    print("PASS: ChemAware V2 frozen action transfer to Noise contracts")


if __name__ == "__main__":
    main()
