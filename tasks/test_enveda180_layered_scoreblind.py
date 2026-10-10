from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scoreblind_retrieval_graph import scoreblind_graph_from_panel


ROOT = Path(__file__).resolve().parents[1]


def test_score_graph_never_requires_truth_arrays(tmp_path):
    panel = tmp_path / "panel.npz"
    np.savez_compressed(
        panel,
        query_row=np.asarray([7, 9]),
        query_ptr=np.asarray([0, 2, 4]),
        molecule_ptr=np.asarray([0, 1, 2, 3, 4]),
        candidate_row=np.asarray([1, 2, 3, 4]),
    )
    graph = scoreblind_graph_from_panel(panel)
    assert graph.n_queries == 2
    assert graph.molecule_label.tolist() == [0, 0, 0, 0]
    assert not hasattr(graph, "query_ik14")


def test_freeze_contract_and_sbatch_are_score_blind():
    freeze = json.loads((ROOT / "tasks/enveda180_layered_freeze_contract_v1.json").read_text())
    assert freeze["freeze_status"] == "FROZEN_BEFORE_ENVEDA_SCORING"
    assert freeze["layered_algorithm"]["primary_top1_output"] == "weighted_spectral_entropy"
    assert freeze["layered_algorithm"]["chemical_evidence"].endswith("not_an_automatic_reranker")
    script = (ROOT / "tasks/run_enveda180_layered_scoreblind.sbatch").read_text()
    assert "evaluate_noise_gnps_article_benchmark.py" not in script
    assert "evaluate_frozen" not in script
    assert "method_scores_unopened.npz" in script
    assert "#SBATCH --cpus-per-task" not in script
