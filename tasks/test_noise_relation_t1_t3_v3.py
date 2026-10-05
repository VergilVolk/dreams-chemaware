#!/usr/bin/env python
"""CPU tests for query-balanced exact-bridge T1/T3 V3."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noise_relation_t1_t3_v3_core import (
    combine_query_losses,
    exact_cosine_triplet_hinge,
    query_balanced_packed_batches,
    query_balanced_event_rotation,
)


def test_one_event_per_query_and_coverage_first_rotation() -> None:
    queries = np.asarray([10, 10, 10, 20, 20, 30], dtype=np.int64)
    actions = np.asarray([3, 1, 2, 8, 7, 9], dtype=np.int64)
    eligible = np.asarray([10, 20, 30, 40], dtype=np.int64)
    selections = []
    for rotation in range(3):
        selected, report = query_balanced_event_rotation(
            queries, actions, eligible, seed=3407, rotation=rotation,
        )
        assert set(selected) == {10, 20, 30}
        assert len(set(selected.values())) == 3
        assert report["one_event_per_action_query"] is True
        selections.append(selected)
    assert len({selections[i][10] for i in range(3)}) == 3
    assert len({selections[i][20] for i in range(2)}) == 2
    assert selections[0][30] == selections[1][30] == selections[2][30]
    replay, _ = query_balanced_event_rotation(
        queries, actions, eligible, seed=3407, rotation=1,
    )
    assert replay == selections[1]
    _, replay_report = query_balanced_event_rotation(
        queries, actions, eligible, seed=3407, rotation=1,
    )
    _, original_report = query_balanced_event_rotation(
        queries, actions, eligible, seed=3407, rotation=1,
    )
    assert replay_report["selected_query_event_sha256"] == original_report[
        "selected_query_event_sha256"
    ]


def test_action_multiplicity_never_changes_query_dose() -> None:
    clean = [torch.tensor(2.0), torch.tensor(4.0), torch.tensor(6.0)]
    action = {0: torch.tensor(10.0), 2: torch.tensor(14.0)}
    loss = combine_query_losses(clean, action, action_fraction=0.5)
    expected = torch.tensor(((2 + 10) / 2 + 4 + (6 + 14) / 2) / 3)
    assert torch.allclose(loss, expected)


def test_exact_triplet_reaches_anchor_positive_and_negative() -> None:
    anchor = torch.tensor([[1.0, 0.2]], requires_grad=True)
    positive = torch.tensor([[0.5, 1.0]], requires_grad=True)
    negative = torch.tensor([[0.9, 0.1]], requires_grad=True)
    loss = exact_cosine_triplet_hinge(anchor, positive, negative).mean()
    assert float(loss.detach()) > 0
    loss.backward()
    for tensor in (anchor, positive, negative):
        assert tensor.grad is not None
        assert float(torch.linalg.vector_norm(tensor.grad)) > 0


def test_targeted_and_control_preoptimizer_gradients_are_distinct() -> None:
    targeted = torch.tensor([[1.0, 0.2, 0.1]], requires_grad=True)
    control = torch.tensor([[0.2, 1.0, 0.1]], requires_grad=True)
    positive = torch.tensor([[0.4, 0.6, 0.3]])
    negative = torch.tensor([[0.8, 0.1, 0.2]])
    targeted_loss = exact_cosine_triplet_hinge(
        targeted, positive, negative, margin=2.0,
    ).mean()
    control_loss = exact_cosine_triplet_hinge(
        control, positive, negative, margin=2.0,
    ).mean()
    targeted_gradient = torch.autograd.grad(targeted_loss, targeted)[0]
    control_gradient = torch.autograd.grad(control_loss, control)[0]
    assert float(torch.linalg.vector_norm(targeted_gradient)) > 0
    assert float(torch.linalg.vector_norm(control_gradient)) > 0
    cosine = torch.nn.functional.cosine_similarity(
        targeted_gradient.flatten(), control_gradient.flatten(), dim=0,
    )
    assert float(cosine) < 0.999


def test_targeted_and_control_share_event_selection() -> None:
    queries = np.asarray([1, 1, 2, 2], dtype=np.int64)
    actions = np.asarray([11, 12, 21, 22], dtype=np.int64)
    eligible = np.asarray([1, 2], dtype=np.int64)
    targeted, _ = query_balanced_event_rotation(
        queries, actions, eligible, seed=99, rotation=0,
    )
    control, _ = query_balanced_event_rotation(
        queries, actions, eligible, seed=99, rotation=0,
    )
    assert targeted == control


def test_packer_counts_each_query_once_with_exact_bridge_overhead() -> None:
    corpus = {
        "query_index": np.asarray([10, 20], dtype=np.int64),
        "query_row": np.asarray([100, 200], dtype=np.int64),
        "action_ptr": np.asarray([0, 0, 0], dtype=np.int64),
        "action_index": np.asarray([], dtype=np.int64),
        "molecule_ptr": np.asarray([0, 2, 4], dtype=np.int64),
        "reference_ptr": np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        "reference_row": np.asarray([101, 102, 201, 202], dtype=np.int64),
    }
    events = {
        "query": np.asarray([10], dtype=np.int64),
        "action_index": np.asarray([7], dtype=np.int64),
        "positive_row": np.asarray([103], dtype=np.int64),
        "negative_row": np.asarray([104], dtype=np.int64),
    }
    batches, report = query_balanced_packed_batches(
        corpus, events, {10: 0}, seed=3,
        maximum_spectra=16, maximum_queries=4,
    )
    assert sorted(value for batch in batches for value in batch) == [0, 1]
    assert report["queries"] == 2
    assert report["action_queries"] == 1
    assert report["optimizer_steps"] == 1
    assert report["action_multiplicity_changes_optimizer_dose"] is False


def test_fold1_summary_requires_warm_and_control_improvement() -> None:
    before = {
        "queries": 10, "recall_at_1_strict": 0.8,
        "mean_reciprocal_rank": 0.85, "mean_margin": 0.1,
        "median_margin": 0.1,
    }
    def report(arm: str, recall: float, mrr: float) -> dict:
        after = {**before, "recall_at_1_strict": recall, "mean_reciprocal_rank": mrr}
        return {
            "status": "noise_relation_t1_t3_v3_training_complete",
            "arm": arm, "seed": 3407, "rotation": 0,
            "selection": {"selected_event_count": 2},
            "schedule": {
                "queries": 10, "action_queries": 2, "optimizer_steps": 3,
                "one_selected_action_per_action_query": True,
            },
            "fold1_diagnostics": {
                "before": before, "after": after,
                "recall1_delta": recall - 0.8,
                "mrr_delta": mrr - 0.85,
                "margin_delta": 0.0,
            },
        }
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        targeted = root / "targeted.json"
        control = root / "control.json"
        output = root / "summary.json"
        targeted.write_text(json.dumps(report("targeted", 0.84, 0.88)), encoding="utf-8")
        control.write_text(json.dumps(report("control", 0.81, 0.86)), encoding="utf-8")
        script = Path(__file__).with_name("summarize_noise_relation_t1_t3_v3_fold1.py")
        subprocess.run([
            sys.executable, str(script), "--targeted-report", str(targeted),
            "--control-report", str(control), "--output", str(output),
        ], check=True, capture_output=True, text=True)
        summary = json.loads(output.read_text(encoding="utf-8"))
        assert summary["proceed_to_outer_held"] is True
        assert summary["deltas"]["targeted_minus_control_recall1_pp"] > 0


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_relation_t1_t3_v3] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
