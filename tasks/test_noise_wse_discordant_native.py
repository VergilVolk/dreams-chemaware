#!/usr/bin/env python
"""CPU-only scientific contracts for WSE-discordant native triplets."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noise_wse_discordant_triplet_core import (
    candidate_molecule_max_rows,
    choose_v1_preservation_triplet,
    choose_wse_supported_triplet,
)


def test_wse_selector_prefers_current_misranking_and_never_uses_wse_as_target() -> None:
    choice = choose_wse_supported_triplet(
        np.asarray([10, 11]), np.asarray([0.60, 0.35]), np.asarray([0.80, 0.75]),
        np.asarray([20, 21]), np.asarray([0.50, 0.45]), np.asarray([0.30, 0.70]),
        native_margin=0.1,
    )
    assert choice is not None
    assert choice.positive_row == 11 and choice.negative_row == 20
    assert choice.v1_margin < 0 and choice.wse_margin > 0
    assert choice.stratum == "misranked"


def test_wse_selector_rejects_inactive_or_wrong_direction_relations() -> None:
    assert choose_wse_supported_triplet(
        np.asarray([1]), np.asarray([0.9]), np.asarray([0.8]),
        np.asarray([2]), np.asarray([0.2]), np.asarray([0.1]),
    ) is None
    assert choose_wse_supported_triplet(
        np.asarray([1]), np.asarray([0.4]), np.asarray([0.2]),
        np.asarray([2]), np.asarray([0.5]), np.asarray([0.7]),
    ) is None


def test_preservation_is_the_largest_available_v1_margin() -> None:
    positive, negative, margin = choose_v1_preservation_triplet(
        np.asarray([1, 2]), np.asarray([0.4, 0.8]),
        np.asarray([3, 4]), np.asarray([0.7, 0.1]),
    )
    assert (positive, negative) == (2, 4)
    assert abs(margin - 0.7) < 1e-12


def test_candidate_rows_are_selected_at_molecule_max() -> None:
    rows, scores = candidate_molecule_max_rows(
        np.asarray([10, 11, 20, 21, 22, 30]),
        np.asarray([0, 2, 5, 6]),
        np.asarray([0.2, 0.8, 0.1, 0.7, 0.5, 0.4]),
    )
    assert np.array_equal(rows, np.asarray([11, 21, 30]))
    assert np.allclose(scores, np.asarray([0.8, 0.7, 0.4]))


def test_route_is_native_triplet_mining_not_distillation() -> None:
    builder = (ROOT / "tasks" / "build_noise_wse_discordant_native_triplets.py").read_text(
        encoding="utf-8"
    )
    trainer = (ROOT / "tasks" / "train_noise_wse_discordant_native.py").read_text(
        encoding="utf-8"
    )
    assert "weighted_entropy_similarity" in builder
    assert "choose_wse_supported_triplet" in builder
    assert "for query in range(graph.n_queries)" in builder
    assert "ContrastiveSpectraDataset" in trainer
    assert "ContrastiveHead" in trainer
    assert "triplet_loss_margin" in trainer
    for forbidden in ("distill", "teacher_embedding", "kl_div", "mse_loss"):
        assert forbidden not in trainer.lower()


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_wse_discordant_native] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
