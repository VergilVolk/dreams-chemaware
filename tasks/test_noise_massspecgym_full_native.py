#!/usr/bin/env python
"""CPU-only contracts for the full-MassSpecGym native Noise route."""
from __future__ import annotations

from pathlib import Path
import tempfile
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from train_noise_massspecgym_full_native import native_dataset

from noise_massspecgym_full_triplet_core import (
    diverse_by_similarity,
    nearest_different_identity_rows,
    query_disjoint_batches,
    query_rotation_positions,
    strict_ppm_rows,
)


def test_difficulty_selection_is_not_hardest_only() -> None:
    rows = np.arange(10, dtype=np.int64)
    scores = np.linspace(0.1, 1.0, 10)
    selected_negative = diverse_by_similarity(rows, scores, 4, hard_first=True)
    selected_positive = diverse_by_similarity(rows, scores, 4, hard_first=False)
    assert 9 in selected_negative and 0 in selected_negative
    assert 0 in selected_positive and 9 in selected_positive
    assert len(selected_negative) == len(np.unique(selected_negative)) == 4


def test_mass_window_and_nearest_fallback_are_real_and_identity_safe() -> None:
    rows = np.arange(6, dtype=np.int64)
    mz = np.asarray([99.0, 99.9995, 100.0, 100.0008, 101.0, 102.0])
    identities = np.asarray(["A", "B", "A", "C", "D", "E"])
    strict = strict_ppm_rows(rows, mz, 100.0, 10.0)
    assert np.array_equal(strict, np.asarray([1, 2, 3]))
    nearest = nearest_different_identity_rows(
        rows, mz, identities, 100.0, "A", 3,
    )
    assert len(nearest) == 3
    assert all(identities[row] != "A" for row in nearest)
    assert set(nearest[:2]) == {1, 3}


def test_query_rotation_is_equal_dose_and_event_complete_over_cycles() -> None:
    queries = np.asarray([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    selections = [query_rotation_positions(queries, epoch, 3407) for epoch in range(4)]
    for selected in selections:
        assert np.array_equal(np.sort(queries[selected]), np.asarray([0, 1, 2]))
        batches = query_disjoint_batches(selected, queries, 4, 3407, 0)
        flattened = [position for batch in batches for position in batch]
        assert sorted(flattened) == sorted(map(int, selected))
        for batch in batches:
            assert len(set(map(int, queries[batch]))) == len(batch)
    for query in np.unique(queries):
        members = set(map(int, np.flatnonzero(queries == query)))
        observed = {
            int(selected[np.flatnonzero(queries[selected] == query)[0]])
            for selected in selections[:len(members)]
        }
        assert observed == members


def test_implementation_uses_all_queries_native_kernel_and_exact_actions() -> None:
    builder = (ROOT / "tasks" / "build_noise_massspecgym_full_triplets.py").read_text(
        encoding="utf-8"
    )
    trainer = (ROOT / "tasks" / "train_noise_massspecgym_full_native.py").read_text(
        encoding="utf-8"
    )
    assert "for query in range(graph.n_queries)" in builder
    assert 'SEVERITIES = ("easy", "medium", "hard")' in builder
    assert "stage1_exact_actions" in builder
    assert "lost a Stage-1 action triplet" in builder
    assert "MassSpecGym triplet library lost a corrected-graph query" in builder
    assert "ContrastiveSpectraDataset" in trainer
    assert "ContrastiveHead" in trainer
    assert "construct_native_model" in trainer
    assert '"max_epochs": 1' in trainer
    assert "custom_loss\": False" in trainer
    assert "query_rotation_positions" in trainer
    assert "every corrected MassSpecGym query exactly once" in trainer


def test_combined_registry_reaches_unmodified_native_dataset() -> None:
    tensor = np.zeros((101, 2), dtype=np.float32)
    tensor[0] = [100.0, 1.1]
    tensor[1:9, 0] = np.linspace(10.0, 80.0, 8)
    tensor[1:9, 1] = np.linspace(0.2, 1.0, 8)
    raw = np.zeros((2, 12), dtype=np.float32)
    raw[0, :8] = tensor[1:9, 0]
    raw[1, :8] = tensor[1:9, 1]
    pool = {
        "registry_kind": np.asarray([0, 0, 1, 2], dtype=np.int8),
        "registry_source_index": np.asarray([0, 1, 0, 0], dtype=np.int64),
        "anchor_idx": np.asarray([2, 3], dtype=np.int64),
        "positive_ptr": np.asarray([0, 1, 2], dtype=np.int64),
        "positive_idx": np.asarray([0, 0], dtype=np.int64),
        "negative_ptr": np.asarray([0, 1, 2], dtype=np.int64),
        "negative_idx": np.asarray([1, 1], dtype=np.int64),
        "event_query": np.asarray([0, 0], dtype=np.int64),
        "event_kind": np.asarray([0, 1], dtype=np.int8),
        "event_action_index": np.asarray([-1, 0], dtype=np.int64),
    }
    preprocessor = SpectrumPreprocessor(
        dformat=DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.hdf5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("spectrum", data=np.stack((raw, raw)))
            handle.create_dataset("precursor_mz", data=np.asarray([100.0, 100.0]))
        dataset, event_indices, report = native_dataset(
            pool, path, tensor[None, ...], tensor[None, ...], preprocessor,
        )
        assert len(event_indices) == 2
        item = dataset[int(event_indices[0])]
        assert item["spec"].shape == (101, 2)
        assert item["pos_specs"].shape == (1, 101, 2)
        assert item["neg_specs"].shape == (1, 101, 2)
        assert report["native_preprocessor_max_abs_error"] <= 1e-6


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_massspecgym_full_native] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
