"""CPU-only regression tests for failures that previously surfaced after GPU work."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from chemaware_direct_training_core import validate_action_role_matrix
from train_chemaware_iceberg_direct_shared import (
    validate_clean_data_and_embedding_cache,
    validate_cli_contract,
    validate_teacher_arrays,
)


def base_args() -> SimpleNamespace:
    return SimpleNamespace(
        folds=5, inner_fold=3, outer_fold=4,
        batch_queries=4, eval_batch_size=32, prefix_cache_batch_size=8,
        n_highest_peaks=100, unfreeze_blocks=1, bootstrap_draws=10_000,
        torch_threads=8, backbone_lr=2e-6, head_lr=1e-5,
        weight_decay=1e-4, temperature=0.1, grad_clip=5.0,
        margin_floor_slack=0.005, max_action_identities=512,
        max_safety_identities=512, max_eval_identities=2000,
    )


def expect_failure(call, text: str) -> None:
    try:
        call()
    except (ValueError, RuntimeError) as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f"expected failure containing {text!r}")


def main() -> None:
    validate_cli_contract(base_args())
    bad_fold = base_args(); bad_fold.inner_fold = 5
    expect_failure(lambda: validate_cli_contract(bad_fold), "must be in [0, folds)")
    bad_batch = base_args(); bad_batch.batch_queries = 0
    expect_failure(lambda: validate_cli_contract(bad_batch), "positive integer")
    bad_temperature = base_args(); bad_temperature.temperature = 0
    expect_failure(lambda: validate_cli_contract(bad_temperature), "temperature")

    action_folds = np.asarray([0, 1, 2, 3, 4], dtype=np.int8)
    action_roles = np.asarray([
        [0, 1, -1, -1, -1],
        [2, 3, 1, -1, -1],
    ], dtype=np.int8)
    validate_action_role_matrix(
        action_roles, 1, action_folds, (0, 1), 2, (0, 1, 2, 3, 4),
    )
    broken_roles = action_roles.copy(); broken_roles[1, 3] = 0
    expect_failure(
        lambda: validate_action_role_matrix(
            broken_roles, 1, action_folds, (0, 1), 2, (0, 1, 2, 3, 4),
        ),
        "sentinel",
    )

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        prediction_path = root / "prediction.npy"
        np.save(prediction_path, np.ones((4, 3), dtype=np.float16))
        score_path = root / "scores.npz"
        np.savez(
            score_path,
            official_rank=np.asarray([1, 2]), official_margin=np.asarray([0.1, -0.1]),
            correct_rank=np.asarray([1, 1]), correct_margin=np.asarray([0.2, 0.1]),
            correct_score=np.ones(4), candidate_swapped_score=np.ones(4),
            peak_permuted_score=np.ones(4), formula=np.asarray(["F0", "F1"]),
        )
        graph = SimpleNamespace(
            n_queries=2,
            query_ptr=np.asarray([0, 2, 4]),
            query_formula=np.asarray(["F0", "F1"]),
        )
        selected = np.asarray([0, 1], dtype=np.int64)
        query_ptr = np.asarray([0, 2, 4], dtype=np.int64)
        with np.load(score_path, allow_pickle=True) as scores:
            validate_teacher_arrays(graph, selected, query_ptr, prediction_path, scores)
        with np.load(score_path, allow_pickle=True) as scores:
            expect_failure(
                lambda: validate_teacher_arrays(
                    graph, selected, np.asarray([0, 3, 5]), prediction_path, scores,
                ),
                "candidate counts",
            )

        hdf5_path = root / "spectra.hdf5"
        with h5py.File(hdf5_path, "w") as handle:
            handle.create_dataset("spectrum", data=np.zeros((3, 2, 2), dtype=np.float32))
            handle.create_dataset("precursor_mz", data=np.ones(3, dtype=np.float32))
        cache_rows = np.asarray([0, 1, 2], dtype=np.int64)
        cache_embedding = np.zeros((3, 1024), dtype=np.float32)
        cache_embedding[np.arange(3), np.arange(3)] = 1.0
        mapped = validate_clean_data_and_embedding_cache(
            hdf5_path, np.asarray([0, 2]), cache_rows, cache_embedding,
        )
        assert set(mapped) == {0, 2}
        expect_failure(
            lambda: validate_clean_data_and_embedding_cache(
                hdf5_path, np.asarray([0, 3]), cache_rows, cache_embedding,
            ),
            "misses graph-reachable rows",
        )
        extended_rows = np.asarray([0, 1, 2, 3], dtype=np.int64)
        extended_embedding = np.zeros((4, 1024), dtype=np.float32)
        extended_embedding[np.arange(4), np.arange(4)] = 1.0
        expect_failure(
            lambda: validate_clean_data_and_embedding_cache(
                hdf5_path, np.asarray([0, 3]), extended_rows, extended_embedding,
            ),
            "outside the HDF5",
        )

    trainer = (Path(__file__).parent / "train_chemaware_iceberg_direct_shared.py").read_text(
        encoding="utf-8",
    )
    assert (
        'None if delta_transfer or near_delta is None else near_delta >= 0'
        in trainer
    )
    assert "report_text = json.dumps(report" in trainer
    assert trainer.index("report_text = json.dumps(report") < trainer.index("atomic_torch_save(checkpoint")
    assert 'args.output / "COMPLETE.json"' in trainer
    assert "cache.graphormer_bias" not in trainer
    assert "cache.graphormer_projected" in trainer
    print("PASS: late ChemAware runtime failures are covered by CPU contracts")


if __name__ == "__main__":
    main()
