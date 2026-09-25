"""CPU contracts for the max-reference-aligned ChemAware curriculum."""
from __future__ import annotations

import sys
import tempfile
import gc
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from build_chemaware_max_boundary_native_triplets import (  # noqa: E402
    DREAMS_NATIVE_REPLAY,
    ERROR_CHEMICAL_MAX_BOUNDARY,
    ERROR_OFFICIAL_MAX_BOUNDARY,
    SAFE_MAX_BOUNDARY,
    FrozenEmbeddings,
    PoolWriter,
    build_current_geometry_pool,
    max_boundary,
)
from build_chemaware_action_hard_native_triplets import ACTION_HARD  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="chem_max_boundary_test_") as raw:
        root = Path(raw)
        rows = np.asarray([10, 11, 12, 13], dtype=np.int64)
        # Anchor 10 is closest to positive 11 and negative 12; negative 13 is
        # outside the 0.1 hinge and must not enter an error event.
        embeddings = np.asarray([
            [1.0, 0.0],
            [0.8, 0.6],
            [0.9, np.sqrt(0.19)],
            [0.0, 1.0],
        ], dtype=np.float32)
        np.save(root / "rows.npy", rows)
        np.save(root / "embeddings.npy", embeddings)
        cache = FrozenEmbeddings(root / "rows.npy", root / "embeddings.npy")
        boundary = max_boundary(
            cache, 10, np.asarray([11]), np.asarray([12, 13]), 0.1,
        )
        assert boundary["positive_row"] == 11
        assert boundary["negative_rows"].tolist() == [12]
        assert float(boundary["negative_hinges"][0]) > 0.0

        writer = PoolWriter()
        writer.append(10, [11], [12], 1, 2, 1, 2)
        writer.append(10, [11], [12], 1, 2, 1, 2)
        writer.append(13, [11, 12], [10], -1, -1, 0, DREAMS_NATIVE_REPLAY)
        arrays = writer.arrays()
        assert len(arrays["anchor_idx"]) == 2
        assert arrays["curriculum_role"].tolist() == [2, DREAMS_NATIVE_REPLAY]
        assert np.diff(arrays["positive_ptr"]).tolist() == [1, 2]

        # Current-geometry re-mining must discover a newly hardest molecule
        # that was absent from the frozen action bank.  It may retain one
        # active ChemAware competitor on an actually wrong query, but it must
        # emit only a safety boundary for a currently correct query.
        remine_rows = np.asarray([10, 11, 12, 13, 20, 21, 22], dtype=np.int64)
        remine_embeddings = np.asarray([
            [1.0, 0.0],
            [0.8, 0.6],
            [0.9, np.sqrt(0.19)],
            [0.95, np.sqrt(0.0975)],
            [0.0, 1.0],
            [0.1, np.sqrt(0.99)],
            [1.0, 0.0],
        ], dtype=np.float32)
        np.save(root / "remine_rows.npy", remine_rows)
        np.save(root / "remine_embeddings.npy", remine_embeddings)
        remine_cache = FrozenEmbeddings(
            root / "remine_rows.npy", root / "remine_embeddings.npy",
        )
        manifest = {
            "query_row": np.asarray([10, 20], dtype=np.int64),
            "query_ptr": np.asarray([0, 3, 5], dtype=np.int64),
            "molecule_ptr": np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64),
            "pair_candidate_row": np.asarray([11, 12, 13, 21, 22], dtype=np.int64),
            "molecule_label": np.asarray([1, 0, 0, 1, 0], dtype=bool),
        }
        base = {
            "source_query": np.asarray([0, 1], dtype=np.int64),
            "negative_candidate": np.asarray([1, 1], dtype=np.int16),
            "source_tag": np.asarray([ACTION_HARD, ACTION_HARD], dtype=np.int16),
        }
        remine_writer = PoolWriter()
        remine_report = build_current_geometry_pool(
            base, {"query": np.asarray([0, 1], dtype=np.int64)}, manifest,
            remine_cache, remine_writer, margin=0.1,
            negative_references_per_error=1,
            chemical_candidates_per_error=1,
        )
        remine_arrays = remine_writer.arrays()
        assert remine_report["official_error_queries"] == 1
        assert remine_report["official_correct_queries"] == 1
        assert remine_report["new_current_hardest_candidates"] == 1
        assert remine_arrays["curriculum_role"].tolist() == [
            ERROR_OFFICIAL_MAX_BOUNDARY,
            ERROR_CHEMICAL_MAX_BOUNDARY,
            SAFE_MAX_BOUNDARY,
        ]
        assert remine_arrays["negative_candidate"].tolist() == [2, 1, 1]
        assert remine_arrays["source_query"].tolist() == [0, 0, 1]
        # Windows keeps an mmap-backed .npy file locked until the final array
        # owner is collected.  Release it before TemporaryDirectory cleanup.
        del boundary, cache, remine_cache
        gc.collect()

    sbatch = (ROOT / "tasks/run_chemaware_max_boundary_native.sbatch").read_text(
        encoding="utf-8",
    )
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem" not in sbatch
    assert "train_chemaware_specific_replay_native.py" in sbatch
    assert "--triplet-loss-margin 0.1" in sbatch
    assert "--official-checkpoint \"$OFFICIAL\"" in sbatch
    assert "--dreams-replay-pool data/e1/e1_train_triplet_pool_10ppm.npz" in sbatch
    assert "--paired-reference stage1_base" in sbatch
    remine = (ROOT / "tasks/run_chemaware_max_boundary_remine.sbatch").read_text(
        encoding="utf-8",
    )
    assert "#SBATCH --gpus=1" in remine
    assert "#SBATCH --mem" not in remine
    assert "--current-geometry-remine" in remine
    assert "encode_chemaware_checkpoint_manifest_rows.py" in remine
    assert "--official-checkpoint \"$PHASE_A_CHECKPOINT\"" in remine
    assert "--triplet-loss-margin 0.1" in remine
    print("PASS: ChemAware max-boundary native-triplet contracts")


if __name__ == "__main__":
    main()
