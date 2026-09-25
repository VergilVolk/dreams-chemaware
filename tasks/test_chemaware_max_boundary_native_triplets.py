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
    FrozenEmbeddings,
    PoolWriter,
    max_boundary,
)


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
        # Windows keeps an mmap-backed .npy file locked until the final array
        # owner is collected.  Release it before TemporaryDirectory cleanup.
        del boundary, cache
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
    print("PASS: ChemAware max-boundary native-triplet contracts")


if __name__ == "__main__":
    main()
