"""CPU contracts for the auditable layered native-triplet corpus."""
from __future__ import annotations

import numpy as np

from build_chemaware_layered_10k_native_pool import (
    LAYERED_OFFICIAL_REPLAY,
    append_pool,
    append_replay,
)
from build_chemaware_max_boundary_native_triplets import PoolWriter
from build_chemaware_sirius_native_triplets import verify_phasea_prefix


def annotated_pool() -> dict[str, np.ndarray]:
    return {
        "anchor_idx": np.asarray([0, 3]),
        "positive_ptr": np.asarray([0, 1, 2]),
        "positive_idx": np.asarray([1, 4]),
        "negative_ptr": np.asarray([0, 1, 2]),
        "negative_idx": np.asarray([2, 5]),
        "source_query": np.asarray([7, 8]),
        "negative_candidate": np.asarray([1, 2]),
        "source_tag": np.asarray([3, 4]),
        "curriculum_role": np.asarray([1, 7]),
    }


def main() -> None:
    source = annotated_pool()
    replay = {
        "anchor_idx": np.asarray([0, 6, 9]),
        "positive_ptr": np.asarray([0, 1, 2, 3]),
        "positive_idx": np.asarray([1, 7, 10]),
        "negative_ptr": np.asarray([0, 1, 2, 3]),
        "negative_idx": np.asarray([2, 8, 11]),
    }
    writer = PoolWriter()
    append_pool(source, writer)
    retained, collisions = append_replay(replay, writer, target=4, seed=1)
    output = writer.arrays()
    assert verify_phasea_prefix(source, output)
    assert retained == 2
    assert collisions in {0, 1}
    assert len(output["anchor_idx"]) == 4
    assert output["curriculum_role"][-2:].tolist() == [
        LAYERED_OFFICIAL_REPLAY, LAYERED_OFFICIAL_REPLAY,
    ]
    assert output["source_query"][-2:].tolist() == [-1, -1]
    print("PASS: ChemAware layered 10k native-pool contracts", flush=True)


if __name__ == "__main__":
    main()
